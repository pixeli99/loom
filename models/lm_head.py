from typing import Tuple

import torch
from torch import nn
from torch import Tensor
import torch.distributed as dist
import torch.nn.functional as F
from pydantic import BaseModel

from models.layers import LinearInit, ScaledEmbeddingInit, Carry
from models.common import IGNORE_LABEL_ID, packing_sequence_sum


class LMHeadConfig(BaseModel):
    vocab_size: int


class LMHead(nn.Module):
    def __init__(self, model: nn.Module, config_dict: dict) -> None:
        super().__init__()
        self.model = model
        # Create cache function
        self.create_cache = self.model.create_cache
        # Train extra args function
        self.compute_train_extra_args = self.model.compute_train_extra_args

        config = LMHeadConfig(**config_dict)
        head_hint: dict = self.model.head_hint  # pyright: ignore[reportAssignmentType]

        # LMHead input and output
        self.embed_tokens = ScaledEmbeddingInit(config.vocab_size, head_hint["in"]["dim"], init_std=head_hint["in"]["init_std"])  # pyright: ignore[reportArgumentType]
        self.lm_head = LinearInit(head_hint["out"]["dim"], config.vocab_size, bias=False, init_std=head_hint["out"]["init_std"])  # pyright: ignore[reportArgumentType]

    def ce_from_logits(self, logits: Tensor, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, tuple[Tensor, Tensor]]]:
        """CE + metrics from hidden-state logits (used by full forward and segmented BP)."""
        labels = batch["labels"]
        masks = labels != IGNORE_LABEL_ID
        loss = F.cross_entropy(logits.to(torch.float32), labels.to(torch.long), ignore_index=IGNORE_LABEL_ID, reduction="sum")
        loss_divisor = masks.sum().to(torch.float32)
        dist.all_reduce(loss_divisor, op=dist.ReduceOp.AVG)
        normalized_loss = loss / loss_divisor
        with torch.no_grad():
            is_correct = torch.argmax(logits, dim=-1) == labels
            local_valid_counts = masks.sum()
            seq_num_tokens_correct = packing_sequence_sum(is_correct, batch["cu_seqlens"])
            seq_num_valid_tokens = packing_sequence_sum(masks, batch["cu_seqlens"])
            seq_is_valid = seq_num_valid_tokens > 0
            metrics = {
                "loss": (normalized_loss.detach() * local_valid_counts, local_valid_counts),
                "accuracy": (is_correct.sum(), local_valid_counts),
                "exact_accuracy": (((seq_num_tokens_correct == seq_num_valid_tokens) & seq_is_valid).sum(), seq_is_valid.sum()),
            }
        return normalized_loss, metrics

    def forward(self, carry: Carry, batch: dict[str, Tensor], **kwargs) -> Tuple[Carry, Tensor] | Tuple[Carry, Tensor, dict[str, Tuple[Tensor, Tensor]]]:
        # Token embedding
        input_embedding = self.embed_tokens(batch["inputs"])

        # Model forward
        new_carry, logits = self.model(carry,
                                       input_embedding,
                                       **{k: v for k, v in batch.items() if k not in ("inputs", "labels")},
                                       **kwargs)
        logits = self.lm_head(logits)

        # Loss & Metrics
        if "labels" in batch:
            # Masks & labels
            labels = batch["labels"]
            masks = labels != IGNORE_LABEL_ID

            # Loss (CE in F32)
            loss = F.cross_entropy(logits.to(torch.float32), labels.to(torch.long), ignore_index=IGNORE_LABEL_ID, reduction="sum")
            # AllReduce loss divisor. Divide by mean of valid tokens across all processes, as gradient will be averaged.
            loss_divisor = masks.sum().to(torch.float32)
            dist.all_reduce(loss_divisor, op=dist.ReduceOp.AVG)

            router_z_coef = getattr(self.model, "router_z_loss_coef", 0.0)
            z_loss = None
            if router_z_coef > 0 and hasattr(self.model, "pop_router_z_loss"):
                z_loss = self.model.pop_router_z_loss()

            normalized_loss = loss / loss_divisor
            if z_loss is not None:
                normalized_loss = normalized_loss + router_z_coef * z_loss

            # Deep supervision: mean CE over intermediate H-cycle states.
            deep_sup_coef = float(getattr(self.model, "deep_sup_coef", 0.0) or 0.0)
            deep_sup_loss = None
            if (
                deep_sup_coef > 0.0
                and self.training
                and hasattr(self.model, "pop_deep_sup_states")
            ):
                inter = self.model.pop_deep_sup_states()
                if inter:
                    aux_sum = None
                    for h in inter:
                        aux_logits = self.lm_head(h)
                        aux_i = F.cross_entropy(
                            aux_logits.to(torch.float32),
                            labels.to(torch.long),
                            ignore_index=IGNORE_LABEL_ID,
                            reduction="sum",
                        )
                        aux_sum = aux_i if aux_sum is None else (aux_sum + aux_i)
                    deep_sup_loss = (aux_sum / float(len(inter))) / loss_divisor
                    normalized_loss = normalized_loss + deep_sup_coef * deep_sup_loss
            elif hasattr(self.model, "pop_deep_sup_states"):
                # Eval / coef=0: drop any stashed states so they don't leak.
                self.model.pop_deep_sup_states()

            # Scale-visible norm penalty (arXiv:2606.24898): λ · mean_k E[‖Hk‖²_rms].
            # ‖v‖²_rms = mean_d(v²); keeps RMSNorm readout but makes scale visible to loss.
            norm_pen_coef = float(getattr(self.model, "state_norm_penalty_coef", 0.0) or 0.0)
            norm_pen_loss = None
            if (
                norm_pen_coef > 0.0
                and self.training
                and hasattr(self.model, "pop_norm_pen_states")
            ):
                states = self.model.pop_norm_pen_states()
                if states:
                    rms2_sum = None
                    for h in states:
                        # E[‖h‖²_rms] over tokens; h is [N, D] packed.
                        rms2 = h.to(torch.float32).pow(2).mean()
                        rms2_sum = rms2 if rms2_sum is None else (rms2_sum + rms2)
                    norm_pen_loss = rms2_sum / float(len(states))
                    normalized_loss = normalized_loss + norm_pen_coef * norm_pen_loss
            elif hasattr(self.model, "pop_norm_pen_states"):
                self.model.pop_norm_pen_states()

            # Loop-Δz RMS match (CoD / residual-scaling): unitless update-norm balance.
            # Final-z RMS is LN-flat (rms2 saw match=0); use RMS(z_t−z_{t−1}) instead.
            # Full-H: L = (rms_ΔH / rms_Δ2 − 1)². Plus stream-gain:
            #   λ* = clamp(λ · ratio, λ_min, λ_max); L += (λ − stopgrad(λ*))²
            # Plus depth prior (fixed-λ BEST / transfer point ≈1.6), gated by |ratio−1|
            # so it only pulls when the stream is imbalanced — not pack-specific LR.
            rms_match = bool(getattr(self.model, "cycle_lambda_rms_match", False))
            rms_match_loss = None
            stream_gain_loss = None
            depth_prior_loss = None
            if rms_match and self.training:
                ref, delta_last = (None, None)
                if hasattr(self.model, "pop_lambda_rms_full_h"):
                    ref, delta_last = self.model.pop_lambda_rms_full_h()
                if ref is not None and delta_last is not None:
                    rms_last = (
                        delta_last.to(torch.float32).pow(2).mean().clamp_min(1e-12).sqrt()
                    )
                    ratio = rms_last / ref
                    rms_match_loss = (ratio - 1.0).pow(2)
                    normalized_loss = normalized_loss + rms_match_loss
                    get_lam = getattr(self.model, "_cycle_lambda_res_live", None)
                    if not callable(get_lam):
                        get_lam = getattr(self.model, "_cycle_lambda_res", None)
                    if callable(get_lam):
                        lam = get_lam()
                        if isinstance(lam, torch.Tensor):
                            lam_f = lam.to(torch.float32).reshape(())
                            lam_min = float(
                                getattr(self.model, "cycle_scale_lambda_min", 1.5) or 1.5
                            )
                            lam_max = float(
                                getattr(self.model, "cycle_scale_lambda_max", 0.0) or 0.0
                            )
                            if lam_max <= 0.0:
                                lam_max = lam_min + 2.0
                            # Operating band (residual-scaling BEST ≈1.6–1.7).
                            lo = float(
                                getattr(self.model, "cycle_lambda_depth_prior_lo", 1.58)
                                or 1.58
                            )
                            hi = float(
                                getattr(self.model, "cycle_lambda_depth_prior_hi", 1.75)
                                or 1.75
                            )
                            # Match residual H-aware hi so prior/stream gates align with detach.
                            H = float(
                                max(int(getattr(self.model, "num_loops", 1) or 1), 1)
                            )
                            hi = min(hi, 1.65 + 0.10 * (9.0 / H) ** 0.5)
                            mid = 0.5 * (lo + hi)  # ≈1.65
                            # Climb zone [lo, mid): keep stream-gain so init15 can rise
                            # (rms7 froze at lo→λ≈1.58→eval 7.2).
                            # rms7b: gate stream in [mid,hi]; rms8: also gate when λ>hi
                            # (CE+stream yank caused soft15→1.80 / soft20→2.17 rebound).
                            mode = str(
                                getattr(self.model, "cycle_lambda_detach_mode", "rms7b")
                                or "rms7b"
                            )
                            if mode == "rms8":
                                stream_off = lam_f >= mid
                            elif mode == "rms9":
                                latched = getattr(
                                    self.model, "_cycle_lambda_settled_latched", None
                                )
                                latched_b = bool(
                                    latched.detach().item()
                                    if isinstance(latched, torch.Tensor)
                                    else bool(latched)
                                )
                                stream_off = latched_b or ((lam_f >= mid) & (lam_f <= hi))
                            elif mode == "rms10":
                                # Stream-gain off at/above hi (ceiling); keep below mid live.
                                stream_off = lam_f >= hi
                            elif mode == "rms11":
                                # Settled band OR above hi (ceiling zone): no stream yank.
                                stream_off = (lam_f >= mid) | (lam_f >= hi)
                            elif mode == "rms12":
                                # STE ceiling: keep stream/prior ON when λ>hi (pull down);
                                # gate only in settled [mid,hi].
                                stream_off = (lam_f >= mid) & (lam_f <= hi)
                            else:
                                stream_off = (lam_f >= mid) & (lam_f <= hi)
                            ratio_d = ratio.detach()
                            lam_star = (lam_f.detach() * ratio_d).clamp(lam_min, lam_max)
                            stream_raw = (lam_f - lam_star).pow(2)
                            stream_gain_loss = torch.where(
                                stream_off, lam_f.new_zeros(()), stream_raw
                            )
                            normalized_loss = normalized_loss + stream_gain_loss
                            lam_ref = float(
                                getattr(self.model, "cycle_lambda_depth_prior", 1.6) or 0.0
                            )
                            if lam_ref > 0.0:
                                # Asymmetric operating band:
                                #   λ > hi → (λ−ref)²
                                #   λ < mid → (λ−mid)²  (climb through [lo,mid) too)
                                #   mid≤λ≤hi → 0         (settled; stream gated above)
                                coef = float(
                                    getattr(self.model, "cycle_lambda_depth_prior_coef", 1.0)
                                    or 1.0
                                )
                                high_pen = torch.where(
                                    lam_f > hi, (lam_f - lam_ref).pow(2), lam_f.new_zeros(())
                                )
                                low_pen = torch.where(
                                    lam_f < mid, (lam_f - mid).pow(2), lam_f.new_zeros(())
                                )
                                depth_prior_loss = coef * (high_pen + low_pen)
                                normalized_loss = normalized_loss + depth_prior_loss
                elif hasattr(self.model, "pop_lambda_rms_states"):
                    # Legacy fallback (non-segmented / short H).
                    states = self.model.pop_lambda_rms_states()
                    if len(states) >= 2:
                        rms_list = [
                            h.to(torch.float32).pow(2).mean().clamp_min(1e-12).sqrt()
                            for h in states
                        ]
                        ref0 = rms_list[0].detach()
                        acc = None
                        for r in rms_list[1:]:
                            term = (r / ref0 - 1.0).pow(2)
                            acc = term if acc is None else (acc + term)
                        rms_match_loss = acc / float(len(rms_list) - 1)
                        normalized_loss = normalized_loss + rms_match_loss
            elif hasattr(self.model, "pop_lambda_rms_states"):
                self.model.pop_lambda_rms_states()


            # Accuracy
            with torch.no_grad():
                is_correct = torch.argmax(logits, dim=-1) == labels
                local_valid_counts = masks.sum()
                # Sequence-level statistics
                seq_num_tokens_correct = packing_sequence_sum(is_correct, batch["cu_seqlens"])
                seq_num_valid_tokens = packing_sequence_sum(masks, batch["cu_seqlens"])
                seq_is_valid = seq_num_valid_tokens > 0
                # Metrics
                metrics = {
                    "loss": (normalized_loss.detach() * local_valid_counts, local_valid_counts),
                    "accuracy": (is_correct.sum(), local_valid_counts),
                    "exact_accuracy": (((seq_num_tokens_correct == seq_num_valid_tokens) & seq_is_valid).sum(), seq_is_valid.sum()),
                }
                if z_loss is not None:
                    metrics["router_z_loss"] = (z_loss.detach(), torch.ones((), device=z_loss.device, dtype=torch.float32))
                if deep_sup_loss is not None:
                    metrics["deep_sup_loss"] = (
                        deep_sup_loss.detach(),
                        torch.ones((), device=deep_sup_loss.device, dtype=torch.float32),
                    )
                if norm_pen_loss is not None:
                    metrics["state_norm_penalty"] = (
                        norm_pen_loss.detach(),
                        torch.ones((), device=norm_pen_loss.device, dtype=torch.float32),
                    )
                if rms_match_loss is not None:
                    metrics["lambda_rms_match"] = (
                        rms_match_loss.detach(),
                        torch.ones((), device=rms_match_loss.device, dtype=torch.float32),
                    )
                if stream_gain_loss is not None:
                    metrics["lambda_stream_gain"] = (
                        stream_gain_loss.detach(),
                        torch.ones((), device=stream_gain_loss.device, dtype=torch.float32),
                    )
                if depth_prior_loss is not None:
                    metrics["lambda_depth_prior"] = (
                        depth_prior_loss.detach(),
                        torch.ones((), device=depth_prior_loss.device, dtype=torch.float32),
                    )

            return new_carry, normalized_loss, metrics

        return new_carry, logits
