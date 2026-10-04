from typing import Literal, Optional
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from pydantic import BaseModel, ConfigDict, model_validator

from models.layers import SwiGLU, AttnType, Attention, Cache, RotaryEmbedding, find_multiple
from models.moe import MoEConfig, MoEFFN


def _res_scale_active(gamma) -> bool:
    if isinstance(gamma, Tensor):
        return True
    return float(gamma) != 1.0


class InitConfig(BaseModel):
    in_std: float

    attn_out_std: float
    ff_out_std: float


class TransformerConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Input config
    max_seq_len: int

    # Transformer config
    n_layers: int

    hidden_size: int
    num_heads: int
    expansion: float
    num_key_value_heads: Optional[int] = None

    ffn_type: Literal["dense", "moe"] = "dense"
    moe: Optional[MoEConfig] = None

    attn_type: AttnType = "prefixlm"

    init_type: Literal["fixed_normal", "lecun_normal", "megatron"]
    init_std: Optional[float] = None

    norm_type: Literal["pre", "post"]
    norm_eps: float

    # Curse-of-Depth depth factor ℓ^{-p} (folded into unified γ with cycle ε).
    layernorm_scaling: bool = False
    layernorm_scaling_power: float = 0.5
    # Layer-axis CoD denominator. Default ell: γ∝ℓ^{-p}, so L00 (ℓ=1) is the largest write.
    #   ell         — ℓ^{-p}
    #   sqrt_n      — every layer uses n_layers^{-p} (fixed √L)
    #   ell0_sqrt_n — only layer 0 uses n_layers^{-p}; later layers keep ℓ^{-p}
    #   ell0_extra  — layer 0 uses (1+extra)^{-p}; later layers keep ℓ^{-p}
    layernorm_scaling_ell_mode: str = "ell"
    layernorm_scaling_ell0_extra: float = 0.0

    # Optional extra depth factor on γ (legacy residual_depth_scale); usually off.
    residual_depth_scale: bool = False
    residual_depth_power: float = 0.5

    # Where to apply unified γ = eps(t) * ℓ^{-p}:
    #   outside: x ← x + γ f(RMS(x))   (Residual Scaling paper form; default)
    #   inside:  x ← x + f(γ RMS(x))
    # Split (LNS inside + ε outside) is removed.
    scale_placement: Literal["inside", "outside"] = "outside"

    # After each residual block, re-RMSNorm (optional; default off).
    per_layer_out_norm: bool = False

    # Peri-LN on the FFN/MoE residual *write* (before γ). none=off.
    # unit = parameter-free RMS; affine = RMS with learnable per-dim scale (Gemma/OLMo2).
    # Applied as x ← x + γ · RMS(MoE(RMS(x))); γ still owns loop/layer step size.
    ffn_branch_rms: Literal["none", "unit", "affine"] = "none"
    # Same peri-LN on the attention residual write (before γ / AttnRes mix).
    # L1 slope lives in loop-1 layer-0 attn, not the MoE write — this is the matching lever.
    attn_branch_rms: Literal["none", "unit", "affine"] = "none"
    # 0 = apply on every loop. N>0 = only t=1..N (0-based loop_idx < N).
    attn_branch_rms_loops: int = 0

    pos_emb_type: Literal["rope", "none"]
    rope_theta: Optional[float] = None

    @model_validator(mode="after")
    def validate_ffn(self) -> "TransformerConfig":
        if self.ffn_type == "moe" and self.moe is None:
            raise ValueError("moe config is required when ffn_type='moe'")
        return self

    @property
    def num_key_value_heads_resolved(self) -> int:
        return self.num_key_value_heads or self.num_heads

    # [Computed properties]
    @property
    def intermediate_size(self):
        # Automatic compute "intermediate_size" from "expansion"
        # NOTE: The formula is to match the number of GLU parameters to a vanilla Transformer with same expansion
        return find_multiple(round(self.expansion * self.hidden_size * 2 / 3), 256)

    @property
    def ff_intermediate_size(self) -> int:
        if self.ffn_type == "moe":
            assert self.moe is not None
            return self.moe.expert_intermediate_size
        return self.intermediate_size
    
    @property
    def init_config(self):
        match self.init_type:
            case "fixed_normal":
                in_std = attn_out_std = ff_out_std = self.init_std if self.init_std is not None else 0.02  # defaults to 0.02, as in OLMo 2
            case "lecun_normal":
                in_std = attn_out_std = 1.0 / math.sqrt(self.hidden_size)
                ff_out_std = 1.0 / math.sqrt(self.ff_intermediate_size)
            case "megatron":
                in_std = self.init_std if self.init_std is not None else 1.0 / math.sqrt(self.hidden_size)
                attn_out_std = ff_out_std = in_std / math.sqrt(2.0 * self.n_layers)
            case _:
                raise NotImplementedError()
            
        return InitConfig(in_std=in_std, attn_out_std=attn_out_std, ff_out_std=ff_out_std)


def build_ffn(config: TransformerConfig) -> nn.Module:
    if config.ffn_type == "moe":
        assert config.moe is not None
        return MoEFFN(
            hidden_size=config.hidden_size,
            moe_config=config.moe,
            init_std_in=config.init_config.in_std,
            init_std_out=config.init_config.ff_out_std,
        )
    return SwiGLU(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        init_std_in=config.init_config.in_std,
        init_std_out=config.init_config.ff_out_std,
    )


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.ffn_type = config.ffn_type
        self.layer_idx = layer_idx
        self.scale_placement = config.scale_placement
        # Depth part of unified γ (ℓ 1-based). L00 is ℓ=1 → no CoD shrink unless remapped.
        p = float(config.layernorm_scaling_power)
        n_layers = float(max(int(getattr(config, "n_layers", 1) or 1), 1))
        ell = float(layer_idx + 1)
        ell_mode = str(getattr(config, "layernorm_scaling_ell_mode", "ell") or "ell").lower()
        ell0_extra = float(getattr(config, "layernorm_scaling_ell0_extra", 0.0) or 0.0)
        if ell_mode in ("sqrt_n", "sqrtl", "sqrt_l"):
            ell = n_layers
        elif ell_mode in ("ell0_sqrt_n", "ell0_sqrtl") and layer_idx == 0:
            ell = n_layers
        elif ell_mode in ("ell0_extra", "ell0") and layer_idx == 0 and ell0_extra > 0.0:
            ell = 1.0 + ell0_extra
        ell_factor = 1.0
        if config.layernorm_scaling:
            ell_factor *= ell ** (-p)
        if config.residual_depth_scale:
            ell_factor *= float(layer_idx + 1) ** (-float(config.residual_depth_power))
        self.ell_factor = ell_factor
        # Loop part of γ; set per H/L call via set_cycle_residual_eps.
        self.cycle_res_eps = 1.0
        self.per_layer_out_norm = bool(config.per_layer_out_norm)
        self.norm_eps = float(config.norm_eps)
        rms_mode = str(getattr(config, "ffn_branch_rms", "none") or "none").lower()
        if rms_mode not in ("none", "unit", "affine"):
            raise ValueError(f"ffn_branch_rms={rms_mode!r}; expected none|unit|affine")
        self.ffn_branch_rms = rms_mode
        self.ffn_branch_rms_weight = (
            nn.Parameter(torch.ones(config.hidden_size)) if rms_mode == "affine" else None
        )
        attn_rms_mode = str(getattr(config, "attn_branch_rms", "none") or "none").lower()
        if attn_rms_mode not in ("none", "unit", "affine"):
            raise ValueError(f"attn_branch_rms={attn_rms_mode!r}; expected none|unit|affine")
        self.attn_branch_rms = attn_rms_mode
        self.attn_branch_rms_weight = (
            nn.Parameter(torch.ones(config.hidden_size)) if attn_rms_mode == "affine" else None
        )
        self.attn_branch_rms_loops = int(getattr(config, "attn_branch_rms_loops", 0) or 0)
        self.attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_key_value_heads_resolved,
            attn_type=config.attn_type,

            init_std_in=config.init_config.in_std,
            init_std_out=config.init_config.attn_out_std
        )
        self.mlp = build_ffn(config)

        self.forward = getattr(self, f"_forward_{config.norm_type}")  # Avoid branching logic in "forward" for torch.compile compatibility
        self.norm = lambda x: F.rms_norm(x, (x.shape[-1], ), eps=config.norm_eps)

    def _gamma(self) -> float | Tensor:
        """Unified γ = loop_factor(t) * ℓ^{-p}. Depth counted once (via ℓ or via √L in loop_factor)."""
        return self.cycle_res_eps * self.ell_factor

    def _run_mlp(self, h: Tensor, loop_idx: int = 0) -> Tensor:
        if self.ffn_type == "moe":
            return self.mlp(h, loop_idx=loop_idx)
        return self.mlp(h)

    def _ffn_branch_rms(self, y: Tensor) -> Tensor:
        """Parameter-free or affine RMS on the FFN/MoE write, before residual γ."""
        mode = getattr(self, "ffn_branch_rms", "none")
        if mode in ("", "none"):
            return y
        w = self.ffn_branch_rms_weight if mode == "affine" else None
        return F.rms_norm(y, (y.shape[-1],), weight=w, eps=self.norm_eps)

    def _attn_branch_rms(self, y: Tensor, loop_idx: int | None = None) -> Tensor:
        """Parameter-free or affine RMS on the attention write, before residual γ / AttnRes."""
        mode = getattr(self, "attn_branch_rms", "none")
        if mode in ("", "none"):
            return y
        n_lim = int(getattr(self, "attn_branch_rms_loops", 0) or 0)
        if n_lim > 0 and loop_idx is not None and int(loop_idx) >= n_lim:
            return y
        w = self.attn_branch_rms_weight if mode == "affine" else None
        return F.rms_norm(y, (y.shape[-1],), weight=w, eps=self.norm_eps)

    def _pop_moe_loop_idx(self, seq_info: dict) -> int:
        return int(seq_info.pop("moe_loop_idx", 0) or 0)

    # [Forward logic]
    def _forward_pre(self, x: Tensor, **seq_info) -> Tensor:
        # AttnRes path must run inside this FSDP-wrapped forward (not via layer.attn).
        bank = seq_info.pop("attn_res_bank", None)
        if bank is not None:
            return self._forward_pre_attn_res(bank, **seq_info)
        dual = seq_info.pop("dual_axis_carry", None)
        if dual is not None:
            return self._forward_pre_dual_axis(x, dual, **seq_info)
        moe_loop_idx = self._pop_moe_loop_idx(seq_info)
        # Unified γ: outside → x + γ f(RMS(x)); inside → x + f(γ RMS(x)).
        gamma = self._gamma()
        h = self.norm(x)
        if self.scale_placement == "inside" and _res_scale_active(gamma):
            h = h * gamma
        attn_out = self.attn(h, **seq_info)
        attn_out = self._attn_branch_rms(attn_out, loop_idx=moe_loop_idx)
        if self.scale_placement == "outside" and _res_scale_active(gamma):
            attn_out = attn_out * gamma
        x = x + attn_out
        self._diag_record("attn", x)

        h = self.norm(x)
        if self.scale_placement == "inside" and _res_scale_active(gamma):
            h = h * gamma
        mlp_out = self._run_mlp(h, moe_loop_idx)
        mlp_out = self._ffn_branch_rms(mlp_out)
        if self.scale_placement == "outside" and _res_scale_active(gamma):
            mlp_out = mlp_out * gamma
        x = x + mlp_out
        self._diag_record("moe" if self.ffn_type == "moe" else "mlp", x)
        if self.per_layer_out_norm:
            x = self.norm(x)
        return x

    def _forward_pre_dual_axis(self, x: Tensor, dual, **seq_info) -> Tensor:
        """N/D dual-axis residual; apply attn/mlp/both via dual.apply_*."""
        moe_loop_idx = self._pop_moe_loop_idx(seq_info)
        gamma = self._gamma()
        h = self.norm(x)
        if self.scale_placement == "inside" and _res_scale_active(gamma):
            h = h * gamma
        attn_out = self.attn(h, **seq_info)
        attn_out = self._attn_branch_rms(attn_out, loop_idx=moe_loop_idx)
        if self.scale_placement == "outside" and _res_scale_active(gamma):
            attn_out = attn_out * gamma
        if dual.apply_attn:
            upd = x if dual.update_src == "x" else attn_out
            r = dual.mix_attn(self.layer_idx, upd)
            x = dual.combine_residual(x, attn_out, r)
        else:
            x = x + attn_out
        self._diag_record("attn", x)

        h = self.norm(x)
        if self.scale_placement == "inside" and _res_scale_active(gamma):
            h = h * gamma
        mlp_out = self._run_mlp(h, moe_loop_idx)
        mlp_out = self._ffn_branch_rms(mlp_out)
        if self.scale_placement == "outside" and _res_scale_active(gamma):
            mlp_out = mlp_out * gamma
        if dual.apply_mlp:
            upd = x if dual.update_src == "x" else mlp_out
            r = dual.mix_mlp(self.layer_idx, upd)
            x = dual.combine_residual(x, mlp_out, r)
        else:
            x = x + mlp_out
        self._diag_record("moe" if self.ffn_type == "moe" else "mlp", x)
        if self.per_layer_out_norm:
            x = self.norm(x)
        return x

    def _forward_pre_attn_res(self, bank, **seq_info) -> Tensor:
        """Full AttnRes branches (attn then mlp) — paper treats each as a depth step."""
        moe_loop_idx = self._pop_moe_loop_idx(seq_info)
        gamma = self._gamma()
        # Attention branch
        h = bank.mix(next_layer=False)
        hn = self.norm(h)
        if self.scale_placement == "inside" and _res_scale_active(gamma):
            hn = hn * gamma
        attn_out = self.attn(hn, **seq_info)
        attn_out = self._attn_branch_rms(attn_out, loop_idx=moe_loop_idx)
        if self.scale_placement == "outside" and _res_scale_active(gamma):
            attn_out = attn_out * gamma
        bank.append(attn_out)
        self._diag_record("attn", bank.mix(next_layer=False))

        # MLP branch
        h = bank.mix(next_layer=False)
        hn = self.norm(h)
        if self.scale_placement == "inside" and _res_scale_active(gamma):
            hn = hn * gamma
        mlp_out = self._run_mlp(hn, moe_loop_idx)
        mlp_out = self._ffn_branch_rms(mlp_out)
        if self.scale_placement == "outside" and _res_scale_active(gamma):
            mlp_out = mlp_out * gamma
        bank.append(mlp_out)
        x_act = bank.mix(next_layer=False)
        if self.per_layer_out_norm:
            x_act = self.norm(x_act)
        self._diag_record("moe" if self.ffn_type == "moe" else "mlp", x_act)
        return x_act

    def _forward_post(self, x: Tensor, **seq_info) -> Tensor:
        moe_loop_idx = self._pop_moe_loop_idx(seq_info)
        gamma = self._gamma()
        attn_in = x * gamma if (self.scale_placement == "inside" and _res_scale_active(gamma)) else x
        attn_out = self.attn(attn_in, **seq_info)
        attn_out = self._attn_branch_rms(attn_out, loop_idx=moe_loop_idx)
        if self.scale_placement == "outside" and _res_scale_active(gamma):
            attn_out = attn_out * gamma
        x = self.norm(x + attn_out)
        self._diag_record("attn", x)

        mlp_in = x * gamma if (self.scale_placement == "inside" and _res_scale_active(gamma)) else x
        mlp_out = self._run_mlp(mlp_in, moe_loop_idx)
        mlp_out = self._ffn_branch_rms(mlp_out)
        if self.scale_placement == "outside" and _res_scale_active(gamma):
            mlp_out = mlp_out * gamma
        x = self.norm(x + mlp_out)
        self._diag_record("moe" if self.ffn_type == "moe" else "mlp", x)
        return x

    def _diag_record(self, stage: str, tensor: Tensor) -> None:
        # Optional module diagnostics (utils.module_diag); no-op when idle.
        try:
            from utils.module_diag import get_active_collector
        except Exception:
            return
        collector = get_active_collector()
        if collector is not None:
            collector.record_activation(self, stage, tensor)


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.head_hint = {"in":  {"dim": config.hidden_size, "init_std": config.init_config.in_std},
                          "out": {"dim": config.hidden_size, "init_std": config.init_config.in_std}}  # Hint for LMHead init

        # Position embeddings
        if config.pos_emb_type == "rope":
            assert config.rope_theta is not None
            self.rotary_emb = RotaryEmbedding(config.hidden_size // config.num_heads, config.max_seq_len, base=config.rope_theta)

        # Layers (pass layer_idx for optional LayerNorm Scaling)
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx=_layer_idx) for _layer_idx in range(config.n_layers)]
        )

        # Use final norm only for prenorm
        self.norm_f = lambda x: x
        if config.norm_type == "pre":
            self.norm_f = lambda x: F.rms_norm(x, (x.shape[-1], ), eps=config.norm_eps)

        # Create cache function
        self.create_cache = lambda **kwargs: [
            Cache.create(
                **kwargs,
                num_heads=config.num_key_value_heads_resolved,
                head_dim=config.hidden_size // config.num_heads,
            )
            for _i in range(config.n_layers)
        ]

    def set_cycle_residual_eps(self, eps: float | Tensor) -> None:
        """Set loop factor eps(t) on every block; γ = eps(t) * ℓ^{-p}.

        If eps is a length-n_layers vector (per-layer λ), assign one value per block.
        """
        if isinstance(eps, Tensor) and eps.numel() > 1:
            if int(eps.numel()) != len(self.layers):
                raise ValueError(
                    f"per-layer cycle eps has {int(eps.numel())} entries, "
                    f"but model has {len(self.layers)} layers"
                )
            for i, layer in enumerate(self.layers):
                layer.cycle_res_eps = eps[i]
            return
        v = eps if isinstance(eps, Tensor) else float(eps)
        for layer in self.layers:
            layer.cycle_res_eps = v

    def forward(self, x: Tensor, cache: Optional[list[Cache]] = None, **seq_info) -> Tensor:
        seq_info["cos_sin"] = self.rotary_emb(seq_info.pop("position_ids", None)) if hasattr(self, "rotary_emb") else None
        # Fully Looped Architecture residual (arXiv:2605.18797 FLTres ablation):
        # distribute previous-loop state to every layer (parameter-free add).
        fla_prev = seq_info.pop("fla_prev", None)
        fla_gamma = float(seq_info.pop("fla_gamma", 0.0) or 0.0)
        # add: x←x+γ·prev (FLTres; can explode); mix: x←(1−γ)x+γ·prev (bounded)
        fla_mode = str(seq_info.pop("fla_mode", "add") or "add")
        # Paper FLT first-layer Input Injection analog: K/V from x_emb on layer 0.
        attn_first_kv = seq_info.pop("attn_first_kv_states", None)
        # Attention Injection Q source (paper FLA). Pop so we can optionally
        # restrict to layer 0 only (Input-AI) without affecting deeper layers.
        attn_query = seq_info.pop("attn_query_states", None)
        attn_query_first_only = bool(seq_info.pop("attn_query_first_layer_only", False))
        # Attention Residuals bank (arXiv:2603.15031). None → standard residual.
        attn_res_bank = seq_info.pop("attn_res_bank", None)
        # Dual-axis recurrent residual (fixed-size H/L carries; no bank).
        dual_axis_carry = seq_info.pop("dual_axis_carry", None)

        if attn_res_bank is not None:
            return self._forward_attn_res(
                x,
                attn_res_bank,
                cache=cache,
                fla_prev=fla_prev,
                fla_gamma=fla_gamma,
                fla_mode=fla_mode,
                attn_first_kv=attn_first_kv,
                attn_query=attn_query,
                attn_query_first_only=attn_query_first_only,
                seq_info=seq_info,
            )

        # Forward layers (standard residual or dual-axis carry)
        for layer_id, layer in enumerate(self.layers):
            if fla_prev is not None and fla_gamma != 0.0:
                if fla_mode == "mix":
                    g = min(max(fla_gamma, 0.0), 1.0)
                    x = (1.0 - g) * x + g * fla_prev
                else:
                    x = x + fla_gamma * fla_prev
            x = self._maybe_inject_loop_id_layer(x, layer_id)
            layer_info = seq_info
            extras = {}
            if attn_first_kv is not None and layer_id == 0:
                extras["attn_kv_states"] = attn_first_kv
            if attn_query is not None and (not attn_query_first_only or layer_id == 0):
                extras["attn_query_states"] = attn_query
            if dual_axis_carry is not None:
                extras["dual_axis_carry"] = dual_axis_carry
            if extras:
                layer_info = {**seq_info, **extras}
            x = layer(x, **layer_info, cache=cache[layer_id] if cache is not None else None)

        return self.norm_f(x)

    def _maybe_inject_loop_id_layer(self, x: Tensor, layer_id: int) -> Tensor:
        """Hook for LoopedTransformer add_layer (H×L) loop-ID; default no-op."""
        return x

    def _forward_attn_res(
        self,
        x: Tensor,
        bank,
        *,
        cache: Optional[list[Cache]],
        fla_prev,
        fla_gamma: float,
        fla_mode: str,
        attn_first_kv,
        attn_query,
        attn_query_first_only: bool,
        seq_info: dict,
    ) -> Tensor:
        """Full AttnRes: call each FSDP-wrapped block (do not touch layer.attn directly)."""
        if fla_prev is not None and fla_gamma != 0.0:
            if fla_mode == "mix":
                g = min(max(fla_gamma, 0.0), 1.0)
                x = (1.0 - g) * x + g * fla_prev
            else:
                x = x + fla_gamma * fla_prev
        # Module input becomes v_0 (or an additional bridge value if bank continues).
        bank.append(x)
        for layer_id, layer in enumerate(self.layers):
            x = self._maybe_inject_loop_id_layer(x, layer_id)
            layer_info = {**seq_info, "attn_res_bank": bank}
            if attn_first_kv is not None and layer_id == 0:
                layer_info["attn_kv_states"] = attn_first_kv
            if attn_query is not None and (not attn_query_first_only or layer_id == 0):
                layer_info["attn_query_states"] = attn_query
            c = cache[layer_id] if cache is not None else None
            # Must go through layer(...) so FSDP2 root hooks run.
            x = layer(x, **layer_info, cache=c)
        return self.norm_f(bank.mix(next_layer=True))

    def update_moe_load_balance_biases(self) -> None:
        """Auxiliary expert-bias update (DeepSeek-style); no-op if MoE unused."""
        import torch.distributed as dist

        for module in self.modules():
            if not isinstance(module, MoEFFN):
                continue
            if module.load_balance_bias_lr <= 0:
                continue
            load = module.pop_expert_load()
            if load is None:
                continue
            if dist.is_initialized():
                dist.all_reduce(load, op=dist.ReduceOp.SUM)
            module.update_expert_bias_from_load(load)

    def discard_moe_load_balance_stats(self) -> None:
        """Drop per-step expert load without touching expert_bias (skipped spikes)."""
        for module in self.modules():
            if isinstance(module, MoEFFN):
                module.pop_expert_load()
