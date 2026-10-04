import os
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from pydantic import BaseModel, ConfigDict, model_validator

from models.common import trunc_normal_init_
from models.layers import SwiGLU


class MoEConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    expert_granularity: int = 8
    coarse_top_k: int = 1

    # Total experts = num_routed + num_shared when shared enabled (e32 = 30+2).
    num_routed_experts: int = 30
    top_k: int = 8

    shared_expert: bool = True
    # Defaults match T_moe_L5 / e32k8s2: 30 routed + 2 always-on shared (total 32),
    # top_k=8 includes shared (routed_top_k=6), slim shared width (= expert width).
    num_shared_experts: int = 2
    # When True, top_k counts shared slots (routed_top_k = top_k - num_shared_experts).
    top_k_includes_shared: bool = True
    shared_expert_size_multiplier: int = 1

    expert_intermediate_size: int

    gate_type: Literal["sigmoid"] = "sigmoid"
    router_dtype: Literal["float32"] = "float32"
    load_balance_bias_lr: float = 1e-3
    router_z_loss_coef: float = 1e-4
    capacity_factor: float = 1.5
    # When True, each loop iteration uses a dedicated router (experts/shared weights tied).
    router_per_loop: bool = False
    # When True, shared router + per-loop additive bias on logits (lightweight routing diversity).
    router_loop_bias: bool = False
    router_num_loops: int = 1

    @model_validator(mode="after")
    def validate_routing(self) -> "MoEConfig":
        if self.top_k <= 0:
            raise ValueError("moe.top_k must be positive")
        if self.num_routed_experts <= 0:
            raise ValueError("moe.num_routed_experts must be positive")
        n_shared = self.num_shared_experts if self.shared_expert else 0
        if self.shared_expert and n_shared <= 0:
            raise ValueError("moe.num_shared_experts must be positive when shared_expert is enabled")
        routed_top_k = (
            self.top_k - n_shared if self.top_k_includes_shared else self.top_k
        )
        if routed_top_k <= 0:
            raise ValueError("routed top_k must be positive after reserving shared experts")
        if routed_top_k > self.num_routed_experts:
            raise ValueError("routed top_k cannot exceed num_routed_experts")
        if self.shared_expert and self.shared_expert_size_multiplier <= 0:
            raise ValueError("moe.shared_expert_size_multiplier must be positive")
        if self.capacity_factor <= 0:
            raise ValueError("moe.capacity_factor must be positive")
        if self.router_per_loop and self.router_num_loops <= 0:
            raise ValueError("moe.router_num_loops must be positive when router_per_loop is enabled")
        if self.router_loop_bias and self.router_num_loops <= 0:
            raise ValueError("moe.router_num_loops must be positive when router_loop_bias is enabled")
        if self.router_per_loop and self.router_loop_bias:
            raise ValueError("moe.router_per_loop and moe.router_loop_bias are mutually exclusive")
        return self

    @property
    def routed_top_k(self) -> int:
        n_shared = self.num_shared_experts if self.shared_expert else 0
        return self.top_k - n_shared if self.top_k_includes_shared else self.top_k


def _moe_backend() -> str:
    return os.environ.get("MOE_BACKEND", "npu_fused").lower()


def _npu_grouped_matmul_available() -> bool:
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available() and hasattr(torch_npu, "npu_grouped_matmul")
    except (ImportError, AttributeError):
        return False


def _npu_fused_moe_available() -> bool:
    """Whether to use the packed [E, ...] expert weights + grouped GEMM path.

    Kept on for CUDA as well: the weight layout (and therefore the
    checkpoint keys) must match the Ascend runs, and the grouped GEMM has a
    portable CUDA implementation below.
    """
    if _moe_backend() in ("loop", "python", "legacy"):
        return False
    if _npu_grouped_matmul_available():
        return True
    return torch.cuda.is_available()


def _npu_moe_routing_available() -> bool:
    try:
        import torch_npu  # noqa: F401

        return (
            torch.npu.is_available()
            and hasattr(torch_npu, "npu_moe_token_permute")
            and hasattr(torch_npu, "npu_moe_token_unpermute")
        )
    except (ImportError, AttributeError):
        return False


def _use_npu_routing() -> bool:
    routing = os.environ.get("MOE_ROUTING", "npu").lower()
    if routing in ("python", "legacy", "loop"):
        return False
    return _npu_moe_routing_available()


def _expert_assignment_counts(top_indices: Tensor, num_experts: int) -> Tensor:
    return torch.bincount(top_indices.reshape(-1), minlength=num_experts)


_CAPACITY_PROFILE = {"fast": 0, "slow": 0}


def pop_capacity_profile(reset: bool = True) -> dict[str, int]:
    stats = dict(_CAPACITY_PROFILE)
    if reset:
        _CAPACITY_PROFILE["fast"] = 0
        _CAPACITY_PROFILE["slow"] = 0
    return stats

def _active_expert_load_counts(top_indices: Tensor, top_scores: Tensor, num_experts: int) -> Tensor:
    active = top_scores > 0
    if not active.any():
        return torch.zeros(num_experts, device=top_indices.device, dtype=torch.long)
    return torch.bincount(top_indices[active].reshape(-1), minlength=num_experts)


def _apply_capacity_loop(
    top_scores: Tensor,
    top_indices: Tensor,
    expert_capacity: int,
    flat_experts: Tensor,
    counts: Tensor,
) -> Tensor:
    """Legacy per-overflow-expert topk (reference implementation)."""
    num_tokens, top_k = top_indices.shape
    kept_scores = top_scores.clone()
    flat_scores = kept_scores.reshape(-1)
    overflow_experts = (counts > expert_capacity).nonzero(as_tuple=True)[0]
    for i in range(overflow_experts.numel()):
        expert_id = overflow_experts[i]
        slot_idx = (flat_experts == expert_id).nonzero(as_tuple=True)[0]
        saved_scores = flat_scores[slot_idx]
        keep_rank = saved_scores.topk(expert_capacity, largest=True).indices
        flat_scores[slot_idx] = 0
        flat_scores[slot_idx[keep_rank]] = saved_scores[keep_rank]
    kept_scores = flat_scores.reshape(num_tokens, top_k)
    row_sum = kept_scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return kept_scores / row_sum


def _capacity_keep_by_order(
    flat_scores: Tensor,
    flat_experts_f: Tensor,
    order: Tensor,
    num_tokens: int,
    top_k: int,
    expert_capacity: int,
) -> Tensor:
    n = flat_scores.numel()
    sorted_experts = flat_experts_f[order]
    idx_all = torch.arange(n, device=flat_scores.device)
    same_as_prev = torch.zeros(n, dtype=torch.bool, device=flat_scores.device)
    same_as_prev[1:] = sorted_experts[1:] == sorted_experts[:-1]
    # Boundary starts: cummax of (i+1) at group heads, else 0 → start index.
    marker = torch.where(same_as_prev, torch.zeros_like(idx_all), idx_all + 1)
    start_of = torch.cummax(marker, dim=0).values - 1
    rank = idx_all - start_of
    keep = torch.zeros(n, dtype=torch.bool, device=flat_scores.device)
    keep[order] = rank < expert_capacity
    kept_scores = torch.where(keep, flat_scores, torch.zeros_like(flat_scores)).reshape(
        num_tokens, top_k
    )
    row_sum = kept_scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return kept_scores / row_sum


def _apply_capacity_sort(
    top_scores: Tensor,
    top_indices: Tensor,
    expert_capacity: int,
    flat_experts: Tensor,
) -> Tensor:
    """Two-pass argsort (score desc, then expert). Available via MOE_CAPACITY_IMPL=sort."""
    num_tokens, top_k = top_indices.shape
    flat_scores = top_scores.reshape(-1).contiguous()
    flat_experts_f = flat_experts.to(torch.float32)
    order = torch.argsort(flat_scores, descending=True, stable=True)
    order = order[torch.argsort(flat_experts_f[order], stable=True)]
    return _capacity_keep_by_order(
        flat_scores, flat_experts_f, order, num_tokens, top_k, expert_capacity,
    )


def _apply_capacity_composite(
    top_scores: Tensor,
    top_indices: Tensor,
    expert_capacity: int,
    flat_experts: Tensor,
) -> Tensor:
    """Single argsort with key=expert*2-score (scores in [0,1]); same keep set as sort."""
    num_tokens, top_k = top_indices.shape
    flat_scores = top_scores.reshape(-1).contiguous()
    flat_experts_f = flat_experts.to(torch.float32)
    order = torch.argsort(flat_experts_f * 2.0 - flat_scores, stable=True)
    return _capacity_keep_by_order(
        flat_scores, flat_experts_f, order, num_tokens, top_k, expert_capacity,
    )


def _apply_capacity(
    top_scores: Tensor,
    top_indices: Tensor,
    capacity_factor: float,
    num_experts: int,
    assignment_counts: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Drop lowest-scoring expert slots beyond per-expert capacity; renormalize rows.

    Returns (top_scores, load_counts) where load_counts tracks active routed slots.
    """
    num_tokens, top_k = top_indices.shape
    expert_capacity = max(1, int(capacity_factor * num_tokens * top_k / num_experts))

    flat_experts = top_indices.reshape(-1)
    counts = (
        assignment_counts
        if assignment_counts is not None
        else _expert_assignment_counts(top_indices, num_experts)
    )
    if counts.max() <= expert_capacity:
        if os.environ.get("MOE_CAPACITY_PROFILE", "0") == "1":
            _CAPACITY_PROFILE["fast"] += 1
        # Return a separate tensor so later load aggregation cannot mutate
        # assignment_counts saved for GMM backward.
        return top_scores, counts.clone()

    if os.environ.get("MOE_CAPACITY_PROFILE", "0") == "1":
        _CAPACITY_PROFILE["slow"] += 1

    # Ascend has no score-based capacity (drop_pad is first-N). Default: composite.
    impl = os.environ.get("MOE_CAPACITY_IMPL", "composite").lower()
    if impl in ("loop", "legacy", "python"):
        kept_scores = _apply_capacity_loop(
            top_scores, top_indices, expert_capacity, flat_experts, counts,
        )
    elif impl == "sort":
        kept_scores = _apply_capacity_sort(
            top_scores, top_indices, expert_capacity, flat_experts,
        )
    elif impl == "adaptive":
        n_overflow = int((counts > expert_capacity).sum().item())
        if n_overflow <= 4:
            kept_scores = _apply_capacity_loop(
                top_scores, top_indices, expert_capacity, flat_experts, counts,
            )
        else:
            kept_scores = _apply_capacity_composite(
                top_scores, top_indices, expert_capacity, flat_experts,
            )
    else:
        kept_scores = _apply_capacity_composite(
            top_scores, top_indices, expert_capacity, flat_experts,
        )
    # After score-topk capacity, each expert keeps exactly min(count, capacity)
    # slots with positive scores (router sigmoid scores are >0). Avoids a second
    # active bincount (~30-40% of capacity time on NPU).
    load_counts = torch.minimum(
        counts, torch.as_tensor(expert_capacity, device=counts.device, dtype=counts.dtype),
    )
    return kept_scores, load_counts


def _prepare_routing(
    x2d: Tensor,
    top_indices: Tensor,
    top_scores: Tensor,
    num_experts: int,
    group_list: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    num_tokens, top_k = top_indices.shape
    flat_expert_idx = top_indices.reshape(-1)
    flat_scores = top_scores.reshape(-1)
    flat_token_idx = torch.arange(num_tokens, device=x2d.device, dtype=torch.long).repeat_interleave(top_k)

    sort_order = flat_expert_idx.to(torch.int32).argsort(stable=True)
    sorted_token_idx = flat_token_idx[sort_order]
    sorted_scores = flat_scores[sort_order]

    permuted_x = x2d[sorted_token_idx]
    if group_list is None:
        group_list = torch.bincount(flat_expert_idx[sort_order], minlength=num_experts).to(torch.int64)
    else:
        group_list = group_list.to(torch.int64)
    return permuted_x, group_list, sorted_token_idx, sorted_scores


def _prepare_routing_npu(
    x2d: Tensor,
    top_indices: Tensor,
    top_scores: Tensor,
    num_experts: int,
    group_list: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    import torch_npu

    expert_idx = top_indices.to(torch.int64)
    permuted_x, sorted_indices = torch_npu.npu_moe_token_permute(
        x2d, expert_idx, expert_idx.numel(),
    )
    if group_list is None:
        group_list = _expert_assignment_counts(top_indices, num_experts).to(torch.int64)
    else:
        group_list = group_list.to(torch.int64)
    return permuted_x, group_list, sorted_indices, top_scores


def _combine_routed_output(
    expert_out: Tensor,
    sorted_token_idx: Tensor,
    sorted_scores: Tensor,
    num_tokens: int,
    hidden_size: int,
    out_dtype: torch.dtype,
) -> Tensor:
    out = torch.zeros(num_tokens, hidden_size, device=expert_out.device, dtype=out_dtype)
    weighted = expert_out * sorted_scores.unsqueeze(-1).to(expert_out.dtype)
    out.index_add_(0, sorted_token_idx, weighted.to(out_dtype))
    return out


def _combine_routed_output_npu(
    expert_out: Tensor,
    sorted_indices: Tensor,
    top_scores: Tensor,
    out_dtype: torch.dtype,
) -> Tensor:
    import torch_npu

    probs = top_scores.to(torch.bfloat16)
    out = torch_npu.npu_moe_token_unpermute(expert_out, sorted_indices, probs=probs)
    return out.to(out_dtype)


def _grouped_matmul_loop(
    x: Tensor, weight: Tensor, offsets: list[int], transpose_b: bool,
) -> Tensor:
    """Grouped GEMM for CUDA: x [M, K] split by offsets against per-expert weights.

    Rows of x are already sorted by expert, so each group is a contiguous slice
    and one mm per expert covers it. There is no native grouped-GEMM kernel here
    (torch._grouped_mm needs compute capability 9.0).

    ``transpose_b`` uses a transposed *view* of the weight, which cuBLAS consumes
    directly. That avoids the materialised [E, K, N] transpose the Ascend kernel
    needs (~100 MB per expert stack per call on this config).
    """
    n_out = weight.shape[1] if transpose_b else weight.shape[2]
    out = x.new_empty(x.shape[0], n_out)
    for expert_id in range(weight.shape[0]):
        start, end = offsets[expert_id], offsets[expert_id + 1]
        if start >= end:
            continue
        w = weight[expert_id]
        torch.mm(x[start:end], w.t() if transpose_b else w, out=out[start:end])
    return out


def _grouped_linear_forward(
    x: Tensor, weight: Tensor, group_list: Tensor, cache: dict, key: str,
    offsets: Optional[list[int]] = None,
) -> Tensor:
    if _npu_grouped_matmul_available():
        import torch_npu

        return torch_npu.npu_grouped_matmul(
            x=[x],
            weight=[_get_gmm_weight(weight, cache, key)],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=group_list,
        )[0]
    # weight is [E, out, in]; take it transposed as a view.
    return _grouped_matmul_loop(
        x, weight, offsets or _grouped_offsets(group_list).tolist(), transpose_b=True,
    )


def _grouped_linear_backward_input(
    grad_output: Tensor, weight: Tensor, group_list: Tensor,
    offsets: Optional[list[int]] = None,
) -> Tensor:
    if _npu_grouped_matmul_available():
        import torch_npu

        return torch_npu.npu_grouped_matmul(
            x=[grad_output],
            weight=[weight],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=group_list,
        )[0]
    return _grouped_matmul_loop(
        grad_output, weight, offsets or _grouped_offsets(group_list).tolist(), transpose_b=False,
    )


def _grouped_offsets(group_list: Tensor) -> Tensor:
    return torch.cat([
        torch.zeros(1, device=group_list.device, dtype=torch.long),
        group_list.cumsum(0),
    ])


def _get_gmm_weight(weight: Tensor, cache: dict, key: str) -> Tensor:
    ptr = weight.data_ptr()
    tag = (ptr, weight.dtype)
    if cache.get(f"{key}_tag") != tag:
        cache[f"{key}_tag"] = tag
        cache[key] = weight.transpose(1, 2).contiguous()
    return cache[key]


def _swiglu_forward(gate_up: Tensor) -> Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


def _grouped_linear_backward_weight(
    x: Tensor, grad_output: Tensor, weight: Tensor, group_list: Tensor,
    offsets: Optional[list[int]] = None,
) -> Tensor:
    grad_weight = torch.zeros_like(weight)
    offsets = offsets or _grouped_offsets(group_list).tolist()
    for expert_id in range(weight.shape[0]):
        start, end = offsets[expert_id], offsets[expert_id + 1]
        if start >= end:
            continue
        x_e = x[start:end]
        grad_e = grad_output[start:end]
        grad_weight[expert_id] = grad_e.transpose(0, 1).matmul(x_e)
    return grad_weight


def _swiglu_backward(grad_output: Tensor, gate_up: Tensor) -> Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    sig_gate = torch.sigmoid(gate)
    silu_gate = gate * sig_gate
    # d/dgate silu(gate) = sigmoid(gate) + gate*sigmoid(gate)*(1-sigmoid(gate)).
    # This used silu(gate) = gate*sigmoid(gate) in place of the leading sigmoid(gate),
    # which made every routed expert's gate_up gradient wrong.
    grad_gate = grad_output * up * (sig_gate + gate * sig_gate * (1 - sig_gate))
    grad_up = grad_output * silu_gate
    return torch.cat([grad_gate, grad_up], dim=-1)


class _GroupedSwiGLUMLPFn(torch.autograd.Function):
    _weight_cache: dict = {}

    @staticmethod
    def forward(
        ctx,
        permuted_x: Tensor,
        gate_up_weight: Tensor,
        down_weight: Tensor,
        group_list: Tensor,
    ) -> Tensor:
        cache = _GroupedSwiGLUMLPFn._weight_cache
        # One host sync per MoE call instead of one per grouped GEMM.
        offsets = None if _npu_grouped_matmul_available() else _grouped_offsets(group_list).tolist()
        ctx.grouped_offsets = offsets
        gate_up_out = _grouped_linear_forward(
            permuted_x, gate_up_weight, group_list, cache, "gate_up", offsets=offsets,
        )
        hidden = _swiglu_forward(gate_up_out)
        expert_out = _grouped_linear_forward(
            hidden, down_weight, group_list, cache, "down", offsets=offsets,
        )
        ctx.save_for_backward(permuted_x, gate_up_out, hidden, gate_up_weight, down_weight, group_list)
        return expert_out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        permuted_x, gate_up_out, hidden, gate_up_weight, down_weight, group_list = ctx.saved_tensors
        offsets = getattr(ctx, "grouped_offsets", None)

        grad_down_weight = _grouped_linear_backward_weight(
            hidden, grad_output, down_weight, group_list, offsets=offsets,
        )
        grad_hidden = _grouped_linear_backward_input(
            grad_output, down_weight, group_list, offsets=offsets,
        )
        grad_gate_up_out = _swiglu_backward(grad_hidden, gate_up_out)
        grad_gate_up_weight = _grouped_linear_backward_weight(
            permuted_x, grad_gate_up_out, gate_up_weight, group_list, offsets=offsets,
        )
        grad_permuted_x = _grouped_linear_backward_input(
            grad_gate_up_out, gate_up_weight, group_list, offsets=offsets,
        )

        return grad_permuted_x, grad_gate_up_weight, grad_down_weight, None


def _grouped_swiglu_mlp(
    permuted_x: Tensor,
    gate_up_weight: Tensor,
    down_weight: Tensor,
    group_list: Tensor,
) -> Tensor:
    compute_dtype = permuted_x.dtype
    gate_up_w = gate_up_weight.to(compute_dtype)
    down_w = down_weight.to(compute_dtype)
    return _GroupedSwiGLUMLPFn.apply(permuted_x, gate_up_w, down_w, group_list)


class MoEFFN(nn.Module):
    """Fine-grained routed MoE FFN with optional shared expert."""

    def __init__(
        self,
        hidden_size: int,
        moe_config: MoEConfig,
        init_std_in: float,
        init_std_out: float,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.routed_top_k = moe_config.routed_top_k
        self.top_k = self.routed_top_k
        self.num_experts = moe_config.num_routed_experts
        self.expert_intermediate_size = moe_config.expert_intermediate_size
        self.capacity_factor = moe_config.capacity_factor
        self.router_z_loss_coef = moe_config.router_z_loss_coef
        self.load_balance_bias_lr = moe_config.load_balance_bias_lr
        self.use_npu_fused = _npu_fused_moe_available()

        if self.use_npu_fused:
            self.gate_up_weight = nn.Parameter(
                trunc_normal_init_(
                    torch.empty(
                        moe_config.num_routed_experts,
                        2 * moe_config.expert_intermediate_size,
                        hidden_size,
                    ),
                    std=init_std_in,
                )
            )
            self.down_weight = nn.Parameter(
                trunc_normal_init_(
                    torch.empty(
                        moe_config.num_routed_experts,
                        hidden_size,
                        moe_config.expert_intermediate_size,
                    ),
                    std=init_std_out,
                )
            )
            self.experts = None
        else:
            self.gate_up_weight = None
            self.down_weight = None
            self.experts = nn.ModuleList([
                SwiGLU(
                    hidden_size=hidden_size,
                    intermediate_size=moe_config.expert_intermediate_size,
                    init_std_in=init_std_in,
                    init_std_out=init_std_out,
                )
                for _ in range(moe_config.num_routed_experts)
            ])

        if moe_config.shared_expert:
            shared_intermediate = (
                moe_config.expert_intermediate_size * moe_config.shared_expert_size_multiplier
            )
            self.shared_experts = nn.ModuleList([
                SwiGLU(
                    hidden_size=hidden_size,
                    intermediate_size=shared_intermediate,
                    init_std_in=init_std_in,
                    init_std_out=init_std_out,
                )
                for _ in range(moe_config.num_shared_experts)
            ])
        else:
            self.shared_experts = nn.ModuleList()

        self.router_per_loop = bool(moe_config.router_per_loop)
        self.router_loop_bias = bool(moe_config.router_loop_bias)
        self.router_num_loops = max(1, int(moe_config.router_num_loops))
        if self.router_per_loop:
            # Fresh router per loop (required tech). Copy-init from router[0] so deep
            # loops start like a shared router and diverge in training — avoids H12
            # cold-start collapse from 60 independently random routers.
            self.routers = nn.ModuleList([
                nn.Linear(hidden_size, moe_config.num_routed_experts, bias=False)
                for _ in range(self.router_num_loops)
            ])
            nn.init.normal_(self.routers[0].weight, mean=0.0, std=init_std_in)
            with torch.no_grad():
                for router in self.routers[1:]:
                    router.weight.copy_(self.routers[0].weight)
            self.router = self.routers[0]  # compat: checkpoint / named_modules
            self.router_loop_biases = None
        else:
            self.routers = None
            self.router = nn.Linear(hidden_size, moe_config.num_routed_experts, bias=False)
            nn.init.normal_(self.router.weight, mean=0.0, std=init_std_in)
            if self.router_loop_bias:
                self.router_loop_biases = nn.Parameter(
                    torch.zeros(self.router_num_loops, moe_config.num_routed_experts)
                )
            else:
                self.router_loop_biases = None
        self.register_buffer("expert_bias", torch.zeros(moe_config.num_routed_experts), persistent=True)
        self._router_z_loss: Optional[Tensor] = None
        self._step_expert_load: Optional[Tensor] = None

    def _router_for_loop(self, loop_idx: int) -> nn.Linear:
        if self.router_per_loop and self.routers is not None:
            idx = max(0, min(int(loop_idx), len(self.routers) - 1))
            return self.routers[idx]
        return self.router

    def _route(self, x2d: Tensor, loop_idx: int = 0) -> Tuple[Tensor, Tensor, Tensor]:
        # Routing is done in fp32 (moe.router_dtype). Under FSDP2 mixed precision the
        # router weight arrives as bfloat16, and cuBLAS rejects mixed-dtype matmul
        # (the Ascend kernel promoted silently), so cast the weight too.
        router = self._router_for_loop(loop_idx)
        router_logits = F.linear(x2d.float(), router.weight.float(), router.bias)
        if self.router_loop_biases is not None:
            idx = max(0, min(int(loop_idx), self.router_loop_biases.shape[0] - 1))
            router_logits = router_logits + self.router_loop_biases[idx].to(
                dtype=router_logits.dtype, device=router_logits.device,
            )
        if self.training and self.router_z_loss_coef > 0:
            log_z = torch.logsumexp(router_logits, dim=-1)
            self._router_z_loss = (log_z * log_z).mean()
        else:
            self._router_z_loss = None

        unbiased_scores = torch.sigmoid(router_logits).to(dtype=x2d.dtype)
        if self.load_balance_bias_lr > 0:
            selection_scores = unbiased_scores + self.expert_bias.to(dtype=unbiased_scores.dtype)
        else:
            selection_scores = unbiased_scores

        _, top_indices = torch.topk(selection_scores, self.top_k, dim=-1)
        top_scores = torch.gather(unbiased_scores, -1, top_indices)
        top_scores = top_scores / top_scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return top_scores, top_indices, unbiased_scores

    def _accumulate_expert_load(self, load_counts: Tensor) -> None:
        if not self.training or self.load_balance_bias_lr <= 0:
            return
        # Clone: on capacity fast-path, load_counts may alias assignment_counts which
        # is also saved for GMM backward; inplace += would poison autograd.
        if self._step_expert_load is None:
            self._step_expert_load = load_counts.detach().clone()
        else:
            self._step_expert_load = self._step_expert_load + load_counts

    def pop_expert_load(self) -> Optional[Tensor]:
        load = self._step_expert_load
        self._step_expert_load = None
        return load

    def update_expert_bias_from_load(self, load_counts: Tensor) -> None:
        if self.load_balance_bias_lr <= 0:
            return
        load = load_counts.to(self.expert_bias.device, dtype=torch.float32)
        error = load.mean() - load
        self.expert_bias.add_(self.load_balance_bias_lr * error.sign().to(self.expert_bias.dtype))

    def _forward_npu_fused(
        self,
        x2d: Tensor,
        top_scores: Tensor,
        top_indices: Tensor,
        group_list: Optional[Tensor] = None,
    ) -> Tensor:
        if _use_npu_routing():
            permuted_x, group_list, sorted_indices, route_scores = _prepare_routing_npu(
                x2d, top_indices, top_scores, self.num_experts, group_list=group_list,
            )
            expert_out = _grouped_swiglu_mlp(
                permuted_x, self.gate_up_weight, self.down_weight, group_list,
            )
            return _combine_routed_output_npu(
                expert_out, sorted_indices, route_scores, x2d.dtype,
            )

        num_tokens = x2d.shape[0]
        permuted_x, group_list, sorted_token_idx, sorted_scores = _prepare_routing(
            x2d, top_indices, top_scores, self.num_experts, group_list=group_list,
        )
        expert_out = _grouped_swiglu_mlp(
            permuted_x, self.gate_up_weight, self.down_weight, group_list,
        )
        return _combine_routed_output(
            expert_out, sorted_token_idx, sorted_scores, num_tokens, self.hidden_size, x2d.dtype,
        )

    def _forward_loop(self, x2d: Tensor, top_scores: Tensor, top_indices: Tensor) -> Tensor:
        num_tokens = x2d.shape[0]
        out = torch.zeros_like(x2d)
        dummy_token = x2d[:1]
        for expert_id, expert in enumerate(self.experts):
            token_weights = torch.zeros(num_tokens, device=x2d.device, dtype=x2d.dtype)
            for slot in range(self.top_k):
                slot_mask = top_indices[:, slot] == expert_id
                if slot_mask.any():
                    token_weights[slot_mask] += top_scores[slot_mask, slot]

            token_mask = token_weights > 0
            if token_mask.any():
                expert_out = expert(x2d[token_mask])
                out[token_mask] += token_weights[token_mask].unsqueeze(-1) * expert_out
            elif self.training:
                out = out + expert(dummy_token).sum() * 0.0
        return out

    def forward(self, x: Tensor, loop_idx: int = 0) -> Tensor:
        orig_shape = x.shape
        x2d = x.reshape(-1, self.hidden_size)

        top_scores, top_indices, _ = self._route(x2d, loop_idx=loop_idx)
        assignment_counts = _expert_assignment_counts(top_indices, self.num_experts)
        if os.environ.get("MOE_SKIP_CAPACITY", "0") == "1":
            load_counts = assignment_counts
        else:
            top_scores, load_counts = _apply_capacity(
                top_scores,
                top_indices,
                self.capacity_factor,
                self.num_experts,
                assignment_counts=assignment_counts,
            )
        self._accumulate_expert_load(load_counts)
        try:
            from utils.module_diag import get_active_collector
            coll = get_active_collector()
            if coll is not None:
                coll.record_moe_expert_load(self, load_counts)
        except Exception:
            pass
        try:
            from utils.expert_freq import get_active_expert_logger
            elog = get_active_expert_logger()
            if elog is not None:
                elog.record(self, load_counts)
        except Exception:
            pass
        if self.use_npu_fused:
            out = self._forward_npu_fused(
                x2d, top_scores, top_indices, group_list=assignment_counts,
            )
        else:
            out = self._forward_loop(x2d, top_scores, top_indices)

        if self.shared_experts:
            for shared in self.shared_experts:
                out = out + shared(x2d)

        return out.reshape(orig_shape)

    def pop_router_z_loss(self) -> Optional[Tensor]:
        z_loss = self._router_z_loss
        self._router_z_loss = None
        return z_loss
