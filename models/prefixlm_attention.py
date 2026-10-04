import os
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

_PREFIXLM_ATTN_BACKEND = os.environ.get("PREFIXLM_ATTN_BACKEND", "auto").lower()
_FUSION_CAUSAL_MASK_CACHE: dict[torch.device, Tensor] = {}
_KVCACHE_SDPA_MASK_CACHE: dict[tuple, Tensor] = {}
_KVCACHE_ARANGE_CACHE: dict[torch.device, Tensor] = {}


def compute_aux_seq_tensors_scalars(prefix_lens: np.ndarray, causal_lens: np.ndarray, batch_max_tokens: int):
    total_lens = prefix_lens + causal_lens
    tensors = {
        "prefix_lens": np.pad(prefix_lens, (0, batch_max_tokens - prefix_lens.shape[0])),
        "causal_lens": np.pad(causal_lens, (0, batch_max_tokens - causal_lens.shape[0])),
        "cu_seqlens": np.pad(np.cumsum(total_lens, dtype=np.int32), (1, batch_max_tokens - total_lens.shape[0] - 1)),
    }
    scalars = {
        "total_seqlen": int(total_lens.sum()),
        "numseqs": total_lens.shape[0],
        "max_seqlen_prefix": int(prefix_lens.max()) if prefix_lens.size else 0,
        "max_seqlen_causal": int(causal_lens.max()) if causal_lens.size else 0,
        "max_seqlen_all": int(total_lens.max()) if total_lens.size else 0,
    }
    return tensors, scalars


def _as_int(value: Union[Tensor, int]) -> int:
    if isinstance(value, Tensor):
        return int(value.item())
    return int(value)


def _npu_fusion_available() -> bool:
    try:
        import torch_npu  # noqa: F401

        return hasattr(torch_npu, "npu_fusion_attention")
    except ImportError:
        return False


def _npu_fusion_fn():
    import torch_npu

    version = os.environ.get("PREFIXLM_FUSION_VERSION", "").lower()
    if not version:
        rope_on = os.environ.get("PREFIXLM_FUSION_ROPE", "1").lower() not in ("0", "false", "off")
        version = "v2" if rope_on else "v1"
    if version == "v2" and hasattr(torch_npu, "npu_fusion_attention_v2"):
        return torch_npu.npu_fusion_attention_v2
    return torch_npu.npu_fusion_attention


def _fa2_available() -> bool:
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401

        return True
    except ImportError:
        return False


_FA2_DTYPES = (torch.float16, torch.bfloat16)


def prefixlm_varlen_attention_fa2(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool,
    prefix_lens: Tensor,
    causal_lens: Tensor,
    cu_seqlens: Tensor,
    total_seqlen: Tensor,
    numseqs: Tensor,
    max_seqlen_prefix: Tensor,
    max_seqlen_causal: Tensor,
    max_seqlen_all: Tensor,
) -> Tensor:
    """FlashAttention-2 varlen path for CUDA.

    q/k/v are [total_tokens, heads, head_dim] packed across documents, which is
    exactly the varlen layout. GQA is handled inside the kernel, so the KV heads
    are not materialised. Rows past total_seqlen are padding and stay zero.
    """
    from flash_attn import flash_attn_varlen_func

    del prefix_lens, causal_lens, max_seqlen_prefix, max_seqlen_causal

    numseqs_int = _as_int(numseqs)
    total_int = _as_int(total_seqlen)
    max_seqlen = _as_int(max_seqlen_all)
    cu = cu_seqlens[: numseqs_int + 1].to(device=q.device, dtype=torch.int32).contiguous()

    out = torch.zeros_like(q)
    if total_int <= 0 or numseqs_int <= 0:
        return out

    attn = flash_attn_varlen_func(
        q[:total_int],
        k[:total_int],
        v[:total_int],
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=is_causal,
    )
    out[:total_int] = attn
    return out


def _resolve_attention_backend(device: torch.device, is_causal: bool) -> str:
    if _PREFIXLM_ATTN_BACKEND in ("loop", "sdpa"):
        return "loop"
    if _PREFIXLM_ATTN_BACKEND in ("fusion", "npu_fusion"):
        if device.type != "npu" or not _npu_fusion_available():
            return "loop"
        return "fusion"
    if _PREFIXLM_ATTN_BACKEND in ("fa2", "flash", "flash_attn"):
        if device.type == "cuda" and is_causal and _fa2_available():
            return "fa2"
        return "loop"
    if device.type == "npu" and _npu_fusion_available():
        return "fusion"
    # CUDA: FA2 varlen covers the full-causal LM path. PrefixLM (prefix_lens>0)
    # is not expressible as one varlen call, so it keeps the SDPA loop.
    if device.type == "cuda" and is_causal and _fa2_available():
        return "fa2"
    return "loop"


def _get_causal_atten_mask(device: torch.device, max_len: int = 2048) -> Tensor:
    mask = _FUSION_CAUSAL_MASK_CACHE.get(device)
    if mask is None:
        mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool, device=device), diagonal=1)
        _FUSION_CAUSAL_MASK_CACHE[device] = mask
    return mask


def _build_fusion_indices(
    prefix_lens: Tensor,
    causal_lens: Tensor,
    cu_seqlens: Tensor,
    numseqs_int: int,
    device: torch.device,
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], list[int], list[int], list[int], int, int]:
    prefix = prefix_lens[:numseqs_int]
    causal = causal_lens[:numseqs_int]
    starts = cu_seqlens[:numseqs_int]
    ends = cu_seqlens[1 : numseqs_int + 1]

    prefix_idx_list: list[int] = []
    causal_q_idx_list: list[int] = []
    kv_idx_list: list[int] = []
    for i in range(numseqs_int):
        start = _as_int(starts[i])
        end = _as_int(ends[i])
        prefix_len = _as_int(prefix[i])
        prefix_idx_list.extend(range(start, start + prefix_len))
        causal_q_idx_list.extend(range(start + prefix_len, end))
        kv_idx_list.extend(range(start, end))

    pre_cu = torch.cumsum(prefix.long(), dim=0).tolist()
    causal_cu = torch.cumsum(causal.long(), dim=0).tolist()
    kv_cu = torch.cumsum((prefix + causal).long(), dim=0).tolist()

    prefix_idx = (
        torch.tensor(prefix_idx_list, device=device, dtype=torch.long) if prefix_idx_list else None
    )
    causal_q_idx = (
        torch.tensor(causal_q_idx_list, device=device, dtype=torch.long) if causal_q_idx_list else None
    )
    kv_idx = torch.tensor(kv_idx_list, device=device, dtype=torch.long) if kv_idx_list else None
    return prefix_idx, causal_q_idx, kv_idx, pre_cu, causal_cu, kv_cu, len(prefix_idx_list), len(causal_q_idx_list)


def _normalize_cache_lengths(cache_lengths: Optional[Union[Tensor, int]], batch_size: int, device: torch.device) -> Tensor:
    if cache_lengths is None:
        return torch.zeros(batch_size, dtype=torch.int32, device=device)
    if isinstance(cache_lengths, int):
        if batch_size == 1:
            return torch.tensor([cache_lengths], dtype=torch.int32, device=device)
        raise ValueError("Scalar cache_lengths requires batch size 1.")
    if cache_lengths.ndim == 0:
        return cache_lengths.reshape(1).to(device=device, dtype=torch.int32)
    return cache_lengths.to(device=device, dtype=torch.int32)


def _scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, is_causal: bool) -> Tensor:
    # q, k, v: [seq, heads, dim] -> SDPA [1, heads, seq, dim]
    k = _repeat_kv_heads(k, q.shape[-2])
    v = _repeat_kv_heads(v, q.shape[-2])
    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    return out.squeeze(0).transpose(0, 1)


def _repeat_kv_heads(kv: Tensor, num_query_heads: int) -> Tensor:
    num_kv_heads = kv.shape[-2]
    if num_kv_heads == num_query_heads:
        return kv
    n_rep = num_query_heads // num_kv_heads
    assert n_rep * num_kv_heads == num_query_heads, (
        f"query heads {num_query_heads} must be divisible by kv heads {num_kv_heads}"
    )
    return kv.repeat_interleave(n_rep, dim=-2)


def _repeat_kv_heads_bhsd(kv: Tensor, num_query_heads: int) -> Tensor:
    num_kv_heads = kv.shape[1]
    if num_kv_heads == num_query_heads:
        return kv
    n_rep = num_query_heads // num_kv_heads
    assert n_rep * num_kv_heads == num_query_heads, (
        f"query heads {num_query_heads} must be divisible by kv heads {num_kv_heads}"
    )
    return kv.repeat_interleave(n_rep, dim=1)


def prefixlm_varlen_attention_sdpa_loop(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool,
    prefix_lens: Tensor,
    causal_lens: Tensor,
    cu_seqlens: Tensor,
    total_seqlen: Tensor,
    numseqs: Tensor,
    max_seqlen_prefix: Tensor,
    max_seqlen_causal: Tensor,
    max_seqlen_all: Tensor,
) -> Tensor:
    del max_seqlen_prefix, max_seqlen_causal, max_seqlen_all

    out = torch.zeros_like(q)
    numseqs_int = _as_int(numseqs)
    total_seqlen_int = _as_int(total_seqlen)
    cu = cu_seqlens[: numseqs_int + 1]

    for i in range(numseqs_int):
        start = _as_int(cu[i])
        end = _as_int(cu[i + 1])
        seq_q = q[start:end]
        seq_k = k[start:end]
        seq_v = v[start:end]

        if is_causal:
            # Full causal LM: one causal attention over the whole packed sequence.
            if end > start:
                out[start:end] = _scaled_dot_product_attention(seq_q, seq_k, seq_v, is_causal=True)
            continue

        prefix_len = _as_int(prefix_lens[i])
        causal_len = _as_int(causal_lens[i])

        if prefix_len > 0:
            out[start : start + prefix_len] = _scaled_dot_product_attention(
                seq_q[:prefix_len], seq_k[:prefix_len], seq_v[:prefix_len], is_causal=False
            )

        if causal_len > 0:
            out[start + prefix_len : end] = _scaled_dot_product_attention(
                seq_q[prefix_len:], seq_k, seq_v, is_causal=True
            )

    if total_seqlen_int < out.shape[0]:
        out[total_seqlen_int:] = 0
    return out


def prefixlm_varlen_attention_npu_fusion(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool,
    prefix_lens: Tensor,
    causal_lens: Tensor,
    cu_seqlens: Tensor,
    total_seqlen: Tensor,
    numseqs: Tensor,
    max_seqlen_prefix: Tensor,
    max_seqlen_causal: Tensor,
    max_seqlen_all: Tensor,
) -> Tensor:
    del max_seqlen_prefix, max_seqlen_causal, max_seqlen_all

    fusion_attn = _npu_fusion_fn()

    numseqs_int = _as_int(numseqs)
    total_seqlen_int = _as_int(total_seqlen)
    if total_seqlen_int == 0:
        return torch.zeros_like(q)

    head_num = q.shape[-2]
    head_dim = q.shape[-1]
    scale = head_dim**-0.5

    qa = q[:total_seqlen_int]
    ka = k[:total_seqlen_int]
    va = v[:total_seqlen_int]
    out = torch.zeros_like(q)

    if is_causal:
        # Full causal LM on packed sequences: one fused causal attn (same kernel as
        # PrefixLM's causal branch), ignoring the PrefixLM bidir split.
        total_lens = (prefix_lens[:numseqs_int] + causal_lens[:numseqs_int]).long()
        full_cu = torch.cumsum(total_lens, dim=0).tolist()
        r = fusion_attn(
            qa,
            ka,
            va,
            head_num,
            "TND",
            scale=scale,
            actual_seq_qlen=full_cu,
            actual_seq_kvlen=full_cu,
            keep_prob=1.0,
            sparse_mode=3,
            atten_mask=_get_causal_atten_mask(q.device),
        )
        out[:total_seqlen_int] = r[0]
        if total_seqlen_int < out.shape[0]:
            out[total_seqlen_int:] = 0
        return out

    prefix_idx, causal_q_idx, kv_idx, pre_cu, causal_cu, kv_cu, n_prefix, n_causal = _build_fusion_indices(
        prefix_lens, causal_lens, cu_seqlens, numseqs_int, q.device
    )

    if n_prefix > 0:
        assert prefix_idx is not None
        q1 = qa.index_select(0, prefix_idx)
        k1 = ka.index_select(0, prefix_idx)
        v1 = va.index_select(0, prefix_idx)
        r1 = fusion_attn(
            q1,
            k1,
            v1,
            head_num,
            "TND",
            scale=scale,
            actual_seq_qlen=pre_cu,
            actual_seq_kvlen=pre_cu,
            keep_prob=1.0,
            sparse_mode=0,
        )
        out = out.index_copy(0, prefix_idx, r1[0])

    if n_causal > 0:
        assert causal_q_idx is not None and kv_idx is not None
        q2 = qa.index_select(0, causal_q_idx)
        k2 = ka.index_select(0, kv_idx)
        v2 = va.index_select(0, kv_idx)
        r2 = fusion_attn(
            q2,
            k2,
            v2,
            head_num,
            "TND",
            scale=scale,
            actual_seq_qlen=causal_cu,
            actual_seq_kvlen=kv_cu,
            keep_prob=1.0,
            sparse_mode=3,
            atten_mask=_get_causal_atten_mask(q.device),
        )
        out = out.index_copy(0, causal_q_idx, r2[0])

    if total_seqlen_int < out.shape[0]:
        out[total_seqlen_int:] = 0
    return out


_LOGGED_BACKEND: set = set()


def _log_backend_once(backend: str, device: torch.device) -> None:
    """One line per process so the training log shows which kernel actually ran."""
    key = (backend, device.type)
    if key in _LOGGED_BACKEND:
        return
    _LOGGED_BACKEND.add(key)
    try:
        import torch.distributed as dist

        if dist.is_initialized() and dist.get_rank() != 0:
            return
    except Exception:
        pass
    print(f"[attention] backend={backend} device={device.type}", flush=True)


def prefixlm_varlen_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool,
    prefix_lens: Tensor,
    causal_lens: Tensor,
    cu_seqlens: Tensor,
    total_seqlen: Tensor,
    numseqs: Tensor,
    max_seqlen_prefix: Tensor,
    max_seqlen_causal: Tensor,
    max_seqlen_all: Tensor,
) -> Tensor:
    backend = _resolve_attention_backend(q.device, is_causal)
    if backend == "fa2" and q.dtype not in _FA2_DTYPES:
        backend = "loop"
    _log_backend_once(backend, q.device)
    if backend == "fa2":
        return prefixlm_varlen_attention_fa2(
            q,
            k,
            v,
            is_causal,
            prefix_lens,
            causal_lens,
            cu_seqlens,
            total_seqlen,
            numseqs,
            max_seqlen_prefix,
            max_seqlen_causal,
            max_seqlen_all,
        )
    if backend == "fusion":
        return prefixlm_varlen_attention_npu_fusion(
            q,
            k,
            v,
            is_causal,
            prefix_lens,
            causal_lens,
            cu_seqlens,
            total_seqlen,
            numseqs,
            max_seqlen_prefix,
            max_seqlen_causal,
            max_seqlen_all,
        )
    return prefixlm_varlen_attention_sdpa_loop(
        q,
        k,
        v,
        is_causal,
        prefix_lens,
        causal_lens,
        cu_seqlens,
        total_seqlen,
        numseqs,
        max_seqlen_prefix,
        max_seqlen_causal,
        max_seqlen_all,
    )


def _resolve_kvcache_backend(device: torch.device) -> str:
    backend = os.environ.get("INFERENCE_KVCACHE_BACKEND", "fusion").lower()
    if backend in ("sdpa", "loop"):
        return backend
    if backend in ("fusion", "npu_fusion", "auto"):
        if device.type == "npu" and _npu_fusion_available():
            return "fusion"
        return "sdpa"
    return "sdpa"


def _get_kv_arange(device: torch.device, length: int) -> Tensor:
    arange = _KVCACHE_ARANGE_CACHE.get(device)
    if arange is None or arange.numel() < length:
        arange = torch.arange(max(length, arange.numel() if arange is not None else 0), device=device)
        _KVCACHE_ARANGE_CACHE[device] = arange
    return arange[:length]


def _write_kvcache(
    k: Tensor,
    v: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    cache_lengths: Tensor,
    seq_len: int,
    prefill_valid_lens: Optional[Tensor] = None,
) -> None:
    batch_size = k.shape[0]
    ends = cache_lengths + seq_len
    if seq_len == 1:
        batch_idx = torch.arange(batch_size, device=k.device)
        pos = ends - 1
        k_cache[batch_idx, pos] = k[:, 0]
        v_cache[batch_idx, pos] = v[:, 0]
        return

    if prefill_valid_lens is not None:
        for batch_idx in range(batch_size):
            valid_len = _as_int(prefill_valid_lens[batch_idx])
            start = _as_int(cache_lengths[batch_idx])
            k_cache[batch_idx, start : start + valid_len] = k[batch_idx, :valid_len]
            v_cache[batch_idx, start : start + valid_len] = v[batch_idx, :valid_len]
        return

    for batch_idx in range(batch_size):
        start = _as_int(cache_lengths[batch_idx])
        end = _as_int(ends[batch_idx])
        k_cache[batch_idx, start:end] = k[batch_idx]
        v_cache[batch_idx, start:end] = v[batch_idx]


def attention_with_kvcache_sdpa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    cache_seqlens: Optional[Union[Tensor, int]],
    causal: bool,
    prefill_valid_lens: Optional[Tensor] = None,
) -> Tensor:
    batch_size, seq_len = q.shape[0], q.shape[1]
    cache_lengths = _normalize_cache_lengths(cache_seqlens, batch_size, q.device)
    if prefill_valid_lens is not None:
        effective_ends = cache_lengths + prefill_valid_lens.to(dtype=torch.int32)
    else:
        effective_ends = cache_lengths + seq_len

    _write_kvcache(k, v, k_cache, v_cache, cache_lengths, seq_len, prefill_valid_lens)

    max_kv_len = k_cache.shape[1]
    if os.environ.get("INFERENCE_KV_SLICE", "0").lower() not in ("0", "false", "off"):
        active_kv = min(int(effective_ends.max().item()), max_kv_len)
    else:
        active_kv = max_kv_len

    q_b = q.transpose(1, 2)
    k_b = k_cache[:, :active_kv].transpose(1, 2)
    v_b = v_cache[:, :active_kv].transpose(1, 2)
    k_b = _repeat_kv_heads_bhsd(k_b, q_b.shape[1])
    v_b = _repeat_kv_heads_bhsd(v_b, q_b.shape[1])

    kv_pos = _get_kv_arange(q.device, active_kv)
    valid = kv_pos.unsqueeze(0) < effective_ends.unsqueeze(1)  # [B, KV]
    if causal and seq_len > 1:
        q_pos = cache_lengths.unsqueeze(1) + _get_kv_arange(q.device, seq_len).unsqueeze(0)
        valid = valid.unsqueeze(1) & (kv_pos.unsqueeze(0).unsqueeze(1) <= q_pos.unsqueeze(-1))  # [B, S, KV]
        if prefill_valid_lens is not None:
            q_valid = _get_kv_arange(q.device, seq_len).unsqueeze(0) < prefill_valid_lens.unsqueeze(1)
            valid = valid & q_valid.unsqueeze(-1)

    # SDPA attn_mask must be [B, 1, S, KV] (or broadcastable). Do not over-unsqueeze.
    if valid.ndim == 2:
        valid_mask = valid[:, None, None, :]  # [B, 1, 1, KV]
    else:
        valid_mask = valid[:, None, :, :]  # [B, 1, S, KV]
    use_mask_cache = os.environ.get("INFERENCE_ATTN_MASK_CACHE", "0").lower() not in ("0", "false", "off")
    mask_key = (q.device, q.dtype, batch_size, seq_len, active_kv, int(causal and seq_len > 1))
    if use_mask_cache and mask_key in _KVCACHE_SDPA_MASK_CACHE:
        attn_mask = _KVCACHE_SDPA_MASK_CACHE[mask_key]
        attn_mask.zero_()
        attn_mask.masked_fill_(~valid_mask, float("-inf"))
    else:
        attn_mask = torch.zeros((batch_size, 1, seq_len, active_kv), device=q.device, dtype=q.dtype)
        attn_mask.masked_fill_(~valid_mask, float("-inf"))
        if use_mask_cache:
            _KVCACHE_SDPA_MASK_CACHE[mask_key] = attn_mask

    out = F.scaled_dot_product_attention(q_b, k_b, v_b, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
    return out.transpose(1, 2)


def attention_with_kvcache_npu_fusion(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    cache_seqlens: Optional[Union[Tensor, int]],
    causal: bool,
    prefill_valid_lens: Optional[Tensor] = None,
) -> Tensor:
    batch_size, seq_len = q.shape[0], q.shape[1]
    if seq_len != 1:
        return attention_with_kvcache_sdpa(
            q, k, v, k_cache, v_cache, cache_seqlens, causal, prefill_valid_lens
        )

    cache_lengths = _normalize_cache_lengths(cache_seqlens, batch_size, q.device)
    ends = cache_lengths + seq_len
    _write_kvcache(k, v, k_cache, v_cache, cache_lengths, seq_len, prefill_valid_lens)

    fusion_attn = _npu_fusion_fn()
    head_num = q.shape[-2]
    head_dim = q.shape[-1]
    scale = head_dim**-0.5

    qa = q[:, 0]
    positions = _get_kv_arange(q.device, k_cache.shape[1]).unsqueeze(0)
    valid = positions < ends.unsqueeze(1)
    batch_idx, pos_idx = valid.nonzero(as_tuple=True)
    ka = k_cache[batch_idx, pos_idx]
    va = v_cache[batch_idx, pos_idx]
    ka = _repeat_kv_heads(ka, head_num)
    va = _repeat_kv_heads(va, head_num)
    q_cu = list(range(1, batch_size + 1))
    kv_cu = ends.cumsum(dim=0).tolist()
    out = fusion_attn(
        qa,
        ka,
        va,
        head_num,
        "TND",
        scale=scale,
        actual_seq_qlen=q_cu,
        actual_seq_kvlen=kv_cu,
        keep_prob=1.0,
        sparse_mode=0,
    )[0]
    return out.unsqueeze(1)


def attention_with_kvcache_loop(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    cache_seqlens: Optional[Union[Tensor, int]],
    causal: bool,
    prefill_valid_lens: Optional[Tensor] = None,
) -> Tensor:
    batch_size, seq_len = q.shape[0], q.shape[1]
    if seq_len != 1:
        return attention_with_kvcache_sdpa(
            q, k, v, k_cache, v_cache, cache_seqlens, causal, prefill_valid_lens
        )

    cache_lengths = _normalize_cache_lengths(cache_seqlens, batch_size, q.device)
    ends = cache_lengths + seq_len
    _write_kvcache(k, v, k_cache, v_cache, cache_lengths, seq_len, prefill_valid_lens)

    out = torch.empty_like(q)
    for batch_idx in range(batch_size):
        end = _as_int(ends[batch_idx])
        seq_q = q[batch_idx : batch_idx + 1]
        seq_k = k_cache[batch_idx : batch_idx + 1, :end]
        seq_v = v_cache[batch_idx : batch_idx + 1, :end]
        seq_k = _repeat_kv_heads(seq_k, seq_q.shape[-2])
        seq_v = _repeat_kv_heads(seq_v, seq_q.shape[-2])
        out[batch_idx : batch_idx + 1] = _scaled_dot_product_attention(seq_q[:, 0], seq_k[:, 0], seq_v[:, 0], is_causal=False).unsqueeze(1)
    return out


def attention_with_kvcache(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    cache_seqlens: Optional[Union[Tensor, int]],
    causal: bool,
    prefill_valid_lens: Optional[Tensor] = None,
) -> Tensor:
    backend = _resolve_kvcache_backend(q.device)
    if backend == "loop":
        return attention_with_kvcache_loop(
            q, k, v, k_cache, v_cache, cache_seqlens, causal, prefill_valid_lens
        )
    if backend == "fusion":
        return attention_with_kvcache_npu_fusion(
            q, k, v, k_cache, v_cache, cache_seqlens, causal, prefill_valid_lens
        )
    return attention_with_kvcache_sdpa(
        q, k, v, k_cache, v_cache, cache_seqlens, causal, prefill_valid_lens
    )
