"""Attention Residuals (AttnRes) — arXiv:2603.15031 (Moonshot / Kimi).

Full AttnRes: replace uniform residual sum with depth-wise softmax over prior
branch outputs. Each destination step has one learnable pseudo-query w ∈ R^d:

    α_i = softmax_i( w^T RMSNorm(v_i) )
    h   = Σ_i α_i v_i

We treat each Attention and MLP as a separate depth step (paper §2).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _as_local(t: Tensor) -> Tensor:
    """Localize DTensor params/acts for depth-mix arithmetic."""
    to_local = getattr(t, "to_local", None)
    if callable(to_local):
        return to_local()
    return t


def _match_ref(out: Tensor, ref: Tensor) -> Tensor:
    """Preserve DTensor wrapper of `ref` when depth-mix ran on local tensors."""
    if type(out) is type(ref):
        return out
    from_local = getattr(type(ref), "from_local", None)
    if from_local is None or not hasattr(ref, "device_mesh"):
        return out
    try:
        return from_local(out, device_mesh=ref.device_mesh, placements=ref.placements)
    except Exception:
        return out


class AttnResBank(nn.Module):
    """Reusable value bank + depth-wise queries for Full AttnRes."""

    def __init__(self, hidden_size: int, max_slots: int = 256, eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.max_slots = int(max_slots)
        self.eps = float(eps)
        # queries[k] used when attending over the first (k+1) values.
        self.queries = nn.Parameter(torch.empty(self.max_slots, self.hidden_size))
        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        self._values: List[Tensor] = []

    def clear(self) -> None:
        self._values = []

    def snapshot(self) -> List[Tensor]:
        """Detached copies so a later segment can mix over prior loops as constants."""
        return [v.detach() for v in self._values]

    def restore(self, values: List[Tensor]) -> None:
        self._values = list(values)

    def detach_inplace(self) -> None:
        self._values = [v.detach() for v in self._values]

    @property
    def size(self) -> int:
        return len(self._values)

    def append(self, v: Tensor) -> None:
        if len(self._values) >= self.max_slots:
            raise RuntimeError(
                f"AttnResBank overflow: {len(self._values)} >= max_slots={self.max_slots}"
            )
        self._values.append(v)

    def mix(self, *, next_layer: bool = False) -> Tensor:
        """Attend over current values.

        next_layer=False: input to the next branch (|V|=n → query slot n-1).
        next_layer=True:  readout after last append (|V|=n → query slot n).
        """
        n = len(self._values)
        if n == 0:
            raise RuntimeError("AttnResBank.mix called with empty values")
        if n == 1 and not next_layer:
            return self._values[0]
        slot = n if next_layer else (n - 1)
        slot = min(max(int(slot), 0), self.max_slots - 1)
        ref = self._values[-1]
        locals_v = [_as_local(v) for v in self._values]
        V = torch.stack(locals_v, dim=0)  # [N, ..., D]
        K = F.rms_norm(V, (V.shape[-1],), eps=self.eps)
        w = _as_local(self.queries[slot])
        # Rank-agnostic: works for [N,B,T,D] or packed [N,tokens,D].
        logits = torch.einsum("d,...d->...", w, K)
        alpha = torch.softmax(logits, dim=0)
        out = (alpha.unsqueeze(-1) * V).sum(dim=0)
        return _match_ref(out, ref)
