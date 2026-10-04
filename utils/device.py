"""Device helpers for training and inference (CUDA and Ascend NPU)."""

import os

import torch

try:  # Ascend NPU is optional; absent on CUDA hosts.
    import torch_npu  # noqa: F401

    _HAS_TORCH_NPU = True
except ImportError:
    _HAS_TORCH_NPU = False


def is_npu_available() -> bool:
    return _HAS_TORCH_NPU and hasattr(torch, "npu") and torch.npu.is_available()


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def device_type() -> str:
    """Accelerator family in use. HRM_DEVICE can force one."""
    forced = os.environ.get("HRM_DEVICE", "").strip().lower()
    if forced in ("cuda", "npu", "cpu"):
        return forced
    if is_cuda_available():
        return "cuda"
    if is_npu_available():
        return "npu"
    return "cpu"


def get_device(device_id: int | None = None) -> torch.device:
    dev = device_type()
    if dev == "cpu":
        return torch.device("cpu")
    if device_id is None:
        # Always index the device: a bare torch.device("cuda") as map_location
        # sends every rank's tensors to card 0.
        mod = torch.cuda if dev == "cuda" else torch.npu
        device_id = mod.current_device()
    return torch.device(f"{dev}:{device_id}")


def set_device(device_id: int) -> None:
    dev = device_type()
    if dev == "cuda":
        torch.cuda.set_device(device_id)
    elif dev == "npu":
        torch.npu.set_device(device_id)


def get_dist_backend() -> str:
    backend = os.environ.get("DIST_BACKEND")
    if backend:
        return backend
    return "nccl" if device_type() == "cuda" else "hccl"


def synchronize() -> None:
    dev = device_type()
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "npu" and is_npu_available():
        torch.npu.synchronize()


def empty_cache() -> None:
    dev = device_type()
    if dev == "cuda":
        torch.cuda.empty_cache()
    elif dev == "npu" and is_npu_available():
        torch.npu.empty_cache()


def use_pin_memory() -> bool:
    # Ascend host copies were slower with pinned memory; CUDA wants it.
    return device_type() == "cuda"
