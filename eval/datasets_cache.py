"""HuggingFace cache location for evaluation (EVAL_HF_ROOT, default eval/hf_cache).

EVAL_OFFLINE=1 reads only what is already cached there; the default downloads on first use.
"""

from __future__ import annotations

import os
from pathlib import Path

from datasets import DownloadMode
from datasets import load_dataset as _load_dataset

_EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_HF_ROOT = Path(os.environ.get("EVAL_HF_ROOT", str(_EVAL_DIR / "hf_cache")))


def is_offline_eval() -> bool:
    return os.environ.get("EVAL_OFFLINE", "0") != "0"


def enable_offline_mode(offline: bool | None = None) -> None:
    """Skip HuggingFace Hub network checks; use only local cache."""
    if offline is None:
        offline = is_offline_eval()
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"


def setup_hf_datasets_cache(hf_root: str | Path | None = None) -> dict[str, str]:
    """Point HF hub/datasets caches at a shared directory (not ~/.cache)."""
    root = Path(hf_root or DEFAULT_HF_ROOT)
    paths = {
        "HF_HOME": str(root / "hf_home"),
        "HF_DATASETS_CACHE": str(root / "hf_datasets"),
        "HUGGINGFACE_HUB_CACHE": str(root / "hf_hub"),
    }
    for key, value in paths.items():
        os.environ[key] = value
        Path(value).mkdir(parents=True, exist_ok=True)
    enable_offline_mode()
    return paths


def load_dataset(*args, **kwargs):
    kwargs.setdefault("trust_remote_code", True)
    if is_offline_eval():
        kwargs.setdefault("download_mode", DownloadMode.REUSE_DATASET_IF_EXISTS)
    return _load_dataset(*args, **kwargs)
