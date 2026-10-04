"""Document-level Causal LM dataset (traditional pretraining packing).

On-disk layout (mirrors V1 QA layout, but with document indices)::

    <dataset_path>/
      metadata.json          # same schema as V1 (tokenizer_info, max_seq_len, total_length)
      tokens.npy             # int token stream (mmap)
      epoch_0/
        doc_start.npy        # int64, start offset into tokens.npy
        doc_len.npy          # int32/int64, length in tokens (>= 2)
      epoch_1/ ...

Each document is packed as a standard next-token LM example:
  inputs = doc[:-1], labels = doc[1:]
  prefix_lens = 0, causal_lens = len(inputs)   # full causal attention

Use with ``data.format=document`` (and preferably ``lm_mode=causal``).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Any
import os
import json

import numpy as np
import pydantic

import torch
from torch.utils.data import IterableDataset, get_worker_info

from models.common import IGNORE_LABEL_ID
from models.prefixlm_attention import compute_aux_seq_tensors_scalars
from models.layers import find_multiple
from multipack_sampler import MultipackDistributedBatchSampler


class DocumentDatasetConfig(pydantic.BaseModel):
    seed: int
    dataset_path: str
    batch_max_length: int
    drop_last_batch: bool

    rank: int
    num_replicas: int
    fixed_epoch: Optional[int] = None
    start_data_epoch: int = 0
    resume_sampler_start_index: int = 0


class DocumentDatasetMeta(pydantic.BaseModel):
    tokenizer_info: dict[str, Any] = {}
    vocab_size: Optional[int] = None
    max_seq_len: int
    total_length: int
    # Optional marker written by 111fineweb / prepare_document_data
    layout: Optional[str] = None


@dataclass
class DocumentDatasetIndices:
    doc_start: np.ndarray
    doc_len: np.ndarray


class DocumentDataset(IterableDataset):
    """Packed document Causal LM loader (drop-in sibling of ``V1Dataset``)."""

    def __init__(self, config: DocumentDatasetConfig):
        super().__init__()
        self.config = config
        self.metadata = self._load_metadata()

        self._data: Optional[np.ndarray] = None
        self._data_indices: Optional[DocumentDatasetIndices] = None
        self._sampler: Optional[MultipackDistributedBatchSampler] = None
        self._epoch = int(config.start_data_epoch)
        self._resume_sampler_start = int(config.resume_sampler_start_index)
        self._active_data_epoch: Optional[int] = None

    def _load_metadata(self) -> DocumentDatasetMeta:
        with open(os.path.join(self.config.dataset_path, "metadata.json"), "r") as f:
            metadata = DocumentDatasetMeta(**json.load(f))
            metadata.max_seq_len -= 1  # autoregressive shift
            assert metadata.vocab_size is None
            metadata.vocab_size = find_multiple(metadata.tokenizer_info.pop("vocab_size"), 256)
            return metadata

    def _count_data_epochs(self) -> int:
        cached = getattr(self, "_num_data_epochs_cached", None)
        if cached is not None:
            return int(cached)
        root = Path(self.config.dataset_path)
        n = sum(1 for p in root.glob("epoch_*") if p.is_dir())
        if n <= 0:
            raise FileNotFoundError(f"no epoch_* dirs under {root}")
        self._num_data_epochs_cached = n
        return n

    def _load_dataset_before_epoch_begin(self):
        if self._data is None:
            self._data = np.load(os.path.join(self.config.dataset_path, "tokens.npy"), mmap_mode="r")

        if self.config.fixed_epoch is not None:
            epoch_idx = int(self.config.fixed_epoch)
        else:
            epoch_idx = int(self._epoch) % self._count_data_epochs()
        self._active_data_epoch = int(epoch_idx)
        epoch_dir = os.path.join(self.config.dataset_path, f"epoch_{epoch_idx}")
        self._data_indices = DocumentDatasetIndices(
            **{
                f.name: np.load(os.path.join(epoch_dir, f"{f.name}.npy"), mmap_mode="r")
                for f in fields(DocumentDatasetIndices)
            }
        )
        if self.config.fixed_epoch is None:
            self._epoch += 1

        # Each doc of length L contributes L-1 supervised positions.
        lengths = np.maximum(self._data_indices.doc_len.astype(np.int64) - 1, 1)
        # seed≠0: permute docs so resume-without-cursor does not replay the same prefix.
        if int(self.config.seed) != 0:
            rng = np.random.RandomState(int(self.config.seed) + int(epoch_idx) * 10007)
            perm = rng.permutation(int(lengths.size))
            lengths = lengths[perm]
            self._data_indices = DocumentDatasetIndices(
                doc_start=self._data_indices.doc_start[perm],
                doc_len=self._data_indices.doc_len[perm],
            )
        self._sampler = MultipackDistributedBatchSampler(
            lengths=lengths,
            batch_max_length=self.config.batch_max_length,
            drop_last_batch=self.config.drop_last_batch,
            rank=self.config.rank,
            num_replicas=self.config.num_replicas,
        )

    def _load_batch(self, indices: np.ndarray):
        assert self._data is not None and self._data_indices is not None

        docs = []
        starts = self._data_indices.doc_start[indices]
        lens = self._data_indices.doc_len[indices]
        for s, n in zip(starts, lens):
            n = int(n)
            if n < 2:
                continue
            docs.append(self._data[int(s) : int(s) + n].astype(np.int32))

        batch = {k: [] for k in ["inputs", "labels", "position_ids"]}
        # prefix_lens = 0 for every doc → full causal path in PrefixLM kernels
        seqlens_prefix = []
        seqlens_causal = []
        for doc in docs:
            inputs = doc[:-1]
            labels = doc[1:]
            batch["inputs"].append(inputs)
            batch["labels"].append(labels)
            batch["position_ids"].append(np.arange(len(inputs), dtype=np.int32))
            seqlens_prefix.append(0)
            seqlens_causal.append(len(inputs))

        if not batch["inputs"]:
            # Degenerate batch: emit a fully-padded empty step (should be rare).
            pad_len = self.config.batch_max_length
            empty = {
                "inputs": np.zeros(pad_len, dtype=np.int32),
                "labels": np.full(pad_len, IGNORE_LABEL_ID, dtype=np.int32),
                "position_ids": np.zeros(pad_len, dtype=np.int32),
            }
            tensors, scalars = compute_aux_seq_tensors_scalars(
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
                self.config.batch_max_length,
            )
            return {k: torch.from_numpy(v) for k, v in (empty | tensors).items()}, scalars

        batch = {k: np.concatenate(v, dtype=np.int32) for k, v in batch.items()}
        pad_len = self.config.batch_max_length - batch["inputs"].shape[0]
        if pad_len > 0:
            pad_values = {
                "inputs": 0,
                "labels": IGNORE_LABEL_ID,
                "position_ids": 0,
            }
            for k, fill in pad_values.items():
                batch[k] = np.pad(batch[k], (0, pad_len), mode="constant", constant_values=fill)

        seqlen_i = np.array(seqlens_prefix, dtype=np.int32)
        seqlen_o = np.array(seqlens_causal, dtype=np.int32)
        tensors, scalars = compute_aux_seq_tensors_scalars(seqlen_i, seqlen_o, self.config.batch_max_length)
        return {k: torch.from_numpy(v) for k, v in (batch | tensors).items()}, scalars

    def __iter__(self):
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1

        self._load_dataset_before_epoch_begin()
        assert self._sampler is not None
        start_index = int(self._resume_sampler_start)
        self._resume_sampler_start = 0
        for indices in self._sampler.iter(start_index=start_index):
            batch, scalars = self._load_batch(indices)
            scalars = {
                **scalars,
                "_resume_data_epoch": int(self._active_data_epoch if self._active_data_epoch is not None else 0),
                "_resume_sampler_start": int(getattr(self._sampler, "cursor", 0)),
                "_resume_epoch_num_samples": int(getattr(self._sampler, "lengths").size),
            }
            yield batch, scalars
