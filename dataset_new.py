from dataclasses import dataclass, fields
from typing import Optional, Any
from pathlib import Path
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


class V1DatasetConfig(pydantic.BaseModel):
    seed: int
    dataset_path: str
    batch_max_length: int
    drop_last_batch: bool

    target_only: bool

    rank: int
    num_replicas: int
    fixed_epoch: Optional[int] = None  # if set, always read this epoch (for eval)
    # Mid-train resume (O(1) seek into multipack cursor; no batch replay).
    start_data_epoch: int = 0
    resume_sampler_start_index: int = 0
    # Oversample packed QA rows whose response is a bare A/B/C/D (+ eoa).
    # Does not change tokens.npy; only repeats those sample indices in the multipack stream.
    letter_answer_upweight: int = 1
    # Oversample CoT-conditioned and/or long responses (reasoning signal), not letter tokens.
    cot_upweight: int = 1
    long_resp_upweight: int = 1
    long_resp_min_tokens: int = 64


class V1DatasetMeta(pydantic.BaseModel):
    tokenizer_info: dict[str, Any] = {}
    vocab_size: Optional[int] = None
    max_seq_len: int
    total_length: int


@dataclass
class V1DatasetIndices:
    inst_start: np.ndarray
    inst_len: np.ndarray
    resp_start: np.ndarray
    resp_len: np.ndarray


class V1Dataset(IterableDataset):
    def __init__(self, config: V1DatasetConfig):
        super().__init__()
        self.config = config
        self.metadata = self._load_metadata()

        # State
        self._data: Optional[np.ndarray] = None
        self._data_indices: Optional[V1DatasetIndices] = None
        self._sampler: Optional[MultipackDistributedBatchSampler] = None
        self._epoch = int(config.start_data_epoch)
        self._resume_sampler_start = int(config.resume_sampler_start_index)
        self._active_data_epoch: Optional[int] = None
        # Maps multipack length-index -> original row index (identity unless letter upweight).
        self._sample_map: Optional[np.ndarray] = None
        self._letter_token_ids = self._resolve_letter_token_ids()
        self._eoa_id = self._resolve_eoa_id()
        self._cot_token_id = self._resolve_condition_token_id("cot")
        self._boq_id = self._resolve_special_token_id("boq")

    def _resolve_letter_token_ids(self) -> set[int]:
        """Bare A/B/C/D token ids for this BPE (verified single-token)."""
        info = self.metadata.tokenizer_info
        path = info.get("tokenizer_path", "")
        try:
            from transformers import AutoTokenizer

            tok_path = path
            if tok_path.endswith("tokenizer.json"):
                tok_path = str(Path(tok_path).parent)
            tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
            ids: set[int] = set()
            for letter in "ABCD":
                enc = tokenizer.encode(letter, add_special_tokens=False)
                if len(enc) == 1:
                    ids.add(int(enc[0]))
            if ids:
                return ids
        except Exception:
            pass
        # Fallback for the project BPE used in HRM-Text-data-tokenized-1024
        return {63, 64, 65, 66}

    def _resolve_eoa_id(self) -> Optional[int]:
        info = self.metadata.tokenizer_info
        eoa = info.get("eoa")
        path = info.get("tokenizer_path", "")
        if not eoa or not path:
            return None
        try:
            from transformers import AutoTokenizer

            tok_path = path
            if tok_path.endswith("tokenizer.json"):
                tok_path = str(Path(tok_path).parent)
            tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
            tid = tokenizer.convert_tokens_to_ids(eoa)
            return int(tid) if tid is not None and tid >= 0 else None
        except Exception:
            return None

    def _resolve_special_token_id(self, key: str) -> Optional[int]:
        info = self.metadata.tokenizer_info
        token = info.get(key)
        path = info.get("tokenizer_path", "")
        if not token or not path:
            return None
        try:
            from transformers import AutoTokenizer

            tok_path = path
            if tok_path.endswith("tokenizer.json"):
                tok_path = str(Path(tok_path).parent)
            tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
            tid = tokenizer.convert_tokens_to_ids(token)
            return int(tid) if tid is not None and tid >= 0 else None
        except Exception:
            return None

    def _resolve_condition_token_id(self, name: str) -> Optional[int]:
        info = self.metadata.tokenizer_info
        mapping = info.get("condition_mapping") or {}
        token = mapping.get(name)
        path = info.get("tokenizer_path", "")
        if not token or not path:
            return None
        try:
            from transformers import AutoTokenizer

            tok_path = path
            if tok_path.endswith("tokenizer.json"):
                tok_path = str(Path(tok_path).parent)
            tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
            tid = tokenizer.convert_tokens_to_ids(token)
            return int(tid) if tid is not None and tid >= 0 else None
        except Exception:
            return None

    def _env_int(self, name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None or str(raw).strip() == "":
            return max(1, int(default))
        return max(1, int(raw))

    def _repeat_indices(self, hits: np.ndarray, upw: int, seed_tag: int) -> np.ndarray:
        if hits.size == 0 or upw <= 1:
            return np.empty(0, dtype=np.int64)
        return np.repeat(hits.astype(np.int64, copy=False), upw - 1)

    def _load_or_build_row_cache(
        self,
        cache_path: str,
        build_fn,
        epoch_idx: int,
        label: str,
    ) -> np.ndarray:
        """Rank0 builds a derived index cache; other ranks wait. Does not modify tokens.npy.

        Always writes the cache file (even if empty) so non-zero ranks never hang.
        """
        if os.path.isfile(cache_path):
            arr = np.load(cache_path)
            if self.config.rank == 0:
                print(f"[data] {label} cache hit: {arr.size} rows (epoch_{epoch_idx})", flush=True)
            return arr

        if self.config.rank == 0:
            arr = build_fn()
            arr = np.asarray(arr, dtype=np.int64).reshape(-1)
            try:
                tmp = cache_path + ".tmp"
                np.save(tmp, arr)
                os.replace(tmp, cache_path)
                print(
                    f"[data] wrote {label} cache: {cache_path} ({arr.size} rows)",
                    flush=True,
                )
            except OSError as e:
                # Still return arr on rank0; signal peers via a sidecar so they do not hang.
                print(f"[data] {label} cache write skipped: {e}", flush=True)
                try:
                    done = cache_path + ".empty"
                    with open(done, "w", encoding="utf-8") as f:
                        f.write(str(arr.size))
                except OSError:
                    pass
            return arr

        import time

        for _ in range(3600):
            if os.path.isfile(cache_path):
                return np.load(cache_path)
            empty_mark = cache_path + ".empty"
            if os.path.isfile(empty_mark):
                return np.empty(0, dtype=np.int64)
            time.sleep(2)
        raise RuntimeError(f"timed out waiting for {label} cache: {cache_path}")

    def _is_letter_answer_row(self, resp_start: int, resp_len: int) -> bool:
        assert self._data is not None
        if resp_len <= 0 or resp_len > 4:
            return False
        resp = self._data[resp_start : resp_start + resp_len]
        tokens = [int(t) for t in resp.tolist()]
        if self._eoa_id is not None:
            tokens = [t for t in tokens if t != self._eoa_id]
        return len(tokens) == 1 and tokens[0] in self._letter_token_ids

    def _load_metadata(self) -> V1DatasetMeta:
        with open(os.path.join(self.config.dataset_path, "metadata.json"), "r") as f:
            metadata = V1DatasetMeta(**json.load(f))
            # Account for autoregressive shift
            metadata.max_seq_len -= 1
            # Compute vocab size from tokenizer info
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

    def _resolve_epoch_idx(self) -> int:
        """Map monotonic self._epoch → existing epoch_k/ (wrap when past last file)."""
        if self.config.fixed_epoch is not None:
            return int(self.config.fixed_epoch)
        return int(self._epoch) % self._count_data_epochs()

    def _load_dataset_before_epoch_begin(self):
        # Load tokens (only if not loaded)
        if self._data is None:
            self._data = np.load(os.path.join(self.config.dataset_path, "tokens.npy"), mmap_mode="r")

        # Load indices (wrap epoch_* when training past the last packed file)
        epoch_idx = self._resolve_epoch_idx()
        self._active_data_epoch = int(epoch_idx)
        self._data_indices = V1DatasetIndices(**{f.name: np.load(os.path.join(self.config.dataset_path, f"epoch_{epoch_idx}", f"{f.name}.npy"), mmap_mode="r")
                                                 for f in fields(V1DatasetIndices)})
        if self.config.fixed_epoch is None:
            self._epoch += 1
            n_ep = self._count_data_epochs()
            if self.config.rank == 0 and self._epoch > n_ep and (self._epoch - 1) % n_ep == 0:
                print(
                    f"[data] cycling data epochs: loaded epoch_{epoch_idx} "
                    f"(loop counter={self._epoch - 1})",
                    flush=True,
                )

        base_lengths = self._data_indices.inst_len + self._data_indices.resp_len - 1  # AR shift
        n = int(base_lengths.shape[0])
        sample_map = np.arange(n, dtype=np.int64)

        upw = int(self.config.letter_answer_upweight)
        upw_env = os.environ.get("LETTER_ANSWER_UPWEIGHT")
        if upw_env is not None and str(upw_env).strip() != "":
            upw = int(upw_env)
        upw = max(1, upw)

        if upw > 1 and self.config.fixed_epoch is None:
            if self._data is None:
                self._data = np.load(os.path.join(self.config.dataset_path, "tokens.npy"), mmap_mode="r")
            rs = self._data_indices.resp_start
            rl = self._data_indices.resp_len
            # Typical Platypus MCQ: one letter token + eoa (resp_len==2). Allow 3 for rare variants.
            cache_dir = os.path.join(self.config.dataset_path, f"epoch_{epoch_idx}")
            cache_rows = os.path.join(cache_dir, "letter_answer_rows.npy")
            cache_first = os.path.join(cache_dir, "letter_answer_first.npy")
            letter_hits_arr = np.empty(0, dtype=np.int64)
            first_letters = np.empty(0, dtype=np.int64)
            if os.path.isfile(cache_rows) and os.path.isfile(cache_first):
                letter_hits_arr = np.load(cache_rows)
                first_letters = np.load(cache_first)
                if self.config.rank == 0:
                    print(
                        f"[data] letter-answer cache hit: {letter_hits_arr.size} rows "
                        f"(epoch_{epoch_idx})",
                        flush=True,
                    )
            else:
                # Only rank 0 scans mmap; others wait for the derived index cache.
                if self.config.rank == 0:
                    cand = (rl >= 2) & (rl <= 3)
                    if np.any(cand):
                        cand_idx = np.flatnonzero(cand)
                        chunk = 1_000_000
                        hit_parts: list[np.ndarray] = []
                        letter_parts: list[np.ndarray] = []
                        letter_id_list = list(self._letter_token_ids)
                        eoa = int(self._eoa_id) if self._eoa_id is not None else None
                        n_chunks = (cand_idx.size + chunk - 1) // chunk
                        for ci, s in enumerate(range(0, cand_idx.size, chunk)):
                            if ci % 20 == 0:
                                print(
                                    f"[data] letter-answer scan {ci}/{n_chunks} "
                                    f"(cand={cand_idx.size})",
                                    flush=True,
                                )
                            sub = cand_idx[s:s + chunk]
                            first = np.asarray(self._data[rs[sub]], dtype=np.int64)
                            letter_local = np.isin(first, letter_id_list)
                            if eoa is not None:
                                letter_local &= first != eoa
                            if np.any(letter_local):
                                hit_parts.append(sub[letter_local].astype(np.int64, copy=False))
                                letter_parts.append(first[letter_local].astype(np.int64, copy=False))
                        if hit_parts:
                            letter_hits_arr = np.concatenate(hit_parts)
                            first_letters = np.concatenate(letter_parts)
                    if letter_hits_arr.size:
                        try:
                            tmp_rows = cache_rows + ".tmp"
                            tmp_first = cache_first + ".tmp"
                            np.save(tmp_rows, letter_hits_arr)
                            np.save(tmp_first, first_letters)
                            os.replace(tmp_rows, cache_rows)
                            os.replace(tmp_first, cache_first)
                            print(
                                f"[data] wrote letter-answer cache: {cache_rows} "
                                f"({letter_hits_arr.size} rows)",
                                flush=True,
                            )
                        except OSError as e:
                            print(f"[data] letter-answer cache write skipped: {e}", flush=True)
                else:
                    import time

                    for _ in range(3600):
                        if os.path.isfile(cache_rows) and os.path.isfile(cache_first):
                            break
                        time.sleep(2)
                    else:
                        raise RuntimeError(
                            f"timed out waiting for letter-answer cache under {cache_dir}"
                        )
                if self.config.rank != 0 or not letter_hits_arr.size:
                    if os.path.isfile(cache_rows) and os.path.isfile(cache_first):
                        letter_hits_arr = np.load(cache_rows)
                        first_letters = np.load(cache_first)
            if letter_hits_arr.size:
                # Class-balance A/B/C/D then scale: naive repeat amplifies majority letter (A),
                # which worsens single-letter collapse. Target per letter ≈ (total_letter * upw) / K.
                letter_ids = sorted(self._letter_token_ids)
                groups = {
                    lid: letter_hits_arr[first_letters == lid]
                    for lid in letter_ids
                    if np.any(first_letters == lid)
                }
                n_letters = max(1, len(groups))
                target_each = max(1, int(letter_hits_arr.size * upw / n_letters))
                # Deterministic across ranks (sampler shards the shared sample_map).
                rng = np.random.RandomState(self.config.seed + 10007 * int(epoch_idx) + upw)
                extra_parts: list[np.ndarray] = []
                for lid, idxs in groups.items():
                    # Base sample_map already includes idxs once; only add extras to reach target_each.
                    need = max(0, target_each - int(idxs.size))
                    if need > 0:
                        extra_parts.append(rng.choice(idxs, size=need, replace=True).astype(np.int64))
                extras = np.concatenate(extra_parts) if extra_parts else np.empty(0, dtype=np.int64)
                sample_map = np.concatenate([sample_map, extras])
                if self.config.rank == 0:
                    counts0 = {lid: int(g.size) for lid, g in groups.items()}
                    print(
                        f"[data] letter-answer upweight={upw} (class-balanced): "
                        f"{letter_hits_arr.size}/{n} rows, counts={counts0}, "
                        f"target_each={target_each} → +{extras.size} extras "
                        f"(stream {sample_map.size})",
                        flush=True,
                    )
            elif self.config.rank == 0:
                print(
                    f"[data] letter-answer upweight={upw}: no letter rows found in epoch_{epoch_idx}",
                    flush=True,
                )

        # --- Reasoning upsampling (CoT condition / long answers); not letter CE hacks ---
        cot_upw = self._env_int("COT_UPWEIGHT", int(self.config.cot_upweight))
        long_upw = self._env_int("LONG_RESP_UPWEIGHT", int(self.config.long_resp_upweight))
        long_min = int(os.environ.get("LONG_RESP_MIN_TOKENS") or self.config.long_resp_min_tokens or 64)
        long_min = max(2, long_min)

        if self.config.fixed_epoch is None and (cot_upw > 1 or long_upw > 1):
            if self._data is None:
                self._data = np.load(os.path.join(self.config.dataset_path, "tokens.npy"), mmap_mode="r")
            cache_dir = os.path.join(self.config.dataset_path, f"epoch_{epoch_idx}")
            rl = self._data_indices.resp_len
            inst_start = self._data_indices.inst_start
            inst_len = self._data_indices.inst_len

            if cot_upw > 1 and self._cot_token_id is not None:
                cot_cache = os.path.join(cache_dir, "cot_rows.npy")
                cot_id = int(self._cot_token_id)
                max_cond = 4  # e.g. synth,cot

                def _build_cot_fast():
                    # Condition tokens sit right after <boq> (inst_start+1 .. +max_cond).
                    cand = inst_len > 1
                    cand_idx = np.flatnonzero(cand)
                    hit_parts: list[np.ndarray] = []
                    chunk = 1_000_000
                    for s in range(0, cand_idx.size, chunk):
                        sub = cand_idx[s : s + chunk]
                        matched = np.zeros(sub.size, dtype=bool)
                        for off in range(1, max_cond + 1):
                            ok = inst_len[sub] > off
                            if not np.any(ok):
                                continue
                            pos = inst_start[sub[ok]] + off
                            toks = np.asarray(self._data[pos], dtype=np.int64)
                            matched_ok = matched[ok]
                            matched_ok |= toks == cot_id
                            matched[ok] = matched_ok
                        if np.any(matched):
                            hit_parts.append(sub[matched].astype(np.int64, copy=False))
                        if self.config.rank == 0 and (s // chunk) % 20 == 0:
                            print(
                                f"[data] cot scan chunk {s // chunk}/{(cand_idx.size + chunk - 1) // chunk}",
                                flush=True,
                            )
                    return np.concatenate(hit_parts) if hit_parts else np.empty(0, dtype=np.int64)

                cot_hits = self._load_or_build_row_cache(cot_cache, _build_cot_fast, epoch_idx, "cot")
                extras = self._repeat_indices(cot_hits, cot_upw, epoch_idx)
                if extras.size:
                    sample_map = np.concatenate([sample_map, extras])
                if self.config.rank == 0:
                    print(
                        f"[data] cot upweight={cot_upw}: {cot_hits.size}/{n} rows → +{extras.size} extras "
                        f"(stream {sample_map.size})",
                        flush=True,
                    )
            elif cot_upw > 1 and self.config.rank == 0:
                print("[data] cot upweight requested but cot token id unresolved; skipped", flush=True)

            if long_upw > 1:
                long_cache = os.path.join(cache_dir, f"long_resp_ge{long_min}_rows.npy")

                def _build_long():
                    return np.flatnonzero(rl >= long_min).astype(np.int64)

                long_hits = self._load_or_build_row_cache(long_cache, _build_long, epoch_idx, f"long_resp>={long_min}")
                extras = self._repeat_indices(long_hits, long_upw, epoch_idx + 17)
                if extras.size:
                    sample_map = np.concatenate([sample_map, extras])
                if self.config.rank == 0:
                    print(
                        f"[data] long-resp upweight={long_upw} (min_len={long_min}): "
                        f"{long_hits.size}/{n} rows → +{extras.size} extras "
                        f"(stream {sample_map.size})",
                        flush=True,
                    )

        self._sample_map = sample_map
        lengths = base_lengths[sample_map]

        # Re-create sampler
        self._sampler = MultipackDistributedBatchSampler(
            lengths=lengths,
            batch_max_length=self.config.batch_max_length,
            drop_last_batch=self.config.drop_last_batch,

            rank=self.config.rank,
            num_replicas=self.config.num_replicas
        )

    def _load_batch(self, indices: np.ndarray):
        # Load instructions and responses
        assert self._data is not None and self._data_indices is not None
        if self._sample_map is not None:
            indices = self._sample_map[indices]
        raw = {k: [] for k in ["inst", "resp"]}
        for k, v in raw.items():
            start = getattr(self._data_indices, f"{k}_start")[indices]
            end = start + getattr(self._data_indices, f"{k}_len")[indices]
            for i in range(len(start)):
                v.append(self._data[start[i]: end[i]].astype(np.int32))

        # Form batch
        batch = {k: [] for k in ["inputs", "labels", "position_ids"]}
        seqlens_i, seqlens_o = [], []
        for i, o in zip(raw["inst"], raw["resp"]):
            # PrefixLM: last input token predict first output token, for compatibility with Causal LM frameworks (prefill-phase)
            # Inputs
            batch["inputs"].append(i)
            batch["inputs"].append(o[:-1])
            # Labels (supervision mask only; attention mask is controlled by arch.attn_type /
            # lm_mode. target_only=False still uses PrefixLM bidir on inst unless attn_type=causal.)
            batch["labels"].append(np.full(len(i) - 1, dtype=i.dtype, fill_value=IGNORE_LABEL_ID) if self.config.target_only else i[1:])
            batch["labels"].append(o)
            # Position IDs
            batch["position_ids"].append(np.arange(len(i), dtype=np.int32))
            batch["position_ids"].append(np.arange(len(i), len(i) + len(o) - 1, dtype=np.int32))
            # Seqlens
            seqlens_i.append(len(i))
            seqlens_o.append(len(o) - 1)

        # Concat
        batch = {k: np.concatenate(v, dtype=np.int32) for k, v in batch.items()}
        # pad to fixed len
        pad_len = self.config.batch_max_length - batch["inputs"].shape[0]
        if pad_len > 0:
            pad_values = {
                "inputs": 0,  # FIXME: Pad with an arbitary token.
                "labels": IGNORE_LABEL_ID,
                "position_ids": 0,
            }
            for k in pad_values.keys():
                batch[k] = np.pad(batch[k], (0, pad_len), mode="constant", constant_values=pad_values[k])

        # Compute cu_seqlens
        seqlen_i, seqlen_o = np.array(seqlens_i, dtype=np.int32), np.array(seqlens_o, dtype=np.int32)
        tensors, scalars = compute_aux_seq_tensors_scalars(seqlen_i, seqlen_o, self.config.batch_max_length)

        # to tensor
        return {k: torch.from_numpy(v) for k, v in (batch | tensors).items()}, scalars

    def __iter__(self):
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1
        # TODO: Feature (Low Priority): Multithreaded data loading

        self._load_dataset_before_epoch_begin()

        assert self._sampler is not None
        start_index = int(self._resume_sampler_start)
        self._resume_sampler_start = 0  # only the first __iter__ after resume seeks
        for indices in self._sampler.iter(start_index=start_index):
            batch, scalars = self._load_batch(indices)
            # Tiny resume cursor for the trainer (next multipack start_index).
            scalars = {
                **scalars,
                "_resume_data_epoch": int(self._active_data_epoch if self._active_data_epoch is not None else 0),
                "_resume_sampler_start": int(getattr(self._sampler, "cursor", 0)),
                # Multipack cursor denominator for "epoch completion %" in the trainer UI.
                "_resume_epoch_num_samples": int(getattr(self._sampler, "lengths").size),
            }
            yield batch, scalars
