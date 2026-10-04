"""Pack a parquet text corpus into the DocumentDataset layout used by pretrain.py.

Layout produced (see dataset_document.py):

    <out>/metadata.json
    <out>/tokens.npy          uint16 token stream
    <out>/epoch_K/doc_start.npy, doc_len.npy

Each document is tokenized, an EOS token is appended, and documents longer than
--seq-len are split into consecutive chunks. Each epoch_K directory is a
different shuffle of the same documents.

Example:
    python tools/pack_document_data.py \
        --parquet 'fineweb-edu/sample/10BT/*.parquet' \
        --tokenizer HuggingFaceTB/SmolLM-360M \
        --out data/fineweb-edu-1024 \
        --seq-len 1024 --epochs 4 --workers 32
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

_TOKENIZER = None
_ARGS = None


def _init_worker(tokenizer_path: str, seq_len: int, text_column: str) -> None:
    global _TOKENIZER, _ARGS
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path)
    _ARGS = (seq_len, text_column)


def _read_texts(path: str, row_group, text_column: str) -> list[str]:
    if row_group is None:  # jsonl / jsonl.zst shard
        import json as _json

        if path.endswith(".zst"):
            import io

            import zstandard as zstd

            with open(path, "rb") as fh:
                reader = zstd.ZstdDecompressor().stream_reader(fh)
                lines = io.TextIOWrapper(reader, encoding="utf-8").readlines()
        else:
            with open(path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
        out = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            rec = _json.loads(line)
            text = rec.get(text_column)
            if text:
                out.append(text)
        return out
    table = pq.ParquetFile(path).read_row_group(row_group, columns=[text_column])
    return table.column(text_column).to_pylist()


def _tokenize_chunk(job):
    """Tokenize one parquet row-group or one jsonl shard. -> (tokens uint16, lens int32, is_holdout)."""
    path, row_group, max_docs, is_holdout = job
    seq_len, text_column = _ARGS
    texts = _read_texts(path, row_group, text_column)
    if max_docs is not None:
        texts = texts[:max_docs]

    eos = _TOKENIZER.eos_token_id
    if eos is None:
        raise ValueError("tokenizer has no eos_token_id")

    out_tokens: list[np.ndarray] = []
    out_lens: list[int] = []
    batch = 1000
    for i in range(0, len(texts), batch):
        encoded = _TOKENIZER(texts[i : i + batch], add_special_tokens=False)["input_ids"]
        for ids in encoded:
            ids = ids + [eos]
            # Split over-long documents; drop tails shorter than 2 tokens
            # (dataset_document.py needs doc_len >= 2 for input/label shift).
            for start in range(0, len(ids), seq_len):
                piece = ids[start : start + seq_len]
                if len(piece) < 2:
                    continue
                out_tokens.append(np.asarray(piece, dtype=np.uint16))
                out_lens.append(len(piece))
    if not out_lens:
        return np.zeros(0, dtype=np.uint16), np.zeros(0, dtype=np.int32), is_holdout
    return np.concatenate(out_tokens), np.asarray(out_lens, dtype=np.int32), is_holdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", help="glob of parquet files")
    ap.add_argument("--jsonl", help="glob of jsonl / jsonl.zst shards (e.g. dolma3)")
    ap.add_argument("--shard-offset", type=int, default=0,
                    help="start index into the sorted shard list; with --shard-stride this "
                         "carves out a slice that is disjoint from other offsets")
    ap.add_argument("--shard-stride", type=int, default=1,
                    help="keep every Nth shard, to subsample a large corpus across all sources")
    ap.add_argument("--shuffle-shards", action="store_true",
                    help="shuffle the kept shards before packing; required with --target-tokens so "
                         "an early stop still covers every source family")
    ap.add_argument("--holdout-stride", type=int, default=0,
                    help="every Nth kept shard becomes holdout instead of training data; "
                         "spreads the validation split across source families (dolma3 style)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=9, help="number of shuffled training orderings (epoch_0..)")
    ap.add_argument("--holdout-epoch", type=int, default=9,
                    help="epoch_N written as a held-out split; must match pretrain.py eval_epoch")
    ap.add_argument("--holdout-max-tokens", type=int, default=50_000_000,
                    help="cap on the holdout split size")
    ap.add_argument("--holdout-tokens", type=int, default=20_000_000,
                    help="tokens reserved for the held-out split (0 = reuse training docs)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--target-tokens", type=int, default=0, help="stop after this many tokens (0 = all)")
    ap.add_argument("--max-docs-per-group", type=int, default=None, help="debug: cap docs per row group")
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if bool(args.parquet) == bool(args.jsonl):
        raise SystemExit("pass exactly one of --parquet / --jsonl")
    pattern = args.parquet or args.jsonl
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no files matched {pattern}")
    if args.shard_stride > 1 or args.shard_offset:
        files = files[args.shard_offset :: args.shard_stride]
        print(
            f"[pack] shard slice [{args.shard_offset}::{args.shard_stride}] "
            f"-> {len(files)} shards kept",
            flush=True,
        )
    if args.shuffle_shards:
        # Shard sizes range from a few documents to tens of MB, and shards are
        # grouped by source family, so processing them in path order biases both
        # --target-tokens early stops and the holdout split.
        np.random.default_rng(args.seed).shuffle(files)
        print(f"[pack] shards shuffled (seed {args.seed})", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for idx, path in enumerate(files):
        holdout = args.holdout_stride > 0 and (idx + 1) % args.holdout_stride == 0
        if args.jsonl:
            jobs.append((path, None, args.max_docs_per_group, holdout))
        else:
            for rg in range(pq.ParquetFile(path).num_row_groups):
                jobs.append((path, rg, args.max_docs_per_group, holdout))
    n_hold = sum(1 for j in jobs if j[3])
    print(
        f"[pack] {len(files)} files, {len(jobs)} chunks ({n_hold} holdout), {args.workers} workers",
        flush=True,
    )

    tokens_path = out / "tokens.npy"
    tmp_bin = out / "tokens.bin.tmp"
    total_tokens = 0
    all_lens: list[np.ndarray] = []
    holdout_flags: list[np.ndarray] = []
    started = time.time()

    with open(tmp_bin, "wb") as sink, Pool(
        args.workers, initializer=_init_worker, initargs=(args.tokenizer, args.seq_len, args.text_column)
    ) as pool:
        for done, (tokens, lens, is_holdout) in enumerate(
            pool.imap(_tokenize_chunk, jobs, chunksize=1), 1
        ):
            if lens.size:
                sink.write(tokens.tobytes())
                all_lens.append(lens)
                holdout_flags.append(np.full(lens.size, is_holdout, dtype=bool))
                total_tokens += int(tokens.size)
            if done % 10 == 0 or done == len(jobs):
                rate = total_tokens / max(time.time() - started, 1e-9)
                print(
                    f"[pack] {done}/{len(jobs)} groups  {total_tokens/1e9:.3f}B tokens  "
                    f"{rate/1e6:.2f}M tok/s",
                    flush=True,
                )
            if args.target_tokens and total_tokens >= args.target_tokens:
                print(f"[pack] reached target {args.target_tokens/1e9:.2f}B tokens, stopping", flush=True)
                pool.terminate()
                break

    doc_len = np.concatenate(all_lens) if all_lens else np.zeros(0, dtype=np.int32)
    doc_start = np.zeros(doc_len.size, dtype=np.int64)
    np.cumsum(doc_len[:-1], dtype=np.int64, out=doc_start[1:])

    # Materialise tokens.npy (np.load mmap needs the .npy header).
    stream = np.memmap(tmp_bin, dtype=np.uint16, mode="r", shape=(total_tokens,))
    arr = np.lib.format.open_memmap(
        tokens_path, mode="w+", dtype=np.uint16, shape=(total_tokens,)
    )
    step = 1 << 26
    for i in range(0, total_tokens, step):
        arr[i : i + step] = stream[i : i + step]
    arr.flush()
    del arr, stream
    tmp_bin.unlink()

    rng = np.random.default_rng(args.seed)

    # Reserve the tail of the corpus as a held-out split. pretrain.py evaluates on
    # epoch_{eval_epoch} (default 9) when data.val_path is null, so writing the
    # holdout there makes val loss a real holdout instead of reshuffled train data.
    n_docs = int(doc_len.size)
    is_holdout = (
        np.concatenate(holdout_flags) if holdout_flags else np.zeros(n_docs, dtype=bool)
    )
    if args.holdout_stride > 0 and is_holdout.any():
        # Shard-level holdout: whole shards, spread across every source family.
        train_idx = np.flatnonzero(~is_holdout)
        holdout_idx_all = np.flatnonzero(is_holdout)
    else:
        holdout_docs = 0
        if args.holdout_tokens > 0:
            cumulative = np.cumsum(doc_len[::-1], dtype=np.int64)
            holdout_docs = int(np.searchsorted(cumulative, args.holdout_tokens) + 1)
            holdout_docs = min(holdout_docs, n_docs // 10)
        train_idx = np.arange(n_docs - holdout_docs)
        holdout_idx_all = np.arange(n_docs - holdout_docs, n_docs)
    train_docs = int(train_idx.size)

    if args.epochs > args.holdout_epoch:
        raise SystemExit(
            f"--epochs {args.epochs} would reuse epoch_{args.holdout_epoch} (the holdout) for training"
        )

    for epoch in range(args.epochs):
        epoch_dir = out / f"epoch_{epoch}"
        epoch_dir.mkdir(exist_ok=True)
        order = train_idx[rng.permutation(train_docs)]
        np.save(epoch_dir / "doc_start.npy", doc_start[order])
        np.save(epoch_dir / "doc_len.npy", doc_len[order])

    holdout_dir = out / f"epoch_{args.holdout_epoch}"
    holdout_dir.mkdir(exist_ok=True)
    holdout_idx = holdout_idx_all if holdout_idx_all.size else rng.permutation(n_docs)
    if args.holdout_max_tokens > 0 and holdout_idx.size:
        keep = np.searchsorted(np.cumsum(doc_len[holdout_idx], dtype=np.int64),
                               args.holdout_max_tokens) + 1
        holdout_idx = holdout_idx[: min(keep, holdout_idx.size)]
    np.save(holdout_dir / "doc_start.npy", doc_start[holdout_idx])
    np.save(holdout_dir / "doc_len.npy", doc_len[holdout_idx])
    # Fill any gap so epoch_* directories are contiguous for _count_data_epochs.
    for epoch in range(args.epochs, args.holdout_epoch):
        gap_dir = out / f"epoch_{epoch}"
        gap_dir.mkdir(exist_ok=True)
        order = train_idx[rng.permutation(train_docs)]
        np.save(gap_dir / "doc_start.npy", doc_start[order])
        np.save(gap_dir / "doc_len.npy", doc_len[order])
    print(
        f"[pack] holdout: {holdout_idx.size} docs ({int(doc_len[holdout_idx].sum())/1e6:.1f}M tokens) "
        f"-> epoch_{args.holdout_epoch}",
        flush=True,
    )

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    metadata = {
        "tokenizer_info": {
            "vocab_size": int(len(tok)),
            "name_or_path": args.tokenizer,
            "eos_token_id": int(tok.eos_token_id),
        },
        "max_seq_len": args.seq_len,
        "total_length": int(total_tokens),
        "layout": "document",
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"[pack] done: {doc_len.size} docs, {total_tokens/1e9:.3f}B tokens, "
        f"{args.epochs} training orderings -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
