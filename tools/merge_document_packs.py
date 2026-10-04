"""Merge several DocumentDataset packs into one corpus.

Concatenates the token streams and shifts each pack's document offsets, so no
re-tokenisation is needed. All packs must share a tokenizer and seq_len.

Training orderings (epoch_0..) are reshuffled across the union of every pack's
documents; the holdout (epoch_9) is the union of the packs' holdouts, so the
validation mix matches the training mix.

    python tools/merge_document_packs.py \
        --inputs data/dolma3-mix-document-1024 data/fineweb-edu-tokenized-document-1024 \
        --out data/mix-dolma3-fineweb-1024 --epochs 9
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def load_pack(path: Path, holdout_epoch: int) -> dict:
    meta = json.loads((path / "metadata.json").read_text())
    train_dir = path / "epoch_0"
    hold_dir = path / f"epoch_{holdout_epoch}"
    pack = {
        "path": path,
        "meta": meta,
        "tokens": np.load(path / "tokens.npy", mmap_mode="r"),
        "train_start": np.load(train_dir / "doc_start.npy"),
        "train_len": np.load(train_dir / "doc_len.npy"),
    }
    if hold_dir.is_dir():
        pack["hold_start"] = np.load(hold_dir / "doc_start.npy")
        pack["hold_len"] = np.load(hold_dir / "doc_len.npy")
    else:
        pack["hold_start"] = np.zeros(0, dtype=np.int64)
        pack["hold_len"] = np.zeros(0, dtype=np.int32)
    return pack


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=9)
    ap.add_argument("--holdout-epoch", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    packs = [load_pack(Path(p), args.holdout_epoch) for p in args.inputs]
    seq_lens = {p["meta"]["max_seq_len"] for p in packs}
    vocabs = {p["meta"]["tokenizer_info"]["vocab_size"] for p in packs}
    tokenizers = {p["meta"]["tokenizer_info"].get("name_or_path") for p in packs}
    if len(seq_lens) != 1 or len(vocabs) != 1 or len(tokenizers) != 1:
        raise SystemExit(f"packs disagree: seq_len={seq_lens} vocab={vocabs} tokenizer={tokenizers}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total_tokens = sum(int(p["tokens"].size) for p in packs)
    print(f"[merge] {len(packs)} packs, {total_tokens/1e9:.3f}B tokens total", flush=True)

    merged = np.lib.format.open_memmap(
        out / "tokens.npy", mode="w+", dtype=np.uint16, shape=(total_tokens,)
    )
    offsets = []
    cursor = 0
    step = 1 << 26
    for pack in packs:
        offsets.append(cursor)
        src = pack["tokens"]
        for i in range(0, src.size, step):
            chunk = src[i : i + step]
            merged[cursor : cursor + chunk.size] = chunk
            cursor += int(chunk.size)
        print(f"[merge] copied {pack['path'].name}: {src.size/1e9:.3f}B tokens", flush=True)
    merged.flush()
    del merged

    train_start = np.concatenate(
        [p["train_start"].astype(np.int64) + off for p, off in zip(packs, offsets)]
    )
    train_len = np.concatenate([p["train_len"] for p in packs])
    hold_start = np.concatenate(
        [p["hold_start"].astype(np.int64) + off for p, off in zip(packs, offsets)]
    )
    hold_len = np.concatenate([p["hold_len"] for p in packs])

    rng = np.random.default_rng(args.seed)
    for epoch in range(args.holdout_epoch):
        epoch_dir = out / f"epoch_{epoch}"
        epoch_dir.mkdir(exist_ok=True)
        order = rng.permutation(train_start.size)
        np.save(epoch_dir / "doc_start.npy", train_start[order])
        np.save(epoch_dir / "doc_len.npy", train_len[order])

    hold_dir = out / f"epoch_{args.holdout_epoch}"
    hold_dir.mkdir(exist_ok=True)
    order = rng.permutation(hold_start.size)
    np.save(hold_dir / "doc_start.npy", hold_start[order])
    np.save(hold_dir / "doc_len.npy", hold_len[order])

    meta = dict(packs[0]["meta"])
    meta["total_length"] = int(total_tokens)
    meta["sources"] = [
        {
            "path": str(p["path"]),
            "tokens": int(p["tokens"].size),
            "train_docs": int(p["train_start"].size),
            "holdout_docs": int(p["hold_start"].size),
            "offset": off,
        }
        for p, off in zip(packs, offsets)
    ]
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(
        f"[merge] done: {train_start.size} train docs "
        f"({int(train_len.sum())/1e9:.3f}B tokens), "
        f"{hold_start.size} holdout docs ({int(hold_len.sum())/1e6:.1f}M tokens) -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
