"""Print the main lm_eval table for one or more eval outputs.

    python eval/collect.py <eval_out_dir | ckpt_dir> [...]

A ckpt dir resolves to its newest eval_results/official_lmeval7_*/results.json.
Metric convention (frozen): ARC / HellaSwag / OBQA / PIQA -> acc_norm, Winogrande / SIQA -> acc.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

METRIC = {
    "arc_easy": "acc_norm",
    "arc_challenge": "acc_norm",
    "hellaswag": "acc_norm",
    "openbookqa": "acc_norm",
    "piqa": "acc_norm",
    "winogrande": "acc",
    "social_iqa": "acc",
}
COLS = [
    ("arc_easy", "ARC-E"),
    ("arc_challenge", "ARC-C"),
    ("hellaswag", "HS"),
    ("openbookqa", "OBQA"),
    ("piqa", "PIQA"),
    ("winogrande", "WG"),
    ("social_iqa", "SIQA"),
]


def _results_json(arg: str) -> Path:
    p = Path(arg)
    if p.is_file():
        return p
    if (p / "results.json").is_file():
        return p / "results.json"
    root = p.parent.parent / "eval_results" / p.name if p.parent.name == "snapshots" else p / "eval_results"
    found = sorted(root.glob("official_lmeval7_*/results.json"))
    if not found:
        raise FileNotFoundError(f"no results.json under {p}")
    return found[-1]


def main(args: list[str]) -> None:
    if not args:
        raise SystemExit(__doc__)
    head = ["run", "H", "step", "dtype", "cap", "limit", "holdout gap"] + [c for _, c in COLS] + ["avg"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for arg in args:
        path = _results_json(arg)
        r = json.loads(path.read_text())
        info = r["checkpoint"]
        hold = r.get("holdout") or {}
        gap = hold.get("gap")
        cap = "off" if (r["provenance"]["env"].get("MOE_SKIP_CAPACITY") == "1") else "on"
        scores = []
        cells = []
        for task, _ in COLS:
            v = (r["results"].get(task) or {}).get(f"{METRIC[task]},none")
            cells.append("-" if v is None else f"{100 * v:.1f}")
            if v is not None:
                scores.append(v)
        avg = f"{100 * sum(scores) / len(scores):.1f}" if len(scores) == len(COLS) else "-"
        row = [
            info.get("run_name") or Path(info["ckpt_path"]).name,
            str(info.get("num_loops")),
            str(info.get("step")),
            str(info.get("param_dtype", info.get("fwd_dtype"))),
            cap,
            str(r.get("limit") or "full"),
            "-" if gap is None else f"{gap:+.4f}",
            *cells,
            avg,
        ]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
