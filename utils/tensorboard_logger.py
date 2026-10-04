from __future__ import annotations

import json
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """TensorBoard + PNG exporter with resume-safe history.

    Scalars are persisted to ``{plot_dir}/../tensorboard_scalars.json`` so that
    pause→resume training redraws the full curve instead of only the current segment.
    """

    def __init__(self, log_dir: str, plot_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.plot_dir = plot_dir
        self.scalars: dict[str, list[tuple[int, float]]] = {}
        self._history_path = os.path.join(os.path.dirname(plot_dir.rstrip("/")), "tensorboard_scalars.json")
        self._load_history()

    def _load_history(self) -> None:
        if not os.path.isfile(self._history_path):
            return
        try:
            with open(self._history_path, encoding="utf-8") as f:
                raw = json.load(f)
            for tag, points in raw.items():
                cleaned: list[tuple[int, float]] = []
                for item in points:
                    if not isinstance(item, (list, tuple)) or len(item) != 2:
                        continue
                    cleaned.append((int(item[0]), float(item[1])))
                # Dedup by step (keep last)
                by_step = {s: v for s, v in cleaned}
                self.scalars[tag] = sorted(by_step.items(), key=lambda x: x[0])
            print(
                f"[TB] Restored {sum(len(v) for v in self.scalars.values())} scalar points "
                f"from {self._history_path}",
                flush=True,
            )
        except Exception as exc:
            print(f"[TB] WARN: failed to load history {self._history_path}: {exc}", flush=True)

    def _save_history(self) -> None:
        os.makedirs(os.path.dirname(self._history_path) or ".", exist_ok=True)
        payload = {tag: [[s, v] for s, v in points] for tag, points in self.scalars.items()}
        tmp = self._history_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, self._history_path)

    @staticmethod
    def _to_scalar(value: Any) -> float | None:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            return float(value.detach().cpu().item())
        return None

    def log(self, metrics: dict[str, Any], step: int) -> None:
        for name, value in metrics.items():
            scalar = self._to_scalar(value)
            if scalar is None:
                continue
            self.writer.add_scalar(name, scalar, step)
            pts = self.scalars.setdefault(name, [])
            # Replace same-step point if re-logged (eval retries etc.)
            if pts and pts[-1][0] == step:
                pts[-1] = (step, scalar)
            else:
                pts.append((step, scalar))

    def save_plots(self) -> None:
        if not self.scalars:
            return

        self._save_history()
        os.makedirs(self.plot_dir, exist_ok=True)

        for tag, points in self.scalars.items():
            if not points:
                continue
            # Stable order
            points = sorted(points, key=lambda x: x[0])
            steps, values = zip(*points)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(steps, values, linewidth=1.5)
            ax.set_xlabel("step")
            ax.set_ylabel(tag)
            ax.set_title(tag)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            out = os.path.join(self.plot_dir, f"{tag.replace('/', '_')}.png")
            fig.savefig(out)
            plt.close(fig)

    def close(self) -> None:
        self.save_plots()
        self.writer.close()
