"""README animation: which experts one token uses in each loop, from routing.json (real routing).

    python tools/animations/routing.py        # writes assets/expert_routing.gif
"""
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FRAMES = HERE / "frames_routing"
OUT = ROOT / "assets" / "expert_routing.gif"

BLUE, TEAL = "#3A5BD9", "#2BA89A"
TINT = "#D6DEFA"
INK, BG, EDGE = "#1F2328", "#FFFFFF", "#C9D1D9"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK})

d = json.loads((HERE / "routing.json").read_text())
LAYER, TOKEN = 7, 9  # layer 8 of 15; token " disease" in the first held-out document
sets = [set(s) for s in np.array(d["example_layer_loop_sets"])[LAYER, :, TOKEN, :].tolist()]
H, N_EXPERTS, K = d["num_loops"], d["n_routed"], d["top_k"]
token = d["example_tokens"][TOKEN].strip()
cum = np.array([len(set().union(*sets[: t + 1])) for t in range(H)], dtype=float)
avg = np.array(d["cum_distinct_by_loop"])
COLS, ROWS = 6, N_EXPERTS // 6
FPS, T_MOVE, T_HOLD, T_INTRO, T_AVG, T_END = 20, 14, 10, 12, 34, 60


def ease(p):
    p = min(max(p, 0.0), 1.0)
    return p * p * (3 - 2 * p)


def mix(c1, c2, p):
    a = np.array(matplotlib.colors.to_rgb(c1))
    b = np.array(matplotlib.colors.to_rgb(c2))
    return tuple(a + (b - a) * p)


def expert_state(e, t, p):
    """Fill colour, edge colour and radius scale of expert e while moving into loop t (0-based)."""
    used_before = any(e in s for s in sets[:t])
    now, prev = e in sets[t], t > 0 and e in sets[t - 1]
    if now:
        start = BLUE if prev else (TINT if used_before else BG)
        pop = 0.0 if used_before else 0.18 * np.sin(np.pi * p)
        return mix(start, BLUE, p), BLUE, 1 + pop
    if prev:
        return mix(BLUE, TINT, p), BLUE, 1.0
    if used_before:
        return TINT, BLUE, 1.0
    return BG, EDGE, 1.0


def draw(t, p, alpha=1.0, avg_p=0.0):
    fig = plt.figure(figsize=(10, 5), dpi=100, facecolor=BG)
    fig.text(0.04, 0.90, "Each loop calls different experts", fontsize=19, weight="bold")
    fig.text(0.04, 0.835, f"Token “{token}”, layer {LAYER + 1} of {d['n_layers']}, "
             f"1.7B LOOM model with {H} loops", fontsize=11.5)

    ax = fig.add_axes([0.04, 0.08, 0.40, 0.64])
    ax.set_xlim(-0.6, COLS - 0.4)
    ax.set_ylim(-1.5, ROWS - 0.2)
    ax.set_aspect("equal")
    ax.axis("off")
    for e in range(N_EXPERTS):
        r, c = divmod(e, COLS)
        fill, edge, s = expert_state(e, t, p) if t >= 0 else (BG, EDGE, 1.0)
        ax.add_patch(Circle((c, ROWS - 1 - r), 0.36 * s, fc=fill, ec=edge, lw=1.4, alpha=alpha))
    for i in range(H):
        on = i < t or (i == t and p > 0.5)
        ax.add_patch(Circle((COLS / 2 - 0.5 + (i - (H - 1) / 2) * 0.42, -1.05), 0.11,
                            fc=BLUE if on else BG, ec=BLUE, lw=1.2, alpha=alpha))
    loop_label = f"loop {t + 1}" if t >= 0 else ""
    ax.text(COLS / 2 - 0.5, -1.45, loop_label, ha="center", va="top", fontsize=12, weight="bold", alpha=alpha)

    bx = fig.add_axes([0.55, 0.17, 0.40, 0.55])
    bx.set_xlim(0.6, H + 0.4)
    bx.set_ylim(0, N_EXPERTS)
    bx.set_xticks(range(1, H + 1))
    bx.set_yticks([0, 6, 12, 18, 24, 30])
    bx.set_xlabel("loop", fontsize=11)
    bx.set_ylabel("distinct experts used so far", fontsize=11)
    for side in ("top", "right"):
        bx.spines[side].set_visible(False)
    bx.axhline(K, color=INK, lw=1.1, ls=(0, (4, 3)), alpha=alpha)
    bx.text(H + 0.35, K + 0.6, "same experts every loop", ha="right", va="bottom", fontsize=10, alpha=alpha)
    if t >= 0:
        xs = list(range(1, t + 1)) + [t + p] if t > 0 else [1]
        ys = list(cum[:t]) + [cum[t - 1] + (cum[t] - cum[t - 1]) * p] if t > 0 else [K * p]
        bx.plot(xs, ys, color=BLUE, lw=2.6, solid_capstyle="round")
        bx.plot(xs[-1], ys[-1], "o", color=BLUE, ms=7)
        bx.text(xs[-1] + 0.18, ys[-1] + 0.4, f"{ys[-1]:.0f}", color=BLUE, fontsize=12, weight="bold", va="bottom")
    if avg_p > 0:
        n = 1 + (H - 1) * avg_p
        xs = np.linspace(1, n, 60)
        bx.plot(xs, np.interp(xs, np.arange(1, H + 1), avg), color=TEAL, lw=2.6, alpha=min(1, 3 * avg_p))
        if avg_p >= 1:
            bx.text(H - 0.1, avg[-1] + 1.6, f"average over {d['n_tokens']:,} tokens: {avg[-1]:.1f}",
                    ha="right", va="bottom", fontsize=10.5, color=TEAL, weight="bold")
    return fig


def main():
    shutil.rmtree(FRAMES, ignore_errors=True)
    FRAMES.mkdir(parents=True)
    plan = [(-1, 0.0, ease(i / T_INTRO), 0.0) for i in range(T_INTRO)]
    for t in range(H):
        plan += [(t, ease(i / T_MOVE), 1.0, 0.0) for i in range(1, T_MOVE + 1)]
        plan += [(t, 1.0, 1.0, 0.0)] * T_HOLD
    plan += [(H - 1, 1.0, 1.0, ease(i / T_AVG)) for i in range(1, T_AVG + 1)]
    plan += [(H - 1, 1.0, 1.0, 1.0)] * T_END
    for i, (t, p, a, ap) in enumerate(plan):
        fig = draw(t, p, a, ap)
        fig.savefig(FRAMES / f"{i:04d}.png", facecolor=BG)
        plt.close(fig)
    OUT.parent.mkdir(exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(FRAMES / "%04d.png"),
                    "-vf", "split[a][b];[a]palettegen=max_colors=48:stats_mode=full[p];[b][p]paletteuse=dither=none",
                    str(OUT)], check=True)
    shutil.rmtree(FRAMES)
    print(f"{OUT} ({OUT.stat().st_size / 1e6:.2f} MB, {len(plan)} frames)")


if __name__ == "__main__":
    main()
