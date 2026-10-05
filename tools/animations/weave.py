"""README animation A: every loop weaves a new row through the experts (real routing, routing.json).

Drawn as a weave draft. Columns are the 30 routed experts of one layer (the warp), rows are
loops (the weft). Where a loop's router picks an expert, the weft floats over the warp (a
coloured horizontal float); everywhere else the warp stays on top (a dark vertical float).
The camera then pulls back to the same weave for the first 48 tokens of the sentence.

    python tools/animations/weave.py      # writes assets/weave.gif and tools/animations/out/weave.mp4
"""
import json
from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection

from style import TEXT, W, Recorder, ease, glow_dot, loop_color, new_frame

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
d = json.loads((HERE / "routing.json").read_text())
LAYER = 7                                            # layer 8 of 15
NTOK = 48
SEL = np.array(d["example_layer_loop_sets"])[LAYER, :, :NTOK]   # [loop, token, k]
NL, _, K = SEL.shape
NE = d["n_routed"]
TOKENS = d["example_tokens"][:NTOK]
FOCUS = 2                                            # " guitar"
COLS, PX, PY = 8, 32.0, 12.2                         # cloth: 6 rows x 8 tokens
WARP_CELL = "#D3D9E3"
WARP_LINE = "#E6EAF0"
VIEW0_W = 42.0
PT_PER_UNIT0 = W * 72 / VIEW0_W                      # points per data unit at the start


def origin(k):
    r, c = divmod(k, COLS)
    return c * PX, -r * PY


def floats(k, rows, head=None, grow=1.2):
    """Segments for weft floats (over) and warp floats (under) of token k, loops in rows.

    With head set, the last row is only woven up to x = head; floats near the head grow in."""
    cx, cy = origin(k)
    over, over_c, over_scale, under = [], [], [], []
    for t in rows:
        chosen = set(SEL[t, k].tolist())
        for e in range(NE):
            x, y = cx + e, cy - t
            g = 1.0
            if head is not None and t == rows[-1]:
                if x > head:
                    continue
                g = min(1.0, (head - x) / grow + 0.15)
            if e in chosen:
                half = 0.21 * g
                over.append([(x - half, y), (x + half, y)])
                over_c.append(loop_color(t, NL))
                over_scale.append(g)
            else:
                half = 0.24 * g
                under.append([(x, y - half), (x, y + half)])
    return over, over_c, over_scale, under


def draw_token(ax, k, rows, u, alpha=1.0, head=None, glow=False, warp_alpha=1.0):
    """u: points per data unit."""
    if not rows:
        return
    cx, cy = origin(k)
    # threads underneath: warp columns and the weft of every finished row
    top, bot = cy + 0.55, cy - rows[-1] - 0.55
    ax.add_collection(LineCollection([[(cx + e, top), (cx + e, bot)] for e in range(NE)],
                                     colors=WARP_LINE, linewidths=0.16 * u, alpha=alpha * warp_alpha, zorder=1))
    for t in rows:
        x_end = cx + NE - 0.5 if (head is None or t != rows[-1]) else min(head, cx + NE - 0.5)
        ax.plot([cx - 0.5, x_end], [cy - t, cy - t], color=loop_color(t, NL), lw=0.10 * u,
                alpha=0.55 * alpha, zorder=2, solid_capstyle="butt")
    over, over_c, over_s, under = floats(k, rows, head)
    if under:
        ax.add_collection(LineCollection(under, colors=WARP_CELL, linewidths=0.40 * u, alpha=alpha,
                                         zorder=3, capstyle="round"))
    if over:
        if glow:
            ax.add_collection(LineCollection(over, colors=over_c, linewidths=[1.0 * u * s for s in over_s],
                                             alpha=0.10 * alpha, zorder=3.5, capstyle="round"))
        ax.add_collection(LineCollection(over, colors=over_c, linewidths=[0.50 * u * s for s in over_s],
                                         alpha=alpha, zorder=4, capstyle="round"))


def render(rec):
    fcx, fcy = origin(FOCUS)
    c0 = np.array([fcx + NE / 2 + 2.0, fcy - (NL - 1) / 2 - 0.6])
    nrow = NTOK // COLS
    cloth_w = (COLS - 1) * PX + NE
    c1 = np.array([cloth_w / 2 - 0.5, -(nrow - 1) * PY / 2 - (NL - 1) / 2 - 0.9])
    w1 = cloth_w * 1.07

    def frame(loop, p, ui=1.0, zoom=0.0, cloth=0.0, tokens=0.0, intro=1.0):
        fig, ax = new_frame()
        z = ease(zoom)
        w = VIEW0_W * (w1 / VIEW0_W) ** z
        cx, cy = c0 + (c1 - c0) * z
        ax.set_xlim(cx - w / 2, cx + w / 2)
        ax.set_ylim(cy - w / 4, cy + w / 4)
        u = W * 72 / w
        s = VIEW0_W / w
        if cloth > 0:
            for k in range(NTOK):
                if k != FOCUS:
                    draw_token(ax, k, list(range(NL)), u, alpha=cloth)
        head = None
        rows = list(range(min(loop + 1, NL)))
        if loop < NL:
            head = fcx - 0.6 + p * (NE + 0.2)
        if intro < 1:   # warp only, fading in
            ax.add_collection(LineCollection(
                [[(fcx + e, fcy + 0.55), (fcx + e, fcy + 0.55 - (NL + 0.1) * ease(intro * 1.4 - e * 0.012))]
                 for e in range(NE)], colors=WARP_LINE, linewidths=0.16 * u, zorder=1))
        else:
            draw_token(ax, FOCUS, rows, u, head=head, glow=True)
        if head is not None and 0 < p < 1 and intro >= 1:
            glow_dot(ax, head, fcy - loop, TEXT, 0.20 * u)
        # distinct experts touched so far
        seen = set()
        for t in rows:
            for e in SEL[t, FOCUS]:
                if head is None or t != rows[-1] or fcx + e <= head:
                    seen.add(int(e))
        if ui > 0:
            ax.text(fcx + NE + 0.8, fcy + 0.5, f"{len(seen)}", fontsize=58 * s, weight="bold", color=TEXT,
                    ha="left", va="top", alpha=ui, zorder=6)
            ax.text(fcx + NE + 0.9, fcy - 3.6, "experts\nused", fontsize=18 * s, color=TEXT, ha="left", va="top",
                    alpha=ui, linespacing=1.25, zorder=6)
            for t in rows if intro >= 1 else []:
                ax.text(fcx - 1.4, fcy - t, f"{t + 1}", color=loop_color(t, NL), fontsize=17 * s, ha="right",
                        va="center", family="DejaVu Sans Mono", alpha=ui, zorder=6)
            ax.text(fcx - 1.4, fcy + 1.3, "loop", fontsize=16 * s, color=TEXT, ha="right", va="center",
                    alpha=ui, zorder=6)
            ax.text(fcx + (NE - 1) / 2, fcy - (NL - 1) - 1.6, f"{NE} experts · layer {LAYER + 1} · token "
                    f"“{TOKENS[FOCUS].strip()}”", fontsize=17 * s, color=TEXT, ha="center", va="top",
                    alpha=ui, zorder=6)
        if tokens > 0:
            for k in range(NTOK):
                x, y = origin(k)
                tok = TOKENS[k].replace("\n", "↵").strip() or "␣"
                ax.text(x + (NE - 1) / 2, y - (NL - 1) - 0.9, tok, fontsize=2.9 * u, ha="center", va="top",
                        color=TEXT, alpha=tokens,
                        weight="bold" if k == FOCUS else "normal")
        rec.add(fig)

    for i in range(1, 31):                              # warp threads drop in
        frame(0, 0.0, intro=i / 30, ui=ease(i / 30))
    for loop in range(NL):                              # nine passes of the shuttle
        for i in range(1, 21):
            frame(loop, ease(i / 20))
        for _ in range(4):
            frame(loop, 1.0)
    for _ in range(20):
        frame(NL, 1.0)
    for i in range(1, 61):                              # pull back to the whole sentence
        z = i / 60
        frame(NL, 1.0, ui=1 - ease(z * 4), zoom=z, cloth=ease(z * 2.5), tokens=ease((z - 0.6) / 0.4))
    for _ in range(96):
        frame(NL, 1.0, ui=0, zoom=1.0, cloth=1.0, tokens=1.0)


def main():
    rec = Recorder(HERE / "frames_weave")
    render(rec)
    size = rec.encode(ROOT / "assets" / "weave.gif", HERE / "out" / "weave.mp4")
    print(f"weave.gif {size:.2f} MB, {rec.n} frames")

if __name__ == "__main__":
    main()
