"""README animation B: how far the hidden state turns in each loop (real states, states.json).

The hidden state leaves every loop with a fixed RMS, so it lives on a sphere and a loop can
only rotate it. Each comet is one token: its 10 states (input + 9 loops) are placed on a
3-D sphere so that the angles between them match the real model (classical MDS of the
token's cosine matrix). Left: LOOM. Right: the same model trained without residual scaling.

    python tools/animations/sphere.py     # writes assets/sphere.gif and tools/animations/out/sphere.mp4
"""
import json
import os
from pathlib import Path

import numpy as np

from style import BG, TEXT, Recorder, ease, glow_dot, loop_color, mix, new_frame

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = json.loads((HERE / "states.json").read_text())
LEFT = os.environ.get("SPHERE_LEFT", "329m_H9_loom")
RIGHT = os.environ.get("SPHERE_RIGHT", "329m_H9_noresscale")
NAMES = {LEFT: os.environ.get("SPHERE_LEFT_NAME", "LOOM"), RIGHT: "without residual scaling"}
N_TOKENS = 22
GRID = "#1C2436"
R = 0.72
CENTERS = {LEFT: np.array([-1.0, 0.16]), RIGHT: np.array([1.0, 0.16])}
TILT = np.radians(18)
SPIN = float(os.environ.get("SPHERE_SPIN", "0"))   # degrees per frame; 0 keeps the GIF small


def embed(gram):
    """Unit 3-D points whose pairwise angles approximate the token's real ones."""
    w, v = np.linalg.eigh(gram)
    idx = np.argsort(w)[::-1][:3]
    x = v[:, idx] * np.sqrt(np.clip(w[idx], 0, None))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def slerp(a, b, p):
    om = np.arccos(np.clip(a @ b, -1, 1))
    if om < 1e-6:
        return a
    return (np.sin((1 - p) * om) * a + np.sin(p * om) * b) / np.sin(om)


def paths(name):
    e = DATA[name]
    rng = np.random.default_rng(3)
    grams = np.array(e["gram"])[:N_TOKENS]
    out = []
    for g in grams:
        x = embed(g) @ random_rotation(rng).T
        out.append(x)
    angles = np.degrees(np.arccos(np.clip(np.array([np.diag(g, 1) for g in np.array(e["gram"])]), -1, 1))).mean(0)
    return np.array(out), angles          # [tokens, H + 1, 3], [H]


def project(v, phi):
    """Rotate about the vertical axis by phi, tilt toward the viewer; return screen xy and depth."""
    c, s = np.cos(phi), np.sin(phi)
    x, y, z = v[..., 0] * c - v[..., 1] * s, v[..., 0] * s + v[..., 1] * c, v[..., 2]
    ct, st = np.cos(TILT), np.sin(TILT)
    y2, z2 = y * ct - z * st, y * st + z * ct
    return np.stack([x, z2], -1), -y2      # depth > 0: front


def draw_sphere(ax, center, phi, alpha):
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(center[0] + R * np.cos(th), center[1] + R * np.sin(th), color=GRID, lw=1.4, alpha=alpha, zorder=1)
    curves = []
    for lat in np.radians([-60, -30, 0, 30, 60]):
        curves.append(np.stack([np.cos(lat) * np.cos(th), np.cos(lat) * np.sin(th), np.full_like(th, np.sin(lat))], -1))
    for lon in np.radians(np.arange(0, 180, 30)):
        curves.append(np.stack([np.cos(th) * np.cos(lon), np.cos(th) * np.sin(lon), np.sin(th)], -1))
    for cv in curves:
        xy, dep = project(cv, phi)
        xy = center + R * xy
        front = dep > 0
        for mask, a in ((front, 0.9), (~front, 0.35)):
            seg = np.where(mask, 1.0, np.nan)
            ax.plot(xy[:, 0] * seg, xy[:, 1] * seg, color=GRID, lw=0.9, alpha=a * alpha, zorder=1)


def draw_comets(ax, name, P, step, p, phi, alpha):
    center = CENTERS[name]
    H = P.shape[1] - 1
    for tok in P:
        for t in range(1, min(step, H) + 1):
            frac = p if t == step else 1.0
            if frac <= 0:
                continue
            arc = np.array([slerp(tok[t - 1], tok[t], q) for q in np.linspace(0, frac, 24)])
            xy, dep = project(arc, phi)
            xy = center + R * xy
            col = loop_color(t - 1, H)
            for mask, a in ((dep > 0, 0.95), (dep <= 0, 0.28)):
                seg = np.where(mask, 1.0, np.nan)
                ax.plot(xy[:, 0] * seg, xy[:, 1] * seg, color=col, lw=1.7, alpha=a * alpha, zorder=3,
                        solid_capstyle="round")
        if step == 0:
            head = tok[0]
        else:
            t = min(step, H)
            head = slerp(tok[t - 1], tok[t], p if step <= H else 1.0)
        hxy, hdep = project(head, phi)
        glow_dot(ax, *(center + R * hxy), "#FFFFFF", 4.2, alpha=alpha * (1.0 if hdep > 0 else 0.35), z=4)


def render(rec, outro=False):
    data = {n: paths(n) for n in (LEFT, RIGHT)}
    H = data[LEFT][0].shape[1] - 1

    def frame(step, p, k, alpha=1.0, zoom=0.0):
        fig, ax = new_frame()
        # zoom > 0 (outro used by the combined animation): fly into the LOOM sphere
        z = ease(zoom)
        cx, cy = CENTERS[LEFT] * z
        half = 2.1 * (1 - z) + 0.55 * z
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half / 2, cy + half / 2)
        phi = np.radians(20 + SPIN * k)
        for name, (P, ang) in data.items():
            c = CENTERS[name]
            draw_sphere(ax, c, phi, alpha)
            draw_comets(ax, name, P, step, p, phi, alpha)
            ax.text(c[0], c[1] - R - 0.10, NAMES[name], ha="center", va="top", fontsize=15, color=TEXT,
                    weight="bold", alpha=alpha)
            if 1 <= step <= H:
                a = ang[step - 1]
                col = loop_color(step - 1, H)
                ax.text(c[0] - 0.02, c[1] - R - 0.26, f"{a:.0f}°", ha="right", va="top", fontsize=20,
                        color=col, weight="bold", alpha=alpha * ease(p * 2))
                ax.text(c[0] + 0.02, c[1] - R - 0.30, "turned in this loop", ha="left", va="top", fontsize=11.5,
                        color=TEXT, alpha=alpha * ease(p * 2))
        if 1 <= step <= H:
            ax.text(0, 0.98, f"loop {step}", ha="center", va="top", fontsize=15, color=loop_color(step - 1, H),
                    weight="bold", alpha=alpha)
        rec.add(fig)

    k = 0
    for i in range(1, 25):
        frame(0, 0.0, k, alpha=ease(i / 24)); k += 1
    for step in range(1, H + 1):
        for i in range(1, 23):
            frame(step, ease(i / 22), k); k += 1
        for _ in range(10):
            frame(step, 1.0, k); k += 1
    for _ in range(40):
        frame(H, 1.0, k); k += 1
    if outro:
        for i in range(1, 31):
            frame(H, 1.0, k, alpha=1 - ease((i / 30 - 0.4) / 0.6), zoom=i / 30); k += 1


def main():
    rec = Recorder(HERE / "frames_sphere")
    render(rec)
    size = rec.encode(ROOT / "assets" / "sphere.gif", HERE / "out" / "sphere.mp4")
    print(f"sphere.gif {size:.2f} MB, {rec.n} frames")


if __name__ == "__main__":
    main()
