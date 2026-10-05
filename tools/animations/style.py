"""Look of the README animation: white background, one hue per loop, one text colour."""
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb

BG = "#FFFFFF"
TEXT = "#1F2328"
WARP = "#D3D9E3"
LOOP_CMAP = LinearSegmentedColormap.from_list("loops", ["#3A5BD9", "#2BA89A", "#F08A4B"])
FPS = 24
W, H_PX, DPI = 12.0, 6.0, 100  # 1200 x 600 px

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT})


def loop_color(t, n):
    return LOOP_CMAP(t / max(n - 1, 1))


def ease(p):
    p = min(max(float(p), 0.0), 1.0)
    return p * p * (3 - 2 * p)


def ease_out(p):
    p = min(max(float(p), 0.0), 1.0)
    return 1 - (1 - p) ** 3


def mix(c1, c2, p):
    a, b = np.array(to_rgb(c1)), np.array(to_rgb(c2))
    return tuple(a + (b - a) * min(max(p, 0.0), 1.0))


def new_frame():
    fig = plt.figure(figsize=(W, H_PX), dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.axis("off")
    return fig, ax


def glow_line(ax, xs, ys, color, lw, alpha=1.0, z=3, glow=True):
    if glow:
        for k, a in ((5.0, 0.05), (2.6, 0.12)):
            ax.plot(xs, ys, color=color, lw=lw * k, alpha=a * alpha, zorder=z, solid_capstyle="round")
    ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, zorder=z + 0.1, solid_capstyle="round")


def glow_dot(ax, x, y, color, size, alpha=1.0, z=5):
    for k, a in ((5.0, 0.06), (2.6, 0.16)):
        ax.scatter([x], [y], s=(size * k) ** 2, color=color, alpha=a * alpha, zorder=z, edgecolors="none")
    ax.scatter([x], [y], s=size ** 2, color=color, alpha=alpha, zorder=z + 0.1, edgecolors="none")


def encode_gif(src_args, gif: Path, width=1000, colors=96, lossy=int(os.environ.get("GIF_LOSSY", "12"))):
    """ffmpeg palette GIF without dithering (dither speckles the dark background), then gifsicle."""
    vf = (f"scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors={colors}:stats_mode=full[p];"
          "[b][p]paletteuse=dither=none")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *src_args, "-vf", vf, str(gif)], check=True)
    if shutil.which("gifsicle"):
        subprocess.run(["gifsicle", "-O3", *([f"--lossy={lossy}"] if lossy else []), "-b", str(gif)], check=True)


class Recorder:
    """Save frames, then encode a GIF (palette per clip) and an MP4."""

    def __init__(self, frames_dir: Path):
        self.dir = frames_dir
        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True)
        self.n = 0
        # PREVIEW=12,80,300 saves only those frames (as PNG) and skips encoding.
        raw = os.environ.get("PREVIEW", "").strip()
        self.preview = {int(x) for x in raw.split(",")} if raw else None

    def add(self, fig):
        if self.preview is None or self.n in self.preview:
            fig.savefig(self.dir / f"{self.n:05d}.png", facecolor=BG)
        plt.close(fig)
        self.n += 1

    def encode(self, gif: Path, mp4: Path = None, width=1000, colors=96):
        if self.preview is not None:
            print("preview frames in", self.dir)
            return 0.0
        gif.parent.mkdir(parents=True, exist_ok=True)
        src = ["-framerate", str(FPS), "-i", str(self.dir / "%05d.png")]
        encode_gif(src, gif, width, colors)
        if mp4 is not None:
            mp4.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *src, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-crf", "18", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(mp4)], check=True)
        if not os.environ.get("KEEP_FRAMES"):
            shutil.rmtree(self.dir)
        return gif.stat().st_size / 1e6
