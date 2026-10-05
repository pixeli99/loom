"""README animation C: B then A in one clip (stability on the sphere, then the weave).

The sphere clip ends by flying into the LOOM sphere and fading out; the weave clip starts
from an empty frame, so the two are joined directly.

    python tools/animations/loom.py       # writes assets/loom.gif and tools/animations/out/loom.mp4
"""
from pathlib import Path

import sphere
import weave
from style import Recorder

HERE = Path(__file__).resolve().parent


def main():
    rec = Recorder(HERE / "frames_loom")
    sphere.render(rec, outro=True)
    weave.render(rec)
    size = rec.encode(HERE.parent.parent / "assets" / "loom.gif", HERE / "out" / "loom.mp4")
    print(f"loom.gif {size:.2f} MB, {rec.n} frames")


if __name__ == "__main__":
    main()
