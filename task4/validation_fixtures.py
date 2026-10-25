"""Prepare the three real upload cases for Task 4.2 screenshots."""

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "outputs/task4/4_2/fixtures"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "wrong_type.txt").write_text("This is a text document, not an image.\n")
    (folder / "fake_image.png").write_bytes(b"This file has a PNG extension but contains no image.")
    pixels = np.random.default_rng(42).integers(0, 256, (1600, 1600, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(folder / "oversized.png")
    assert (folder / "oversized.png").stat().st_size > 5 * 1024 * 1024
    print("Screenshot fixtures:", folder)
    for path in sorted(folder.iterdir()):
        print(path.name, path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
