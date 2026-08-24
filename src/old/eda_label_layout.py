"""EDA: build a grid of randomly sampled specimen photos.

Writes `tables_figures/eda/layout_grid.png`, for eyeballing how labels are laid
out within a photo. One-off script.
"""

import os
import random

from PIL import Image

IMAGES_DIR = os.path.join("data", "raw", "images")
OUT_PATH = os.path.join("tables_figures", "eda", "layout_grid.png")


def sample_image_files(n=12, seed=42):
    files = sorted(os.listdir(IMAGES_DIR))
    random.Random(seed).shuffle(files)
    return files[:n]


def build_grid(files, thumb_size=(400, 267)):
    thumbs = [Image.open(os.path.join(IMAGES_DIR, f)).resize(thumb_size) for f in files]
    cols = 4
    rows = (len(thumbs) + cols - 1) // cols
    grid = Image.new("RGB", (thumb_size[0] * cols, thumb_size[1] * rows), "white")
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * thumb_size[0]
        y = (idx // cols) * thumb_size[1]
        grid.paste(thumb, (x, y))
    return grid


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    files = sample_image_files()
    grid = build_grid(files)
    grid.save(OUT_PATH)
    print(f"Sampled: {files}")


if __name__ == "__main__":
    main()
