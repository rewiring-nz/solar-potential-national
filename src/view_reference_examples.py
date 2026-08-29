"""
Lay out every reference_examples/<name>/{bare,installed}.* pair
side by side into one montage PNG, for quick visual review.

Usage: python src/view_reference_examples.py
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REF_DIR = Path(__file__).resolve().parent.parent / "reference_examples"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "reference_examples_montage.png"
CELL_H = 320
EXTS = (".jpg", ".jpeg", ".png")


def find_image(folder, stem):
    for ext in EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main():
    examples = sorted(p for p in REF_DIR.iterdir() if p.is_dir())
    if not examples:
        print(f"No examples found in {REF_DIR} -- see reference_examples/README.md")
        return

    rows = []
    for folder in examples:
        bare_path = find_image(folder, "bare")
        installed_path = find_image(folder, "installed")
        if not bare_path or not installed_path:
            print(f"skipping {folder.name}: missing bare.* or installed.*")
            continue

        bare_img = Image.open(bare_path).convert("RGB")
        installed_img = Image.open(installed_path).convert("RGB")
        for img in (bare_img, installed_img):
            img.thumbnail((CELL_H * 2, CELL_H))
        rows.append((folder.name, bare_img, installed_img))

    if not rows:
        print("No complete (bare + installed) example pairs found.")
        return

    row_h = CELL_H + 30
    width = max(b.width + i.width for _, b, i in rows) + 40
    montage = Image.new("RGB", (width, row_h * len(rows)), "white")
    draw = ImageDraw.Draw(montage)

    for i, (name, bare_img, installed_img) in enumerate(rows):
        y = i * row_h
        draw.text((10, y + 4), f"{name}  (left: bare, right: installed)", fill="black")
        montage.paste(bare_img, (10, y + 24))
        montage.paste(installed_img, (20 + bare_img.width, y + 24))

    montage.save(OUT_PATH)
    print(f"Saved {OUT_PATH} ({len(rows)} example(s))")


if __name__ == "__main__":
    main()
