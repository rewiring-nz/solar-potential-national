"""RID2's roof-centred segment masks as face-region pretraining.

The face-region model (tools/train_face_regions.py) is limited by data:
96 marked roofs. RID2 ships 1,819 roof-centred 512px images with a
per-pixel SEGMENT mask, which is the same supervision in the same format
-- roof centred, one sample per roof, faces as regions.

NOT THE SAME BET AS LAST TIME. RID2 pretraining was measured a dead end for
the LINE detector (three variants, all lost on union), where the targets had
to be DERIVED from segment azimuths and the derivation was lossy. Here the
masks are the target directly, so the supervision is exact.

WHAT RID2 GIVES AND WHAT IT DOES NOT. Mask values are azimuth CLASSES, not
instance ids, so two adjacent faces pointing the same way merge into one --
an under-count this cannot fix, and a reason to fine-tune on the markup rather
than trust RID alone. There is no LiDAR: the four LiDAR channels are fed as
the neutral value they take on flat ground, so the network learns to read
imagery first and the fine-tune teaches it what the LiDAR channels add.
German roofs are also not NZ roofs; this is a prior, not an answer.

Usage:
    python tools/export_rid_regions.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, binary_dilation

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/rid2/case_study_roof_centered"
OUT = ROOT / "data/face_regions_rid"
SIZE = 256
BOUNDARY_PX = 3
CORE_ERODE_PX = 4
EAVE_ERASE_PX = 5


def main():
    imgs = sorted((SRC / "images_roof_centered").glob("*.png")) or \
        sorted((SRC / "images_roof_centered").glob("*.tif"))
    masks = {p.stem: p for p in
             (SRC / "masks_roof_centered/masks_segments").glob("*")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train").mkdir(exist_ok=True)
    n = skipped = 0
    for ip in imgs:
        mp = masks.get(ip.stem)
        if mp is None:
            skipped += 1
            continue
        rgb = np.array(Image.open(ip).convert("RGB").resize((SIZE, SIZE),
                                                            Image.BILINEAR))
        m = np.array(Image.open(mp).resize((SIZE, SIZE), Image.NEAREST))
        roof = m > 0
        if roof.sum() < 1500:
            skipped += 1
            continue
        # boundary = where neighbouring pixels disagree about which face
        b = np.zeros_like(roof)
        b[:-1, :] |= (m[:-1, :] != m[1:, :])
        b[:, :-1] |= (m[:, :-1] != m[:, 1:])
        b &= roof
        b = binary_dilation(b, np.ones((BOUNDARY_PX, BOUNDARY_PX), bool))
        # erase the eave, exactly as for the marked roofs: the outline is known
        edge = roof & ~binary_erosion(
            roof, np.ones((3, 3), bool), iterations=EAVE_ERASE_PX)
        b &= ~edge
        core = binary_erosion(roof & ~b, np.ones((3, 3), bool),
                              iterations=CORE_ERODE_PX)
        # LiDAR channels absent: feed the flat-ground neutral value
        chans = np.dstack([
            rgb.astype(np.float32) / 255.0,
            np.zeros((SIZE, SIZE), np.float32),      # height above base
            np.zeros((SIZE, SIZE), np.float32),      # slope
            np.full((SIZE, SIZE), 0.5, np.float32),  # aspect x
            np.full((SIZE, SIZE), 0.5, np.float32),  # aspect y
        ])
        np.savez_compressed(
            OUT / "train" / f"{ip.stem}.npz",
            image=(chans * 255).astype("uint8"),
            target=(np.dstack([b, core]) * 255).astype("uint8"),
            weight=roof.astype("uint8"))
        n += 1
    (OUT / "manifest.json").write_text(json.dumps(
        {"source": "RID2 case_study_roof_centered", "samples": n,
         "size": SIZE, "lidar": False}, indent=1))
    print(f"exported {n} RID roofs   skipped {skipped}")


if __name__ == "__main__":
    sys.exit(main() or 0)
