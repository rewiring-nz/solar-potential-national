"""
Where roof-line candidates come from -- imagery today, a vision model later.

Track D of the vision pathway: the seam a trained model plugs into, built and
testable BEFORE the model exists.

The partition does not need to know who proposed a line. It needs (angle,
offset) in the footprint's own frame, and it then decides for itself whether the
LiDAR agrees -- `_line_is_real` in roof_partition only cuts where the surface
actually turns or the two sides sit at different heights. That gate is the whole
fusion story and it stays exactly where it is:

    THE MODEL PROPOSES, THE LIDAR DISPOSES.

which is also why swapping the proposer is safe. A model that hallucinates a
ridge produces a candidate that fails the height test and is discarded, exactly
as a stain or a tonal band in the imagery is discarded today.

Two kinds of candidate, and the distinction matters:

  ORDINARY   offered to the partition and kept only if cutting improves the fit.
             The right test when both sensors can see the feature.
  STRONG     acted on WITHOUT the LiDAR agreeing, because some roofs are
             invisible to it -- 7 Anderson Heights has two hip creases that are
             unmistakable in imagery and almost absent from a near-flat point
             cloud, so every LiDAR-scored candidate there is rejected and the
             faces run straight over both hips.

Model lines currently enter through the SAME gates as imagery lines, and are NOT
automatically promoted to strong. That is deliberate: whether a model deserves
more trust than a Hough fragment is a question for the evaluation harness
(tools/score_geometry.py), not an assumption to bake in before any model has
been measured.

WITH NO MODEL FILE PRESENT THIS CHANGES NOTHING. It delegates straight to
roof_lines, so the current behaviour is bit-for-bit what it was.

Model predictions are read from:
    data/vision_lines/<building_id>.json
      {"lines": [[x1,y1,x2,y2], ...],      # NZTM metres
       "scores": [0.91, ...],              # optional, 0-1
       "model": "wireframe-v1"}            # optional, for provenance
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VISION_DIR = DATA_DIR / "vision_lines"

# A predicted line below this confidence is not offered at all. Deliberately
# permissive: the LiDAR gate downstream is the real filter, and throwing away
# candidates here would hide model recall problems the scorer needs to see.
MIN_SCORE = 0.25


def _to_angle_offset(x1, y1, x2, y2):
    """Segment endpoints -> the (angle, offset) normal form the partition cuts
    with. Angle is the line's direction in radians; offset is its signed
    perpendicular distance from the origin, matching roof_lines' convention."""
    ang = math.atan2(y2 - y1, x2 - x1)
    # perpendicular distance from origin to the line through (x1,y1) at `ang`
    off = -math.sin(ang) * x1 + math.cos(ang) * y1
    length = math.hypot(x2 - x1, y2 - y1)
    return ang, off, length


def model_lines(building_id):
    """Predicted lines for one building, or None if the model has not run.

    Returns a list of (angle, offset, length, score)."""
    if building_id is None:
        return None
    p = VISION_DIR / f"{building_id}.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None
    segs = d.get("lines") or []
    scores = d.get("scores") or [1.0] * len(segs)
    out = []
    for s, sc in zip(segs, scores):
        if len(s) != 4 or sc < MIN_SCORE:
            continue
        ang, off, ln = _to_angle_offset(*s)
        out.append((ang, off, ln, float(sc)))
    return out


def has_model(building_id):
    return building_id is not None and (VISION_DIR / f"{building_id}.json").exists()


def strong_lines(imagery_ds, footprint, building_id=None):
    """Lines to act on even where the LiDAR cannot confirm them.

    Falls straight through to roof_lines.strong_roof_lines when no model
    prediction exists, so today's behaviour is unchanged."""
    from src.roof_lines import strong_roof_lines
    lines = model_lines(building_id)
    if lines is None:
        return strong_roof_lines(imagery_ds, footprint)
    # A model line is only promoted to "strong" on the same evidence an imagery
    # line needs -- length relative to the building. Confidence alone is not
    # enough: a model can be confident and wrong, and the whole point of the
    # strong path is that nothing downstream will check it.
    from src.roof_lines import STRONG_LINE_MIN_M, STRONG_LINE_AREA_COEF, MAX_STRONG_LINES
    bar = max(STRONG_LINE_MIN_M,
              STRONG_LINE_AREA_COEF * math.sqrt(max(footprint.area, 1.0)))
    strong = [(a, o) for a, o, ln, sc in sorted(lines, key=lambda t: -t[2])
              if ln >= bar]
    return strong[:MAX_STRONG_LINES]


def candidate_lines(imagery_ds, footprint, building_id=None):
    """Every line worth OFFERING to the partition, which keeps only the ones
    that improve the fit. Model lines are merged with imagery lines rather than
    replacing them -- until the scorer says otherwise, more candidates offered
    to a gate that rejects bad ones is strictly better than fewer."""
    from src.roof_lines import roof_line_candidates
    base = list(roof_line_candidates(imagery_ds, footprint))
    lines = model_lines(building_id)
    if lines is None:
        return base
    return base + [(a, o) for a, o, ln, sc in lines]


def provenance(building_id):
    """Which proposer was used, for the scorecard and the build log."""
    if not has_model(building_id):
        return "imagery"
    try:
        d = json.loads((VISION_DIR / f"{building_id}.json").read_text())
        return f"model:{d.get('model', 'unknown')}"
    except Exception:
        return "model:unreadable"
