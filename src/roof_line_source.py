"""
Where roof-line candidates come from -- imagery today, a vision model later.

Track D of the vision pathway: the seam a trained model plugs into, built and
testable BEFORE the model exists.

The partition does not need to know who proposed a line. It needs (angle,
offset) in the footprint's own frame.

"THE MODEL PROPOSES, THE LIDAR DISPOSES" was the design, and it survives only
in a much weaker form than it claims. Line by line, `_line_is_real` keeps 86.1%
of model lines that match Josh's drawings and 83.7% of those that do not --
2.4 points of separation, near enough to a coin flip.

End to end it is still worth keeping, which is not the same thing and was
worth measuring rather than assuming from the line-level number. Turned off,
invented edges rise 22.4% -> 25.0% and clutter 119 m -> 128 m for 0.3 points
more real creases found. A weak filter applied to hundreds of candidates still
removes more bad cuts than good ones. It stays -- as a mild net positive, not
as "the whole fusion story".

Nor can it do better. The survey is 1.7 returns per m2, roughly 0.77 m
spacing, and a hip crease is decimetre-scale geometry that falls between
samples. Asked to confirm creases Josh drew by eye on 0.1 m imagery, the gate
rejects a quarter of them.

So the fusion is the other way round, which is what Josh described:

    THE IMAGERY FINDS THE LINES. THE LIDAR FITS THE ANGLES.

The imagery is the only sensor here that can see a fold at all, so a bad
proposal has to be filtered on the model's own confidence -- MIN_SCORE below,
which separates true from false by 59 points where the gate manages 2. The
LiDAR's real job is downstream: the plane, slope and aspect of each face once
the faces exist, which it does well because a plane needs many points and not
a sharp edge.

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

# A predicted line below this confidence is not offered at all.
#
# WAS 0.25, on the reasoning that "the LiDAR gate downstream is the real
# filter". Measured, that gate is not a filter at all. Against 659 model lines
# matching Josh's drawings and 2,043 that do not, _line_is_real keeps 86.1% of
# the true ones and 83.7% of the false -- 2.4 points of separation, which is
# noise. It cannot do better: the point cloud is 1.7 returns/m2 and a hip
# crease falls between samples, so the gate is asking the LiDAR to confirm
# something it cannot resolve.
#
# The model's own confidence separates them properly: at 0.90 it keeps 81.3%
# of true lines and 22.2% of false, and the old 0.25 admitted essentially
# every false line (baseline precision 24% -- three cuts in four were wrong).
#
# Swept end to end against Josh's markup on 85 roofs, model path only:
#
#   MIN_SCORE   lines found   edges he did NOT draw   clutter   facets
#      0.25        83.6%              25.3%            122       8.4
#      0.90        82.5%              22.4%            119       8.0
#      0.95        81.2%              22.6%            113       7.5
#
# 0.90 buys the largest fall in invented edges for the smallest loss of real
# ones. Precision matters more than recall here: a wrong cut fragments a roof
# and is what got imagery cuts called "actively harmful" once before, while a
# missed cut just leaves the LiDAR partition to do what it already does.
MIN_SCORE = 0.90

# A predicted line shorter than this fraction of sqrt(roof area) is not cut on.
#
# CONFIDENCE WAS THE WRONG AXIS ON ITS OWN. Raising MIN_SCORE made cuts more
# likely to be REAL; it did nothing about them being SHORT, and _cut extends
# whatever it is given into an infinite line across the whole cell. Josh found
# it on 107 Beach Street (#4725721): a 127 m2 roof, about 11 m across, sliced
# by five model lines of 1.8, 2.0, 2.1, 3.1 and 5.5 m. A 1.8 m observation
# became an 11 m assertion.
#
# Across ~4,000 currently-cutting lines the median is 0.25 of sqrt(roof area),
# so HALF the cuts come from stubs under a quarter of the roof's own scale.
#
# Swept on Josh's 85 labelled roofs, model path only:
#
#   bar     lines found   edges he did NOT draw   clutter   facets
#   none       82.5%             22.4%              119      8.0
#   0.25       82.5%             21.4%              115      7.7
#   0.35       82.7%             20.5%              112      7.5
#   0.50       82.7%             20.8%              114      7.4
#
# 0.35 improves BOTH directions -- marginally more real creases found, 8.5%
# fewer invented ones -- which is rare enough to be worth trusting. Past 0.50
# it starts discarding real short creases faster than noise.
#
# This is the first change that helps roofs he has NOT labelled. Everything
# else fixed today only reaches the 114 he drew.
MIN_LEN_FRAC = 0.35


def _to_angle_offset(x1, y1, x2, y2, footprint):
    """Segment endpoints -> the (angle_deg, offset) form roof_partition._cut takes.

    DELEGATES to roof_lines._angle_offset rather than reimplementing it. The
    first version of this function did reimplement it and got two things wrong,
    and because nothing exercised the path with a real model on disk, both sat
    undetected from August until 2 September:

      * it returned RADIANS. _cut takes degrees and calls np.radians on them, so
        a line at 45 degrees was cut at 0.785 degrees.
      * it measured offset from the ORIGIN. _cut measures from the polygon
        centroid, and in NZTM the origin is about 1.2 million metres away.

    Every proposed line was therefore tested somewhere meaningless, and the
    LiDAR gate rejected 100% of them -- 68 of 68 on one roof -- which read
    exactly like a model with nothing to say.

    One implementation, in the module that owns the convention."""
    from shapely.geometry import LineString
    from src.roof_lines import _angle_offset
    c = footprint.centroid
    ang, off = _angle_offset(LineString([(x1, y1), (x2, y2)]), c.x, c.y)
    return ang, off, math.hypot(x2 - x1, y2 - y1)


def model_lines(building_id, footprint=None):
    """Predicted lines for one building, or None if the model has not run.

    Returns a list of (angle_deg, offset, length, score). The offset is relative
    to `footprint`'s centroid, so the footprint is required -- passing None
    yields the raw segments instead, for callers that want geometry."""
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
    # A stub cannot justify a cut across the whole roof. Scale is the roof's
    # own, so the bar means the same thing on a garage and a warehouse.
    min_len = 0.0
    if footprint is not None:
        try:
            min_len = MIN_LEN_FRAC * math.sqrt(max(footprint.area, 1.0))
        except Exception:
            min_len = 0.0
    out = []
    for s, sc in zip(segs, scores):
        if len(s) != 4 or sc < MIN_SCORE:
            continue
        if min_len and math.hypot(s[2] - s[0], s[3] - s[1]) < min_len:
            continue
        if footprint is None:
            out.append((None, None, math.hypot(s[2] - s[0], s[3] - s[1]),
                        float(sc), list(s)))
            continue
        ang, off, ln = _to_angle_offset(*s, footprint)
        out.append((ang, off, ln, float(sc)))
    return out


def has_model(building_id):
    return building_id is not None and (VISION_DIR / f"{building_id}.json").exists()


# ------------------------------------------------------------------ drawn

# Josh's own lines, for the roofs he has actually marked up.
#
# WHY THIS EXISTS. Until now his markups reached the build only by training the
# line model, whose predictions were then offered as proposals and gated by the
# LiDAR. So on a roof he had drawn himself, the build used the model's guess
# (held-out F1 0.43 on ridges, 0.13 on cliffs) instead of his ground truth, and
# the gate could veto his creases exactly as it vetoes a model's. He found this
# from the map: 7 Anderson Heights (#5371108, 14 drawn lines) and 1 Memorial
# Street (#5372565, 19 drawn lines) are both marked complete and both came out
# with facets that look nothing like what he drew.
#
# A drawn line is not a proposal. He looked at the imagery and said "there is a
# fold here", which is the same evidence the LiDAR gate is a proxy for -- and a
# better one, because the gate misses creases the point cloud is flat across.
# The module header note about 7 Anderson says so outright: its hip creases are
# "unmistakable in 0.1 m imagery and nearly absent from a point cloud".
_LABELS_CACHE = [None]
LABELS_PATH = DATA_DIR / "roof_labels.json"
FOLD_KINDS = {"ridge", "valley", "cliff"}
# Flags that say the drawn geometry describes nothing trustworthy. bad_outline
# is deliberately NOT here: an offset outline is exactly the case where the
# drawn lines are the more reliable description of the roof.
VOID_FLAGS = {"absent", "not_building", "unclear"}


def _labels():
    if _LABELS_CACHE[0] is None:
        try:
            _LABELS_CACHE[0] = json.loads(
                LABELS_PATH.read_text()).get("buildings", {})
        except Exception:
            _LABELS_CACHE[0] = {}
    return _LABELS_CACHE[0]


def drawn_segments(building_id):
    """Raw [(x1,y1,x2,y2), ...] in NZTM for a roof Josh has marked, else []."""
    if building_id is None:
        return []
    lab = _labels().get(str(building_id))
    if not lab or lab.get("problem") in VOID_FLAGS:
        return []
    out = []
    for l in lab.get("lines") or []:
        if l.get("kind") not in FOLD_KINDS:
            continue
        pts = l.get("points")
        if pts and len(pts) >= 2:
            for i in range(len(pts) - 1):
                a, b = pts[i], pts[i + 1]
                out.append([a[0], a[1], b[0], b[1]])
        elif l.get("a") and l.get("b"):
            a, b = l["a"], l["b"]
            out.append([a[0], a[1], b[0], b[1]])
    return [s for s in out if math.hypot(s[2] - s[0], s[3] - s[1]) > 0.3]


def drawn_faces(building_id):
    """The faces the LABELLING TOOL derived from Josh's lines, as NZTM rings.

    These were in roof_labels.json the whole time and nothing read them. The
    tool runs facesFor() in the browser as he draws -- the same construction he
    is looking at when he decides a roof is finished -- and exports the result
    with an area and a `usable` flag per face. #5371108 carries 9 of them.

    Re-deriving faces from the lines in Python was the wrong instinct and cost
    a lot: sealing rules, noding bugs, and two measured regressions, all to
    reconstruct something already computed and agreed. Worse, a second
    derivation can silently disagree with the one he saw, so the geometry the
    build uses would not be the geometry he approved.

    `usable=False` faces are kept here and dropped by the caller: they are
    exactly the "no panels here" areas he clicked, and knowing a face exists
    but takes no panels is more useful than not knowing it exists.
    """
    if building_id is None:
        return []
    lab = _labels().get(str(building_id))
    if not lab or lab.get("problem") in VOID_FLAGS:
        return []
    out = []
    for f in lab.get("faces") or []:
        ring = f.get("ring") or []
        if len(ring) >= 3:
            out.append({"ring": [(float(x), float(y)) for x, y in ring],
                        "m2": float(f.get("m2") or 0.0),
                        "usable": bool(f.get("usable", True))})
    return out


def has_drawn_faces(building_id):
    return bool(drawn_faces(building_id))


def has_drawn(building_id):
    return bool(drawn_segments(building_id))


def drawn_lines(building_id, footprint=None):
    """Josh's lines in the same (angle, offset, length, score) shape as the
    model's, scored 1.0 because they are not predictions."""
    segs = drawn_segments(building_id)
    if not segs:
        return []
    out = []
    for s in segs:
        if footprint is None:
            out.append((None, None, math.hypot(s[2] - s[0], s[3] - s[1]),
                        1.0, list(s)))
            continue
        ang, off, ln = _to_angle_offset(*s, footprint)
        out.append((ang, off, ln, 1.0))
    return out


def strong_lines(imagery_ds, footprint, building_id=None):
    """Lines to act on even where the LiDAR cannot confirm them.

    Falls straight through to roof_lines.strong_roof_lines when no model
    prediction exists, so today's behaviour is unchanged."""
    from src.roof_lines import strong_roof_lines
    lines = model_lines(building_id, footprint)
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
    if not strong:
        # A model that proposes nothing long enough must not DELETE the imagery
        # strong lines. Returning [] here would have silently removed them on
        # every building carrying a prediction file.
        return strong_roof_lines(imagery_ds, footprint)
    return strong[:MAX_STRONG_LINES]


def candidate_lines(imagery_ds, footprint, building_id=None):
    """Every line worth OFFERING to the partition, which keeps only the ones
    that improve the fit. Model lines are merged with imagery lines rather than
    replacing them -- until the scorer says otherwise, more candidates offered
    to a gate that rejects bad ones is strictly better than fewer."""
    from src.roof_lines import roof_line_candidates
    base = list(roof_line_candidates(imagery_ds, footprint))
    lines = model_lines(building_id, footprint)
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
