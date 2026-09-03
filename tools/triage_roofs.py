"""
Find the roofs the build got wrong, without anyone having to look at them.

Josh: "It would be good if there was an easy way to find failed rooftops, which
I can then mark up for you to fix. So we can have a loop that gets every rooftop
fixed."

The loop only closes if the FINDING step is automatic. There are ~15,000 roofs
in the district and 114 labelled ones; scrolling for failures is the bottleneck,
not marking them up. So this scores every built roof for suspicion using only
what the build already wrote to disk -- no LiDAR, no rebuild, no ground truth --
and hands back a ranked queue that feeds straight into the labelling bundle.

THE SIGNALS NEED NO GROUND TRUTH. Nothing here knows what a correct roof looks
like; it knows where the build contradicts itself or its own vision model.

WHY THIS TOOL VALIDATES ITSELF. A ranked list of roofs is trivial to produce and
almost as easy to get silently wrong: any weighted sum of plausible-sounding
signals yields a confident ordering, including one that is pure noise. Josh's
time is the scarce resource in this loop, and a triage that ranks noise spends
all of it. So --validate scores every signal against the roofs he has marked
complete, where truth is known: panels from THIS build measured against the
lines HE drew. A signal that does not predict real failure is dropped from the
score and said so out loud, rather than quietly carried.

MEASURED on 84 complete roofs, mean real crossing rate 20.4% of panels
(Spearman against that rate):

  +0.70  CROSS_MODEL   panels crossing a line the vision model predicted
  +0.42  FACETS/100M2  facet density -- but scale-confounded, see below
  +0.36  SMALL_FACET   area-weighted typical facet size (the scale-fair one)
  +0.16  UNUSED        share of roof area carrying no panels
  -0.10  CONFIDENCE    the build's own roof_confidence -- NO SIGNAL, dropped
   n/a   CROSS_FACET   panels spanning two of the build's own facets

Three of those findings were not what was expected going in:

  THE MODEL IS THE BEST PREDICTOR, at +0.70, despite held-out F1 of only 0.43 on
  ridges and 0.13 on cliffs. Being good enough to say "look at this roof" is a
  far lower bar than being good enough to say "the ridge is exactly here", and
  the model clears the first bar comfortably while failing the second. Because
  it is trained on these very labels, --validate re-runs its correlation on the
  model's own held-out split every time: +0.72 seen vs +0.68 unseen, a 0.04 gap,
  so it generalises rather than remembering. That check must keep running -- each
  retrain on new labels renews the risk.

  ROOF_CONFIDENCE DOES NOT PREDICT *THIS* FAILURE (-0.10), which is not the same
  as not working. It is _area_weighted_inlier: the share of a roof's LiDAR points
  within 30 cm of the planes we place panels on -- plane-FIT quality. A panel
  crossing a ridge is a boundary-PLACEMENT error, and a roof can fit its planes
  beautifully while the folds between them are in the wrong place. Tested against
  what it should predict, it earns its keep: -0.26 against the share of roof left
  carrying no panels, and roofs Josh flagged average 0.879 against 0.927 for
  unflagged. It is dropped from THIS score and left alone everywhere else.

  CROSS_FACET IS A NON-EVENT, AND MEASURED THE WRONG THING AT FIRST. Expected to
  be the strongest signal; it fires on 2 roofs in 15,261 on the old build, which
  is a clean bill of health for sibling_facets in the panel fitter. On the new
  build it fired on 75 roofs across 16 regions and looked like a serious
  regression -- until 21 of 25 flagged roofs turned out to have OVERLAPPING
  facets. A panel inside the region where two facets overlap sits on one piece
  of roof that has been described twice; it bridges nothing and is perfectly
  placeable. Excluding overlapping pairs takes the count from 75 to 3.

  The overlap itself is now reported as its own thing, because it is a real if
  minor defect: on a random sample of 400 multi-facet roofs, 2.2% have
  overlapping facets covering 0.65% of facet area. Worth watching, not worth
  alarm -- and emphatically not worth telling Josh that 75 roofs have
  unplaceable panels when 3 do.

FACETS/100M2 SCORED HIGHER BUT IS NOT USED. Its median is 7.85 on roofs under
30 m2 and 2.41 on roofs over 150 m2, so it largely measures how small a roof is
and would fill the queue with garden sheds. Area-weighted typical facet size
asks the same question -- is a typical section big enough to lay panels on --
without the size confound, and is preferred despite scoring lower.

Usage:
    python tools/triage_roofs.py --validate          # do the signals work?
    python tools/triage_roofs.py --bundle 40         # worst 40 -> markup HTML
    python tools/triage_roofs.py --flagged 5371001 4735015    # force to top
    python tools/triage_roofs.py --compare data/triage/prev.json
"""

import argparse
import json
import math
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

DATA = ROOT / "data"
LAYOUTS = DATA / "panel_layouts.geojson"
POTENTIAL = DATA / "solar_potential.geojson"
LABELS = DATA / "roof_labels.json"
VISION = DATA / "vision_lines"
OUT_DIR = DATA / "triage"
LABEL_SET = DATA / "label_set"

# A panel overlapping a boundary by less than this is clipping it, not
# straddling it. Matches tools/measure_panel_crossings.py so the two tools
# cannot disagree about what a crossing is.
GRAZE_M2 = 0.10 * 0.10
PANEL_M2 = 1.7                  # a facet smaller than one panel can hold nothing
MODEL_SCORE_MIN = 0.60          # only reasonably confident predicted lines count
SKIP_FLAGS = {"absent", "not_building", "unclear"}


# ---------------------------------------------------------------- loading


def _load_nztm():
    """Facets and panels from the build, in metres.

    The build writes WGS84 for the map; every measurement here is an area or a
    distance, so it is reprojected once to NZTM rather than fudged with degrees.
    """
    import geopandas as gpd

    if not LAYOUTS.exists():
        print(f"no {LAYOUTS} -- run a build first")
        return None, None
    d = json.loads(LAYOUTS.read_text())
    feats = d["features"]
    rows = defaultdict(list)
    for f in feats:
        rows[f["properties"].get("kind")].append(f)

    out = {}
    for kind in ("facet", "panel"):
        fs = rows.get(kind, [])
        if not fs:
            out[kind] = None
            continue
        g = gpd.GeoDataFrame.from_features(fs, crs="EPSG:4326").to_crs(2193)
        out[kind] = g
    return out.get("facet"), out.get("panel")


def _model_lines(bid):
    """Predicted roof lines for one building, already in NZTM world metres."""
    p = VISION / f"{bid}.json"
    if not p.exists():
        return []
    try:
        d = json.loads(p.read_text())
    except Exception:
        return []
    segs = []
    for seg, sc in zip(d.get("lines", []), d.get("scores", [])):
        if sc is not None and sc < MODEL_SCORE_MIN:
            continue
        if len(seg) == 4:
            segs.append(((seg[0], seg[1]), (seg[2], seg[3])))
    return segs


def _drawn_lines(lab):
    """Josh's drawn lines for one roof as segments, in NZTM metres."""
    segs = []
    for l in lab.get("lines", []):
        pts = l.get("points")
        if pts and len(pts) >= 2:
            for i in range(len(pts) - 1):
                segs.append((tuple(pts[i]), tuple(pts[i + 1])))
        elif l.get("a") and l.get("b"):
            segs.append((tuple(l["a"]), tuple(l["b"])))
    return [s for s in segs
            if math.dist(s[0], s[1]) > 1e-6]


# ---------------------------------------------------------------- geometry


def _straddles(panel, lines_union):
    """Does this panel sit on both sides of a line, substantially?

    Splitting rather than intersecting: a panel whose corner clips a line yields
    one real piece and a sliver, while a panel laid across a fold yields two
    real pieces. Only the second is a placement error.
    """
    if lines_union is None or not panel.intersects(lines_union):
        return False, 0.0
    try:
        pieces = panel.difference(lines_union.buffer(0.01))
    except Exception:
        return False, 0.0
    geoms = list(getattr(pieces, "geoms", [pieces]))
    big = sorted((g.area for g in geoms if g.area > GRAZE_M2), reverse=True)
    if len(big) < 2:
        return False, 0.0
    return True, sum(big[1:])          # area on the wrong side(s)


def _facet_overlap(facets):
    """Facet pairs that overlap each other, and how much area is double-covered.

    Facets are supposed to partition a roof. When they overlap, the same square
    metre belongs to two planes at once -- a real (if minor) geometry defect,
    measured district-wide at 2.2% of multi-facet roofs and 0.65% of facet area.
    """
    pairs = set()
    area = 0.0
    for i in range(len(facets)):
        for j in range(i + 1, len(facets)):
            try:
                a = facets[i].intersection(facets[j]).area
            except Exception:
                continue
            if a > GRAZE_M2:
                pairs.add((i, j))
                area += a
    return pairs, area


def _cross_facet_count(panels, facets):
    """Panels spanning two of the pipeline's own facets -- genuinely.

    THIS MEASURED THE WRONG THING FIRST TIME. A panel overlapping two facets by
    a real amount looks like it is bridging a fold, and on the old build it
    essentially never happened (2 roofs in 15,261). On the new build it fired on
    75 roofs across 16 regions, which read as a serious regression -- until
    checking showed 21 of 25 flagged roofs simply had OVERLAPPING facets. A
    panel sitting inside the region where two facets overlap is on one piece of
    roof that has been described twice; it is not bridging anything, and it is
    perfectly placeable.

    So overlapping pairs are excluded, and the overlap is reported separately as
    what it actually is. What survives is a panel spanning two facets that do
    NOT overlap, which is a real fold and a real placement error.
    """
    overlapping, ov_area = _facet_overlap(facets)
    n = 0
    wrong = 0.0
    for pnl in panels:
        parts = []
        for k, fc in enumerate(facets):
            if not pnl.intersects(fc):
                continue
            try:
                a = pnl.intersection(fc).area
            except Exception:
                continue
            if a > GRAZE_M2:
                parts.append((k, a))
        if len(parts) < 2:
            continue
        # Every pair this panel touches must be a genuine neighbour, not two
        # descriptions of the same roof.
        idx = [k for k, _ in parts]
        real = any((min(a, b), max(a, b)) not in overlapping
                   for i, a in enumerate(idx) for b in idx[i + 1:])
        if not real:
            continue
        n += 1
        wrong += sum(sorted((a for _, a in parts), reverse=True)[1:])
    return n, wrong, ov_area


# ---------------------------------------------------------------- signals


def compute_signals(facets_gdf, panels_gdf, potential, use_model=True):
    """One row per building: the raw, uncombined signals."""
    from shapely.geometry import LineString
    from shapely.ops import unary_union

    fac_by = defaultdict(list)
    for rec in facets_gdf.itertuples():
        fac_by[int(rec.building_id)].append(rec)
    pan_by = defaultdict(list)
    for rec in panels_gdf.itertuples():
        pan_by[int(rec.building_id)].append(rec.geometry)

    rows = {}
    ids = set(fac_by) | set(pan_by) | set(potential)
    for bid in ids:
        facs = fac_by.get(bid, [])
        pans = pan_by.get(bid, [])
        prop = potential.get(bid, {})

        fgeoms = [f.geometry for f in facs]
        farea = sum(g.area for g in fgeoms) or 0.0
        parea = sum(p.area for p in pans) or 0.0

        # facets carrying nothing
        empty_area = 0.0
        for f in facs:
            pc = getattr(f, "panel_count", 0) or 0
            if not pc:
                empty_area += f.geometry.area
        small = sum(1 for g in fgeoms if g.area < PANEL_M2)

        confs = [(getattr(f, "roof_confidence", None), f.geometry.area)
                 for f in facs]
        confs = [(c, a) for c, a in confs if c is not None]
        conf_w = (sum(c * a for c, a in confs) / sum(a for _, a in confs)
                  if confs else None)

        n_cross, cross_area, ov_area = _cross_facet_count(pans, fgeoms) if (
            pans and len(fgeoms) > 1) else (0, 0.0, 0.0)

        n_model, model_area = 0, 0.0
        if use_model and pans:
            segs = _model_lines(bid)
            if segs:
                mu = unary_union([LineString(s) for s in segs])
                for p in pans:
                    hit, w = _straddles(p, mu)
                    if hit:
                        n_model += 1
                        model_area += w

        # Facet size, not facet density. facets_per_100m2 looked like the
        # obvious fragmentation measure and is scale-broken: measured across
        # this build its median is 7.85 on roofs under 30 m2 and 2.41 on roofs
        # over 150 m2, so it ranks every garden shed as catastrophically
        # fragmented. Median facet area asks the question that actually
        # matters -- is a typical section here big enough to lay panels on --
        # and asks it the same way at every roof size.
        # AREA-WEIGHTED, which matters more than it looks. A plain median over
        # facets [5.0, 5.5, 332.0] returns 5.5 and condemns a roof that is one
        # clean plane with two slivers hanging off it. Weighting each facet by
        # its own area asks the question from the roof's point of view -- how
        # big is the facet under a typical square metre -- and that roof
        # correctly returns 332.
        med_facet = 0.0
        if fgeoms:
            ordered = sorted((g.area for g in fgeoms))
            half = sum(ordered) / 2.0
            run = 0.0
            for x in ordered:
                run += x
                if run >= half:
                    med_facet = x
                    break

        rows[bid] = {
            "building_id": bid,
            "no_estimate_reason": prop.get("no_estimate_reason"),
            "facet_count": len(facs),
            "median_facet_m2": round(med_facet, 2),
            "small_facet_inv": round(1.0 / max(med_facet, 0.5), 4),
            "roof_area_m2": round(farea, 1),
            "panel_count": len(pans),
            "panel_area_m2": round(parea, 1),
            "cross_facet_n": n_cross,
            "cross_facet_frac": round(n_cross / len(pans), 4) if pans else 0.0,
            "cross_facet_area": round(cross_area, 2),
            "facet_overlap_m2": round(ov_area, 2),
            "facet_overlap_frac": round(ov_area / farea, 4) if farea else 0.0,
            "cross_model_n": n_model,
            "cross_model_frac": round(n_model / len(pans), 4) if pans else 0.0,
            "unused_frac": round(empty_area / farea, 4) if farea else 0.0,
            "facets_per_100m2": round(100.0 * len(facs) / farea, 3) if farea else 0.0,
            "small_facet_frac": round(small / len(facs), 4) if facs else 0.0,
            "confidence": round(conf_w, 4) if conf_w is not None else None,
        }
    return rows


# ---------------------------------------------------------------- truth


def truth_crossings(panels_gdf, labels):
    """Real failure rate on the roofs Josh has finished, for validation only.

    Truth is this build's panels measured against the lines he drew. Roofs
    flagged absent / not_building / unclear carry no geometry anyone believes
    and are excluded; roofs not marked complete are excluded too, because a
    partially drawn roof produces false crossings where he simply stopped.
    """
    from shapely.geometry import LineString
    from shapely.ops import unary_union

    pan_by = defaultdict(list)
    for rec in panels_gdf.itertuples():
        pan_by[int(rec.building_id)].append(rec.geometry)

    out = {}
    for k, lab in labels.items():
        bid = int(k)
        if lab.get("problem") in SKIP_FLAGS or not lab.get("complete"):
            continue
        segs = _drawn_lines(lab)
        pans = pan_by.get(bid, [])
        if not segs or not pans:
            continue
        mu = unary_union([LineString(s) for s in segs])
        n = 0
        wrong = 0.0
        for p in pans:
            hit, w = _straddles(p, mu)
            if hit:
                n += 1
                wrong += w
        pa = sum(p.area for p in pans)
        out[bid] = {
            "panels": len(pans),
            "crossing": n,
            "crossing_frac": n / len(pans),
            "wrong_area_frac": wrong / pa if pa else 0.0,
        }
    return out


def _spearman(xs, ys):
    """Rank correlation, written out rather than pulling in scipy."""
    n = len(xs)
    if n < 4:
        return float("nan")

    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else float("nan")


# Everything measurable gets validated, including the ones expected to fail --
# a signal is dropped because it did not predict, not because it was left out.
SIGNALS = ["cross_facet_frac", "unused_frac", "facets_per_100m2",
           "small_facet_inv", "small_facet_frac", "cross_model_frac"]


def validate(rows, truth):
    """Does each signal actually predict real failure? Report, do not assume."""
    common = [b for b in truth if b in rows]
    if len(common) < 8:
        print(f"only {len(common)} complete labelled roofs in this build -- "
              "too few to validate; treat the ranking as unproven")
        return {}
    ys = [truth[b]["crossing_frac"] for b in common]
    print(f"\nSIGNAL VALIDATION over {len(common)} roofs marked complete")
    print("  truth = panels from THIS build crossing the lines you drew\n")
    print(f"  mean real crossing rate: "
          f"{100 * sum(ys) / len(ys):.1f}% of panels\n")
    print(f"  {'signal':22s} {'spearman':>9s}   reading")
    got = {}
    for s in SIGNALS:
        xs = [rows[b].get(s) or 0.0 for b in common]
        r = _spearman(xs, ys)
        if math.isnan(r):
            note = "not enough spread"
        elif r >= 0.35:
            note = "predicts failure -- keep"
        elif r >= 0.15:
            note = "weak but positive"
        elif r > -0.15:
            note = "no signal -- DROP"
        else:
            note = "predicts the OPPOSITE -- drop"
        got[s] = r
        print(f"  {s:22s} {r:>+9.3f}   {note}")
    # confidence reads the other way round: low confidence should mean failure
    xs = [rows[b].get("confidence") if rows[b].get("confidence") is not None
          else 1.0 for b in common]
    r = _spearman(xs, ys)
    got["confidence"] = r
    print(f"  {'confidence':22s} {r:>+9.3f}   "
          f"{'low conf predicts failure -- keep' if r <= -0.15 else 'no signal -- DROP'}")
    print("\n  A signal that does not predict failure here does not go in the")
    print("  score. Ranking on a plausible-sounding number that is really noise")
    print("  would spend your labelling time on randomly chosen roofs.")

    # THE LEAK CHECK. cross_model_frac is derived from a model trained on these
    # very labels, and most labelled roofs are training roofs. A model that had
    # merely memorised their lines would score brilliantly here and be useless
    # on the 15,000 roofs nobody has drawn -- which is the entire job. So the
    # correlation is re-run on the model's own held-out split. This has to stay
    # automatic: every retrain on new labels renews the risk.
    man = DATA / "training" / "manifest.json"
    if man.exists() and "cross_model_frac" in got:
        try:
            val_ids = {int(x) for x in
                       json.loads(man.read_text()).get("val_building_ids", [])}
        except Exception:
            val_ids = set()
        seen = [b for b in common if b not in val_ids]
        held = [b for b in common if b in val_ids]
        if len(held) >= 4 and len(seen) >= 4:
            def rho(idl):
                return _spearman([rows[b].get("cross_model_frac") or 0.0
                                  for b in idl],
                                 [truth[b]["crossing_frac"] for b in idl])
            rs, rh = rho(seen), rho(held)
            gap = rs - rh
            print(f"\n  LEAK CHECK on cross_model_frac "
                  f"({len(seen)} training-seen, {len(held)} held-out)")
            print(f"    training-seen {rs:+.3f}   held-out {rh:+.3f}   "
                  f"gap {gap:+.3f}")
            if rh < 0.15:
                print("    -> the signal does NOT survive on unseen roofs. It is")
                print("       memorisation. Dropping it from the score.")
                got["cross_model_frac"] = 0.0
            elif gap > 0.30:
                print("    -> most of the strength is memorisation; it still")
                print("       predicts on unseen roofs but trust it less.")
            else:
                print("    -> holds up on roofs the model never saw. Real signal.")
        else:
            print("\n  LEAK CHECK on cross_model_frac: too few held-out roofs "
                  "to run; treat its strength as unproven.")
    return got


# ---------------------------------------------------------------- scoring


def score(rows, weights):
    """Combine the validated signals into one suspicion number per roof."""
    def norm(vals):
        v = sorted(x for x in vals if x is not None)
        if not v:
            return lambda x: 0.0
        lo, hi = v[0], v[int(len(v) * 0.98)] if len(v) > 4 else v[-1]
        rng = (hi - lo) or 1.0
        return lambda x: 0.0 if x is None else max(0.0, min(1.0, (x - lo) / rng))

    norms = {s: norm([r.get(s) for r in rows.values()]) for s in weights
             if s != "confidence"}
    for bid, r in rows.items():
        total = 0.0
        parts = {}
        for s, w in weights.items():
            if s == "confidence":
                c = r.get("confidence")
                v = 0.0 if c is None else max(0.0, min(1.0, 1.0 - c))
            else:
                v = norms[s](r.get(s))
            parts[s] = round(v * w, 4)
            total += v * w
        r["score"] = round(total, 4)
        r["score_parts"] = parts
        r["reasons"] = _reasons(r)
    return rows


def _reasons(r):
    """Why this roof is on the list: a stable code plus words for the bundle.

    The code is what gets counted in the failure-mode summary. An earlier
    version counted the prose and produced categories like "facets on 15 m2",
    which is one category per roof size and tells nobody anything.
    """
    out = []

    def add(code, text):
        out.append({"code": code, "text": text})

    if r.get("no_estimate_reason"):
        add("not_estimated", f"not estimated: {r['no_estimate_reason']}")
    if r.get("cross_facet_n"):
        add("panel_spans_facets",
            f"{r['cross_facet_n']} panels span two separate facets")
    if r.get("facet_overlap_frac", 0) > 0.05:
        add("facets_overlap",
            f"facets overlap each other over "
            f"{r['facet_overlap_m2']:.0f} m2 -- same roof described twice")
    if r.get("unused_frac", 0) > 0.45 and r.get("roof_area_m2", 0) > 40:
        add("roof_unused",
            f"{100 * r['unused_frac']:.0f}% of roof carries no panels")
    if r.get("median_facet_m2", 99) < 8 and r.get("roof_area_m2", 0) > 40:
        add("fragmented",
            f"{r['facet_count']} facets, typical one only "
            f"{r['median_facet_m2']:.1f} m2")
    if r.get("confidence") is not None and r["confidence"] < 0.5:
        add("low_confidence", f"roof confidence {r['confidence']:.2f}")
    if r.get("cross_model_n"):
        add("crosses_predicted_line",
            f"{r['cross_model_n']} panels cross a predicted line")
    return out


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="check the signals against roofs marked complete")
    ap.add_argument("--bundle", type=int, default=0,
                    help="write the worst N into a markup queue")
    ap.add_argument("--flagged", nargs="*", type=int, default=None,
                    help="building ids you flagged on the map -- forced to top")
    ap.add_argument("--include-labelled", action="store_true",
                    help="do not skip roofs you have already marked")
    ap.add_argument("--no-model", action="store_true",
                    help="skip the vision-line signal (faster)")
    ap.add_argument("--compare", default=None,
                    help="a previous triage json, to show movement")
    ap.add_argument("--min-area", type=float, default=40.0, dest="min_area",
                    help="ignore roofs smaller than this (m2)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    facets, panels = _load_nztm()
    if facets is None or panels is None:
        return 1
    print(f"{len(facets)} facets, {len(panels)} panels from {LAYOUTS.name}")

    potential = {}
    if POTENTIAL.exists():
        for f in json.loads(POTENTIAL.read_text())["features"]:
            p = f["properties"]
            if p.get("building_id") is not None:
                potential[int(p["building_id"])] = p

    labels = {}
    if LABELS.exists():
        labels = json.loads(LABELS.read_text()).get("buildings", {})

    ids = None
    if a.limit:
        ids = set(sorted({int(b) for b in facets["building_id"]})[:a.limit])
        facets = facets[facets["building_id"].isin(ids)]
        panels = panels[panels["building_id"].isin(ids)]

    print("computing signals...")
    rows = compute_signals(facets, panels, potential, use_model=not a.no_model)
    print(f"  {len(rows)} buildings scored")

    # Weights follow the measured Spearman correlations against real failure
    # (see --validate), not intuition. The ordering was a surprise: the vision
    # model's predicted lines are the single best predictor at +0.70 even
    # though the model is far too weak to define geometry itself. Being good
    # enough to say "look at this roof" is a much lower bar than being good
    # enough to say "the ridge is here".
    weights = {"cross_model_frac": 0.50, "small_facet_inv": 0.25,
               "unused_frac": 0.10, "cross_facet_frac": 0.10,
               "confidence": 0.05}

    if a.validate:
        truth = truth_crossings(panels, labels)
        got = validate(rows, truth)
        # drop anything that failed to predict, so the ranking uses only what
        # actually works on this build
        kept = {}
        for s, w in weights.items():
            r = got.get(s)
            if r is None or math.isnan(r):
                continue
            ok = (r <= -0.15) if s == "confidence" else (r >= 0.15)
            if ok:
                kept[s] = w
        if kept:
            tot = sum(kept.values())
            weights = {k: v / tot for k, v in kept.items()}
            print(f"\n  scoring on: {', '.join(weights)}")
        else:
            print("\n  NO SIGNAL PREDICTED FAILURE. The ranking below is not")
            print("  trustworthy -- do not spend labelling time on it.")

    rows = score(rows, weights)

    ranked = sorted(rows.values(), key=lambda r: -r["score"])
    if not a.include_labelled:
        done = {int(k) for k, v in labels.items() if v.get("complete")}
        ranked = [r for r in ranked if r["building_id"] not in done]

    # A 7 m2 shed is not worth a markup session. The queue is a claim on Josh's
    # attention, so anything too small to carry a worthwhile array is dropped
    # from it rather than ranked and then ignored.
    ranked = [r for r in ranked if r["roof_area_m2"] >= a.min_area]

    # Two different jobs, so two different lists. A roof with no estimate at all
    # needs a decision about WHY (absent? not a building? unreadable?); a roof
    # with a bad layout needs its lines drawn. Mixing them buries the second
    # under the first -- there are 857 of the first kind.
    no_est = [r for r in ranked if r.get("no_estimate_reason")]
    suspect = [r for r in ranked if not r.get("no_estimate_reason")]

    if a.flagged:
        fl = set(a.flagged)
        for r in suspect:
            if r["building_id"] in fl:
                r["score"] = 1.0 + r["score"]
                r["reasons"] = [{"code": "you_flagged",
                                 "text": "YOU FLAGGED THIS ON THE MAP"}] \
                    + r["reasons"]
        suspect.sort(key=lambda r: -r["score"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # The run CONFIG is saved, not just the weights. --no-model leaves the
    # weights dict untouched and simply never computes cross_model_frac, so two
    # runs can carry identical weights and still be incomparable -- which is
    # exactly what happened: comparing a --no-model baseline against a full run
    # reported "9152 roofs worse" when nothing about the roofs had changed.
    (OUT_DIR / "roof_triage.json").write_text(json.dumps(
        {"weights": weights, "min_area_m2": a.min_area,
         "config": {"use_model": not a.no_model, "min_area": a.min_area,
                    "validated": bool(a.validate)},
         "roofs": suspect, "not_estimated": no_est}, indent=1))

    print(f"\nWORST LAYOUTS  (estimated, but the geometry looks wrong)\n")
    for r in suspect[:20]:
        why = "; ".join(x["text"] for x in r["reasons"][:2]) or "low confidence"
        print(f"  {r['score']:.3f}  #{r['building_id']:<9} "
              f"{r['panel_count']:>3}p {r['roof_area_m2']:>6.0f}m2  {why}")

    modes = Counter()
    for r in suspect:
        for w in r["reasons"]:
            modes[w["code"]] += 1
    print(f"\nFAILURE MODES across {len(suspect)} suspect roofs "
          f"(>= {a.min_area:.0f} m2)")
    for m, c in modes.most_common(8):
        print(f"  {c:>6}  {m}")
    print(f"\n  plus {len(no_est)} roofs with no estimate at all "
          f"-- separate list, they need a reason not a markup")

    if a.compare and Path(a.compare).exists():
        prev_doc = json.loads(Path(a.compare).read_text())
        # THE COMPARISON IS ONLY VALID IF BOTH RUNS SCORED THE SAME WAY.
        # --validate re-derives the weights every run by dropping whatever
        # failed to predict, so two runs can easily use different signal sets --
        # and then the mean score moves because the scoring moved, not because
        # the roofs got better. That is a fake result of exactly the kind this
        # tool exists to avoid, so it is checked rather than assumed.
        pw = prev_doc.get("weights") or {}
        pc = prev_doc.get("config") or {}
        now_cfg = {"use_model": not a.no_model, "min_area": a.min_area,
                   "validated": bool(a.validate)}
        cfg_differs = bool(pc) and pc != now_cfg
        if cfg_differs or set(pw) != set(weights) or any(
                abs(pw.get(k, 0) - weights.get(k, 0)) > 1e-6 for k in
                set(pw) | set(weights)):
            print(f"\n  REFUSING TO COMPARE against {a.compare}")
            print(f"    then: {', '.join(f'{k}={v:.2f}' for k, v in sorted(pw.items())) or '(none)'}"
                  f"   config {pc or '(not recorded)'}")
            print(f"    now : {', '.join(f'{k}={v:.2f}' for k, v in sorted(weights.items())) or '(none)'}"
                  f"   config {now_cfg}")
            print("    The two runs scored on different signals, so any change")
            print("    in mean score would be the scoring changing, not the")
            print("    roofs. Re-run both the same way (e.g. both --no-model).")
            prev = {}
        else:
            prev = {r["building_id"]: r for r in prev_doc["roofs"]}
        moved = [(r["building_id"], prev[r["building_id"]]["score"], r["score"])
                 for r in suspect if r["building_id"] in prev]
        if moved:
            better = sum(1 for _, p, c in moved if c < p - 0.02)
            worse = sum(1 for _, p, c in moved if c > p + 0.02)
            mp = sum(p for _, p, _ in moved) / len(moved)
            mc = sum(c for _, _, c in moved) / len(moved)
            print(f"\nAGAINST {a.compare}: mean score {mp:.3f} -> {mc:.3f}  "
                  f"({better} roofs better, {worse} worse, "
                  f"{len(moved) - better - worse} unchanged)")

    if a.bundle:
        LABEL_SET.mkdir(parents=True, exist_ok=True)
        pick = suspect[:a.bundle]
        ids = [r["building_id"] for r in pick]
        (LABEL_SET / "queue.json").write_text(json.dumps(
            {"ids": ids, "source": "triage_roofs.py",
             "reasons": {str(r["building_id"]):
                         [x["text"] for x in r["reasons"]] for r in pick}},
            indent=1))
        print(f"\nqueued the worst {len(ids)} roofs -> {LABEL_SET / 'queue.json'}")
        print("  next:  python tools/build_label_bundle.py --out mark_worst.html")

    print(f"\nwrote {OUT_DIR / 'roof_triage.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
