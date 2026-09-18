"""Read a roof's faces from the model trained on Josh's drawn faces.

The fourth candidate generator, and the first that learns the OUTPUT we
actually want. sam/line/lidar/hypothesis all produce polygons by assembling
or assuming: SAM segments appearance, the line net finds crease pixels that
a partition then has to close into cells, LiDAR grows planar regions, and
hypothesis fits a closed vocabulary of shapes. Each fails in its own way and
the failures are visible on the map -- a dropped line, an invented flat top.

Here the model predicts, per pixel, BOUNDARY and CORE; a watershed turns the
pair into regions that tile the roof exactly once. A region cannot fail to
close, and nothing is assumed about what shape a roof "should" be.

Over-segmentation is controlled where Josh asked for it to be -- "we need to
avoid too many lines or random invented lines" -- by MIN_FACE_FRAC and the
seed threshold, both measurable against his face counts rather than tuned by
eye.

STATUS 18 Sep: MEASURED, NOT SHIPPING. On the 35 held-out roofs it has
never trained on (tools/eval_face_sources.py), against Josh's own faces:

    reading                    agree   matched   faces vs Josh
    the chain that ships now   0.714     77.6%   +1.9 (over on 19 roofs)
    learned from his markup    0.577     55.9%   -1.0 (under on 14)

It is not close enough to replace anything, so nothing in the build calls
it. But note HOW each one fails: the shipping chain INVENTS faces, which is
the complaint Josh keeps raising, and this one draws too few. Every polygon
it does draw is closed and straight -- there is no junk to clean up, only
detail to gain.

TWO MEASURED DEAD ENDS, so neither is retried blind:
  * RID2 pretraining. 1,819 roof-centred German roofs reach 0.624 boundary
    F1 on their OWN held-out split -- the architecture learns creases well
    given data -- but fine-tuned onto Josh's roofs it scores 0.544/48.8%,
    WORSE than training on his 96 alone (0.567/54.0%). RID has no LiDAR
    (four of seven channels neutral), its masks are azimuth classes rather
    than face instances, and German roofs are not NZ roofs. This is the
    same corpus that was a dead end for the line detector; it is now a
    measured dead end twice, for different targets.
  * Watershed knobs. Swept core threshold 0.50/0.70/0.85 against minimum
    face 0.012/0.005: the best pairing (0.50, 0.005) buys 0.567 -> 0.577.
    The shortfall is not in the post-processing.

WHAT WOULD ACTUALLY MOVE IT: more of Josh's roofs. 96 training roofs is the
binding constraint, and unlike the shape-fitting path, this one converts his
marking effort directly into the output -- the only lever measured to work.
"""

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(os.environ.get("SOLAR_FACE_REGION_MODEL",
                            ROOT / "data" / "models" / "face_regions_v1.pt"))
SIZE = 256
FILL_PX = 208
MIN_FACE_FRAC = float(os.environ.get("SOLAR_FR_MIN_FACE", "0.005"))
CORE_THR = float(os.environ.get("SOLAR_FR_CORE_THR", "0.50"))
_NET = [None]


def _net():
    if _NET[0] is None:
        import torch
        sys.path.insert(0, str(ROOT / "tools"))
        from train_face_regions import UNet
        if not MODEL.exists():
            return None
        ck = torch.load(MODEL, map_location="cpu", weights_only=False)
        n = UNet(cin=ck.get("cin", 7), cout=ck.get("cout", 2))
        n.load_state_dict(ck["state_dict"])
        n.eval()
        _NET[0] = n
    return _NET[0]


def _frame(geom):
    minx, miny, maxx, maxy = geom.bounds
    span = max(maxx - minx, maxy - miny)
    m_per_px = span / FILL_PX
    half = SIZE / 2 * m_per_px
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    return (cx - half, cy - half, cx + half, cy + half), m_per_px


def predict_maps(geom, img_ds, pts):
    """(boundary, core) probability maps for this roof, or None."""
    import torch
    import rasterio.windows
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree
    net = _net()
    if net is None:
        return None
    b, m_per_px = _frame(geom)
    win = rasterio.windows.from_bounds(*b, img_ds.transform)
    rgb = np.moveaxis(img_ds.read([1, 2, 3], window=win, boundless=True,
                                  fill_value=0, out_shape=(3, SIZE, SIZE)),
                      0, -1).astype(np.float32) / 255.0
    z = np.zeros((SIZE, SIZE), np.float32)
    if pts is not None and len(pts) >= 50:
        gx, gy = np.meshgrid(
            np.linspace(b[0] + m_per_px / 2, b[2] - m_per_px / 2, SIZE),
            np.linspace(b[3] - m_per_px / 2, b[1] + m_per_px / 2, SIZE))
        tree = cKDTree(pts[:, :2])
        dist, idx = tree.query(np.c_[gx.ravel(), gy.ravel()], k=1)
        z = pts[idx, 2].reshape(SIZE, SIZE).astype(np.float32)
        z[dist.reshape(SIZE, SIZE) > 2.5] = np.nan
    base = np.nanpercentile(z, 5) if np.isfinite(z).any() else 0.0
    zf = np.nan_to_num(z - base, nan=0.0)
    zf = gaussian_filter(zf, max(2.0, 1.2 / m_per_px))
    dzy, dzx = np.gradient(zf, m_per_px)
    slope = np.degrees(np.arctan(np.hypot(dzx, dzy)))
    mag = np.hypot(dzx, dzy) + 1e-9
    x = np.dstack([rgb, np.clip(zf / 10.0, 0, 1), np.clip(slope / 45.0, 0, 1),
                   (-dzx / mag + 1) / 2, (dzy / mag + 1) / 2]).astype(np.float32)
    with torch.no_grad():
        t = torch.from_numpy(x).permute(2, 0, 1)[None]
        p = torch.sigmoid(net(t))[0].numpy()
    return p[0], p[1], b, m_per_px


def learned_faces(geom, img_ds, pts):
    """Face polygons in NZTM, read from the model. [] if unavailable."""
    from scipy import ndimage
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    try:
        from skimage.segmentation import watershed
    except Exception:
        return []
    out = predict_maps(geom, img_ds, pts)
    if out is None:
        return []
    bnd, core, b, m_per_px = out

    # roof mask in frame pixels
    from PIL import Image, ImageDraw
    im = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(im).polygon(
        [((x - b[0]) / (b[2] - b[0]) * SIZE,
          (1 - (y - b[1]) / (b[3] - b[1])) * SIZE)
         for x, y in geom.exterior.coords], fill=255)
    roof = np.array(im) > 0
    if roof.sum() < 400:
        return []

    seeds, nseed = ndimage.label((core > CORE_THR) & roof)
    if nseed == 0:
        return []
    # drop specks before they become faces -- this is the over-segmentation
    # control Josh asked for, expressed as a share of THIS roof
    min_px = max(30, int(MIN_FACE_FRAC * roof.sum()))
    keep = [i for i in range(1, nseed + 1) if (seeds == i).sum() >= min_px]
    if not keep:
        return []
    relabel = np.zeros(nseed + 1, np.int32)
    for n, i in enumerate(keep, start=1):
        relabel[i] = n
    seeds = relabel[seeds]
    lab = watershed(bnd, markers=seeds, mask=roof)

    inv_x = lambda px: b[0] + px / SIZE * (b[2] - b[0])
    inv_y = lambda py: b[1] + (1 - py / SIZE) * (b[3] - b[1])
    faces = []
    for i in range(1, len(keep) + 1):
        m = lab == i
        if m.sum() < min_px:
            continue
        try:
            from rasterio.features import shapes as rio_shapes
            polys = [Polygon(g["coordinates"][0])
                     for g, v in rio_shapes(m.astype("uint8"),
                                            mask=m.astype(bool))
                     if v == 1 and len(g["coordinates"][0]) >= 4]
        except Exception:
            continue
        if not polys:
            continue
        p = max(polys, key=lambda q: q.area)
        # frame px -> world, then straighten: a learned region has a ragged
        # raster edge, and a roof face does not.
        ring = [(inv_x(px), inv_y(py)) for px, py in p.exterior.coords]
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.geom_type != "Polygon":
            continue
        poly = poly.simplify(max(0.25, 1.2 * m_per_px), preserve_topology=True)
        if poly.area >= MIN_FACE_FRAC * geom.area:
            faces.append(poly)
    return faces
