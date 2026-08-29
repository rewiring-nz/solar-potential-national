"""
Flag roof-surface anomalies (skylights, existing solar panels, vents,
AC units, moss/staining patches) from the 0.1m aerial imagery, so
panel_fitting can avoid placing panels on top of them.

This exists because LiDAR height data has a real blind spot: anything
close to flush with the roof plane (skylights, existing panels) produces
no elevation signal for RANSAC to catch as an outlier. Colour is often
the *only* signal available for those cases -- a skylight or an existing
panel array looks nothing like surrounding roofing material even when
its height is identical.

Method is a heuristic, not a trained classifier: flag pixels whose colour
deviates from their own *local* neighbourhood (not the whole facet), keep
only blobs in a plausible object-size range, then generously buffer each
one. This will mis-flag some things (a hard shadow edge, a patch of moss,
ridge flashing) and miss others (a skylight that happens to closely match
the surrounding roofing colour) -- documented limitation, not hidden. The
pipeline treats a flag as "exclude from panel placement," which is the
conservative direction to be wrong in: worst case it costs a bit of
otherwise-usable roof area, not a panel mounted over a skylight.

Local rather than whole-facet-relative contrast: an earlier version
compared each pixel to the *whole facet's* median colour and flagged
outliers past a few standard deviations of the whole facet's own colour
spread. Confirmed directly against several real buildings that this
misses real equipment on any facet that *also* contains one clearly
different, larger anomaly elsewhere (a plant/equipment yard, a big skylit
area) -- that region inflates the facet-wide spread enough that a
subtler, smaller vent elsewhere on the same facet no longer reads as an
outlier relative to it, even though it's clearly a different object close
up. Comparing each pixel to a heavily blurred version of *its own
neighbourhood* instead (a standard unsharp-mask/high-pass construction)
sidesteps that: a large uniform equipment yard blurs into its own
neighbourhood just fine and stays quiet, while a compact object of either
size stands out against its immediate surroundings regardless of what
else is happening elsewhere on the same facet.

This only reliably lights up an object's higher-contrast edge, not its
full uniform interior (confirmed directly: a rooftop unit's interior often
blurs cleanly into its own local neighbourhood once that neighbourhood is
mostly the unit's own pixels, so only the transition ring to the
surrounding roof shows real local contrast). Rather than chase a precise
interior fill, each detected blob is hulled and then buffered outward by
a fixed margin -- deliberately erring toward *over*-covering a detected
object (consistent with this module's whole conservative philosophy)
instead of tuning morphology per-building to chase an exact outline.
"""

import sys
from pathlib import Path

import numpy as np
import rasterio
import shapely.vectorized
from rasterio.features import rasterize, shapes as rasterio_shapes
from rasterio.mask import mask as rasterio_mask
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components as sparse_connected_components
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, Point, shape
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

MIN_BLOB_AREA_M2 = 0.09  # ~30x30cm -- smaller than this is more likely JPEG/compression noise
MAX_BLOB_AREA_FRACTION = 0.4  # beyond this it's more likely a lighting gradient than a discrete object
Z_THRESHOLD = 2.75  # local-contrast-from-neighbourhood std devs to count as anomalous
MIN_VALID_PIXELS = 20  # facets smaller than this in the imagery can't be usefully analysed
BOUNDARY_ERODE_M = config.PANEL_EDGE_SETBACK_M  # keep the analysed region inside where panels could go anyway;
# also keeps mixed/blended pixels right on the facet's own edge (real ridge/valley flashing, a sliver of a
# neighbouring facet's different-coloured roofing) from reading as a false anomaly right along that boundary
LOCAL_BACKGROUND_RADIUS_M = 1.2  # how far a pixel's "local neighbourhood" reaches for the blurred-background
# estimate -- wide enough to average out normal roofing texture/seams, narrow enough that a typical vent/AC
# unit doesn't dominate its own background estimate (which would make it invisible to itself)
SATURATION_RGB_THRESHOLD = 250  # all three channels above this counts as "blown out"
SATURATION_LOCAL_STD_MAX = 4.0  # confirmed directly on a real building (#4735613): a sharp-edged,
# perfectly flat white wedge cutting across an otherwise normal roof, zero internal texture unlike
# every real surface around it -- the signature of sensor glare/lens flare in the source aerial
# photo, not a real light-coloured roof material (which keeps *some* texture: seams, panel joins,
# weathering). The local-contrast test above misses this entirely by construction: a uniformly
# saturated patch matches its own blurred neighbourhood estimate closely, so it reads as "quiet",
# not anomalous. Treated as unknown, not confirmed-clear -- excluded the same as a real
# obstruction, since a saturated patch carries no real information about what's actually there.
SATURATION_WINDOW_PX = 7  # local texture window for the saturation check, ~0.7m at this imagery's
# 0.1m/px resolution -- wide enough to distinguish real roofing texture from a flat glare patch,
# narrow enough not to blur across a genuine small light-coloured feature's own edge
# Buffering out merges scattered flagged points into one coherent object. It
# also leaves the object BIGGER than the thing it represents, and nothing on
# these two paths ever pulled it back (only the strong-cluster path trims).
# Josh, on 28 Rees St: obstructions "show up wider in the lidar than they
# really are, because they often have vertical edges but don't necessarily show
# as vertical edges in the lidar" -- a return near a vertical face lands at an
# intermediate height, so the flagged cluster is already dilated before we
# buffer it further. BLOB_TRIM_FRACTION pulls back that share of the buffer
# after the merge: a morphological close rather than a plain dilate.
# 0.4, not the full 1.0 -- and the reason is a mistake worth keeping written
# down. A full close was chosen because it placed 165 more panels than 0.4,
# which is optimising the one number Josh had already said is not a measure of
# quality. He found the consequence immediately: "lots of examples of panels
# overlapping obstructions now".
#
# Eroding by the whole buffer removes the LiDAR smear AND the safety margin,
# and on a small obstruction it removes nearly everything -- measured across
# eight reported buildings, the full close cut total obstruction area by 20%,
# but by 78% on a 12 m2 one and 60% on another. A 2 m2 vent simply vanished,
# so panels went straight over it.
#
# The floor below matters as much as the fraction: erosion is a fixed distance
# and small blobs have far more perimeter per unit area, so a proportional
# limit is what stops them being destroyed.
BLOB_TRIM_FRACTION = 0.4
BLOB_TRIM_MIN_AREA_KEPT = 0.7   # never erode a blob below this share of itself
BLOB_BUFFER_M = 0.25  # each detected blob is expanded by this much before being kept -- direct testing
# confirmed detection reliably finds an object's edge/transition ring but not its full uniform interior, so
# a plain outline of *detected pixels* under-covers the real object; buffering outward trades some excess
# margin (this module's own stated conservative direction) for reliably covering the whole real object
# instead of just the ring around it


def _glare_mask(rgb, valid):
    """Flags pixels that are both fully saturated (blown-out white) and
    locally textureless -- see SATURATION_LOCAL_STD_MAX's comment for why
    that combination, not saturation alone, is the real tell for a glare/
    flare artifact rather than genuine light-coloured roofing."""
    local_std = np.zeros(rgb.shape[1:])
    for c in range(3):
        local_mean = ndimage.uniform_filter(rgb[c], SATURATION_WINDOW_PX)
        local_sqmean = ndimage.uniform_filter(rgb[c] ** 2, SATURATION_WINDOW_PX)
        local_std = np.maximum(local_std, np.sqrt(np.maximum(local_sqmean - local_mean ** 2, 0)))
    saturated = np.all(rgb > SATURATION_RGB_THRESHOLD, axis=0)
    return saturated & (local_std < SATURATION_LOCAL_STD_MAX) & valid


def _local_contrast_map(imagery_ds, sample_geom):
    """Shared by detect_obstructions and the LiDAR/image cross-check below:
    per-pixel colour distance from a blurred estimate of each pixel's own
    local neighbourhood (see the module comment for why local, not
    whole-facet). Returns (dist, transform, mean_dist, std_dist, valid, rgb) or
    None if there's no usable imagery here."""
    try:
        arr, transform = rasterio_mask(imagery_ds, [sample_geom], crop=True, filled=True, nodata=0)
    except ValueError:
        return None

    if arr.shape[0] < 3:
        return None
    rgb = arr[:3].astype(float)
    alpha = arr[3] if arr.shape[0] > 3 else None
    valid = (alpha > 0) if alpha is not None else np.ones(rgb.shape[1:], dtype=bool)
    if valid.sum() < MIN_VALID_PIXELS:
        return None

    pixel_size_m = abs(transform.a)
    sigma_px = LOCAL_BACKGROUND_RADIUS_M / pixel_size_m

    # Local background per channel: a coverage-normalised Gaussian blur
    # (weight the blur by `valid` and divide it back out) so nodata/masked
    # pixels right at the facet's own edge don't drag the background
    # estimate toward black there.
    local_background = np.empty_like(rgb)
    valid_f = valid.astype(float)
    blurred_weight = ndimage.gaussian_filter(valid_f, sigma_px)
    for c in range(3):
        blurred_channel = ndimage.gaussian_filter(np.where(valid, rgb[c], 0.0), sigma_px)
        local_background[c] = blurred_channel / np.maximum(blurred_weight, 1e-6)

    dist = np.sqrt(((rgb - local_background) ** 2).sum(axis=0))
    # Median and MAD, not mean and standard deviation. The threshold downstream
    # is `dist > centre + z * spread`, so the statistic that sets it must not be
    # moved by the very objects it is meant to find: every skylight and duct
    # inflates a standard deviation, which raises the bar against itself. That
    # self-masking is a plausible part of the under-detection Josh marked on
    # 29 Edinburgh Dr -- roughly ten obstructions marked, and the colour path
    # was finding two.
    #
    # Measured, mean/std -> median/MAD, on the two roofs with obstruction ground
    # truth: 29 Edinburgh 2 blobs / 2.6 m2 -> 5 blobs / 9.1 m2, toward the ~10
    # marked; 17 Cardigan St unchanged at 10 blobs / 12.3 m2, which is the roof
    # that previously caught this module inventing objects, so the extra
    # sensitivity is not manufacturing detections on a uniform roof.
    #
    # 1.4826 makes MAD a consistent estimator of sigma for Gaussian data, so
    # the existing z thresholds keep their meaning and did not need retuning.
    d = dist[valid]
    centre = float(np.median(d))
    spread = float(np.median(np.abs(d - centre))) * 1.4826
    return dist, transform, centre, spread, valid, rgb


def _trim_blob(geom, buffer_m):
    """Pull a merged blob back in, without letting it vanish.

    The outward buffer exists to join scattered points into one object; keeping
    that full expansion also inflates the object. Trimming restores the size
    while keeping the merge. A blob that trimming would erase is kept whole --
    a real 0.4 m vent modelled slightly large beats one not modelled at all.
    """
    if BLOB_TRIM_FRACTION <= 0 or geom.is_empty:
        return geom
    trimmed = geom.buffer(-BLOB_TRIM_FRACTION * buffer_m)
    if trimmed.is_empty or trimmed.geom_type not in ("Polygon", "MultiPolygon"):
        return geom
    if trimmed.area < BLOB_TRIM_MIN_AREA_KEPT * geom.area:
        return geom   # small blob: the erosion would take most of it, so skip it
    return trimmed


# Skylights are the clearest thing on a roof to a camera and among the hardest
# for everything else here. Josh marked 29 Edinburgh Dr: roughly ten
# obstructions, mostly 1-2 m skylights. The pipeline found four. They are near
# flush, so the height path cannot see them, and the colour path scores LOCAL
# contrast -- a z-score against a blurred neighbourhood -- which a bright panel
# on a variably-lit roof does not always clear.
#
# Measured on that roof, the simplest possible test does better: the top decile
# of brightness gives nine compact blobs over 0.5 m2 where the full detector
# gives four. A roof is mostly one material, so its brightness distribution has
# a long thin upper tail and that tail is glass and metal.
#
# Kept deliberately narrow. Only compact, skylight-sized blobs qualify -- a
# large bright region is a membrane roof or sun on a slope, not an object.
# The threshold is taken over the WHOLE ROOF, never a single facet, and that is
# the difference between this working and inventing objects. A percentile always
# finds something: by construction a tenth of any facet is in its own top
# decile, so run per facet the detector cannot return nothing, and on 17
# Cardigan St -- a uniformly sunlit terracotta roof with, per Josh, seven small
# obstructions -- it manufactured eight blobs totalling 21 m2, carving 17% of a
# 149 m2 roof down to 9 panels. Against the whole roof that same roof yields two
# small blobs, while 29 Edinburgh Dr keeps eight real skylights.
BRIGHT_PERCENTILE = 95
BRIGHT_MIN_AREA_M2 = 0.5
BRIGHT_MAX_AREA_M2 = 4.0        # a skylight or vent; larger bright regions are roof
BRIGHT_MAX_ELONGATION = 4.0
BRIGHT_MAX_TOTAL_SHARE = 0.15   # if this much of a facet is "bright" it is the roof


def detect_bright_objects(imagery_ds, facet_geom, roof_geom=None):
    """Compact bright blobs: skylights, vents, flashing, plant housings.

    roof_geom is the whole building's roof; the brightness threshold comes from
    it, and only blobs inside facet_geom are returned. Without it the threshold
    falls back to the facet, which is the failure described above."""
    if imagery_ds is None or facet_geom.is_empty:
        return []
    scope = roof_geom if (roof_geom is not None and not roof_geom.is_empty) else facet_geom
    minx, miny, maxx, maxy = scope.bounds
    try:
        window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, imagery_ds.transform)
        arr = imagery_ds.read([1, 2, 3], window=window)
    except Exception:
        return []
    if arr.size == 0 or arr.shape[1] < 6 or arr.shape[2] < 6:
        return []
    wt = imagery_ds.window_transform(window)
    rgb = np.moveaxis(arr, 0, -1).astype(float)
    scope_mask = rasterize([(scope, 1)], out_shape=rgb.shape[:2], transform=wt).astype(bool)
    mask = rasterize([(facet_geom, 1)], out_shape=rgb.shape[:2], transform=wt).astype(bool)
    if scope_mask.sum() < MIN_VALID_PIXELS or mask.sum() < MIN_VALID_PIXELS:
        return []
    lum = rgb.mean(axis=2)
    thresh = np.percentile(lum[scope_mask], BRIGHT_PERCENTILE)
    bright = (lum > thresh) & mask
    if bright.sum() > BRIGHT_MAX_TOTAL_SHARE * mask.sum():
        return []
    labels, n = ndimage.label(bright)
    if n == 0:
        return []
    px_area = abs(wt.a) * abs(wt.e)
    out = []
    for i in range(1, n + 1):
        blob = labels == i
        area = float(blob.sum()) * px_area
        if not (BRIGHT_MIN_AREA_M2 <= area <= BRIGHT_MAX_AREA_M2):
            continue
        polys = [shape(gj) for gj, v in rasterio_shapes(blob.astype(np.uint8),
                                                        mask=blob, transform=wt) if v == 1]
        if not polys:
            continue
        poly = max(polys, key=lambda q: q.area).buffer(BLOB_BUFFER_M)
        if poly.is_empty or poly.geom_type != "Polygon":
            continue
        if _elongation_ratio(poly) > BRIGHT_MAX_ELONGATION:
            continue          # a long thin bright strip is flashing along a ridge
        out.append(poly.intersection(facet_geom))
    return [q for q in out if not q.is_empty and q.geom_type == "Polygon"]


def detect_obstructions(imagery_ds, facet_geom, z_threshold=None, boundary_erode_m=None):
    """Returns a list of shapely Polygons (in imagery_ds's CRS -- same as
    facet_geom) flagged as likely roof obstructions/anomalies within this
    facet. Empty list if nothing flagged or the facet is too small/lacks
    imagery coverage to analyse. z_threshold/boundary_erode_m override the
    module defaults -- exposed for the live parameter-tuning server."""
    z_threshold = Z_THRESHOLD if z_threshold is None else z_threshold
    boundary_erode_m = BOUNDARY_ERODE_M if boundary_erode_m is None else boundary_erode_m
    sample_geom = facet_geom.buffer(-boundary_erode_m)
    if sample_geom.is_empty:
        return []
    if sample_geom.geom_type == "MultiPolygon":
        sample_geom = max(sample_geom.geoms, key=lambda p: p.area)
    elif sample_geom.geom_type != "Polygon":
        return []

    contrast = _local_contrast_map(imagery_ds, sample_geom)
    if contrast is None:
        return []
    dist, transform, mean_dist, std_dist, valid, rgb = contrast
    glare = _glare_mask(rgb, valid)

    if std_dist < 1e-6:  # perfectly uniform local contrast -- nothing more for the colour-outlier
        color_flagged = np.zeros_like(valid)  # test to add, but a saturated/glare patch is still real
    else:
        color_flagged = (dist > mean_dist + z_threshold * std_dist) & valid

    flagged = color_flagged | glare
    flagged = ndimage.binary_opening(flagged, structure=np.ones((3, 3)))

    labeled, n_blobs = ndimage.label(flagged, structure=np.ones((3, 3)))
    if n_blobs == 0:
        return []

    pixel_area_m2 = abs(transform.a) * abs(transform.e)
    facet_area_m2 = valid.sum() * pixel_area_m2

    obstructions = []
    for blob_id in range(1, n_blobs + 1):
        blob_mask = labeled == blob_id
        blob_area_m2 = blob_mask.sum() * pixel_area_m2
        if blob_area_m2 < MIN_BLOB_AREA_M2:
            continue
        rows, cols = np.where(blob_mask)
        xs, ys = rasterio.transform.xy(transform, rows, cols)
        hull = MultiPoint(np.column_stack([xs, ys])).convex_hull.buffer(BLOB_BUFFER_M, join_style="round")
        hull = _trim_blob(hull, BLOB_BUFFER_M)
        if hull.geom_type != "Polygon" or hull.area > MAX_BLOB_AREA_FRACTION * facet_area_m2:
            continue
        obstructions.append(hull)

    return obstructions


# --- Height-based detection (point cloud) -----------------------------
#
# The colour-based detector above exists specifically because LiDAR height
# has a blind spot for anything flush with the roof (skylights, existing
# panels). But that blind spot cuts both ways: colour has its own blind
# spot for anything that's a genuinely different *shape* but a similar
# *colour* to the surrounding roof -- a light-coloured vent housing, a
# pale skylight dome, a tank -- especially once the colour detector is
# only comparing against a *local* neighbourhood (see the module comment
# above), which a large, gently-shaded object can blend into just as
# easily as a small one. Height doesn't share that failure mode: a raised
# object reads as a large, unambiguous residual against the facet's own
# fitted plane regardless of what colour it happens to be.
#
# Reuses the facet's own already-fitted plane (a, b, c) rather than
# fitting anything new -- every point in the facet's footprint that
# RANSAC didn't claim as an inlier for that plane is, by construction,
# either on a different real facet (shouldn't be spatially inside this
# one to begin with) or a genuine height anomaly, so an independent
# re-query of the point cloud against the facet's own plane finds exactly
# the points segmentation already implicitly excluded.

HEIGHT_RESIDUAL_THRESHOLD_M = 0.4  # a real inlier for THIS plane already sits within
# roof_segmentation.RANSAC_DISTANCE_THRESHOLD_M (0.35m) of it by construction -- set just above
# that so this only fires on points the plane fit itself already couldn't explain, not on the
# same roofing-texture noise RANSAC's own tolerance was widened to absorb
HEIGHT_CLUSTER_RADIUS_M = 1.0  # matches roof_segmentation.POINTCLOUD_CLUSTER_RADIUS_M's own
# precedent for "still one physical object", not one point cloud noise floor
HEIGHT_MIN_CLUSTER_POINTS = 4
# DO NOT lower this, or HEIGHT_RESIDUAL_THRESHOLD_M, without re-running
# src/validate_obstructions.py. Tried 0.30m / 3 points to catch the plant that
# panels are still being placed on: the validated equipment reference
# (#5370338, 223 m2 of real ducting) collapsed from 42% of its roof to 1%.
# The mechanism is perverse and worth remembering -- flagging MORE points grows
# each cluster, the grown cluster then exceeds HEIGHT_STRONG_MAX_FACET_FRACTION
# and is rejected whole, so raising sensitivity LOWERS detection. Under-detect
# barely moved either (1 Earl St 12 -> 12 panels on raised structure). The real
# fix is reworking that cap, not turning up the gain in front of it.
HEIGHT_MIN_BLOB_AREA_M2 = MIN_BLOB_AREA_M2
HEIGHT_MAX_BLOB_AREA_M2 = 20.0  # absolute cap, not just MAX_BLOB_AREA_FRACTION -- confirmed
# directly on a real building: a whole *separate, lower roof section* wrongly merged into this
# facet during segmentation reads as one large, spatially contiguous height residual (every point
# on it is genuinely offset from THIS facet's plane, consistently, not noise) and can still clear
# a fractional cap on a large roof (171m2 blob on an 835m2 facet is only 20%) while being nothing
# like a discrete rooftop object -- that's a segmentation-quality symptom, not an obstruction, and
# flagging it would silently write off a fifth of a roof's real usable area. 20m2 comfortably
# covers a large single real object (a big chiller/tank) while rejecting a mis-merged sub-facet.
HEIGHT_BLOB_BUFFER_M = 0.15  # smaller than the colour detector's -- unlike a colour transition
# ring, a height anomaly is detected across an object's whole footprint (every point on top of
# it has a large residual, not just its edge), so there's much less real coverage to make up

# "Strong" height evidence: a blob whose member points stand well clear of the plane AND at
# varied heights is unambiguous 3D equipment and earns exemptions from the safety guards below.
# Both conditions matter and each kills a different false positive:
# - median |residual| > STRONG: way beyond plane-fit noise (real ducts/plant stand 0.8m+; the
#   0.4m detection threshold alone also catches marginal stuff worth guarding).
# - residual spread (p90-p10) > SPREAD: real equipment clusters are LUMPY (units, gaps, pipes at
#   assorted heights -- confirmed on a real plant-covered roof spanning 0.4-11m of residual),
#   while the one confirmed historical false positive -- a separate flat roof section wrongly
#   merged into the facet -- is a parallel plane: large median offset but nearly UNIFORM, tiny
#   spread. Spread separates the two exactly where median alone cannot.
# Confirmed necessary on two user-reported buildings: a commercial roof with 40% of its points
# >0.4m off-plane emitted almost no obstructions because the equipment clusters exceeded the
# 20m2 cap (plant decks) or failed the grey-duct-on-grey-roof photo cross-check (elongated).
HEIGHT_STRONG_MEDIAN_RESIDUAL_M = 0.8
HEIGHT_STRONG_SPREAD_M = 0.3
HEIGHT_STRONG_POINT_RADIUS_M = 0.5  # per-point footprint radius for strong clusters (see below)
# -- roughly the LiDAR point spacing on equipment, so one object's points merge into one shape
# while gaps between separate objects survive as usable lanes
OBSTRUCTION_TRIM_M = 0.2  # ...and then pull that merged shape back in by this much. The radius
# above exists for CONNECTIVITY (one duct run = one shape, not a string of beads), but it also
# leaves a 0.5m skirt around the real object: a single flagged point became a 0.79m2 exclusion,
# roughly 9x a small vent. Josh, on 35 Gorge Rd: "some obstructions getting drawn larger than
# they are which is interrupting what could otherwise be a clean array". Merge wide, then hug.
# Fragmenting into several smaller obstructions here is fine and usually more accurate.
HEIGHT_STRONG_MIN_PART_AREA_M2 = 0.4  # a strong-footprint fragment smaller than this is a lone
# stray point, not equipment worth carving a panel exclusion around
HEIGHT_STRONG_PLANAR_RMS_M = 0.12  # a large strong-footprint part whose own points fit their own
# plane this tightly is a MISSED ROOF LEVEL, not equipment -- confirmed on a real multi-level
# building where under-segmentation left other roof storeys inside one facet: chained into one
# strong cluster (multiple levels at different offsets = big spread, defeating the whole-cluster
# spread test), their footprints wrote off 426m2 and 120 panels of real usable roof. Equipment
# (units, pipes, vents at jumbled heights) never fits one plane at 0.12m RMS over 10+ m2; a flat
# or pitched roof level always does. Rejected parts are left unflagged rather than turned into
# facets here -- resolving them properly is segmentation's job, not this module's.
HEIGHT_STRONG_PLANAR_MIN_AREA_M2 = 10.0
HEIGHT_STRONG_ABOVE_FRACTION = 0.75  # a strong part only counts as equipment if its flagged
# points sit predominantly ABOVE the roof plane. Real ducts/plant/chimneys protrude upward
# (fraction ~1.0); a curved/ribbed/sawtooth surface mis-modelled as one flat facet deviates in
# BOTH directions (fraction ~0.5) -- the fingerprint separating every blanket-exclusion field
# report (airport #4722059, canopy #4721762, hangar #4721734, membrane #4679079, 29 Park St)
# from genuine rooftop equipment.
HEIGHT_STRONG_ABOVE_MARGIN_M = 0.25
# How certain the above-plane fingerprint must be before a part is allowed past
# the facet-fraction cap. Deliberately far stricter than HEIGHT_STRONG_ABOVE_
# FRACTION: passing the cap means claiming a large share of a roof, so it should
# take near-unanimous evidence, not merely enough to call something equipment.
HEIGHT_STRONG_UNCAPPED_ABOVE_FRACTION = 0.95
# A ceiling the bypass above CANNOT cross, whatever the fingerprint says.
#
# The bypass was justified on the argument that frac_above answers "equipment or
# mis-modelled surface" directly, so the crude size cap was redundant. That
# argument is wrong, and this file already said why: when a fitted plane sits
# BELOW the true roof, essentially every point reads as above it, so a
# mis-modelled surface scores frac_above near 1.0 and sails through a 95% bar.
#
# Measured on 22 Willow Place: the bypass carved 167.5 m2 of a 205 m2 roof --
# 82% -- against 1% before, taking the building from 86 panels to zero. Ten
# others went the same way.
#
# No plausible roof is more than half equipment. Past that the fingerprint has
# demonstrably failed, whatever it reports, so the oversized parts are refused.
HEIGHT_STRONG_TOTAL_CEILING = 0.5
HEIGHT_STRONG_MAX_FACET_FRACTION = 0.35  # a single strong-evidence part covering more of the
# facet than this is a mis-modelled roof SURFACE, not rooftop equipment -- no real duct/plant
# cluster blankets most of a roof. Catches curved/barrel roofs that region-growing collapses
# into one "flat" facet: every point sits far off the plane (median residual + spread both
# trip), and the planarity rescue above can't save it because a curved surface fails a plane
# fit exactly the way real equipment does. Confirmed real case: 29 Park St hall (#4726041),
# one 2386m2 facet at 0.2 deg whose "obstruction" was 1837m2 = 77% of the roof.


# Deck-seeking plane fit. The segmentation plane is a plain least-squares fit
# over ALL of a facet's points, so parapets and rooftop plant drag it off the
# deck -- and because obstructions pull the plane toward themselves, they hide
# themselves. Measured: on 17 Marine Pde the plane sits 0.82 m high, which is
# twice the 0.4 m detection threshold, and the share of points reading as
# raised goes 18.6% -> 40.1% once the fit is trimmed. On the equipment
# reference it goes 8.0% -> 14.8%.
DECK_FIT_ITERATIONS = 8
DECK_FIT_BAND_M = 0.25          # points this close to the current fit define the deck
DECK_FIT_MIN_POINTS = 25
# A deck fit is only trusted when a clear dominant surface exists to lock onto.
# Below this share the facet is not one surface -- typically roof levels that
# segmentation failed to separate -- and re-fitting its plane only moves the
# error around rather than removing it. 55 Arrowtown-Lake Hayes Rd is one
# 507 m2 facet at 46% inlier: with the segmentation plane 83% of its flagged
# points stand above (passing the equipment fingerprint), with a deck plane
# only 59% do, so its real 175 m2 structure was rejected as "roof shape" and
# 53 panels were placed on top of it.
#
# CAUTION: this bar is set on eight buildings and the two populations nearly
# touch -- 51% and 48% for facets that want the deck fit against 49% and 46%
# for facets that must not have it. It separates the current validation set
# and no more than that. Widen the set before trusting it further; a building
# that lands at 50% could go either way.
DECK_TRUST_MIN_INLIER = 0.50


def _fit_deck_plane(pts, band_m=DECK_FIT_BAND_M, iterations=DECK_FIT_ITERATIONS):
    """Iteratively trimmed plane fit: fit, keep the points near the fit, refit.

    Returns (plane, inlier_fraction). Converges onto the largest coherent
    surface -- the roof deck -- instead of the average of deck and equipment."""
    # Seed from the LOWER half, not from everything. The deck is the lowest
    # major surface on the roof -- equipment sits on top of it, never under it
    # -- and an unbiased trimmed fit converges onto whichever surface is
    # largest, which on 55 Arrowtown-Lake Hayes Rd is a 175 m2 structure
    # covering a third of the facet. The fit locked onto that instead, the real
    # roof then read as "below", the both-ways fingerprint (correctly) called
    # it roof shape rather than equipment, and the obstruction vanished
    # entirely: 175 m2 -> 0, with 53 panels landing on top of it.
    A0 = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))])
    try:
        (a0, b0, c0), *_ = np.linalg.lstsq(A0, pts[:, 2], rcond=None)
    except np.linalg.LinAlgError:
        return None, 0.0
    r0 = pts[:, 2] - (a0 * pts[:, 0] + b0 * pts[:, 1] + c0)
    keep = r0 <= np.median(r0)
    if keep.sum() < DECK_FIT_MIN_POINTS:
        keep = np.ones(len(pts), bool)
    a = b = c = None
    for _ in range(iterations):
        P = pts[keep]
        if len(P) < DECK_FIT_MIN_POINTS:
            break
        A = np.column_stack([P[:, 0], P[:, 1], np.ones(len(P))])
        try:
            (a, b, c), *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)
        except np.linalg.LinAlgError:
            return None, 0.0
        r = pts[:, 2] - (a * pts[:, 0] + b * pts[:, 1] + c)
        nxt = np.abs(r - np.median(r[keep])) < band_m
        if nxt.sum() < DECK_FIT_MIN_POINTS:
            break
        if (nxt == keep).all():
            keep = nxt
            break
        keep = nxt
    if a is None:
        return None, 0.0
    return (float(a), float(b), float(c)), float(keep.mean())


def detect_obstructions_from_height(pc_source, facet_geom, plane, residual_threshold_m=None,
                                     return_strength=False):
    """Returns a list of shapely Polygons (in the point cloud's CRS -- same
    as facet_geom) flagged as raised/recessed relative to the facet's own
    fitted plane. `plane` is that facet's (a, b, c) from roof_segmentation
    (z = a*x + b*y + c). Empty list if nothing flagged or no point-cloud
    coverage here -- callers should still run the colour-based detector
    regardless, this is a complement, not a replacement (see module
    comment). return_strength=True returns (polygon, is_strong) pairs
    instead -- see HEIGHT_STRONG_MEDIAN_RESIDUAL_M above."""
    residual_threshold_m = HEIGHT_RESIDUAL_THRESHOLD_M if residual_threshold_m is None else residual_threshold_m
    minx, miny, maxx, maxy = facet_geom.bounds
    pts = pc_source.points_in_bbox(minx - 1, miny - 1, maxx + 1, maxy + 1, building_only=True)
    if len(pts) < HEIGHT_MIN_CLUSTER_POINTS:
        return []

    inside = shapely.vectorized.contains(facet_geom, pts[:, 0], pts[:, 1])
    pts = pts[inside]
    if len(pts) < HEIGHT_MIN_CLUSTER_POINTS:
        return []

    # Re-fit the plane to the DOMINANT roof surface before measuring anything
    # against it. The segmentation plane is a plain least-squares fit over all
    # the facet's points, so parapets and rooftop plant drag it off the deck:
    # on 28 Rees St it sat 0.63m BELOW the roof, leaving 76% of points
    # "raised" -- the over-carve guard then (correctly) refused to flag most
    # of the roof, and the building ended up with ZERO obstructions despite
    # obvious ducting. The same bad reference produces the opposite failure
    # elsewhere (37 Camp St: half a clean roof flagged as one blob). A trimmed
    # fit locks onto the deck, which is what residuals should be measured from.
    # NOTE: a deck-seeking re-fit of this plane was tried here (both a
    # symmetric trim and a lower-envelope bias) to fix 28 Rees St, where the
    # segmentation plane sits 0.63m BELOW the roof and leaves 76% of points
    # "raised" so the over-carve guard suppresses everything. Both variants
    # DID surface the missed plant, but both also gutted the validated
    # equipment reference (#5370338: 220m2 of real ducting -> 33m2), because
    # a deck-hugging plane pushes densely-covered roofs into the strong-
    # evidence path where the 35% over-carve cap then rejects the big parts.
    # Reverted rather than shipped: the reference case is the canary. The
    # real fix is to make the strong path's caps aware that a correct plane
    # changes what "most of the roof is raised" means -- doing that safely
    # needs the cap logic reworked with its own validation set, not a
    # threshold nudge here.
    # See _fit_deck_plane: measuring residuals against a plane the obstructions
    # themselves pulled upward is what made this detector miss them.
    deck, deck_inlier = _fit_deck_plane(pts)
    deck_trusted = deck is not None and deck_inlier >= DECK_TRUST_MIN_INLIER
    if deck_trusted:
        plane = deck
    a, b, c = plane
    residual = pts[:, 2] - (a * pts[:, 0] + b * pts[:, 1] + c)
    flagged = np.abs(residual) > residual_threshold_m
    if flagged.sum() < HEIGHT_MIN_CLUSTER_POINTS:
        return []
    fpts = pts[flagged, :2]

    tree = cKDTree(fpts)
    pairs = tree.query_pairs(HEIGHT_CLUSTER_RADIUS_M, output_type="ndarray")
    n = len(fpts)
    if len(pairs) == 0:
        labels = np.arange(n)
    else:
        row = np.concatenate([pairs[:, 0], pairs[:, 1]])
        col = np.concatenate([pairs[:, 1], pairs[:, 0]])
        graph = coo_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
        _, labels = sparse_connected_components(graph, directed=False)

    facet_area_m2 = facet_geom.area
    abs_residual = np.abs(residual[flagged])
    obstructions = []
    for label_id in np.unique(labels):
        member_idx = np.where(labels == label_id)[0]
        if len(member_idx) < HEIGHT_MIN_CLUSTER_POINTS:
            continue
        member_res = abs_residual[member_idx]
        is_strong = (np.median(member_res) > HEIGHT_STRONG_MEDIAN_RESIDUAL_M
                     and (np.percentile(member_res, 90) - np.percentile(member_res, 10)) > HEIGHT_STRONG_SPREAD_M)

        if is_strong:
            # Tight, non-convex footprint from the flagged points themselves. A convex hull is
            # the wrong shape for a big STRONG cluster: on a plant-covered commercial roof the
            # 1m adjacency clustering legitimately chains dozens of separate units/ducts into
            # one cluster (confirmed real case: 2760 points whose hull covered 755m2 of a 991m2
            # facet -- 76%, instantly over any sane ceiling), and the hull blankets the real
            # usable lanes BETWEEN equipment along with the equipment itself. Per-point buffers
            # hug what the LiDAR actually saw and leave the lanes usable.
            member_pts3d = pts[flagged][member_idx]
            footprint = unary_union([Point(p).buffer(HEIGHT_STRONG_POINT_RADIUS_M) for p in fpts[member_idx]])
            trimmed = footprint.buffer(-OBSTRUCTION_TRIM_M)
            if not trimmed.is_empty:
                footprint = trimmed   # keep the untrimmed shape only if trimming erased it entirely
            parts = list(footprint.geoms) if footprint.geom_type == "MultiPolygon" else [footprint]
            # Largest first, so if the ceiling binds it is the most convincing
            # equipment that survives rather than whichever part came first.
            parts.sort(key=lambda g: -g.area)
            strong_area_used = 0.0
            plane_a, plane_b, plane_c = plane
            for part in parts:
                if part.geom_type != "Polygon" or part.area < HEIGHT_STRONG_MIN_PART_AREA_M2:
                    continue
                # Order matters here, and it used to be wrong. The size cap ran
                # FIRST and rejected any large part outright -- but the cap is
                # only a proxy for "this is a mis-modelled roof surface, not
                # equipment", and the very next test answers that question
                # directly: equipment protrudes upward (frac_above ~1.0) while a
                # curved/ribbed surface deviates both ways (~0.5). Asking the
                # proxy before the evidence is what made raising sensitivity
                # LOWER detection: more flagged points grow a cluster, the grown
                # cluster trips the cap, and the whole thing is thrown away.
                in_part = shapely.vectorized.contains(part, member_pts3d[:, 0], member_pts3d[:, 1])
                part_pts3d = member_pts3d[in_part]
                frac_above = None
                if len(part_pts3d) >= 5:
                    plane_z = (plane_a * part_pts3d[:, 0] + plane_b * part_pts3d[:, 1] + plane_c)
                    frac_above = float(np.mean(part_pts3d[:, 2] > plane_z + HEIGHT_STRONG_ABOVE_MARGIN_M))
                    if frac_above < HEIGHT_STRONG_ABOVE_FRACTION:
                        continue  # deviates both ways -> curved/ribbed roof shape, not equipment
                if part.area > HEIGHT_STRONG_MAX_FACET_FRACTION * facet_geom.area:
                    # Big. Keep it only where the fingerprint is unambiguous --
                    # essentially every point standing clear above the plane. A
                    # part with too few points to fingerprint keeps the old
                    # conservative behaviour and is dropped.
                    if frac_above is None or frac_above < HEIGHT_STRONG_UNCAPPED_ABOVE_FRACTION:
                        continue
                    # ...and never past the ceiling, however confident it looks.
                    if (strong_area_used + part.area
                            > HEIGHT_STRONG_TOTAL_CEILING * facet_geom.area):
                        continue
                strong_area_used += part.area
                if part.area >= HEIGHT_STRONG_PLANAR_MIN_AREA_M2:
                    in_part = shapely.vectorized.contains(part, member_pts3d[:, 0], member_pts3d[:, 1])
                    part_pts = member_pts3d[in_part]
                    if len(part_pts) >= 6:
                        x0, y0 = part_pts[:, 0].mean(), part_pts[:, 1].mean()
                        A = np.column_stack([part_pts[:, 0] - x0, part_pts[:, 1] - y0,
                                              np.ones(len(part_pts))])
                        try:
                            coeffs, *_ = np.linalg.lstsq(A, part_pts[:, 2], rcond=None)
                            fit_res = A @ coeffs - part_pts[:, 2]
                            if np.sqrt(np.mean(fit_res ** 2)) < HEIGHT_STRONG_PLANAR_RMS_M:
                                continue  # a missed roof level, not equipment (see constant above)
                        except np.linalg.LinAlgError:
                            pass
                obstructions.append((part.simplify(0.1), True))
            continue

        # Same above-plane physics as the strong branch: a cluster whose
        # points straddle the plane is roof shape (vault/rib/valley), not a
        # protruding object -- the sawtooth-canopy stripes came through here.
        member3d = pts[flagged][member_idx]
        if len(member3d) >= 5:
            pz = plane[0] * member3d[:, 0] + plane[1] * member3d[:, 1] + plane[2]
            if float(np.mean(member3d[:, 2] > pz + HEIGHT_STRONG_ABOVE_MARGIN_M)) < HEIGHT_STRONG_ABOVE_FRACTION:
                continue
        hull = MultiPoint(fpts[member_idx]).convex_hull.buffer(HEIGHT_BLOB_BUFFER_M, join_style="round")
        hull = _trim_blob(hull, HEIGHT_BLOB_BUFFER_M)
        if hull.geom_type != "Polygon":
            continue
        max_area = min(HEIGHT_MAX_BLOB_AREA_M2, MAX_BLOB_AREA_FRACTION * facet_area_m2)
        if hull.area < HEIGHT_MIN_BLOB_AREA_M2 or hull.area > max_area:
            continue
        obstructions.append((hull, False))

    if return_strength:
        return obstructions
    return [hull for hull, _ in obstructions]


# --- Combined detection: LiDAR + image, cross-checked ------------------
#
# Neither signal is trusted blind. Colour has the flush-object blind spot
# documented at the top of this file; height has its own, confirmed
# directly above on a real building: a whole neighbouring roof section
# that segmentation didn't cleanly separate from this facet reads as a
# large, genuine, but *wrong* height residual, and -- more subtly -- a
# facet's own edge is where its fitted plane is least reliable (fewest
# neighbours on one side during segmentation), so a real but non-object
# residual tends to show up there too, shaped like a thin wedge or strip
# hugging the boundary rather than a compact blob. An absolute area cap
# handles the first case. For the second, shape is the tell: a genuine
# rooftop object is roughly as wide as it is long; a plane-fit edge
# artifact is usually long and thin. Compact height detections are kept
# on their own -- that's the whole reason to have height at all, catching
# a same-coloured object colour alone can't see. Elongated ones are held
# to a higher bar: the 0.1m image is the higher-resolution, more directly
# interpretable signal of the two, so an elongated LiDAR blob is only
# kept if the image *also* shows some anomaly there, even a mild one well
# below the standalone colour detector's own confident threshold.

ELONGATION_RATIO_THRESHOLD = 3.0  # long-side/short-side of a height blob's own minimum
# rotated rectangle -- past this it reads as a strip/wedge, not a compact object
CROSS_CHECK_Z_THRESHOLD = 1.75  # deliberately looser than Z_THRESHOLD (2.75): this only asks
# "does the image show anything unusual across this whole shape", as corroboration for a LiDAR
# blob shaped like a plane-fit artifact, not standing alone as its own detection the way
# Z_THRESHOLD must. Checked against the *median* contrast value over the blob's own footprint,
# not a single pixel -- a lucky/noisy single pixel confirming an entire elongated strip would
# defeat the point of cross-checking at all


def _elongation_ratio(polygon):
    mrr = polygon.minimum_rotated_rectangle
    if mrr.geom_type != "Polygon":
        return 1.0
    coords = list(mrr.exterior.coords)
    if len(coords) < 3:
        return 1.0
    edge1 = np.hypot(coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
    edge2 = np.hypot(coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])
    long_side, short_side = max(edge1, edge2), max(min(edge1, edge2), 1e-6)
    return long_side / short_side



def _lidar_signature(blob, pc_source, plane):
    """Do the raw returns under this colour blob show ANY relief off the plane?

    The gate for uncorroborated colour detections. Requires enough points to
    judge (a blob over a LiDAR gap is dropped -- we cannot confirm it), and a
    quarter of them, or at least six, sitting more than 0.12 m off the facet
    plane."""
    if pc_source is None:
        return True                     # no LiDAR at all: keep old behaviour
    minx, miny, maxx, maxy = blob.bounds
    pts = pc_source.points_in_bbox(minx, miny, maxx, maxy, building_only=True)
    if len(pts) < 4:
        return False
    import shapely.vectorized as _sv
    inside = _sv.contains(blob, pts[:, 0], pts[:, 1])
    bp = pts[inside]
    if len(bp) < 4:
        return False
    a, b, c = plane
    res = np.abs(bp[:, 2] - (a * bp[:, 0] + b * bp[:, 1] + c))
    off = int((res > 0.12).sum())
    return off >= 6 or off / len(bp) >= 0.25


def detect_obstructions_combined(imagery_ds, pc_source, facet_geom, plane,
                                  z_threshold=None, boundary_erode_m=None,
                                  residual_threshold_m=None, roof_geom=None):
    """Runs both detectors and reconciles them per the module comment
    above: colour-based obstructions always kept; compact height-based
    obstructions always kept (colour structurally can't see a flush,
    same-coloured object); elongated/edge-hugging height-based ones kept
    only if the image corroborates them too -- UNLESS the height evidence
    is strong (see HEIGHT_STRONG_MEDIAN_RESIDUAL_M): a duct or plant run
    standing far off the plane at varied heights is real regardless of its
    shape, and the photo cross-check structurally fails on grey-equipment-
    on-grey-roof (confirmed on a real commercial roof where nearly all its
    extensive ducting was silently dropped by that check). Falls back to
    colour-only if pc_source is None or has no coverage here."""
    # Rural gap regions have no 0.1m urban imagery (LINZ layer is urban-only),
    # so colour detection is unavailable there -- fall back to LiDAR height
    # evidence alone rather than skipping the area entirely. Documented
    # degradation: flush objects (skylights, existing panels) go undetected.
    # Colour-only blobs must look like a real object. The colour detector
    # exists for FLUSH objects height can't see (skylights, existing panel
    # arrays) -- those are >=~1.2m2 and compact. Everything it flags that is
    # tiny or streaky is stains, shadow edges and parapet lines: 9 of 10
    # blobs on 17/60 Hallenstein St were 1-2m2 scatter, and 6 Shotover St
    # lost most of a good flat roof to edge streaks. Height-corroborated
    # blobs bypass both tests, so genuine equipment is never dropped.
    COLOUR_ONLY_MIN_AREA_M2 = 1.2
    COLOUR_ONLY_MAX_ELONGATION = 3.5
    # ...and a MAXIMUM. There was only a floor here, so a large compact colour
    # region passed as a "flush object". On a bright flat membrane roof that is
    # what shadow, staining and seam discolouration look like, and it is the
    # whole of the over-carve problem. Split by detector on the labelled set in
    # src/validate_obstructions.py:
    #     #4734932  height  52.7 m2   colour 258.3 m2   -> 55% of roof carved
    #     #5370372  height   6.3 m2   colour 249.0 m2   -> 41%
    #     #4735250  height  15.1 m2   colour  53.7 m2   -> 46%
    #     #5370338  height 225.3 m2   colour   8.6 m2   -> the VALIDATED
    #                                                      equipment reference
    # Every over-carve is colour; the real plant is height. That is the
    # discriminator the two earlier attempts at this went looking for and
    # missed -- they tried area caps and planarity tests that cannot separate
    # these, because the reference roof is legitimately 42% obstruction while
    # the worst over-carve is 55%.
    # The cap applies ONLY to blobs with no height corroboration: real
    # equipment that also shows up in the photo still bypasses it entirely.
    COLOUR_ONLY_MAX_AREA_M2 = 15.0
    COLOUR_CORROBORATION_MIN_AREA_M2 = 1.0
    COLOUR_CORROBORATION_MIN_FRACTION = 0.15
    # How far past the raised core a corroborated colour blob is kept. The
    # colour edge is real evidence of the object's extent, just not of the whole
    # tonal region it sits in.
    COLOUR_CORE_MARGIN_M = 0.4

    color_obs = [] if imagery_ds is None else detect_obstructions(
        imagery_ds, facet_geom, z_threshold, boundary_erode_m)
    if pc_source is None:
        return color_obs

    # Colour-blob valley veto: a colour anomaly whose underlying LiDAR sits
    # clearly BELOW the facet plane is the shadowed valley of a vault/rib/
    # sawtooth (roof geometry photographing dark), not an object. Flush
    # objects -- the colour path's real quarry (skylights, existing panels)
    # -- sit ON the plane and pass untouched. Field case: white sawtooth
    # canopy #4721762, every vault valley striped as an obstruction.
    if color_obs:
        va, vb, vc = plane
        kept_color = []
        for blob in color_obs:
            minx, miny, maxx, maxy = blob.bounds
            bpts = pc_source.points_in_bbox(minx, miny, maxx, maxy, building_only=True)
            if len(bpts) >= 5:
                import shapely.vectorized as _sv
                inside = _sv.contains(blob, bpts[:, 0], bpts[:, 1])
                bp = bpts[inside]
                if len(bp) >= 5:
                    res = bp[:, 2] - (va * bp[:, 0] + vb * bp[:, 1] + vc)
                    if np.median(res) < -HEIGHT_STRONG_ABOVE_MARGIN_M:
                        continue  # below-plane valley shadow, not an object
            kept_color.append(blob)
        color_obs = kept_color

    height_obs = detect_obstructions_from_height(pc_source, facet_geom, plane, residual_threshold_m,
                                                  return_strength=True)

    height_union = unary_union([h for h, _ in height_obs]) if height_obs else None
    filtered_color = []
    for blob in color_obs:
        # Corroboration must be REAL overlap, not a touch. This was
        # blob.intersects(height_union), so a 100 m2 shadow region that
        # happened to graze a 0.4 m2 vent was fully exempt from every colour
        # test -- which is why #4734932 kept carving 55% of its roof away even
        # after the size cap: its shadow blobs each clipped one of the small
        # height parts scattered across the roof.
        inter = blob.intersection(height_union).area if height_union is not None else 0.0
        corroborated = inter >= max(COLOUR_CORROBORATION_MIN_AREA_M2,
                                    COLOUR_CORROBORATION_MIN_FRACTION * blob.area)
        if corroborated:
            # Keep the corroborated PORTION, not the whole tonal region. Colour
            # says where a tonal region is; height says what is actually raised.
            # On a flat commercial roof the colour path flags membrane tone and
            # shadow -- 7 Shotover St: 242.7 m2 of a 462 m2 roof -- so keeping a
            # 73 m2 region whole because one small vent sits inside it carved a
            # quarter of that roof away (Josh: "incorrect obstructions").
            #
            # A blob small enough to be one plausible object is still kept
            # whole: trimming a 3 m2 vent housing back to its raised core would
            # lose the part of it the LiDAR simply missed.
            if blob.area <= COLOUR_ONLY_MAX_AREA_M2:
                filtered_color.append(blob)
            else:
                core = blob.intersection(height_union).buffer(COLOUR_CORE_MARGIN_M)
                core = core.intersection(blob)
                for part in (core.geoms if core.geom_type == "MultiPolygon" else [core]):
                    if part.geom_type == "Polygon" and part.area >= COLOUR_ONLY_MIN_AREA_M2:
                        filtered_color.append(part)
        elif (COLOUR_ONLY_MIN_AREA_M2 <= blob.area <= COLOUR_ONLY_MAX_AREA_M2
              and _elongation_ratio(blob) <= COLOUR_ONLY_MAX_ELONGATION
              and _lidar_signature(blob, pc_source, plane)):
            # Josh's fusion rule (29 Aug, 6 Weaver St): "an obstruction should
            # show some signature on both. For example you don't want to
            # identify painting discolouration as an obstruction, which lidar
            # can tell you is flat." Live testing showed exactly that failure
            # at scale: ~10 paint patches carved on 6 Weaver where he counts
            # ONE object, and dozens speckled across 101/8 Duke St. A colour
            # blob that overlaps a height detection is corroborated above;
            # one that does not must at least sit over raw returns that are
            # off the plane. Paint is flat; equipment is not. Skylights are
            # unaffected -- they come from detect_bright_objects, which joins
            # separately below.
            filtered_color.append(blob)
    color_obs = filtered_color

    if not height_obs:
        return color_obs

    compact = [h for h, strong in height_obs
               if strong or _elongation_ratio(h) <= ELONGATION_RATIO_THRESHOLD]
    elongated = [h for h, strong in height_obs
                 if not strong and _elongation_ratio(h) > ELONGATION_RATIO_THRESHOLD]

    confirmed_elongated = []
    if elongated and imagery_ds is None:
        confirmed_elongated = elongated  # no photo to cross-check against
    elif elongated:
        boundary_erode_m = BOUNDARY_ERODE_M if boundary_erode_m is None else boundary_erode_m
        sample_geom = facet_geom.buffer(-boundary_erode_m)
        if sample_geom.geom_type == "MultiPolygon":
            sample_geom = max(sample_geom.geoms, key=lambda p: p.area)
        # is_empty as well as the type check: eroding a thin facet leaves an
        # EMPTY Polygon, whose geom_type is still "Polygon", and rasterio then
        # fails computing bounds of a geometry with no coordinates. The path
        # above already guards this; this one did not, and reconstruction --
        # which produces more slender facets -- crashed a whole area's build on
        # the first roof that hit it.
        contrast = (_local_contrast_map(imagery_ds, sample_geom)
                    if sample_geom.geom_type == "Polygon" and not sample_geom.is_empty
                    else None)
        color_union = unary_union(color_obs) if color_obs else None
        for h in elongated:
            if color_union is not None and h.intersects(color_union):
                confirmed_elongated.append(h)
                continue
            if contrast is None:
                continue
            dist, transform, mean_dist, std_dist, valid, rgb = contrast
            if std_dist < 1e-6:
                continue
            hb_minx, hb_miny, hb_maxx, hb_maxy = h.bounds
            (r0, c0) = rasterio.transform.rowcol(transform, hb_minx, hb_maxy)
            (r1, c1) = rasterio.transform.rowcol(transform, hb_maxx, hb_miny)
            r0, r1 = sorted((max(0, r0), min(dist.shape[0], r1 + 1)))
            c0, c1 = sorted((max(0, c0), min(dist.shape[1], c1 + 1)))
            if r1 <= r0 or c1 <= c0:
                continue
            rows_idx, cols_idx = np.mgrid[r0:r1, c0:c1]
            xs, ys = rasterio.transform.xy(transform, rows_idx.ravel(), cols_idx.ravel())
            inside = shapely.vectorized.contains(h, np.array(xs), np.array(ys)).reshape(rows_idx.shape)
            if not inside.any():
                continue
            shape_dist = dist[r0:r1, c0:c1][inside]
            if np.median(shape_dist) > mean_dist + CROSS_CHECK_Z_THRESHOLD * std_dist:
                confirmed_elongated.append(h)

    # Bright compact objects join on their own terms -- see detect_bright_objects.
    # They are the skylight case, which the height path cannot see (near flush)
    # and the local-contrast colour path only sometimes clears. Josh marked
    # roughly ten obstructions on 29 Edinburgh Dr and this pipeline found four;
    # the brightness tail finds nine.
    try:
        bright = detect_bright_objects(imagery_ds, facet_geom, roof_geom)
    except Exception:
        bright = []

    all_obs = color_obs + compact + confirmed_elongated + bright
    if not all_obs:
        return []
    merged = unary_union(all_obs)
    return list(merged.geoms) if merged.geom_type == "MultiPolygon" else [merged]
