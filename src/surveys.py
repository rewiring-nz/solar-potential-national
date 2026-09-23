"""Which LiDAR/imagery survey covers a place, and where its data lives.

WHY THIS EXISTS. config.py names one layer per product:

    LINZ_DSM_LAYER     = 105855   # Otago - Queenstown LiDAR 1m DSM (2021)
    LINZ_DEM_LAYER     = 105898   # Otago - Queenstown LiDAR 1m DEM (2021)
    LINZ_IMAGERY_LAYER = 124754   # Queenstown 0.1m Urban Aerial Photos (2026)
    POINTCLOUD_BULK_URL = ".../pc-bulk/NZ21_Otago"

Every one of those is a Queenstown answer to a national question. New Zealand's
LiDAR is a patchwork of regional surveys flown in different years and published
as separate layers, so "the DSM layer" is not a property of the pipeline -- it
is a property of the place being built. Wellington needed a different
point-cloud store and was silently pointed at Otago's until 31 August; the
fetch succeeded and the data was simply somewhere else.

A constant cannot express that, and a hand-edited constant per deployment is
how the same mistake happens again. So a survey is a record with a coverage
bbox, and the fetchers ask which one covers the area they are building.

WHAT THIS DOES NOT DO. It does not discover surveys. The registry is still a
list someone maintains -- in config.SURVEYS, per deployment, because that is
the file that already differs between the Queenstown and Wellington repos.
What changes is that the list is DATA with coverage attached, so adding a
region cannot silently inherit the wrong survey, and a region outside every
listed survey is an error rather than a plausible-looking wrong answer.

The defaults keep working: with no config.SURVEYS at all, every lookup returns
the module-level constants, which is exactly today's behaviour.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

KEYS = ("dsm_layer", "dem_layer", "imagery_layer", "lidar_tile_index_layer",
        "pointcloud_bulk_url", "pointcloud_tile_year")

# What a survey record falls back to for any key it does not set.
_DEFAULT_FROM_CONFIG = {
    "dsm_layer": "LINZ_DSM_LAYER",
    "dem_layer": "LINZ_DEM_LAYER",
    "imagery_layer": "LINZ_IMAGERY_LAYER",
    "lidar_tile_index_layer": "LINZ_LIDAR_TILE_INDEX_LAYER",
    "pointcloud_bulk_url": "POINTCLOUD_BULK_URL",
    "pointcloud_tile_year": "POINTCLOUD_TILE_YEAR",
}


def defaults():
    return {k: getattr(config, var, None)
            for k, var in _DEFAULT_FROM_CONFIG.items()}


def _covers(survey, bbox):
    """Does this survey's coverage contain the whole of bbox?

    Whole, not partly: a region straddling two surveys would otherwise be
    fetched from whichever happened to be listed first, with half its data
    quietly missing. Straddling is a real case at survey boundaries and it
    needs splitting the region, which is a decision for a person.
    """
    w, s, e, n = bbox
    cw, cs, ce, cn = survey["bbox"]
    return cw <= w and cs <= s and ce >= e and cn >= n


def survey_for(bbox, name=None):
    """The survey covering `bbox`, as a dict with every key in KEYS.

    With no registry configured this is the single Queenstown survey the
    constants describe, so existing deployments behave identically.
    """
    registry = getattr(config, "SURVEYS", None)
    base = defaults()
    if not registry:
        base["name"] = getattr(config, "SURVEY_NAME", "default")
        return base
    hits = [s for s in registry if _covers(s, bbox)]
    if not hits:
        raise LookupError(
            f"no survey covers {name or bbox}. Add one to config.SURVEYS with "
            f"its coverage bbox and layer ids, or widen an existing one -- "
            f"known: {', '.join(s.get('name', '?') for s in registry)}")
    # Smallest coverage wins: a city-scale capture inside a regional one is
    # the more specific answer, and usually the newer and higher resolution.
    hits.sort(key=lambda s: (s["bbox"][2] - s["bbox"][0])
              * (s["bbox"][3] - s["bbox"][1]))
    out = dict(base)
    # KEY PRESENCE, NOT TRUTHINESS. This skipped None values, so a survey
    # saying "this product does not exist for me" -- pointcloud_bulk_url:
    # None for Kingston, whose 2025 LiDAR OpenTopography has not published --
    # silently inherited Queenstown's Otago store instead. It would have
    # fetched 2025 tile names from a 2021 dataset, got 404s, and fallen back
    # to the DSM: right answer, wrong reason, and a config that reads as
    # though Kingston uses Otago's point cloud. An explicit None is an
    # answer, and it is "none".
    out.update({k: hits[0][k] for k in KEYS if k in hits[0]})
    out["name"] = hits[0].get("name", "unnamed")
    return out


def survey_for_region(name):
    from src.region_build import area_bbox_wgs84
    return survey_for(area_bbox_wgs84(name), name)
