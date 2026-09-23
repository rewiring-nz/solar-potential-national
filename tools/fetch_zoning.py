"""District-plan zoning for every building, so a home is not called a business.

Homes were being misclassified as businesses; a better home/business
classification was needed. Is
there building data on this?"

What was being used: roof area over 400 m2. That is not a use signal at all,
and it cannot be -- a 450 m2 house in a suburb and a 450 m2 small business
look identical from above, so the rule mislabels large houses by
construction. Measured on the pilot: 132 of 1,066 buildings called business.

What was checked and rejected:
  * LINZ building outlines carry a `use` column. 99.5% of it reads "Unknown"
    across the district -- only 65 schools, 8 supermarkets and 5 hospitals
    are filled in. Worth using where present, useless as the main signal.
  * OpenStreetMap types about 10% of buildings here (38 retail, 32 house, 21
    hotel...). Real, free, and far too sparse to rely on.
  * "Out of scale with your neighbours" scores well until you notice it calls
    a large town-centre building residential because the shops around it are
    also large. Geometry cannot answer a question about use.

What this uses instead: the council's own district plan. Queenstown Lakes
publishes its combined plan zones as an ArcGIS service, and a zone is a
statement about what may legally be built there -- Queenstown Town Centre,
General Industrial, Low Density Residential. That is the actual answer to
the question, from the authority that decides it.

Writes `zone` and `zone_class` onto every building in the region outlines.
Other councils publish the same thing at different endpoints; ZONE_SOURCES
is where a new district gets added.

Usage: python tools/fetch_zoning.py [region ...]
"""

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# territorial authority -> zoning service. The layer must expose a coded
# `Zone` field; add a district by adding a line.
ZONE_SOURCES = {
    "Queenstown-Lakes District": {
        "url": "https://gis.qldc.govt.nz/server/rest/services/DistrictPlan/"
               "Combined_District_Plan/MapServer/54",
        "field": "Zone",
    },
}

# Which zones mean a business tariff. Residential zones, rural living and the
# lifestyle precincts are homes; town centres, industrial, commercial, the
# airport and the resort zones are not. Where a zone genuinely holds both
# (Business Mixed Use, Settlements) the building falls back to the geometry
# test rather than being asserted either way.
BUSINESS_ZONES = {
    "Airport", "Arrowtown Town Centre", "Queenstown Town Centre",
    "Wanaka Town Centre", "Local Shopping Centre", "General Industrial and Service",
    "Coneburn Industrial", "Three Parks Commercial", "Three Parks Business",
    "Active Sports and Recreation", "Community Purposes", "Civic Spaces",
    "Gibbston Resort", "Jacks Point Resort", "Millbrook Resort",
    "Waterfall Park Resort", "Hogans Gully Resort", "The Hills Resort",
    "Rural Visitor", "Te Pūtahi Ladies Mile",
}
HOME_ZONES = {
    "High Density Residential", "High Density Residential A",
    "Medium Density Residential", "Medium Density Residential A",
    "Large Lot Residential A", "Large Lot Residential B",
    "Arrowtown Residential Historic Management", "Rural Residential",
    "Rural Lifestyle", "Wakatipu Basin Lifestyle Precinct",
    "Wakatipu Basin Rural Amenity Zone", "Rural", "Gibbston Character",
}


def _get(url, params, tries=3):
    q = urllib.parse.urlencode(params)
    for i in range(tries):
        try:
            with urllib.request.urlopen(f"{url}?{q}", timeout=90) as r:
                return json.loads(r.read())
        except Exception as exc:
            if i == tries - 1:
                raise
            print(f"    retry after {exc!r}", flush=True)
            time.sleep(3)


def zone_lookup(src):
    meta = _get(src["url"], {"f": "json"})
    for f in meta.get("fields", []):
        if f["name"] == src["field"] and f.get("domain", {}).get("codedValues"):
            return {str(cv["code"]): cv["name"]
                    for cv in f["domain"]["codedValues"]}
    return {}


def fetch_zones(src, bbox):
    """Zone polygons intersecting bbox (minx,miny,maxx,maxy in WGS84)."""
    import geopandas as gpd
    codes = zone_lookup(src)
    out, offset = [], 0
    while True:
        d = _get(src["url"] + "/query", {
            "where": "1=1", "outFields": src["field"], "f": "geojson",
            "geometry": ",".join(str(v) for v in bbox),
            "geometryType": "esriGeometryEnvelope", "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "resultOffset": offset, "resultRecordCount": 1000,
        })
        feats = d.get("features", [])
        out.extend(feats)
        if len(feats) < 1000:
            break
        offset += 1000
        if offset > 20000:
            break
    if not out:
        return None
    g = gpd.GeoDataFrame.from_features(out, crs=4326)
    g["zone_name"] = g[src["field"]].astype(str).map(
        lambda c: codes.get(c, f"code {c}"))
    return g


def classify(name):
    if name in BUSINESS_ZONES:
        return "business"
    if name in HOME_ZONES:
        return "home"
    return "mixed"      # caller falls back to geometry


def main(argv):
    import geopandas as gpd
    from src.region_build import area_paths
    d = ROOT / "data/regions"
    regions = argv or sorted(p.name for p in d.iterdir() if p.is_dir())
    for region in regions:
        op = Path(area_paths(region)["outlines"])
        if not op.exists():
            continue
        g = gpd.read_file(op)
        ta = (g["territorial_authority"].dropna().iloc[0]
              if "territorial_authority" in g.columns and len(g.dropna(subset=["territorial_authority"])) else None)
        src = ZONE_SOURCES.get(ta)
        if not src:
            print(f"{region}: no zoning source for {ta!r}", flush=True)
            continue
        wgs = g.to_crs(4326)
        bbox = wgs.total_bounds
        zones = fetch_zones(src, bbox)
        if zones is None:
            print(f"{region}: no zones returned", flush=True)
            continue
        zones = zones.to_crs(g.crs)
        cen = gpd.GeoDataFrame(geometry=g.geometry.representative_point(),
                               crs=g.crs)
        j = gpd.sjoin(cen, zones[["zone_name", "geometry"]],
                      how="left", predicate="within")
        j = j[~j.index.duplicated(keep="first")]
        g["zone"] = j["zone_name"].values
        g["zone_class"] = [classify(z) if isinstance(z, str) else "unknown"
                           for z in g["zone"]]
        g.to_file(op, driver="GeoJSON")
        from collections import Counter
        c = Counter(g["zone_class"])
        print(f"{region}: {dict(c)}", flush=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]) or 0)
