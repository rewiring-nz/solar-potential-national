"""Score our faces against Josh's TRACED markup, not against a face count."""
import sys, json, warnings; sys.path.insert(0,"."); sys.path.insert(0,"src")
warnings.filterwarnings("ignore")
import numpy as np, geopandas as gpd, shapely.vectorized
from shapely.geometry import Polygon, LineString
from src.region_build import area_paths
from src.pointcloud_source import PointCloudSource
import src.roof_partition as rp

T = [x for x in json.load(open("data/roof_truth.json"))["roofs"]
     if x["building_id"] == 5371108][0]["traced_lines_nztm"]
band = Polygon(T["middle_section_corners"])
pc = PointCloudSource()
gdf = gpd.read_file(area_paths("pilot")["outlines"]).set_index("building_id", drop=False)
g = gdf.loc[5371108].geometry; mnx,mny,mxx,mxy = g.bounds
raw = pc.points_in_bbox(mnx-1,mny-1,mxx+1,mxy+1,building_only=True)
pts = raw[shapely.vectorized.contains(g, raw[:,0], raw[:,1])]
faces = rp.partition_roof(5371108, g.buffer(0), pts)

# the band's own long axis separates the NW section from the SE section
c = np.array(band.centroid.coords[0])
m2 = np.array(T["middle_section_corners"][1]); m4 = np.array(T["middle_section_corners"][3])
axis = m2 - m4; axis = axis / np.linalg.norm(axis)      # runs across the roof
normal = np.array([-axis[1], axis[0]])                   # runs ALONG the roof

print(f"faces: {len(faces)}  (Josh: 8)")
print(f"\nJosh's structure: 2 hips + 4 slope quarters + 2 middle-section faces")
print(f"decisive test -- no face may span the middle section:\n")
straddlers = 0
for f in sorted(faces, key=lambda x:-x["area_m2"]):
    gg = f["geometry"]
    s = (np.array(gg.exterior.coords) - c) @ normal
    nw = float((s > 0.5).sum()); se = float((s < -0.5).sum())
    crosses = gg.intersection(band).area
    flag = ""
    if nw > 0 and se > 0 and crosses > 0.5:
        flag = "  <-- SPANS the middle section"
        straddlers += 1
    print(f"  {gg.area:6.1f} m2  aspect {f['aspect_deg']:6.1f}  "
          f"overlap with band {crosses:5.1f} m2{flag}")
inband = [f for f in faces if f["geometry"].intersection(band).area > 0.4 * band.area]
print(f"\nfaces covering the middle section: {len(inband)}  (Josh: 2)")
print(f"faces spanning across it:          {straddlers}  (Josh: 0)")
