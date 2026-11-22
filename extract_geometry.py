"""
Extract exact architectural floor polygons and site/podium geometry
from 'One Highline floor plans.pdf'.
"""
import os
import json
import numpy as np
import pymupdf
from shapely.geometry import Polygon, MultiPoint, LineString, Point
from shapely.ops import unary_union
import matplotlib.pyplot as plt

PDF_PATH = os.environ.get("OHL_PDF_PATH", "One Highline floor plans.pdf")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extracted")
os.makedirs(OUT_DIR, exist_ok=True)

doc = pymupdf.open(PDF_PATH)

# Scale factors verified against RL elevations and scale bars:
# Site Plan (P001) & 1st-4th Floors (P008-P011): 1.1268 pt/ft
# Elevations (P002-P005): 0.9013 pt/ft
# Tower Floor Plans (P012-P295): 3.606 pt/ft
SCALE_SITE = 1.1268
SCALE_ELEV = 0.9013
SCALE_TOWER = 3.606

# Full floor pages identified earlier:
WEST_PAGES = [
    (5, 12), (6, 19), (7, 28), (8, 36), (9, 44), (10, 52),
    (11, 60), (12, 67), (14, 74), (15, 81), (16, 88), (17, 95),
    (18, 102), (19, 109), (20, 116), (21, 123), (22, 130), (23, 135),
    (24, 140), (25, 145), (26, 150), (27, 155), (28, 160), (29, 165),
    (30, 170), (31, 175), (32, 180), (33, 183), (34, 186), (35, 189), (36, 192),
]

EAST_PAGES = [
    (11, 194), (12, 202), (14, 209), (15, 217), (16, 225), (17, 233),
    (18, 241), (19, 249), (20, 257), (21, 265), (22, 273), (23, 279),
    (24, 285), (25, 291), (26, 294),
]

def extract_floor_hull_and_core(page_num):
    """
    On a tower floor plan page, extract:
    1) All filled grey (0.5,0.5,0.5) wall/pier polygons in the main drawing area (y < 600)
    2) The convex hull of all grey piers/walls (outer facade envelope!)
    3) The bounding box of the central elevator core (for sub-inch floor-to-floor registration)
    """
    page = doc[page_num - 1]
    pts = []
    core_pts = []
    for d in page.get_drawings():
        r = d["rect"]
        # Main floor plan is above y=610 (keyplan & title are below y=610)
        if r.y1 > 605 or r.y0 < 15:
            continue
        fill = d.get("fill")
        if fill is not None:
            # Medium grey walls & perimeter structural columns (~0.45..0.55)
            if all(0.35 <= c <= 0.65 for c in fill):
                for item in d["items"]:
                    op = item[0]
                    if op == "l":
                        pts.append((item[1].x, item[1].y))
                        pts.append((item[2].x, item[2].y))
                    elif op == "re":
                        rr = item[1]
                        pts.extend([(rr.x0, rr.y0), (rr.x1, rr.y0), (rr.x1, rr.y1), (rr.x0, rr.y1)])
                    elif op == "qu":
                        q = item[1]
                        pts.extend([(q.ul.x, q.ul.y), (q.ur.x, q.ur.y), (q.lr.x, q.lr.y), (q.ll.x, q.ll.y)])
                    elif op == "c":
                        pts.append((item[1].x, item[1].y))
                        pts.append((item[4].x, item[4].y))
    if not pts:
        return None
    mp = MultiPoint(pts)
    hull = mp.convex_hull
    # Convert PDF y-down to y-up feet
    coords = [(x / SCALE_TOWER, -y / SCALE_TOWER) for x, y in hull.exterior.coords]
    return Polygon(coords)

# Extract all West Tower floors and check their progression!
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

west_polys = {}
for fl, pno in WEST_PAGES:
    poly = extract_floor_hull_and_core(pno)
    west_polys[fl] = poly
    if fl in [5, 7, 10, 15, 20, 25, 30, 36]:
        x, y = poly.exterior.xy
        color = plt.cm.viridis((fl - 5) / 31.0)
        axes[0].plot(x, y, label=f"W{fl}", color=color, linewidth=1.5)
axes[0].set_aspect("equal")
axes[0].set_title("West Tower Floor Hulls (Raw Page Coords, ft)")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

east_polys = {}
for fl, pno in EAST_PAGES:
    poly = extract_floor_hull_and_core(pno)
    east_polys[fl] = poly
    if fl in [11, 14, 17, 20, 23, 26]:
        x, y = poly.exterior.xy
        color = plt.cm.plasma((fl - 11) / 15.0)
        axes[1].plot(x, y, label=f"E{fl}", color=color, linewidth=1.5)
axes[1].set_aspect("equal")
axes[1].set_title("East Tower Floor Hulls (Raw Page Coords, ft)")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/tower_hulls_raw.png", dpi=150)
print("Saved tower_hulls_raw.png")

for name, d in [("West", west_polys), ("East", east_polys)]:
    print(f"\n=== {name} Tower Hull Areas (sq ft) & Bounds ===")
    for fl, p in sorted(d.items()):
        b = p.bounds
        print(f"  Fl {fl:2d}: area={p.area:7.0f} sf, w={b[2]-b[0]:5.1f} ft, h={b[3]-b[1]:5.1f} ft, cx={(b[0]+b[2])/2:6.1f}, cy={(b[1]+b[3])/2:6.1f}")
