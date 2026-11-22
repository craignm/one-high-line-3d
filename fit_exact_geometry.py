"""
Fit exact, noise-free floor polygons for West Tower (W7-W37) and East Tower (E7-E27)
directly from the FFT-registered PDF floor plans, plus verify 3D projections against
ALL FOUR PDF Elevations (N, E, S, W) and the Site Plan!
"""
import os, json
import numpy as np
import pymupdf
from shapely.geometry import Polygon, box
import matplotlib.pyplot as plt

PDF_PATH = os.environ.get("OHL_PDF_PATH", "One Highline floor plans.pdf")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extracted")
doc = pymupdf.open(PDF_PATH)

with open(f"{OUT_DIR}/tower_floors.json") as f:
    tower_data = json.load(f)

# Helper: intersect half-planes a_i * x + b_i * y <= c_i inside a big bounding box
def halfplane_polygon(planes):
    """planes: list of (nx, ny, c) where nx*x + ny*y <= c, unit normal (nx, ny)."""
    poly = box(-500, -500, 500, 500)
    for nx, ny, c in planes:
        # Build a large half-plane polygon
        tx, ty = -ny, nx  # tangent
        p0 = np.array([nx * c, ny * c])
        R = 1000.0
        hp = Polygon([
            p0 - R * np.array([tx, ty]),
            p0 + R * np.array([tx, ty]),
            p0 + R * np.array([tx, ty]) - R * np.array([nx, ny]),
            p0 - R * np.array([tx, ty]) - R * np.array([nx, ny]),
        ])
        poly = poly.intersection(hp)
    return poly

# For each floor polygon in local coords, measure its support function h(theta) = max_{(x,y) in P} (x cos t + y sin t)
# at the 6 exact face normal directions!
# Let's first discover the exact 6 face normal angles of West Tower and East Tower from W15 and E18.
def edge_normals(coords):
    pts = np.asarray(coords)
    n = len(pts)
    res = []
    for i in range(n):
        p1, p2 = pts[i], pts[(i+1)%n]
        d = p2 - p1
        L = np.linalg.norm(d)
        if L > 8.0:
            # Polygon from shapely convex_hull is CCW, so outward normal is (dy, -dx)/L
            nx, ny = d[1]/L, -d[0]/L
            ang = np.degrees(np.arctan2(ny, nx))
            c = np.max(pts @ np.array([nx, ny]))
            res.append((round(ang, 2), round(L, 1), round(c, 2)))
    return res

print("W7  normals:", edge_normals(tower_data["west"]["7"]["coords"]))
print("W15 normals:", edge_normals(tower_data["west"]["15"]["coords"]))
print("W25 normals:", edge_normals(tower_data["west"]["25"]["coords"]))
print("W36 normals:", edge_normals(tower_data["west"]["36"]["coords"]))
print("E15 normals:", edge_normals(tower_data["east"]["15"]["coords"]))
print("E20 normals:", edge_normals(tower_data["east"]["20"]["coords"]))
print("E26 normals:", edge_normals(tower_data["east"]["26"]["coords"]))
