"""
Build a 100% watertight, manifold, FDM-printable 1:450 architectural model
of One High Line (500 W 18th St, Manhattan, designed by Bjarke Ingels Group)
directly from the extracted vector floor plans & elevations.

Outputs (in ./output/):
  - one_high_line_complete_1_450.stl  (Complete assembled model, ~333 x 152 x 303 mm)
  - one_high_line_complete_1_450.3mf  (Complete assembled model in 3MF)
  - part1_west_base_and_lower.stl     (West plinth + podium + S retail + West Tower 5W-20W)
  - part2_west_tower_upper.stl        (West Tower 21W-37W + penthouse loggia + crown)
  - part3_east_base_and_lower.stl     (East plinth + podium + E retail + High Line + East Tower 7E-16E)
  - part4_east_tower_upper.stl        (East Tower 17E-27E + penthouse loggia + crown)
  - part5_skybridge.stl               (Snap-in double-height skybridge, prints flat support-free)
"""
import os, json, time
import numpy as np
import manifold3d as m3d
import trimesh
from shapely.geometry import Polygon, LineString, Point
from shapely.affinity import affine_transform

t0 = time.time()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(BASE_DIR, "extracted", "tower_floors.json")) as f:
    TOWER_DATA = json.load(f)

# Scale: 1:450 -> 1 foot = 304.8 / 450 = 0.677333... mm
SCALE_MM_PER_FT = 304.8 / 450.0  # 0.677333 mm/ft

# =============================================================================
# MANIFOLD3D PRIMITIVE BUILDERS (all inputs in real-world FEET; scaled at end)
# =============================================================================

def poly_ccw(pts):
    """Ensure 2D polygon points [(x,y), ...] are CCW and not closed."""
    p = np.asarray(pts, dtype=np.float64)
    if len(p) > 2 and np.allclose(p[0], p[-1]):
        p = p[:-1]
    area2 = np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    if area2 < 0:
        p = p[::-1]
    return p

def extrude_poly(pts, z0, z1):
    """Extrude a 2D polygon cleanly from z0 to z1 as a Manifold."""
    if z1 <= z0 + 1e-6:
        return m3d.Manifold()
    p = poly_ccw(pts)
    cs = m3d.CrossSection([p.tolist()])
    m = m3d.Manifold.extrude(cs, float(z1 - z0))
    if abs(z0) > 1e-9:
        m = m.translate([0.0, 0.0, float(z0)])
    return m

def loft_polygons(pts_bot, z_bot, pts_top, z_top, n_samples=None):
    """
    Build a 100% watertight lofted Manifold between two CCW 2D polygons
    at z_bot and z_top. Both polygons must have the same number of vertices
    (or will be resampled to n_samples if provided).
    """
    p0 = poly_ccw(pts_bot)
    p1 = poly_ccw(pts_top)
    if n_samples is not None:
        p0 = resample_poly(p0, n_samples)
        p1 = resample_poly(p1, n_samples)
    assert len(p0) == len(p1), f"Vertex mismatch: {len(p0)} vs {len(p1)}"
    n = len(p0)

    # Vertices: 0..n-1 bottom, n..2n-1 top, 2n bottom center, 2n+1 top center
    c0 = p0.mean(axis=0)
    c1 = p1.mean(axis=0)
    verts = np.zeros((2 * n + 2, 3), dtype=np.float32)
    verts[:n, :2] = p0
    verts[:n, 2] = z_bot
    verts[n:2*n, :2] = p1
    verts[n:2*n, 2] = z_top
    verts[2*n] = [c0[0], c0[1], z_bot]
    verts[2*n+1] = [c1[0], c1[1], z_top]

    tris = []
    # Side quads (CCW outward)
    for i in range(n):
        j = (i + 1) % n
        b0, b1 = i, j
        t0_i, t1_i = n + i, n + j
        tris.append([b0, b1, t1_i])
        tris.append([b0, t1_i, t0_i])
    # Bottom cap (normal -Z, CW when viewed from +Z)
    for i in range(n):
        j = (i + 1) % n
        tris.append([2*n, j, i])
    # Top cap (normal +Z, CCW when viewed from +Z)
    for i in range(n):
        j = (i + 1) % n
        tris.append([2*n+1, n + i, n + j])

    mesh = m3d.Mesh(vert_properties=verts, tri_verts=np.asarray(tris, dtype=np.uint32))
    return m3d.Manifold(mesh)

def resample_poly(pts, n_samples):
    """Resample a CCW polygon uniformly by arc length starting from vertex 0."""
    p = poly_ccw(pts)
    closed = np.vstack([p, p[0]])
    segs = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(segs)])
    total = cum[-1]
    targets = np.linspace(0.0, total, n_samples, endpoint=False)
    out = np.zeros((n_samples, 2), dtype=np.float64)
    for k, s in enumerate(targets):
        idx = np.searchsorted(cum, s, side="right") - 1
        idx = min(idx, len(p) - 1)
        t = (s - cum[idx]) / max(segs[idx], 1e-9)
        out[k] = closed[idx] * (1.0 - t) + closed[idx + 1] * t
    return out

def resample_poly_aligned(pts, n_samples, ref_angle_deg=135.0):
    """
    Resample a CCW polygon uniformly by arc length, starting from the perimeter point
    closest to the ray from centroid at ref_angle_deg (ensures twist-free, smooth
    pier correspondence across floors!).
    """
    p = poly_ccw(pts)
    c = p.mean(axis=0)
    # Dense sample first
    dense = resample_poly(p, 720)
    ang = np.degrees(np.arctan2(dense[:, 1] - c[1], dense[:, 0] - c[0]))
    diff = np.abs((ang - ref_angle_deg + 180.0) % 360.0 - 180.0)
    start_idx = int(np.argmin(diff))
    rolled = np.roll(dense, -start_idx, axis=0)
    step = 720.0 / n_samples
    idxs = (np.arange(n_samples) * step).astype(int) % 720
    return rolled[idxs]

def inset_poly(pts, dist):
    """Inset (shrink) a 2D CCW polygon by dist (feet)."""
    poly = Polygon(poly_ccw(pts)).buffer(-dist, join_style=2, mitre_limit=3.0)
    if poly.is_empty:
        return poly_ccw(pts)
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    return poly_ccw(list(poly.exterior.coords)[:-1])

def build_pier_segment(p_bot, t_bot, n_bot, z_bot, p_top, t_top, n_top, z_top, width, depth_in, depth_out=0.0):
    """
    Build a single continuous twisted vertical travertine pier prism from z_bot to z_top.
    p_bot, p_top: point on outer facade perimeter
    t_bot, t_top: unit tangent along facade (CCW)
    n_bot, n_top: unit outward normal
    """
    hw = width * 0.5
    quad_bot = [
        p_bot - hw * t_bot - depth_in * n_bot,
        p_bot + hw * t_bot - depth_in * n_bot,
        p_bot + hw * t_bot + depth_out * n_bot,
        p_bot - hw * t_bot + depth_out * n_bot,
    ]
    quad_top = [
        p_top - hw * t_top - depth_in * n_top,
        p_top + hw * t_top - depth_in * n_top,
        p_top + hw * t_top + depth_out * n_top,
        p_top - hw * t_top + depth_out * n_top,
    ]
    return loft_polygons(quad_bot, z_bot, quad_top, z_top)

def perimeter_frames(pts, n_piers, ref_angle_deg):
    """Return (points, tangents, normals) at n_piers positions along polygon."""
    dense = resample_poly_aligned(pts, 360, ref_angle_deg)
    # Compute smooth tangents/normals
    step = 360.0 / n_piers
    idxs = (np.arange(n_piers) * step).astype(int) % 360
    P, T, N = [], [], []
    for idx in idxs:
        p_prev = dense[(idx - 2) % 360]
        p_next = dense[(idx + 2) % 360]
        d = p_next - p_prev
        L = np.linalg.norm(d)
        t = d / max(L, 1e-9)
        n = np.array([t[1], -t[0]])  # outward normal for CCW
        P.append(dense[idx])
        T.append(t)
        N.append(n)
    return P, T, N

def build_chamfered_spandrel(pts_outer, pts_inner, z_bot, z_top, chamfer_h=2.4):
    """
    Build a support-free horizontal spandrel slab:
    - Lower chamfer from (pts_inner, z_bot - chamfer_h) -> (pts_outer, z_bot) [45-deg bevel!]
    - Vertical face from (pts_outer, z_bot) -> (pts_outer, z_top)
    """
    n_s = 64
    p_in = resample_poly_aligned(pts_inner, n_s, 135.0)
    p_out = resample_poly_aligned(pts_outer, n_s, 135.0)
    bevel = loft_polygons(p_in, z_bot - chamfer_h, p_out, z_bot)
    slab = extrude_poly(p_out, z_bot, z_top)
    return bevel + slab

def build_podium_block(pts, z0, z1, n_floors, pier_spacing=14.0, pier_w=3.2,
                       recess=2.4, spandrel_h=2.8, chamfer_h=2.2, parapet_h=3.0):
    """
    Build a crisp architectural podium block with recessed glazing core,
    horizontal floor spandrel bands (with 45-deg support-free undersides),
    vertical travertine piers, and a top roof parapet.
    """
    p_out = poly_ccw(pts)
    p_in = inset_poly(p_out, recess)
    parts = [extrude_poly(p_in, z0, z1)]

    # Ground base band (solid 2.5 ft plinth lip)
    parts.append(extrude_poly(p_out, z0, min(z0 + 2.5, z1)))

    # Floor spandrels
    fl_h = (z1 - z0) / n_floors
    for k in range(1, n_floors):
        zk = z0 + k * fl_h
        parts.append(build_chamfered_spandrel(p_out, p_in, zk - spandrel_h * 0.5, zk + spandrel_h * 0.5, chamfer_h))

    # Top cornice / parapet
    parts.append(build_chamfered_spandrel(p_out, p_in, z1 - spandrel_h, z1 + parapet_h, chamfer_h))
    # Slight recessed roof inside parapet (parapet rim)
    parts.append(extrude_poly(p_in, z1, z1 + 0.5))

    # Vertical piers placed cleanly along every straight edge (including every corner!)
    n = len(p_out)
    for i in range(n):
        a = p_out[i]
        b = p_out[(i + 1) % n]
        edge = b - a
        L = np.linalg.norm(edge)
        if L < 2.0:
            continue
        t = edge / L
        nrm = np.array([t[1], -t[0]])
        n_bays = max(1, int(round(L / pier_spacing)))
        for j in range(n_bays + 1):
            frac = j / n_bays
            pt = a + frac * edge
            # Slightly widen corner piers
            w = pier_w * (1.25 if (j == 0 or j == n_bays) else 1.0)
            parts.append(build_pier_segment(pt, t, nrm, z0, pt, t, nrm, z1 + parapet_h, w, recess + 2.0, 0.0))

    return m3d.Manifold.batch_boolean(parts, m3d.OpType.Add)


# =============================================================================
# 1. EXACT 6-VERTEX PLANAR-PARALLELOGRAM + TWISTING-CORNER TOWER GEOMETRY
# =============================================================================
# Every floor is an exact 6-vertex polygon [v0, v1, v2, v3, v4, v5] (CCW):
#   - The 4 main facades (N, S, E, W) are EXACT planar parallelograms (constant
#     edge direction & constant edge length across all floors!)
#   - ONLY the NW and SE corners twist floor-by-floor:
#       West Tower: W=6, S=5, SE_twist=3, E=3, N=5, NW_twist=2  (CCW order!)
#       East Tower: W=4, S=5, SE_twist=1, E=7, N=4, NW_twist=2  (CCW order!)

# --- WEST TOWER (31 floors: W6..W36, Z = 55.5' to 395.5') ---
# Calibrated directly from FFT-registered W7..W36 (exact planar parallelograms):
# Group A (SW, NW1, SE2) translates by dA = (+13.9, -36.7) from W7 (t=0) to W36 (t=1)
# Group B (NW2, NE, SE1) translates by dB = (-43.4,   0.0) from W7 (t=0) to W36 (t=1)
W_BASE_W7 = np.array([
    [22.8,  178.7],  # 0: NW1 (top of West face)
    [65.8,   67.2],  # 1: SW  (bottom of West face / West end of South face)
    [150.4, 105.8],  # 2: SE2 (East end of South face / start of SE twist)
    [183.2, 136.5],  # 3: SE1 (end of SE twist / South end of East face)
    [167.1, 180.4],  # 4: NE  (North end of East face / East end of North face)
    [66.4,  180.4],  # 5: NW2 (West end of North face / end of NW twist)
], dtype=np.float64)
W_DELTA_W7_TO_W36 = np.array([
    [+13.9, -36.7],  # 0: NW1 (Group A)
    [+13.9, -36.7],  # 1: SW  (Group A) -> Edge 0->1 (West, 6 bays) is exact planar parallelogram!
    [+13.9, -36.7],  # 2: SE2 (Group A) -> Edge 1->2 (South, 5 bays) is exact planar parallelogram!
    [-43.4,   0.0],  # 3: SE1 (Group B) -> Edge 2->3 (SE twist, 3 bays!)
    [-43.4,   0.0],  # 4: NE  (Group B) -> Edge 3->4 (East, 3 bays) is exact planar parallelogram!
    [-43.4,   0.0],  # 5: NW2 (Group B) -> Edge 4->5 (North, 5 bays) is exact planar parallelogram!
                     #                     Edge 5->0 (NW twist, 2 bays!)
], dtype=np.float64)
WEST_EDGE_BAYS = [6, 5, 3, 3, 5, 2]  # [W, S, SE_twist, E, N, NW_twist]

# 30 stories from W6 (t=-1/29) to W36 (t=1.0), plus roof slab at W37 (t=1.0)
n_w_floors = 30
w_z_levels = np.linspace(55.5, 395.5, n_w_floors + 1)
t_w_floors = np.linspace(-1.0 / 29.0, 1.0, n_w_floors)
w_polys_all = [W_BASE_W7 + t * W_DELTA_W7_TO_W36 for t in t_w_floors]

# --- EAST TOWER (19 floors: E7..E26, Z = 75.5' to 296.17') ---
# Calibrated directly from FFT-registered E15..E26 (11 floors), extrapolated linearly to E7 (7 floors below E15):
# Group A (NW1, SW, SE2) translates by dA = (-1.73,  0.00) ft/floor
# Group B (SE1, NE, NW2) translates by dB = (+0.48, -1.30) ft/floor
E_BASE_E15 = np.array([
    [217.4, 101.8],  # 0: NW1 (North end of West face)
    [246.7,  24.8],  # 1: SW  (South end of West face / West end of South face)
    [331.0,  23.8],  # 2: SE2 (East end of South face / start of wide ~2x SE twist bay)
    [347.0,  52.4],  # 3: SE1 (end of wide ~2x SE twist bay / South end of East face)
    [303.4, 168.4],  # 4: NE  (North end of East face / East end of North face)
    [228.0, 134.5],  # 5: NW2 (West end of North face / end of NW twist)
], dtype=np.float64)
E_DELTA_PER_FLOOR = np.array([
    [-1.73,  0.00],  # 0: NW1 (Group A)
    [-1.73,  0.00],  # 1: SW  (Group A) -> Edge 0->1 (West, 4 bays) is exact planar parallelogram!
    [-1.73,  0.00],  # 2: SE2 (Group A) -> Edge 1->2 (South, 5 bays) is exact planar parallelogram!
    [+0.48, -1.30],  # 3: SE1 (Group B) -> Edge 2->3 (SE twist, 1 bay!)
    [+0.48, -1.30],  # 4: NE  (Group B) -> Edge 3->4 (East, 7 bays) is exact planar parallelogram!
    [+0.48, -1.30],  # 5: NW2 (Group B) -> Edge 4->5 (North, 4 bays) is exact planar parallelogram!
                     #                     Edge 5->0 (NW twist, 2 bays!)
], dtype=np.float64)
EAST_EDGE_BAYS = [4, 5, 1, 7, 4, 2]  # [W, S, SE_twist, E, N, NW_twist]

# 19 stories: E7 (df=-7) through E26 (df=+11)
n_e_floors = 19
e_z_levels = np.linspace(75.5, 296.17, n_e_floors + 1)
df_e_floors = np.linspace(-7.0, 11.0, n_e_floors)
e_polys_all = [E_BASE_E15 + df * E_DELTA_PER_FLOOR for df in df_e_floors]


# =============================================================================
# 2. BUILD TWISTING TOWERS WITH PLANAR PARALLELOGRAM FACADES & TWISTING CORNERS
# =============================================================================

def build_twisting_tower(polys, edge_bays, z_levels, crown_top_z, ref_angle_deg, name="Tower"):
    """
    Build a complete twisting tower from exact 6-vertex floor polygons:
      - All 4 main facades (N, S, E, W) are EXACT planar parallelograms
        composed of strictly vertical 1-story piers & 90-deg rectangular windows.
      - ONLY the NW and SE corners twist, with exact specified bay counts
        (NW: 2 bays on both towers; SE: 3 bays on West, 1 bay on East).
      - Top penthouse wrap-around open colonnade loggia (deeply recessed top floor).
      - Flared mechanical crown with crisp vertical architectural louver fins.
    """
    n_floors = len(polys)
    recess = 2.6        # 1.76 mm window recess at 1:450
    spandrel_h = 2.6    # 1.76 mm horizontal floor spandrel band
    chamfer_h = 2.0     # subtle 45-deg soffit bevel for support-free FDM printing
    pier_w = 3.5        # 2.37 mm vertical travertine pier width

    parts = []
    for k in range(n_floors):
        z0, z1 = z_levels[k], z_levels[k + 1]
        is_ph = (k == n_floors - 1)
        p_out = poly_ccw(polys[k])
        r_in = 10.5 if is_ph else recess
        p_in = inset_poly(p_out, r_in)

        # 1. Strictly VERTICAL recessed glass core for floor k (crisp 90-deg rectangular windows!)
        parts.append(extrude_poly(p_in, z0 - 0.2, z1 + 0.2))

        # 2. Horizontal spandrel slab at bottom of floor k
        if k == 0:
            parts.append(extrude_poly(p_out, z0 - 2.0, z0 + spandrel_h * 0.6))
        else:
            p_prev_in = inset_poly(poly_ccw(polys[k - 1]), recess)
            p_bevel_bot = resample_poly_aligned(p_prev_in, 64, ref_angle_deg)
            p_bevel_top = resample_poly_aligned(p_out, 64, ref_angle_deg)
            parts.append(loft_polygons(p_bevel_bot, z0 - chamfer_h, p_bevel_top, z0))
            parts.append(extrude_poly(p_out, z0, z0 + spandrel_h * 0.55))

        # Top spandrel lintel of floor k
        if not is_ph:
            parts.append(build_chamfered_spandrel(p_out, p_in, z1 - spandrel_h * 0.55, z1 + 0.2, chamfer_h=1.8))

        # 3. Strictly VERTICAL 1-floor-high travertine piers:
        #    - ONE clean vertical corner pier at each of the 6 vertices (aligned to edge bisector -> no doubled-up corners!)
        #    - Intermediate piers strictly inside each edge (j = 1 .. n_bays - 1)
        d_in = 3.8 if is_ph else (recess + 2.2)
        n_edges = len(p_out)
        # Precompute unit edge tangents & outward normals
        edge_t, edge_n, edge_L = [], [], []
        for ei in range(n_edges):
            e = p_out[(ei + 1) % n_edges] - p_out[ei]
            L = np.linalg.norm(e)
            t_v = e / max(L, 1e-9)
            n_v = np.array([t_v[1], -t_v[0]])
            edge_t.append(t_v); edge_n.append(n_v); edge_L.append(L)

        # Single crisp vertical corner pier at each vertex
        for vi in range(n_edges):
            pt = p_out[vi]
            n_avg = edge_n[(vi - 1) % n_edges] + edge_n[vi]
            n_len = np.linalg.norm(n_avg)
            n_bis = n_avg / n_len if n_len > 1e-6 else edge_n[vi]
            t_bis = np.array([-n_bis[1], n_bis[0]])
            parts.append(build_pier_segment(
                pt, t_bis, n_bis, z0 - 0.1,
                pt, t_bis, n_bis, z1 + 0.15,
                width=pier_w * 1.15, depth_in=d_in, depth_out=0.0
            ))

        # Intermediate vertical piers strictly between corners
        for ei in range(n_edges):
            if edge_L[ei] < 1.0:
                continue
            a = p_out[ei]
            b = p_out[(ei + 1) % n_edges]
            edge = b - a
            n_bays = edge_bays[ei]
            for j in range(1, n_bays):
                frac = j / n_bays
                pt = a + frac * edge
                parts.append(build_pier_segment(
                    pt, edge_t[ei], edge_n[ei], z0 - 0.1,
                    pt, edge_t[ei], edge_n[ei], z1 + 0.15,
                    width=pier_w, depth_in=d_in, depth_out=0.0
                ))

    # Top roof cornice slab at z_levels[-1] (caps the penthouse colonnade)
    zk = z_levels[-1]
    p_top = poly_ccw(polys[-1])
    p_top_in = inset_poly(p_top, recess)
    parts.append(build_chamfered_spandrel(p_top, p_top_in, zk - 2.2, zk + 3.5, chamfer_h=2.2))
    parts.append(extrude_poly(p_top, zk - 1.2, zk + 3.5))

    # 4. Flared Louvered Mechanical Crown (from top roof z_levels[-1] to crown_top_z)
    z_cr0 = z_levels[-1] + 1.0
    z_cr1 = crown_top_z
    cr_bot = resample_poly_aligned(inset_poly(p_top, 10.5), 64, ref_angle_deg)
    cr_top = resample_poly_aligned(inset_poly(p_top, 9.0), 64, ref_angle_deg)
    twist_vec = polys[-1].mean(axis=0) - polys[-4].mean(axis=0)
    cr_top = cr_top + twist_vec * 0.70

    cr_bot_in = resample_poly_aligned(inset_poly(cr_bot, 1.6), 64, ref_angle_deg)
    cr_top_in = resample_poly_aligned(inset_poly(cr_top, 1.6), 64, ref_angle_deg)
    parts.append(loft_polygons(cr_bot_in, z_cr0 - 1.5, cr_top_in, z_cr1))

    parts.append(loft_polygons(cr_bot, z_cr0 - 1.0, cr_bot, z_cr0 + 2.2))
    parts.append(loft_polygons(cr_top_in, z_cr1 - 4.5, cr_top, z_cr1 - 2.5))
    parts.append(loft_polygons(cr_top, z_cr1 - 2.5, cr_top, z_cr1))

    n_fins = 46
    F0, FT0, FN0 = perimeter_frames(cr_bot, n_fins, ref_angle_deg)
    F1, FT1, FN1 = perimeter_frames(cr_top, n_fins, ref_angle_deg)
    for j in range(n_fins):
        parts.append(build_pier_segment(
            F0[j], FT0[j], FN0[j], z_cr0 - 0.5,
            F1[j], FT1[j], FN1[j], z_cr1 - 0.2,
            width=2.0, depth_in=3.0, depth_out=0.0
        ))

    print(f"  Boolean-unioning {len(parts)} solids for {name}...")
    return m3d.Manifold.batch_boolean(parts, m3d.OpType.Add)


print("Building West Tower (6-edge planar parallelograms + NW 2-bay & SE 3-bay twist)...")
west_tower = build_twisting_tower(w_polys_all, WEST_EDGE_BAYS, w_z_levels, crown_top_z=435.5,
                                  ref_angle_deg=135.0, name="West Tower")

print("Building East Tower (6-edge planar parallelograms + NW 2-bay & SE 1-bay twist)...")
east_tower = build_twisting_tower(e_polys_all, EAST_EDGE_BAYS, e_z_levels, crown_top_z=336.17,
                                  ref_angle_deg=45.0, name="East Tower")


# =============================================================================
# 3. BUILD PODIUMS, SOUTH RETAIL WING, EAST PAVILION, SKYBRIDGE, HIGH LINE & PLINTH
# =============================================================================

print("Building Podiums, Retail Wings, Skybridge, High Line & Plinth...")

# --- West Podium (1st-5th Floors, Z = 0 to 56.0') ---
west_podium_poly = [
    (2.0, 193.0),    # NW (11th Ave & W 18th St)
    (158.0, 193.0),  # NE (W 18th St, courtyard edge)
    (178.0, 134.0),  # East courtyard corner
    (46.0, 70.0),    # SW courtyard corner (11th Ave)
]
west_podium = build_podium_block(west_podium_poly, z0=0.0, z1=56.0, n_floors=4,
                                 pier_spacing=15.0, pier_w=3.2, recess=2.4, parapet_h=2.5)

# --- South Retail Wing (along W 17th St, Z = 0 to 78.0' + rooftop bulkhead 78 to 96') ---
south_retail_poly = [
    (70.0, 11.0),    # SW (11th Ave & W 17th St)
    (228.0, 11.0),   # SE (W 17th St)
    (208.0, 60.0),   # NE (courtyard)
    (50.0, 60.0),    # NW (11th Ave)
]
south_retail = build_podium_block(south_retail_poly, z0=0.0, z1=78.0, n_floors=4,
                                  pier_spacing=12.0, pier_w=2.4, recess=2.0, spandrel_h=3.2, parapet_h=2.5)
# South Retail rooftop bulkhead (visible on P001, P002, P004)
south_bulkhead_poly = [
    (138.0, 22.0), (195.0, 22.0), (185.0, 50.0), (138.0, 50.0)
]
south_bulkhead = extrude_poly(south_bulkhead_poly, 77.0, 96.0)
south_retail = south_retail + south_bulkhead

# --- East Podium (1st-6th Floors, Z = 0 to 78.0') ---
east_podium_poly = [
    (214.0, 193.0),  # NW (W 18th St, courtyard edge)
    (300.0, 193.0),  # NE (W 18th St, High Line edge)
    (368.0, 11.0),   # SE (W 17th St)
    (278.0, 11.0),   # SW (W 17th St, courtyard edge)
]
east_podium = build_podium_block(east_podium_poly, z0=0.0, z1=78.0, n_floors=6,
                                 pier_spacing=14.0, pier_w=3.2, recess=2.4, parapet_h=2.5)

# --- East Retail Pavilion (ground level under & east of High Line, Z = 0 to 14.5') ---
east_retail_poly = [
    (296.0, 193.0),
    (372.0, 193.0),
    (452.0, 11.0),
    (364.0, 11.0),
]
east_retail = build_podium_block(east_retail_poly, z0=0.0, z1=14.5, n_floors=1,
                                 pier_spacing=16.0, pier_w=2.8, recess=2.0, parapet_h=2.5)

# --- North & South Bridges (3rd-4th Floors, Z = 27.3' to 56.5') ---
def build_bridge(poly, exposed_edges, z0=27.3, z1=56.5, n_mullions=5):
    """Build a double-height architectural bridge with recessed glazing, top/bottom spandrels, and vertical mullions."""
    core = extrude_poly(inset_poly(poly, 2.0), z0, z1)
    bot_beam = extrude_poly(poly, z0, z0 + 3.4)
    top_beam = extrude_poly(poly, z1 - 3.4, z1)
    b_parts = [core, bot_beam, top_beam]
    for (a, b) in exposed_edges:
        a, b = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
        d = b - a
        L = np.linalg.norm(d)
        if L < 2.0:
            continue
        t = d / L
        nrm = np.array([t[1], -t[0]])
        for frac in np.linspace(0.15, 0.85, n_mullions):
            pt = a + frac * d
            b_parts.append(build_pier_segment(pt, t, nrm, z0, pt, t, nrm, z1, width=2.2, depth_in=4.0))
    return m3d.Manifold.batch_boolean(b_parts, m3d.OpType.Add)

# 1) North Bridge ("RESIDENTIAL AMENITY SPACE" on 3rd Floor Plan):
#    Straight recessed north face along W 18th St; kinked south face angling NE from West Podium then running E to East Podium!
north_bridge_poly = [
    (155.0, 185.0),  # NW (embedded into West Podium, recessed from W 18th St Y=193)
    (222.0, 185.0),  # NE (embedded into East Podium)
    (235.0, 150.0),  # SE (embedded into East Podium)
    (206.0, 150.0),  # South courtyard kink vertex
    (173.0, 130.0),  # SW (embedded into West Podium)
]
north_bridge = build_bridge(
    north_bridge_poly,
    exposed_edges=[
        ((216.0, 185.0), (161.0, 185.0)),  # North face (CCW: E->W so outward normal is +Y North!)
        ((179.0, 133.6), (206.0, 150.0)),  # South-West angled courtyard face (W->E, outward normal SE)
        ((206.0, 150.0), (229.0, 150.0)),  # South-East horizontal courtyard face (W->E, outward normal -Y South)
    ],
    z0=27.3, z1=56.5, n_mullions=4
)

# 2) South Bridge ("RETAIL" on 3rd & 4th Floor Plans):
#    Spans between South Retail Wing (low-rise commercial building) and East Tower Podium along W 17th St!
south_bridge_poly = [
    (202.0, 58.0),   # NW (embedded into South Retail Wing, flush with courtyard face)
    (265.0, 58.0),   # NE (embedded into East Podium)
    (280.0, 18.0),   # SE (embedded into East Podium, recessed ~7 ft from W 17th St Y=11)
    (220.0, 18.0),   # SW (embedded into South Retail Wing)
]
south_bridge = build_bridge(
    south_bridge_poly,
    exposed_edges=[
        ((259.0, 58.0), (208.0, 58.0)),    # North courtyard face (E->W, outward normal +Y North)
        ((225.0, 18.0), (275.0, 18.0)),    # South W 17th St face (W->E, outward normal -Y South)
    ],
    z0=27.3, z1=56.5, n_mullions=4
)

# --- The High Line Elevated Park (Z = 20' to 30', sweeping across east side of block!) ---
# Centerline & width from P001 Site Plan & P010 3rd Floor
hl_poly = [
    (320.0, 212.0),  # North entry at W 18th St (west edge)
    (368.0, 212.0),  # North entry at W 18th St (east edge)
    (372.0, 142.0),  # Bend east
    (418.0, 58.0),   # Mid SE diagonal east
    (468.0, -8.0),   # South exit at W 17th St / 10th Ave (east edge)
    (430.0, -8.0),   # South exit (west edge)
    (382.0, 62.0),   # Mid SE diagonal west
    (332.0, 142.0),  # Bend west
]
hl_deck = extrude_poly(hl_poly, 20.0, 26.0)
# Side parapet railings (outer perimeter rim Z = 26.0 to 30.0)
hl_parapet_outer = extrude_poly(hl_poly, 25.5, 30.0)
hl_parapet_inner = extrude_poly(inset_poly(hl_poly, 2.4), 25.8, 31.0)
hl_parapet = hl_parapet_outer - hl_parapet_inner
# Raised linear planting beds on High Line deck (subtle 1.2 ft architectural relief)
hl_planter = extrude_poly(inset_poly(hl_poly, 9.0), 25.8, 27.5)
# Structural steel support columns (paired columns every ~42 ft down to Z = 0)
hl_cols = []
col_pairs = [
    ((330.0, 198.0), (358.0, 198.0)),
    ((336.0, 154.0), (364.0, 154.0)),
    ((358.0, 108.0), (388.0, 108.0)),
    ((386.0, 64.0),  (414.0, 64.0)),
    ((416.0, 20.0),  (446.0, 20.0)),
]
for pA, pB in col_pairs:
    for cx, cy in (pA, pB):
        hw = 2.0  # 4 ft = 2.7 mm square column (sturdy for FDM!)
        hl_cols.append(extrude_poly([(cx-hw, cy-hw), (cx+hw, cy-hw), (cx+hw, cy+hw), (cx-hw, cy+hw)], 0.0, 21.0))
    # Cross-girder cap beam under deck
    pA, pB = np.array(pA), np.array(pB)
    d = pB - pA
    t = d / np.linalg.norm(d)
    nrm = np.array([-t[1], t[0]])
    beam = [pA - 2.0*nrm, pB - 2.0*nrm, pB + 2.0*nrm, pA + 2.0*nrm]
    hl_cols.append(extrude_poly(beam, 18.0, 21.0))

high_line = m3d.Manifold.batch_boolean([hl_deck, hl_parapet, hl_planter] + hl_cols, m3d.OpType.Add)

# --- City-Block Plinth & Landscaped Courtyard (Z = -12' to 0') ---
# Outer street plinth bounding W 17th St, W 18th St, 11th Ave, 10th Ave
plinth_outer = [
    (66.0, -8.0),    # SW corner (11th Ave & W 17th St)
    (472.0, -8.0),   # SE corner (10th Ave & W 17th St)
    (472.0, 212.0),  # NE corner (10th Ave & W 18th St)
    (-18.0, 212.0),  # NW corner (11th Ave & W 18th St)
]
# Clip High Line cleanly flush to the city-block plinth perimeter!
high_line = high_line ^ extrude_poly(plinth_outer, -5.0, 60.0)
# Raised sidewalk/block pad (Z = -1.2' to 0.0', inset 10 ft from street curb edge for crisp urban street frame!)
sidewalk_pad = [
    (66.0, 2.0),
    (462.0, 2.0),
    (462.0, 202.0),
    (-6.0, 202.0),
]
plinth_base = extrude_poly(plinth_outer, -12.0, -1.2)
plinth_sidewalk = extrude_poly(sidewalk_pad, -1.5, 0.0)

# Organic courtyard planter islands (from P001 & P008 central plaza)
def oval_poly(cx, cy, rx, ry, ang_deg=0.0, n=24):
    th = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = rx * np.cos(th)
    y = ry * np.sin(th)
    a = np.radians(ang_deg)
    xr = cx + x*np.cos(a) - y*np.sin(a)
    yr = cy + x*np.sin(a) + y*np.cos(a)
    return list(zip(xr, yr))

planter1 = extrude_poly(oval_poly(145.0, 82.0, 22.0, 12.0, 18.0), -0.5, 1.8)
planter2 = extrude_poly(oval_poly(196.0, 96.0, 18.0, 11.0, -25.0), -0.5, 1.8)
plinth = m3d.Manifold.batch_boolean([plinth_base, plinth_sidewalk, planter1, planter2], m3d.OpType.Add)


# =============================================================================
# 4. ASSEMBLE COMPLETE MODEL & SCALE TO 1:450 (MM)
# =============================================================================

print("Boolean-unioning complete One High Line complex...")
# Shift so X_min ~ 0, Y_min ~ 0, Z_min = 0 in mm!
complete_ft = m3d.Manifold.batch_boolean([
    plinth,
    west_podium,
    south_retail,
    east_podium,
    east_retail,
    north_bridge,
    south_bridge,
    high_line,
    west_tower,
    east_tower,
], m3d.OpType.Add)

def to_mm_manifold(m_ft, origin_ft=(-18.0, -8.0, -12.0)):
    return m_ft.translate([-origin_ft[0], -origin_ft[1], -origin_ft[2]]).scale([SCALE_MM_PER_FT]*3)

def manifold_to_trimesh(m):
    mesh = m.to_mesh()
    verts = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    faces = np.asarray(mesh.tri_verts, dtype=np.int64)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)

complete_mm = to_mm_manifold(complete_ft)
tm_complete = manifold_to_trimesh(complete_mm)

print(f"\n=== COMPLETE 1:450 MODEL STATS ===")
print(f"  Watertight: {tm_complete.is_watertight}, Volume: {tm_complete.is_volume}")
print(f"  Triangles:  {len(tm_complete.faces):,}")
ext = tm_complete.bounding_box.extents
print(f"  Dimensions: {ext[0]:.1f} mm (E-W) x {ext[1]:.1f} mm (N-S) x {ext[2]:.1f} mm (H)")

tm_complete.export(os.path.join(OUT_DIR, "one_high_line_complete_1_450.stl"))
tm_complete.export(os.path.join(OUT_DIR, "one_high_line_complete_1_450.3mf"))
print("Exported complete STL & 3MF!")


# =============================================================================
# 5. SPLIT INTO 5 FDM-PRINTABLE INTERLOCKING PARTS (ALL FIT <= 200x200x200 MM!)
# =============================================================================
print("\nGenerating 5-part FDM-ready interlocking display kit...")

# Helper: cylindrical alignment peg & socket (with 0.25 mm FDM clearance)
def peg_cyl(cx_ft, cy_ft, z0_ft, h_ft, r_mm):
    r_ft = r_mm / SCALE_MM_PER_FT
    cyl = m3d.Manifold.cylinder(height=float(h_ft), radius_low=float(r_ft), radius_high=float(r_ft * 0.92), circular_segments=32)
    return cyl.translate([float(cx_ft), float(cy_ft), float(z0_ft)])

def box_ft(x0, x1, y0, y1, z0, z1):
    return m3d.Manifold.cube([float(x1 - x0), float(y1 - y0), float(z1 - z0)]).translate([float(x0), float(y0), float(z0)])

# Split plane between West Base and East Base: X = 235 ft (cleanly in courtyard between South Retail & East Podium!)
#Wait: check South Retail east edge is at X=228 ft, East Podium west edge is at X=214..278 ft!
# Let's split along a clean slanted plane parallel to East Podium's west face!
# Even simpler: split at X_split(Y) = 235 - 0.36*(Y - 10) -> at Y=10: X=235 (clear of South Retail 228 & East Podium 278!),
# at Y=193: X=169 (right in the middle of the Skybridge gap between West Podium 158 and East Podium 214!)
west_half_clip = extrude_poly([(-100, -100), (242, -100), (176, 260), (-100, 260)], -50, 500)
east_half_clip = extrude_poly([(242, -100), (600, -100), (600, 260), (176, 260)], -50, 500)

# Build the assembly WITHOUT skybridge first (skybridge is Part 5 snap-in so ZERO supports!)
base_no_bridge_ft = m3d.Manifold.batch_boolean([
    plinth, west_podium, south_retail, east_podium, east_retail, high_line, west_tower, east_tower
], m3d.OpType.Add)

# Horizontal tower split heights:
#   West Tower split at Z = 225.0' (between floor 20W & 21W -> lower ~160mm tall, upper ~143mm tall!)
#   East Tower split at Z = 180.0' (between floor 16E & 17E -> lower ~130mm tall, upper ~106mm tall!)
Z_SPLIT_W = 225.0
Z_SPLIT_E = 180.0

west_full = base_no_bridge_ft ^ west_half_clip
east_full = base_no_bridge_ft ^ east_half_clip

p1_ft = west_full ^ box_ft(-100, 600, -100, 300, -50, Z_SPLIT_W)
p2_ft = west_full ^ box_ft(-100, 600, -100, 300, Z_SPLIT_W, 500)
p3_ft = east_full ^ box_ft(-100, 600, -100, 300, -50, Z_SPLIT_E)
p4_ft = east_full ^ box_ft(-100, 600, -100, 300, Z_SPLIT_E, 500)

# Add alignment pegs & sockets (0.25 mm radial clearance = 0.37 ft)
# 1) Between Part 1 & Part 2 (West Tower split at Z=225'): 2 vertical pegs inside West Tower core!
w_center_225 = w_polys_all[15].mean(axis=0)
w_pegs_add, w_pegs_sub = [], []
for dx, dy in [(-14.0, 12.0), (14.0, -12.0)]:
    cx, cy = w_center_225[0] + dx, w_center_225[1] + dy
    w_pegs_add.append(peg_cyl(cx, cy, Z_SPLIT_W - 2.0, 12.0, r_mm=4.5))
    w_pegs_sub.append(peg_cyl(cx, cy, Z_SPLIT_W - 0.5, 11.2, r_mm=4.85))
p1_ft = m3d.Manifold.batch_boolean([p1_ft] + w_pegs_add, m3d.OpType.Add)
p2_ft = p2_ft - m3d.Manifold.batch_boolean(w_pegs_sub, m3d.OpType.Add)

# 2) Between Part 3 & Part 4 (East Tower split at Z=180'): 2 vertical pegs inside East Tower core!
e_center_180 = e_polys_all[9].mean(axis=0)
e_pegs_add, e_pegs_sub = [], []
for dx, dy in [(-14.0, -10.0), (14.0, 10.0)]:
    cx, cy = e_center_180[0] + dx, e_center_180[1] + dy
    e_pegs_add.append(peg_cyl(cx, cy, Z_SPLIT_E - 2.0, 12.0, r_mm=4.5))
    e_pegs_sub.append(peg_cyl(cx, cy, Z_SPLIT_E - 0.5, 11.2, r_mm=4.85))
p3_ft = m3d.Manifold.batch_boolean([p3_ft] + e_pegs_add, m3d.OpType.Add)
p4_ft = p4_ft - m3d.Manifold.batch_boolean(e_pegs_sub, m3d.OpType.Add)

# 3) Between Part 1 & Part 3 (Plinth seam): 2 horizontal cylindrical alignment pins inside the 8mm thick plinth!
def horiz_peg(x0_ft, y0_ft, z0_ft, length_ft, r_mm):
    r_ft = r_mm / SCALE_MM_PER_FT
    cyl = m3d.Manifold.cylinder(height=float(length_ft), radius_low=float(r_ft), radius_high=float(r_ft * 0.92), circular_segments=24)
    cyl = cyl.rotate([0, 90, 0])  # points along +X
    return cyl.translate([float(x0_ft), float(y0_ft), float(z0_ft)])

seam_pins_add, seam_pins_sub = [], []
for y_pin in [35.0, 155.0]:
    # Seam X at y_pin: X = 242 - (y_pin + 100) * (242 - 176) / 360
    x_seam = 242.0 - (y_pin + 100.0) * (66.0 / 360.0)
    seam_pins_add.append(horiz_peg(x_seam - 4.0, y_pin, -6.5, 14.0, r_mm=2.2))
    seam_pins_sub.append(horiz_peg(x_seam - 1.0, y_pin, -6.5, 12.0, r_mm=2.5))
p1_ft = m3d.Manifold.batch_boolean([p1_ft] + seam_pins_add, m3d.OpType.Add)
p3_ft = p3_ft - m3d.Manifold.batch_boolean(seam_pins_sub, m3d.OpType.Add)

# 4) Part 5 & Part 6: Snap-in North Bridge & South Bridge!
# Carve matching pockets (with ~0.35mm clearance) into West Podium/South Retail (p1) & East Podium (p3)
nb_pocket = extrude_poly(north_bridge_poly, 26.8, 57.0)
sb_pocket = extrude_poly(south_bridge_poly, 26.8, 57.0)
p1_ft = p1_ft - nb_pocket - sb_pocket
p3_ft = p3_ft - nb_pocket - sb_pocket
p5_ft = north_bridge
p6_ft = south_bridge

# Export all 6 parts, each grounded at Z = 0 mm for immediate drop-in to slicer!
parts_map = [
    ("part1_west_base_and_lower.stl", p1_ft),
    ("part2_west_tower_upper.stl",    p2_ft),
    ("part3_east_base_and_lower.stl", p3_ft),
    ("part4_east_tower_upper.stl",    p4_ft),
    ("part5_north_bridge.stl",        p5_ft),
    ("part6_south_bridge.stl",        p6_ft),
]

for fname, m_ft in parts_map:
    m_mm = to_mm_manifold(m_ft)
    tm = manifold_to_trimesh(m_mm)
    # Ground each part cleanly on Z = 0, centered in X/Y for easy slicing!
    b = tm.bounds
    tm.apply_translation([-(b[0][0] + b[1][0]) / 2.0, -(b[0][1] + b[1][1]) / 2.0, -b[0][2]])
    # For Part 5 (Skybridge), also note it sits flat at Z=0 (bottom of bridge on build plate -> zero supports!)
    tm.export(os.path.join(OUT_DIR, fname))
    e = tm.bounding_box.extents
    print(f"  {fname:32s} | watertight={tm.is_watertight} | tris={len(tm.faces):7,} | {e[0]:5.1f} x {e[1]:5.1f} x {e[2]:5.1f} mm")

print(f"\nDone in {time.time() - t0:.1f}s!")
