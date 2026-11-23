"""
Render studio-lit orthographic and perspective views of the generated 3D STL
plus side-by-side comparison against the PDF N/S/E/W Elevations.
Uses a fast vectorized z-buffer rasterizer in pure NumPy + PIL so we see
exact triangle geometry with crisp travertine architectural shading.
"""
import os
import numpy as np
import trimesh
from PIL import Image
import pymupdf
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "output")
REND_DIR = os.path.join(BASE_DIR, "renders")
os.makedirs(REND_DIR, exist_ok=True)

mesh = trimesh.load(os.path.join(OUT_DIR, "one_high_line_complete_1_450.stl"))
V = np.asarray(mesh.vertices, dtype=np.float32)
F = np.asarray(mesh.faces, dtype=np.int32)
FN = np.asarray(mesh.face_normals, dtype=np.float32)

# Center mesh in X, Y for easy camera orbits
center = np.array([(V[:,0].min()+V[:,0].max())/2, (V[:,1].min()+V[:,1].max())/2, V[:,2].max()*0.42], dtype=np.float32)

def render_view(V, F, FN, azim_deg, elev_deg, img_w=1000, img_h=800, ortho=True, fov_deg=30.0, bg=(242, 240, 236)):
    """Fast scanline-less painter's-algorithm / z-sorted flat-shaded architectural render."""
    az = np.radians(azim_deg)
    el = np.radians(elev_deg)
    # Camera looks from (cos el cos az, cos el sin az, sin el) toward center
    cam_dir = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)], dtype=np.float32)
    world_up = np.array([0, 0, 1], dtype=np.float32)
    if abs(np.dot(cam_dir, world_up)) > 0.99:
        world_up = np.array([0, 1, 0], dtype=np.float32)
    right = np.cross(world_up, cam_dir); right /= np.linalg.norm(right)
    up = np.cross(cam_dir, right); up /= np.linalg.norm(up)

    Vc = V - center
    # Camera coordinates: x_c = right, y_c = up, z_c = toward camera (depth)
    xc = Vc @ right
    yc = Vc @ up
    zc = Vc @ cam_dir

    if not ortho:
        cam_dist = 820.0
        persp = cam_dist / np.maximum(cam_dist - zc, 50.0)
        xc = xc * persp
        yc = yc * persp

    # Back-face culling
    cam_dot = FN @ cam_dir
    vis = cam_dot > -0.05
    F_vis = F[vis]
    FN_vis = FN[vis]

    # Lighting: warm studio key light + cool fill + subtle rim (travertine palette!)
    L1 = np.array([0.55, -0.45, 0.70], dtype=np.float32); L1 /= np.linalg.norm(L1)
    L2 = np.array([-0.60, 0.55, 0.40], dtype=np.float32); L2 /= np.linalg.norm(L2)
    d1 = np.clip(FN_vis @ L1, 0.0, 1.0)
    d2 = np.clip(FN_vis @ L2, 0.0, 1.0)
    # Height-based subtle AO (darker near ground courtyard)
    fz = V[F_vis, 2].mean(axis=1)
    ao = 0.85 + 0.15 * np.clip(fz / 200.0, 0.0, 1.0)
    # Warm travertine stone color: (238, 232, 220)
    lum = (0.30 + 0.58 * d1 + 0.22 * d2) * ao
    col = np.clip(np.outer(lum, np.array([240.0, 234.0, 222.0])), 0, 255).astype(np.uint8)

    # Sort visible triangles back-to-front by centroid zc
    z_tri = zc[F_vis].mean(axis=1)
    order = np.argsort(z_tri)
    F_vis = F_vis[order]
    col = col[order]

    # Map to image coordinates
    xmin, xmax = xc.min(), xc.max()
    ymin, ymax = yc.min(), yc.max()
    margin = 0.08
    span_x = (xmax - xmin) * (1 + 2*margin)
    span_y = (ymax - ymin) * (1 + 2*margin)
    scale = min(img_w / span_x, img_h / span_y)
    px = (xc - (xmin + xmax)/2) * scale + img_w / 2
    py = img_h / 2 - (yc - (ymin + ymax)/2) * scale

    # Render using matplotlib PolyCollection (antialiased, crisp!)
    fig = plt.figure(figsize=(img_w/100, img_h/100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(np.array(bg)/255.0)
    from matplotlib.collections import PolyCollection
    polys = np.stack([px[F_vis], py[F_vis]], axis=-1)
    pc = PolyCollection(polys, facecolors=col/255.0, edgecolors="none", antialiaseds=False)
    ax.add_collection(pc)
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.axis("off")
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf

views = [
    ("hero_sw.png",   -125, 22, False, "Hero View (SW from 11th Ave & W 17th St)"),
    ("hero_ne.png",     48, 24, False, "Hero View (NE from High Line & W 18th St)"),
    ("hero_nw.png",    132, 22, False, "Hero View (NW from 11th Ave & W 18th St)"),
    ("elev_north.png",  90,  0, True,  "North Elevation (Orthographic)"),
    ("elev_south.png", -90,  0, True,  "South Elevation (Orthographic)"),
    ("elev_west.png",  180,  0, True,  "West Elevation (Orthographic)"),
    ("aerial.png",    -110, 58, False, "Aerial View"),
]

imgs = {}
for fname, az, el, ortho, title in views:
    print(f"Rendering {fname}...")
    im = render_view(V, F, FN, az, el, img_w=900, img_h=720, ortho=ortho)
    Image.fromarray(im).save(os.path.join(REND_DIR, fname))
    imgs[fname] = (im, title)

# Create a 2x2 showcase sheet of the 4 hero/aerial views
fig, axes = plt.subplots(2, 2, figsize=(16, 12.8))
for ax, key in zip(axes.flat, ["hero_sw.png", "hero_ne.png", "hero_nw.png", "aerial.png"]):
    im, title = imgs[key]
    ax.imshow(im)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(REND_DIR, "showcase_4views.png"), dpi=130)
plt.close(fig)

# Create a side-by-side comparison of N Elevation & S Elevation: PDF vs Our 3D Model (if PDF is available)
PDF_PATH = os.environ.get("OHL_PDF_PATH", "One Highline floor plans.pdf")
if os.path.exists(PDF_PATH):
    doc = pymupdf.open(PDF_PATH)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for row, (pno, rkey, lbl) in enumerate([(2, "elev_north.png", "North Elevation"), (4, "elev_south.png", "South Elevation")]):
        pix = doc[pno - 1].get_pixmap(dpi=120)
        pdf_im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        # Crop PDF elevation region
        h, w = pdf_im.shape[:2]
        axes[row, 0].imshow(pdf_im[int(h*0.16):int(h*0.73), int(w*0.08):int(w*0.92)])
        axes[row, 0].set_title(f"PDF {lbl}", fontsize=12)
        axes[row, 0].axis("off")
        axes[row, 1].imshow(imgs[rkey][0])
        axes[row, 1].set_title(f"3D Model {lbl}", fontsize=12)
        axes[row, 1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(REND_DIR, "verify_elevations.png"), dpi=130)
    plt.close(fig)
print("All renders complete!")
