"""
Register all West Tower (5W-36W) and East Tower (11E-26E) floor plans
via FFT phase correlation on the elevator core, and extract clean
6-corner/4-corner outer facade polygons for every floor.
"""
import os, json
import numpy as np
import pymupdf
from scipy import ndimage
from shapely.geometry import MultiPoint, Polygon
import matplotlib.pyplot as plt

PDF_PATH = os.environ.get("OHL_PDF_PATH", "One Highline floor plans.pdf")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extracted")
os.makedirs(OUT_DIR, exist_ok=True)

doc = pymupdf.open(PDF_PATH)
SCALE_TOWER = 3.606  # pt/ft
DPI = 144
PX_PER_FT = SCALE_TOWER * (DPI / 72.0)  # 7.212 px/ft

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

def render_crop(page_num):
    page = doc[page_num - 1]
    pix = page.get_pixmap(dpi=DPI, colorspace=pymupdf.csGRAY)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    y0, y1 = int(15 * DPI / 72), int(605 * DPI / 72)
    return arr[y0:y1, :], y0

def fft_shift(ref_img, mov_img):
    """Find (dy, dx) to shift mov_img so it aligns with ref_img using FFT correlation."""
    # Use inverted intensity (walls = bright, background = 0)
    a = (255.0 - ref_img.astype(np.float32))
    b = (255.0 - mov_img.astype(np.float32))
    a = (a > 80).astype(np.float32)
    b = (b > 80).astype(np.float32)
    fa = np.fft.rfft2(a)
    fb = np.fft.rfft2(b)
    cc = np.fft.irfft2(fa * np.conj(fb), s=a.shape)
    iy, ix = np.unravel_index(np.argmax(cc), cc.shape)
    h, w = a.shape
    dy = iy if iy < h // 2 else iy - h
    dx = ix if ix < w // 2 else ix - w
    return dy, dx

def extract_tower(pages, ref_idx):
    # Render reference floor
    ref_crop, y0 = render_crop(pages[ref_idx][1])
    # Mask central region of ref_crop where the elevator core lives, to avoid outer facade shift biasing correlation
    # Actually, full solid walls (including elevator shafts + stairs) have strongest correlation in the center!
    h, w = ref_crop.shape
    results = {}
    for fl, pno in pages:
        crop, _ = render_crop(pno)
        dy, dx = fft_shift(ref_crop, crop)
        # Extract solid piers & walls
        dark = crop < 165
        solid = ndimage.binary_opening(dark, structure=np.ones((5, 5)))
        ys, xs = np.where(solid)
        # Shift into ref_crop coordinates!
        xs_reg = (xs + dx) / PX_PER_FT
        ys_reg = -(ys + dy + y0) / PX_PER_FT
        pts = np.column_stack([xs_reg, ys_reg])
        hull = MultiPoint(pts[::3]).convex_hull.simplify(1.2)
        coords = list(hull.exterior.coords)[:-1]
        results[fl] = {
            "coords": [(round(float(x), 2), round(float(y), 2)) for x, y in coords],
            "area": round(float(hull.area), 1),
            "dx_ft": round(dx / PX_PER_FT, 2),
            "dy_ft": round(-dy / PX_PER_FT, 2),
        }
    return results

print("Registering West Tower (31 floors)...")
west_data = extract_tower(WEST_PAGES, ref_idx=15)  # ref = 20th floor
print("Registering East Tower (15 floors)...")
east_data = extract_tower(EAST_PAGES, ref_idx=7)   # ref = 18th floor

with open(f"{OUT_DIR}/tower_floors.json", "w") as f:
    json.dump({"west": west_data, "east": east_data}, f, indent=2)

# Plot registered progression!
fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))

for fl in sorted(west_data.keys()):
    c = np.array(west_data[fl]["coords"] + [west_data[fl]["coords"][0]])
    col = plt.cm.viridis((fl - 5) / 31.0)
    lw = 2.0 if fl in [5, 7, 15, 22, 30, 36] else 0.7
    alpha = 1.0 if fl in [5, 7, 15, 22, 30, 36] else 0.35
    lbl = f"W{fl} ({west_data[fl]['area']:.0f}sf)" if fl in [5, 7, 15, 22, 30, 36] else None
    axes[0].plot(c[:,0], c[:,1], color=col, linewidth=lw, alpha=alpha, label=lbl)
axes[0].set_aspect("equal"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[0].set_title("West Tower: All 31 Floors FFT-Registered (ft)")

for fl in sorted(east_data.keys()):
    c = np.array(east_data[fl]["coords"] + [east_data[fl]["coords"][0]])
    col = plt.cm.plasma((fl - 11) / 15.0)
    lw = 2.0 if fl in [11, 15, 20, 24, 26] else 0.7
    alpha = 1.0 if fl in [11, 15, 20, 24, 26] else 0.4
    lbl = f"E{fl} ({east_data[fl]['area']:.0f}sf)" if fl in [11, 15, 20, 24, 26] else None
    axes[1].plot(c[:,0], c[:,1], color=col, linewidth=lw, alpha=alpha, label=lbl)
axes[1].set_aspect("equal"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
axes[1].set_title("East Tower: All 15 Floors FFT-Registered (ft)")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/tower_registered.png", dpi=150)
print("Saved tower_registered.png and tower_floors.json")
