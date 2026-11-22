"""
Measure exact site coordinates (in feet, X=East, Y=North, origin near SW corner of block)
for the city block, podiums, skybridge, High Line, West/East towers, and crowns,
plus exact vertical heights from the 4 elevations.
"""
import os, json
import numpy as np
import pymupdf
import matplotlib.pyplot as plt

PDF_PATH = os.environ.get("OHL_PDF_PATH", "One Highline floor plans.pdf")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extracted")
doc = pymupdf.open(PDF_PATH)

SCALE_SITE = 1.1268  # pt/ft on P001, P008-P011
SCALE_ELEV = 0.9013  # pt/ft on P002-P005

# Render P001 (Site), P008 (1st), P010 (3rd), P011 (4th) with an exact FEET coordinate grid!
# Let origin (0, 0) in feet be at PDF point (x=48.0, y=482.0) -> roughly SW corner of building on W 17th & 11th Ave
OX_PT, OY_PT = 48.0, 482.0

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
for ax, (pno, title) in zip(axes.flat, [(1, "P001 Site Plan"), (8, "P008 1st Floor"), (10, "P010 3rd Floor (Skybridge)"), (11, "P011 4th Floor")]):
    page = doc[pno - 1]
    dpi = 150
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    # Image extent in feet:
    # x_pt goes from 0 to page.rect.width; y_pt goes from 0 (top) to page.rect.height (bottom)
    # X_ft = (x_pt - OX_PT) / SCALE_SITE
    # Y_ft = (OY_PT - y_pt) / SCALE_SITE
    xmin = (0 - OX_PT) / SCALE_SITE
    xmax = (page.rect.width - OX_PT) / SCALE_SITE
    ymax = (OY_PT - 0) / SCALE_SITE
    ymin = (OY_PT - page.rect.height) / SCALE_SITE
    ax.imshow(img, extent=[xmin, xmax, ymin, ymax])
    ax.set_xlim(-35, 500)
    ax.set_ylim(-30, 260)
    ax.set_xticks(np.arange(-20, 500, 20))
    ax.set_yticks(np.arange(-20, 260, 20))
    ax.grid(True, color="red", alpha=0.35, linewidth=0.6)
    ax.tick_params(labelsize=7)
    ax.set_title(title)
    ax.set_xlabel("X (ft, East)")
    ax.set_ylabel("Y (ft, North)")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/site_grid.png", dpi=150)

# Also render P002 (N Elev) and P004 (S Elev) with an exact FEET height grid (Z from RL 0')!
fig2, axes2 = plt.subplots(1, 2, figsize=(18, 9))
for ax, (pno, y_rl0, title) in zip(axes2, [(2, 544.26, "P002 N Elevation (E on left, W on right)"), (4, 542.44, "P004 S Elevation (W on left, E on right)")]):
    page = doc[pno - 1]
    dpi = 150
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    xmin = (0 - 100.0) / SCALE_ELEV
    xmax = (page.rect.width - 100.0) / SCALE_ELEV
    zmax = (y_rl0 - 0) / SCALE_ELEV
    zmin = (y_rl0 - page.rect.height) / SCALE_ELEV
    ax.imshow(img, extent=[xmin, xmax, zmin, zmax])
    ax.set_xlim(-20, 480)
    ax.set_ylim(-10, 450)
    ax.set_xticks(np.arange(0, 480, 40))
    ax.set_yticks(np.arange(0, 450, 20))
    ax.grid(True, color="blue", alpha=0.3, linewidth=0.6)
    ax.tick_params(labelsize=7)
    ax.set_title(title)
    ax.set_ylabel("Z (ft above RL 0')")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/elev_grid.png", dpi=150)
print("Saved site_grid.png and elev_grid.png")
