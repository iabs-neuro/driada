"""
Compose Fig 1 — extract the two pages of fig1_assembled.pdf, place side by side
with A/B labels.

fig1_assembled.pdf is the export of a slide deck in which the outputs of
fig1_architecture.py, fig1_objects.py, fig1_network.py and fig1_traces.py were
arranged by hand; that manual step is not scripted.

Usage:
    python fig1_compose.py          # saves fig1.pdf + .png
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import elife_paths

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import io

PDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig1_assembled.pdf")
DPI = 600

# ── Render each page to high-res image ────────────────────────────
doc = fitz.open(PDF_PATH)

images = []
for page_num in range(2):
    page = doc[page_num]
    zoom = DPI / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    # Clamp near-white pixels (>=250) to pure white to remove PDF border artifacts
    arr = np.array(img)
    arr[arr >= 250] = 255
    img = Image.fromarray(arr)
    images.append(img)

doc.close()

# ── Compose side by side ──────────────────────────────────────────
w0, h0 = images[0].size
w1, h1 = images[1].size

# Scale both to same height
target_h = max(h0, h1)
scale0 = target_h / h0
scale1 = target_h / h1
w0s = int(w0 * scale0)
w1s = int(w1 * scale1)

img0 = images[0].resize((w0s, target_h), Image.LANCZOS)
img1 = images[1].resize((w1s, target_h), Image.LANCZOS)

# Figure: both panels side by side, B shifted down
gap = int(DPI * 0.9)  # wider gap between panels
b_drop = int(target_h * 0.06)  # shift B downward
total_w = w0s + gap + w1s
total_h = target_h + b_drop

fig_w_in = total_w / DPI
fig_h_in = total_h / DPI

fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=DPI)
ax.set_xlim(0, total_w)
ax.set_ylim(total_h, 0)  # y inverted for image coords
ax.set_axis_off()
fig.patch.set_facecolor("white")

ax.imshow(img0, extent=[0, w0s, target_h, 0], aspect="auto")
ax.imshow(img1, extent=[w0s + gap, total_w, target_h + b_drop, b_drop], aspect="auto")

# ── Panel labels ──────────────────────────────────────────────────
label_fontsize = 48
label_y = DPI * 0.05  # near top

ax.text(DPI * 0.02, label_y, "A", fontsize=label_fontsize,
        fontweight="bold", va="top", ha="left", color="black")
ax.text(w0s + gap + DPI * 0.02, label_y, "B", fontsize=label_fontsize,
        fontweight="bold", va="top", ha="left", color="black")

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# ── Save ──────────────────────────────────────────────────────────
fig.savefig(elife_paths.out_path("fig1.png"),
            bbox_inches="tight", dpi=DPI, pad_inches=0)
fig.savefig(elife_paths.out_path("fig1.pdf"),
            bbox_inches="tight", dpi=DPI, pad_inches=0)
print("Saved fig1.pdf and .png")
