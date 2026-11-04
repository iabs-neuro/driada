"""Nature figure format shared by the figure scripts of the INTENSE paper.

Nature Portfolio rules applied here (research-figure-guide.nature.com, read
through figaudit.py): 183 mm wide (two columns), at most 247 mm high, Arial
5-7 pt, panel letters 8 pt bold lowercase, RGB, vector PDF with embedded
TrueType fonts, no transparency.

Transparency is removed by ``flatten_alpha``: every colour drawn with alpha < 1
is replaced by the opaque colour it shows on a white background. Where
translucent elements overlapped (shaded periods over a trace), the scripts put
the shading under the trace so that nothing is hidden. Dense scatter plots are
drawn by ``raster_scatter`` into an opaque RGB image, which keeps the look of
overlapping translucent points without an alpha channel in the PDF.
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402

import figaudit as fa  # noqa: E402

JOURNAL, COLUMN = "nature", "2"
WIDTH_MM = fa.target_width_mm(JOURNAL, COLUMN)
MAX_HEIGHT_MM = fa.JOURNALS[JOURNAL]["max_height_mm"]
MM = 1 / 25.4

LABEL_PT = 7      # axis labels
TICK_PT = 6       # tick labels, legends, annotations
SMALL_PT = 5      # dense annotations (heatmap labels, significance stars)
LETTER_PT = 8     # panel letters
LINE_PT = 0.75    # data lines
AXIS_PT = 0.5     # spines and ticks


def apply_style():
    """rcParams of the Nature format; call before any plotting."""
    family = fa.use_style(JOURNAL, base_pt=LABEL_PT)
    plt.rcParams.update({
        "axes.labelsize": LABEL_PT, "axes.titlesize": LABEL_PT,
        "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT, "legend.fontsize": TICK_PT,
        "axes.linewidth": AXIS_PT, "xtick.major.width": AXIS_PT, "ytick.major.width": AXIS_PT,
        "xtick.minor.width": AXIS_PT * 0.8, "ytick.minor.width": AXIS_PT * 0.8,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0, "xtick.minor.size": 1.2,
        "ytick.minor.size": 1.2, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
        "axes.labelpad": 2.0, "lines.linewidth": LINE_PT, "patch.linewidth": AXIS_PT,
        "legend.handlelength": 1.6, "legend.handletextpad": 0.4, "legend.borderaxespad": 0.2,
        "legend.labelspacing": 0.3, "legend.columnspacing": 1.0,
        # mathtext in the same font as the text, so the PDF embeds Arial only
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial", "mathtext.default": "regular",
        "savefig.facecolor": "white", "figure.facecolor": "white", "savefig.transparent": False,
        "image.cmap": "viridis",
    })
    if family not in ("Arial", "Helvetica"):
        print(f"[WARNING] Arial is not installed; text falls back to {family}", file=sys.stderr)
    return family


def style_preset():
    """StylePreset of driada.utils.publication with the sizes above (panel scripts take it)."""
    from driada.utils.publication import StylePreset

    return StylePreset.fixed_size(
        name="nature", base_spine_width=AXIS_PT, base_tick_width=AXIS_PT, base_tick_length=2.0,
        base_tick_pad=1.5, base_tick_labelsize=TICK_PT, base_label_size=LABEL_PT,
        base_title_size=LABEL_PT, base_legend_fontsize=TICK_PT, base_line_width=LINE_PT,
        base_marker_size=1.5, base_annotation_fontsize=TICK_PT,
    )


def figure(height_mm):
    """Figure 183 mm wide; height capped by the journal's page depth."""
    if height_mm > MAX_HEIGHT_MM:
        raise ValueError(f"height {height_mm} mm exceeds the Nature maximum {MAX_HEIGHT_MM} mm")
    return plt.figure(figsize=(WIDTH_MM * MM, height_mm * MM))


def axes_mm(fig, x, y, w, h, **kwargs):
    """Axes at x, y (top-left corner, mm from the top-left of the figure), w x h mm."""
    W, H = fig.get_size_inches() / MM
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kwargs)


def ymax_in_view(ax):
    """Largest y of the data lines of `ax` within its current x limits."""
    lo, hi = sorted(ax.get_xlim())
    top = 0.0
    for ln in ax.get_lines():
        x, y = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
        m = (x >= lo) & (x <= hi)
        if m.any():
            top = max(top, float(np.nanmax(y[m])))
    return top


def letter_mm(fig, letter, x, y):
    """Panel letter with its top-left corner at x, y mm from the top-left of the figure."""
    W, H = fig.get_size_inches() / MM
    return fig.text(x / W, 1 - y / H, letter.lower(), fontsize=LETTER_PT, fontweight="bold",
                    ha="left", va="top")


def panel_letter(container, letter, x=0.0, y=1.0, dx_mm=0.0, dy_mm=0.0):
    """Bold lowercase panel letter at (x, y) of a figure or subfigure (top-left by default)."""
    fig = container.figure if hasattr(container, "figure") and container.figure is not None else container
    from matplotlib.transforms import ScaledTranslation

    tr = getattr(container, "transSubfigure", container.transFigure)
    tr = tr + ScaledTranslation(dx_mm * MM, dy_mm * MM, fig.dpi_scale_trans)
    return container.text(x, y, letter.lower(), transform=tr, fontsize=LETTER_PT,
                          fontweight="bold", ha="left", va="top")


def on_white(img):
    """RGB float image of an RGB(A) image composited on white."""
    img = np.asarray(img, dtype=float)
    if img.max() > 1.0:
        img = img / 255.0
    if img.ndim == 3 and img.shape[2] == 4:
        a = img[..., 3:4]
        img = img[..., :3] * a + (1.0 - a)
    return np.clip(img[..., :3], 0, 1)


def _blend(rgba, alpha=None):
    """Opaque colour that `rgba` (with optional extra alpha) shows on white."""
    c = np.array(mcolors.to_rgba(rgba), dtype=float)
    a = c[3] * (1.0 if alpha is None else alpha)
    if a <= 0:
        return (0, 0, 0, 0)
    return tuple(c[:3] * a + (1.0 - a)) + (1.0,)


def _blend_array(arr, alpha=None):
    arr = np.atleast_2d(np.asarray(arr, dtype=float))
    if arr.size == 0:
        return arr
    out = arr.copy()
    a = arr[:, 3] * (1.0 if alpha is None else alpha)
    vis = a > 0
    out[vis, :3] = arr[vis, :3] * a[vis, None] + (1.0 - a[vis, None])
    out[vis, 3] = 1.0
    out[~vis] = 0.0
    return out


def flatten_alpha(fig):
    """Replace every translucent colour of the figure by its opaque look on white."""
    from matplotlib.collections import Collection
    from matplotlib.image import AxesImage
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.text import Text

    for art in fig.findobj():
        alpha = art.get_alpha() if hasattr(art, "get_alpha") else None
        if isinstance(art, Line2D):
            mec = art.get_markeredgecolor()
            if isinstance(mec, str) and mec.lower() == "none":
                # an invisible marker edge is written as a stroke with opacity 0
                mfc = art.get_markerfacecolor()
                art.set_markeredgecolor("white" if isinstance(mfc, str) and mfc.lower() == "none" else mfc)
                art.set_markeredgewidth(0)
            for get, set_ in (("get_color", "set_color"), ("get_markerfacecolor", "set_markerfacecolor"),
                              ("get_markeredgecolor", "set_markeredgecolor")):
                c = getattr(art, get)()
                if isinstance(c, str) and c in ("none", "None", "auto"):
                    continue
                getattr(art, set_)(_blend(c, alpha))
            art.set_alpha(None)
        elif isinstance(art, Patch):
            fc, ec = art.get_facecolor(), art.get_edgecolor()   # already include alpha
            art.set_alpha(None)
            art.set_facecolor(_blend(fc) if fc[3] > 0 else "none")
            if ec[3] > 0:
                art.set_edgecolor(_blend(ec))
            else:
                # an invisible edge is written as a stroke with opacity 0; draw none instead
                art.set_edgecolor(_blend(fc) if fc[3] > 0 else "white")
                art.set_linewidth(0)
        elif isinstance(art, Collection):
            if art.get_array() is not None:
                if alpha is None or alpha >= 1:
                    continue        # colour-mapped and opaque: leave the mapping alone
                art.update_scalarmappable()
                fc, ec = art.get_facecolor(), art.get_edgecolor()
                art.set_array(None)
            else:
                fc, ec = art.get_facecolor(), art.get_edgecolor()
            art.set_alpha(None)
            if len(fc):
                art.set_facecolor(_blend_array(fc))
            if len(ec):
                eb = _blend_array(ec)
                if np.all(eb[:, 3] == 0):
                    art.set_edgecolor(_blend_array(fc) if len(fc) else "white")
                    art.set_linewidth(0)
                else:
                    art.set_edgecolor(eb)
        elif isinstance(art, AxesImage):
            a = art.get_array()
            if a is not None and a.ndim == 3 and a.shape[2] == 4:
                art.set_data(on_white(a))
            art.set_alpha(None)
        elif isinstance(art, Text):
            if alpha is not None and alpha < 1:
                art.set_color(_blend(art.get_color(), alpha))
                art.set_alpha(None)
    return fig


def raster_scatter(ax, x, y, colors, s, alpha=1.0, dpi=600, order=None, zorder=1):
    """Draw a dense scatter as one opaque RGB image filling the data area of `ax`.

    Axis limits must be final; the axes position is taken after a draw.
    """
    fig = ax.figure
    fig.canvas.draw()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    xl, yl = ax.get_xlim(), ax.get_ylim()
    off = plt.figure(figsize=(bbox.width, bbox.height), dpi=dpi)
    oax = off.add_axes([0, 0, 1, 1])
    oax.set_xlim(xl)
    oax.set_ylim(yl)
    oax.set_axis_off()
    oax.set_facecolor("white")
    if order is not None:
        x, y, colors = np.asarray(x)[order], np.asarray(y)[order], np.asarray(colors, dtype=object)[order]
    oax.scatter(x, y, c=list(colors), s=s, alpha=alpha, linewidths=0)
    off.canvas.draw()
    rgb = np.asarray(off.canvas.buffer_rgba())[..., :3].copy()
    plt.close(off)
    ax.imshow(rgb, extent=(xl[0], xl[1], yl[0], yl[1]), aspect="auto", interpolation="nearest",
              origin="upper", zorder=zorder)
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    return rgb


def check_pdf(path):
    """Problems of a written PDF that figaudit's checks do not cover: transparency, colour space."""
    import fitz

    out = []
    doc = fitz.open(path)
    for xref in range(1, doc.xref_length()):
        try:
            obj = doc.xref_object(xref, compressed=False)
        except Exception:  # noqa: BLE001 - unreadable objects are not ours to judge
            continue
        if "/SMask" in obj and "/SMask /None" not in obj:
            out.append(("ERROR", f"object {xref} has a soft mask (transparency)"))
        for key in ("/CA ", "/ca "):
            i = obj.find(key)
            while i >= 0:
                val = float(obj[i + 4:].split()[0].strip("/>"))
                if val < 1:
                    out.append(("ERROR", f"object {xref}: {key.strip()} {val} (transparency)"))
                i = obj.find(key, i + 1)
        if "/DeviceCMYK" in obj:
            out.append(("ERROR", f"object {xref} is CMYK, Nature wants RGB"))
        if "/Subtype /Image" in obj and "/DeviceRGB" not in obj and "/DeviceGray" not in obj \
                and "/ImageMask true" not in obj:
            out.append(("CHECK", f"image {xref} has an unusual colour space"))
    for p in doc:
        for img in p.get_images(full=True):
            if img[1] != 0:
                out.append(("ERROR", f"image {img[0]} has an alpha mask"))
    return out


def save(fig, stem, flatten=True):
    """Flatten transparency, write <stem>.pdf at the final size, audit the figure and the PDF.

    Returns the list of (severity, message) of all checks.
    """
    if flatten:
        flatten_alpha(fig)
    fig.savefig(f"{stem}_preview_300dpi.png", dpi=300)
    problems = fa.save(fig, stem, JOURNAL, COLUMN, formats=["pdf"])
    pdf = f"{stem}.pdf"
    pdf_problems = fa.audit_pdf(pdf, JOURNAL, COLUMN) + check_pdf(pdf)
    fa.report(pdf_problems, pdf)
    return problems + pdf_problems
