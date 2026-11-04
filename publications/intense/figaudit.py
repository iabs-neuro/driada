"""Check a publication figure against a journal's figure rules.

Copy this file next to the figure scripts of a project (the figure must stay
reproducible after the session ends), then:

    import figaudit as fa
    fa.use_style("eneuro")                                  # once, before plotting
    fig, axes = fa.subplots("eneuro", "1.5", height_mm=60, ncols=2)
    ...                                                     # plot
    fa.save(fig, "fig2", "eneuro", "1.5")                   # writes files, prints the audit

For a figure made elsewhere (Prism, Illustrator, R) that exists as a PDF:

    python figaudit.py pdf Figure2.pdf --journal nature --column 1

A TIFF's header (printed size, dpi, alpha, compression, file size), pixels untouched:

    python figaudit.py tiff Fig2.tif --journal plos --column text

List the journal table with sources:

    python figaudit.py journals

The width/overlap/label checks are adapted from `orx_figstyle.py` in
alphaXiv/OpenResearch (MIT License, Copyright (c) 2026 alphaXiv).
"""
from __future__ import annotations

import argparse
import os
import sys

MM_PER_IN = 25.4

# Every number below was read from the journal's own page on the date given.
# A journal that is not here is not guessed: pass width_mm= and the rules the
# user gives you. Re-check a journal's page when its row is more than a year old.
JOURNALS = {
    "nature": {
        # Nature Portfolio titles publish their own pages too and can differ
        # (Nature Neuroscience: max 180 mm) - use their row when there is one.
        "name": "Nature (Nature Portfolio figure guide; other titles may add their own limits)",
        "checked": "2026-09-19",
        "source": "https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/ ; "
        "https://www.nature.com/nature/for-authors/final-submission",
        "widths_mm": {"1": 89.0, "1.5": 120.0, "2": 183.0},  # 1.5 column may be 120-136 mm
        "width_rule": "exact",
        "text_pt": (5.0, 7.0),  # panel labels: 8 pt bold lowercase a, b, c
        "panel_label_pt": 8.0,
        "fonts": ["Arial", "Helvetica"],
        "vector_fonts": "embed",  # embed TrueType (Type 42), do NOT outline text
        "min_line_pt": 0.25,
        "formats": ["pdf"],
        "raster_dpi": 450,
        "raster_dpi_min": 300,
        "max_height_mm": 247.0,  # full page depth
        "max_mb": None,
        "panel_letters": "lower",
        "extra": [],
    },
    "cell": {
        "name": "Cell Press two-column journals (Cell, Neuron, Cell Reports, Current Biology, ...)",
        "checked": "2026-09-19",
        "source": "https://www.cell.com/figureguidelines",
        "widths_mm": {"1": 85.0, "1.5": 114.0, "2": 174.0},
        "width_rule": "max",
        "text_pt": (None, None),  # not stated; the universal 5 pt floor applies
        "fonts": ["Arial"],  # "Always embed fonts, and use only Arial fonts"
        "vector_fonts": "embed",
        "min_line_pt": None,
        "formats": ["pdf"],
        "raster_dpi": 300,
        "raster_dpi_min": None,
        "max_height_mm": 200.0,  # recommended maximum 16.5 x 20 cm
        "max_mb": 20.0,
        "panel_letters": None,
        "extra": [],
    },
    "eneuro": {
        "name": "eNeuro (Society for Neuroscience)",
        "checked": "2026-09-19",
        "source": "https://www.eneuro.org/content/preparing-manuscript",
        "widths_mm": {"1": 85.0, "1.5": 116.0, "2": 176.0},
        "width_rule": "max",
        "text_pt": (None, None),
        "fonts": None,
        # "For figures in vector-based format, all fonts should be converted to
        # outlines and saved as EPS". matplotlib cannot outline EPS text, so the
        # deliverable is a TIFF at final size, which the journal also accepts.
        "vector_fonts": "outline",
        "min_line_pt": None,
        "formats": ["tif"],
        "raster_dpi": 600,  # colour/greyscale minimum is 300 dpi
        "raster_dpi_min": 300,
        "max_height_mm": None,
        "max_mb": None,
        "panel_letters": None,
        "extra": ["no_box", "no_two_bar"],
    },
    "plos": {
        "name": "PLOS journals (PLOS Biology, PLOS Comput Biol, PLOS ONE, ...)",
        "checked": "2026-09-19",
        "source": "https://journals.plos.org/plosone/s/figures",
        "widths_mm": {"min": 66.8, "text": 132.0, "page": 190.5},
        "width_rule": "range",  # between min and page; "text" aligns with the text column
        "text_pt": (8.0, 12.0),
        "fonts": ["Arial", "Times New Roman", "Times", "Symbol"],
        "vector_fonts": "embed",
        "min_line_pt": None,
        "formats": ["tif"],  # Fig1.tif, ...; LZW compression required, no alpha channel
        "raster_dpi": 300,  # 300-600 dpi allowed
        "raster_dpi_min": 300,
        "max_height_mm": 222.3,
        "max_mb": 10.0,
        "panel_letters": None,
        "extra": [],
    },
    "elife": {
        "name": "eLife",
        "checked": "2026-09-19",
        "source": "https://reviewer.elifesciences.org/author-guide/revised",
        "widths_mm": {"min": 100.0, "page": 200.0},
        "width_rule": "atleast",  # at least 10 cm; a full-page figure at least 20 cm
        "text_pt": (None, None),
        "fonts": None,
        "vector_fonts": "embed",
        "min_line_pt": None,
        "formats": ["pdf", "tif"],
        "raster_dpi": 300,
        "raster_dpi_min": 300,
        "max_height_mm": None,
        "max_mb": None,
        "panel_letters": None,
        "extra": [],
    },
}

# Nature Neuroscience: the Nature Portfolio figure guide (row "nature") applies
# to every Portfolio title; the journal's own page adds a 180 mm maximum width
# and states no single-column width, so "1" is the Portfolio's 89 mm.
JOURNALS["natneurosci"] = {
    **JOURNALS["nature"],
    "name": "Nature Neuroscience (Nature Portfolio guide + the journal's own page)",
    "source": JOURNALS["nature"]["source"] + " ; https://www.nature.com/neuro/submission-guidelines/aip-and-formatting",
    "widths_mm": {"1": 89.0, "max": 180.0},  # "1" from the Portfolio guide, "max" from the journal page
    "raster_dpi": 300,
    "max_height_mm": None,  # the 247 mm page depth is stated for Nature itself
}

UNIVERSAL_MIN_PT = 5.0  # below this nothing is legible in print, whatever the journal says
BAD_CMAPS = {"jet", "rainbow", "hsv", "gist_rainbow", "nipy_spectral", "gist_ncar", "turbo"}
SANS = ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "TeX Gyre Heros", "DejaVu Sans"]


def target_width_mm(journal: str | None, column: str | None = None, width_mm: float | None = None) -> float:
    """Width the figure must be built at, in mm."""
    if width_mm is not None:
        return float(width_mm)
    spec = JOURNALS[journal]
    widths = spec["widths_mm"]
    if column not in widths:
        raise ValueError(f"{journal}: column must be one of {sorted(widths)}, got {column!r}")
    return widths[column]


def size(journal, column=None, height_mm=60.0, width_mm=None):
    """figsize in inches for plt.figure/plt.subplots."""
    return (target_width_mm(journal, column, width_mm) / MM_PER_IN, height_mm / MM_PER_IN)


def use_style(journal: str | None = None, base_pt: float | None = None) -> str | None:
    """Journal fonts and sizes, Type 42 font embedding. Returns the font actually used."""
    import matplotlib as mpl
    from matplotlib import font_manager

    spec = JOURNALS.get(journal, {}) if journal else {}
    wanted = (spec.get("fonts") or []) + [f for f in SANS if f not in (spec.get("fonts") or [])]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in wanted if f in installed), None)
    lo, hi = spec.get("text_pt", (None, None)) or (None, None)
    if base_pt is None:
        base_pt = 7.0 if hi is None else min(hi, max(lo or 0, 7.0))
        if lo and lo > base_pt:
            base_pt = lo
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [family] + wanted if family else wanted,
        "font.size": base_pt,
        "axes.titlesize": base_pt,
        "axes.labelsize": base_pt,
        "xtick.labelsize": base_pt - 1 if (lo is None or base_pt - 1 >= lo) else base_pt,
        "ytick.labelsize": base_pt - 1 if (lo is None or base_pt - 1 >= lo) else base_pt,
        "legend.fontsize": base_pt - 1 if (lo is None or base_pt - 1 >= lo) else base_pt,
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 150,
        "savefig.dpi": spec.get("raster_dpi", 600),
    })
    return family


def subplots(journal, column=None, height_mm=60.0, width_mm=None, **kwargs):
    """plt.subplots at the final size, constrained layout unless layout= is given.

    For a figure laid out on one master GridSpec (figure-alignment), skip this and
    pass figsize=size(...) to plt.figure instead.
    """
    import matplotlib.pyplot as plt

    kwargs.setdefault("layout", "constrained")
    return plt.subplots(figsize=size(journal, column, height_mm, width_mm), **kwargs)


def panel_labels(axes, labels=None, journal=None, uppercase=False):
    """Bold panel letters at the top-left of each axes, outside the plot area."""
    import string

    axes = list(getattr(axes, "flat", axes))
    letters = string.ascii_uppercase if uppercase else string.ascii_lowercase
    labels = labels or letters[: len(axes)]
    pt = JOURNALS.get(journal, {}).get("panel_label_pt", 8.0) if journal else 8.0
    for ax, lab in zip(axes, labels):
        ax.text(-0.02, 1.02, lab, transform=ax.transAxes, fontsize=pt, fontweight="bold",
                ha="right", va="bottom")


# --------------------------------------------------------------------------- audit


def _looks_numeric(label: str) -> bool:
    cleaned = label.replace("−", "-").replace("$", "").replace("{", "").replace("}", "")
    cleaned = cleaned.replace("\\mathdefault", "").replace("^", "e").replace("%", "").strip()
    try:
        float(cleaned.replace("10e", "1e"))
        return True
    except ValueError:
        return False


def _font_name(text) -> str:
    from matplotlib import font_manager

    try:
        path = font_manager.findfont(text.get_fontproperties(), fallback_to_default=True)
        return font_manager.FontProperties(fname=path).get_name()
    except Exception:  # noqa: BLE001 - a lookup failure must not stop the audit
        return "?"


def _width_scale(actual_mm, journal, column, width_mm):
    """(scale to the printed size, error message or None) for a given canvas width."""
    if width_mm is None and not (journal and column):
        return 1.0, None
    target = target_width_mm(journal, column, width_mm)
    rule = JOURNALS.get(journal, {}).get("width_rule", "exact") if width_mm is None else "exact"
    if column == "max":
        rule = "upto"
    if rule == "upto":
        if actual_mm <= target + 0.5:
            return 1.0, None
        scale = target / actual_mm
        return scale, (f"width {actual_mm:.1f} mm exceeds the {journal} maximum of {target:g} mm: "
                       f"it will be rescaled by {scale:.2f}, fonts and lines with it")
    ok = (target - 3.0 <= actual_mm <= target + 0.5) if rule == "max" else abs(actual_mm - target) <= 0.5
    if ok:
        return 1.0, None
    scale = target / actual_mm
    return scale, (f"width {actual_mm:.1f} mm, but the {column or 'requested'}-column slot is {target:.1f} mm: "
                   f"the journal will rescale it by {scale:.2f} and every font and line with it")


def _is_colorbar(ax) -> bool:
    return getattr(ax, "_colorbar", None) is not None or ax.get_label() == "<colorbar>"


def audit(fig, journal: str | None = None, column: str | None = None, width_mm: float | None = None) -> list[tuple[str, str]]:
    """Return [(severity, message)]; severity is 'ERROR' (must fix) or 'CHECK' (look and decide)."""
    import matplotlib as mpl

    spec = JOURNALS.get(journal, {}) if journal else {}
    out: list[tuple[str, str]] = []
    err = lambda m: out.append(("ERROR", m))  # noqa: E731
    chk = lambda m: out.append(("CHECK", m))  # noqa: E731

    # 1. Width. Everything else is measured at the printed size, so the scale
    #    factor between the canvas and the page is applied to every size below.
    actual = fig.get_size_inches()[0] * MM_PER_IN
    rule = spec.get("width_rule", "exact")
    widths = spec.get("widths_mm", {})
    scale, message = _width_scale(actual, journal, column, width_mm)
    if message:
        err(message)
    if width_mm is not None or (journal and column):
        pass
    elif rule == "range":
        if not widths["min"] - 0.5 <= actual <= widths["page"] + 0.5:
            scale = min(max(actual, widths["min"]), widths["page"]) / actual
            err(f"width {actual:.1f} mm is outside the allowed {widths['min']}-{widths['page']} mm")
    elif rule == "atleast":
        if actual < widths["min"] - 0.5:
            err(f"width {actual:.1f} mm is below the journal minimum of {widths['min']} mm")
    else:
        chk("no journal/column given: width not checked against a column")
    max_h = spec.get("max_height_mm")
    height = fig.get_size_inches()[1] * MM_PER_IN * scale
    if max_h and height > max_h + 0.5:
        err(f"height {height:.1f} mm when printed exceeds the {journal} maximum of {max_h:g} mm")

    # 2. Font embedding for vector output.
    if mpl.rcParams["pdf.fonttype"] != 42 or mpl.rcParams["ps.fonttype"] != 42:
        err("pdf/ps.fonttype is not 42: PDF/EPS text embeds as Type 3, which journals and arXiv reject "
            "- call use_style() before plotting")

    texts = [t for t in fig.findobj(mpl.text.Text) if t.get_visible() and t.get_text().strip()]

    # 3. Font family actually resolved (not what was asked for).
    used = {}
    for t in texts:
        used.setdefault(_font_name(t), []).append(t.get_text()[:20])
    allowed = spec.get("fonts")
    if allowed:
        bad = {f: s for f, s in used.items() if f not in allowed}
        for f, s in bad.items():
            err(f"font {f!r} is not allowed by {journal} ({', '.join(allowed)}); e.g. {s[:2]}"
                + (" - the allowed font is not installed on this machine: say so, do not install fonts silently"
                   if f in ("DejaVu Sans", "Liberation Sans") else ""))
    elif any(f.startswith("DejaVu") for f in used):
        chk("text is in matplotlib's default DejaVu Sans: Arial/Helvetica is the safe choice for journals")

    # 4. Text sizes at the printed size.
    lo, hi = spec.get("text_pt", (None, None)) if spec else (None, None)
    floor = max(lo or 0, UNIVERSAL_MIN_PT)
    label_pt = spec.get("panel_label_pt")
    small, large = {}, {}
    for t in texts:
        eff = round(t.get_fontsize() * scale, 1)
        if eff < floor - 0.05:
            small.setdefault(eff, []).append(t.get_text()[:25])
        is_panel_label = label_pt and len(t.get_text().strip()) == 1 and t.get_text().isalpha()
        if hi and eff > hi + 0.05 and not (is_panel_label and eff <= label_pt + 0.05):
            large.setdefault(eff, []).append(t.get_text()[:25])
    for eff, s in sorted(small.items()):
        err(f"text at {eff} pt when printed (minimum {floor:g} pt): {s[:3]}")
    for eff, s in sorted(large.items()):
        err(f"text at {eff} pt when printed ({journal} maximum {hi:g} pt): {s[:3]}")

    # 5. Per-axes content rules.
    panels = [ax for ax in fig.axes if ax.axison and not _is_colorbar(ax)]
    for i, ax in enumerate(panels):
        name = f"panel {i + 1}"
        if ax.get_title() or ax.get_title("left") or ax.get_title("right"):
            err(f"{name}: axes title {(ax.get_title() or ax.get_title('left'))!r} - the figure legend (caption) "
                "carries the title; delete it")
        has_image = bool(ax.images)
        for axis, xy, shared in ((ax.xaxis, "x", ax.get_shared_x_axes()), (ax.yaxis, "y", ax.get_shared_y_axes())):
            label = axis.get_label().get_text()
            ticks = [t.get_text() for t in axis.get_ticklabels() if t.get_text().strip()]
            numeric = bool(ticks) and all(_looks_numeric(t) for t in ticks)
            if not label:
                if any(getattr(s, f"get_{xy}label")() for s in shared.get_siblings(ax) if s is not ax):
                    continue
                if ticks and not numeric and axis.get_scale() == "linear":
                    continue  # categorical ticks label themselves
                if not ticks:
                    continue
                err(f"{name}: {xy} axis has numeric ticks but no label")
            elif numeric and not any(c in label for c in "([") and "#" not in label:
                chk(f"{name}: {xy} label {label!r} has no unit in brackets - add one unless it is unitless")
        for im in ax.images + [c for c in ax.collections if hasattr(c, "get_cmap")]:
            cmap = getattr(im.get_cmap(), "name", "")
            if cmap.removesuffix("_r") in BAD_CMAPS:
                err(f"{name}: colormap {cmap!r} is not perceptually uniform and fails for colour-blind readers "
                    "- use viridis/cividis (one quantity) or a diverging map such as RdBu_r (signed)")
        if "no_box" in spec.get("extra", []) and not has_image:
            if ax.spines["top"].get_visible() or ax.spines["right"].get_visible():
                err(f"{name}: {journal} requires removing the top and right axis lines of graphs")
        if "no_two_bar" in spec.get("extra", []):
            bars = [p for c in ax.containers for p in getattr(c, "patches", [])
                    if isinstance(c, mpl.container.BarContainer)]
            if len(bars) == 2:
                err(f"{name}: a two-bar graph - {journal} does not accept them; state the values in the text "
                    "or show the individual data points instead")

    # 6. Panel letters for multi-panel figures.
    if len(panels) > 1:
        letters = [t for t in texts if len(t.get_text().strip()) in (1, 3)
                   and t.get_text().strip("() ").isalpha() and len(t.get_text().strip("() ")) == 1]
        if len(letters) < len(panels):
            chk(f"{len(panels)} panels but {len(letters)} panel letters - label every panel (panel_labels())")
        elif spec.get("panel_letters") == "lower" and any(t.get_text().strip("() ").isupper() for t in letters):
            err(f"{journal} wants lowercase bold panel letters (a, b), found uppercase")

    # 7. Thin lines.
    min_lw = spec.get("min_line_pt")
    if min_lw:
        thin = set()
        for ax in panels:
            for ln in ax.lines:
                if ln.get_visible() and ln.get_linestyle() not in ("None", "") and ln.get_linewidth() * scale < min_lw:
                    thin.add(round(ln.get_linewidth() * scale, 2))
        if thin:
            err(f"lines at {sorted(thin)} pt when printed ({journal} minimum {min_lw} pt)")

    out.extend(_text_collisions(fig))
    return out


def _text_collisions(fig) -> list[tuple[str, str]]:
    """Text overlapping other text or running off the canvas (adapted from OpenResearch)."""
    import matplotlib as mpl

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    offscreen = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            span = (hi - lo) or 1.0
            for tick in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
                if not lo - span * 1e-6 <= tick.get_loc() <= hi + span * 1e-6:
                    offscreen.update({id(tick.label1), id(tick.label2)})
    hidden = set()
    for ax in fig.axes:
        if not ax.axison:
            for part in (ax.xaxis, ax.yaxis):
                hidden.update(id(t) for t in part.findobj(mpl.text.Text))
    boxes = []
    for t in fig.findobj(mpl.text.Text):
        if id(t) in offscreen or id(t) in hidden or not t.get_visible() or not t.get_text().strip():
            continue
        try:
            box = t.get_window_extent(renderer)
        except (RuntimeError, ValueError):
            continue
        if box.width > 0 and box.height > 0:
            boxes.append((t, box))
    out, tol = [], 1.5
    fb = fig.bbox
    clipped = sorted({t.get_text()[:25] for t, b in boxes
                      if b.x0 < fb.x0 - tol or b.x1 > fb.x1 + tol or b.y0 < fb.y0 - tol or b.y1 > fb.y1 + tol})
    if clipped:
        out.append(("ERROR", f"text runs off the canvas and will be cut: {clipped[:3]}"))
    hits = []
    for i, (ta, ba) in enumerate(boxes):
        for tb, bb in boxes[i + 1:]:
            ov = mpl.transforms.Bbox.intersection(ba, bb)
            if ov is not None and ov.width > tol and ov.height > tol:
                hits.append(f"{ta.get_text()[:22]!r} over {tb.get_text()[:22]!r}")
    if hits:
        more = f" (+{len(hits) - 3} more)" if len(hits) > 3 else ""
        out.append(("ERROR", "overlapping text: " + "; ".join(hits[:3]) + more))
    # Text on top of data: a label inside the axes that covers plotted lines/bars.
    for ax in fig.axes:
        if not ax.axison or _is_colorbar(ax):
            continue
        artists = [a for a in ax.lines + ax.patches if a.get_visible()]
        for t in ax.texts:
            if not t.get_visible() or not t.get_text().strip() or t.get_bbox_patch() is not None:
                continue
            try:
                tb = t.get_window_extent(renderer)
            except (RuntimeError, ValueError):
                continue
            for a in artists:
                try:
                    ab = a.get_window_extent(renderer)
                except (RuntimeError, ValueError, AttributeError):
                    continue
                if isinstance(a, mpl.lines.Line2D):
                    path = a.get_transform().transform_path(a.get_path())
                    if path.intersects_bbox(tb, filled=False):
                        out.append(("CHECK", f"text {t.get_text()[:25]!r} crosses a plotted line"))
                        break
                elif mpl.transforms.Bbox.intersection(tb, ab) is not None and \
                        mpl.transforms.Bbox.intersection(tb, ab).width > tol:
                    out.append(("CHECK", f"text {t.get_text()[:25]!r} sits on a bar/patch"))
                    break
    return out


def report(problems, stem="figure", stream=sys.stderr) -> None:
    errors = [m for s, m in problems if s == "ERROR"]
    checks = [m for s, m in problems if s == "CHECK"]
    if not problems:
        print(f"FIGURE AUDIT {stem}: clean", file=stream)
        return
    print(f"FIGURE AUDIT {stem}: {len(errors)} error(s), {len(checks)} to check", file=stream)
    for m in errors:
        print(f"  ERROR  {m}", file=stream)
    for m in checks:
        print(f"  CHECK  {m}", file=stream)


def save(fig, stem, journal=None, column=None, width_mm=None, formats=None, dpi=None,
         preview=True, close=True):
    """Write the journal's deliverable format(s) at the final size, audit, print the report.

    formats: default = what the journal accepts; pass e.g. ["pdf"] to override.
    dpi: raster resolution; default = the journal's row.
    preview: also write <stem>_preview.png at 150 dpi to look at - never for submission.
    No bbox_inches="tight": trimming changes the physical width the figure was built at.
    Returns the list of (severity, message).
    """
    import matplotlib.pyplot as plt

    spec = JOURNALS.get(journal, {}) if journal else {}
    formats = list(dict.fromkeys(formats or spec.get("formats") or ["pdf"]))
    dpi = dpi or spec.get("raster_dpi", 600)
    parent = os.path.dirname(stem)
    if parent:
        os.makedirs(parent, exist_ok=True)
    paths = []
    for ext in formats:
        path = f"{stem}.{ext}"
        if ext in ("tif", "tiff"):
            _save_tiff(fig, path, dpi)
        elif ext in ("png", "jpg", "jpeg"):
            fig.savefig(path, dpi=dpi)
        else:
            fig.savefig(path, format=ext)
        paths.append(path)
    if preview:
        fig.savefig(f"{stem}_preview.png", dpi=150)
    try:
        problems = audit(fig, journal, column, width_mm)
    finally:
        if close:
            plt.close(fig)
    max_mb = spec.get("max_mb")
    for path in paths:
        mb = os.path.getsize(path) / 1e6
        if max_mb and mb > max_mb:
            problems.append(("ERROR", f"{path} is {mb:.1f} MB, over the {journal} limit of {max_mb:g} MB "
                                      "- lower dpi= (not below the journal minimum)"))
    report(problems, stem)
    note = f"; preview {stem}_preview.png (not for submission)" if preview else ""
    print("written: " + ", ".join(paths) + note, file=sys.stderr)
    return problems


def _save_tiff(fig, path, dpi):
    """RGB (no alpha), LZW-compressed TIFF with the resolution tags set.

    Written by hand rather than through Pillow: Pillow compresses TIFF through
    libtiff, and a broken libtiff (seen in Anaconda on Windows) kills the Python
    process outright instead of raising. PLOS requires LZW; no journal here
    rejects it.
    """
    import io

    import numpy as np
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    rgba = Image.open(buf).convert("RGBA")
    rgb = Image.new("RGB", rgba.size, "white")
    rgb.paste(rgba, mask=rgba.getchannel("A"))
    _write_tiff_lzw(path, np.asarray(rgb), dpi)


def _lzw(data: bytes) -> bytes:
    """TIFF LZW as libtiff writes it: MSB-first codes of 9-12 bits."""
    out = bytearray()
    acc = nbits = 0
    width = 9

    def emit(code):
        nonlocal acc, nbits
        acc = (acc << width) | code
        nbits += width
        while nbits >= 8:
            nbits -= 8
            out.append((acc >> nbits) & 0xFF)
        acc &= (1 << nbits) - 1

    emit(256)  # Clear
    table = {}
    next_code = 258
    it = iter(data)
    w = next(it, None)
    if w is not None:
        for b in it:
            code = table.get((w, b))
            if code is not None:
                w = code
                continue
            emit(w)
            table[(w, b)] = next_code
            next_code += 1
            w = b
            if next_code == 4094:  # table full: Clear at 12 bits, start over
                emit(256)
                table.clear()
                next_code = 258
                width = 9
            elif next_code > (1 << width) - 1:
                width += 1
        emit(w)
        # the decoder adds one more entry after reading w; widen if that crosses
        next_code += 1
        if next_code > (1 << width) - 1 and width < 12:
            width += 1
    emit(257)  # End of information
    if nbits:
        out.append((acc << (8 - nbits)) & 0xFF)
    return bytes(out)


def _write_tiff_lzw(path, rgb, dpi, rows_per_strip=16):
    import struct

    import numpy as np

    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape
    strips = [_lzw(rgb[r:r + rows_per_strip].tobytes()) for r in range(0, h, rows_per_strip)]
    n = len(strips)
    blob = bytearray()
    offsets = []
    pos = 8
    for strip in strips:
        offsets.append(pos)
        blob.extend(strip)
        pos += len(strip)

    def extra(data):
        nonlocal pos
        if pos % 2:
            blob.append(0)
            pos += 1
        at = pos
        blob.extend(data)
        pos += len(data)
        return at

    bps_at = extra(struct.pack("<HHH", 8, 8, 8))
    so_at = extra(struct.pack(f"<{n}I", *offsets))
    sbc_at = extra(struct.pack(f"<{n}I", *[len(s) for s in strips]))
    res_at = extra(struct.pack("<II", int(round(dpi)), 1))
    if pos % 2:
        blob.append(0)
        pos += 1
    ifd = [  # (tag, type 3=SHORT 4=LONG 5=RATIONAL, count, value or offset)
        (256, 4, 1, w), (257, 4, 1, h), (258, 3, 3, bps_at), (259, 3, 1, 5), (262, 3, 1, 2),
        (273, 4, n, so_at if n > 1 else offsets[0]), (277, 3, 1, 3), (278, 4, 1, rows_per_strip),
        (279, 4, n, sbc_at if n > 1 else len(strips[0])), (282, 5, 1, res_at), (283, 5, 1, res_at),
        (284, 3, 1, 1), (296, 3, 1, 2),
    ]
    body = struct.pack("<H", len(ifd))
    for tag, typ, cnt, val in ifd:
        if typ == 3 and cnt == 1:
            body += struct.pack("<HHIHH", tag, typ, cnt, val, 0)
        else:
            body += struct.pack("<HHII", tag, typ, cnt, val)
    body += struct.pack("<I", 0)
    with open(path, "wb") as f:
        f.write(b"II*\x00" + struct.pack("<I", pos) + bytes(blob) + body)


# ----------------------------------------------------------------- existing PDF


def audit_pdf(path, journal=None, column=None, width_mm=None) -> list[tuple[str, str]]:
    """Audit a figure that already exists as a PDF (Prism, Illustrator, R). Needs pymupdf."""
    try:
        import fitz  # pymupdf
    except ImportError:
        return [("ERROR", "pymupdf is not installed: pip install pymupdf")]
    spec = JOURNALS.get(journal, {}) if journal else {}
    out = []
    doc = fitz.open(path)
    page = doc[0]
    actual = page.rect.width / 72 * MM_PER_IN
    scale, message = _width_scale(actual, journal, column, width_mm)
    if message:
        out.append(("ERROR", "page " + message))
    fonts = page.get_fonts(full=True)
    for f in fonts:
        ftype, basefont, ext = f[2], f[3], f[1]
        if ftype == "Type3":
            out.append(("ERROR", f"font {basefont!r} is Type 3 - re-export with TrueType/Type 42 fonts"))
        if ext == "n/a" and ftype not in ("Type3",):
            out.append(("ERROR", f"font {basefont!r} is not embedded"))
    allowed = spec.get("fonts")
    if allowed:
        for f in fonts:
            base = f[3].split("+")[-1].replace("MT", "").split("-")[0].replace(",", "")
            if not any(base.lower().startswith(a.lower().replace(" ", "")) for a in allowed):
                out.append(("ERROR", f"font {f[3]!r} not in {journal}'s list ({', '.join(allowed)})"))
    if spec.get("vector_fonts") == "outline" and fonts:
        out.append(("ERROR", f"{journal} wants text converted to outlines in vector files (or a TIFF); "
                              "this PDF has live fonts"))
    lo, hi = spec.get("text_pt", (None, None)) if spec else (None, None)
    floor = max(lo or 0, UNIVERSAL_MIN_PT)
    sizes = {}
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if span["text"].strip():
                    sizes.setdefault(round(span["size"] * scale, 1), []).append(span["text"][:20])
    for s, t in sorted(sizes.items()):
        if s < floor - 0.05:
            out.append(("ERROR", f"text at {s} pt when printed (minimum {floor:g} pt): {t[:3]}"))
        elif hi and s > hi + 0.05 and not (spec.get("panel_label_pt") and s <= spec["panel_label_pt"] + 0.05
                                           and all(len(x.strip()) == 1 for x in t)):
            out.append(("ERROR", f"text at {s} pt when printed ({journal} maximum {hi:g} pt): {t[:3]}"))
    singles = {(x.strip(), s) for s, t in sizes.items() for x in t if len(x.strip()) == 1 and x.strip().isalpha()}
    if spec.get("panel_letters") == "lower" and any(x.isupper() for x, _ in singles):
        out.append(("ERROR", f"{journal} wants lowercase bold panel letters (a, b); found "
                             f"{sorted({x for x, _ in singles})}"))
    label_pt = spec.get("panel_label_pt")
    if label_pt:
        off = sorted({s for _, s in singles if abs(s - label_pt) > 0.3})
        if off:
            out.append(("CHECK", f"single letters at {off} pt when printed; {journal} panel letters are "
                                 f"{label_pt:g} pt bold - if these are panel letters, resize them"))
    min_lw = spec.get("min_line_pt")
    if min_lw:
        thin = sorted({round(d["width"] * scale, 2) for d in page.get_drawings()
                       if d.get("color") is not None and d.get("width") and d["width"] * scale < min_lw})
        if thin:
            out.append(("ERROR", f"stroked lines at {thin[:5]} pt when printed ({journal} minimum {min_lw} pt)"))
    if not fonts and not sizes:
        out.append(("CHECK", "no live text found: either fully outlined or a raster image inside a PDF"))
    return out


def audit_tiff(path, journal=None, column=None, width_mm=None) -> list[tuple[str, str]]:
    """Check a TIFF's header (size, dpi, colour mode, compression, file size) without decoding pixels.

    Reading the pixels of an LZW TIFF goes through libtiff too, which can crash
    Python on a broken installation; the header alone is parsed by Pillow itself.
    """
    from PIL import Image

    spec = JOURNALS.get(journal, {}) if journal else {}
    out = []
    with Image.open(path) as im:  # lazy: header only
        w_px, h_px = im.size
        dpi = float((im.info.get("dpi") or (0, 0))[0])
        mode = im.mode
        compression = im.info.get("compression", "raw")
    if not dpi:
        return [("ERROR", "no resolution (dpi) stored in the file - the printed size is undefined")]
    actual = w_px / dpi * MM_PER_IN
    _, message = _width_scale(actual, journal, column, width_mm)
    if message:
        out.append(("ERROR", message))
    need = spec.get("raster_dpi_min")
    if need and dpi < need - 0.5:
        out.append(("ERROR", f"{dpi:g} dpi, below the {need} dpi minimum"))
    if "A" in mode:
        out.append(("ERROR", f"colour mode {mode} has an alpha channel - journals want RGB"))
    if journal == "plos" and compression != "tiff_lzw":
        out.append(("ERROR", f"compression {compression!r}: PLOS requires LZW"))
    mb = os.path.getsize(path) / 1e6
    if spec.get("max_mb") and mb > spec["max_mb"]:
        out.append(("ERROR", f"{mb:.1f} MB, over the {journal} limit of {spec['max_mb']:g} MB"))
    max_h = spec.get("max_height_mm")
    if max_h and h_px / dpi * MM_PER_IN > max_h + 0.5:
        out.append(("ERROR", f"height {h_px / dpi * MM_PER_IN:.1f} mm exceeds the {journal} maximum of {max_h:g} mm"))
    out.append(("CHECK", f"{w_px}x{h_px} px at {dpi:g} dpi = {actual:.1f} x {h_px / dpi * MM_PER_IN:.1f} mm, "
                         f"{mode}, {compression}, {mb:.2f} MB; text sizes, fonts and overlaps cannot be read "
                         "from pixels - audit the figure before rasterizing"))
    return out


def _cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("journals", help="list journals, widths and sources")
    for name, text in (("pdf", "audit an existing PDF figure"), ("tiff", "check a TIFF's header")):
        q = sub.add_parser(name, help=text)
        q.add_argument("path")
        q.add_argument("--journal", choices=sorted(JOURNALS))
        q.add_argument("--column")
        q.add_argument("--width-mm", type=float)
    a = p.parse_args(argv)
    if a.cmd == "journals":
        for key, s in JOURNALS.items():
            w = ", ".join(f"{k}: {v:g} mm" for k, v in s["widths_mm"].items())
            print(f"{key:7s} {s['name']}\n"
                  f"        column keys -> widths ({s['width_rule']}): {w}; max height {s['max_height_mm']} mm\n"
                  f"        text pt (min, max) {s['text_pt']}; fonts {s['fonts']}; vector text: {s['vector_fonts']}; "
                  f"panel letters: {s['panel_letters'] or 'not stated'}\n"
                  f"        submit {s['formats']} @ {s['raster_dpi']} dpi; max file {s['max_mb']} MB; "
                  f"other rules {s['extra']}\n        checked {s['checked']}: {s['source']}")
        return 0
    check = audit_pdf if a.cmd == "pdf" else audit_tiff
    problems = check(a.path, a.journal, a.column, a.width_mm)
    report(problems, a.path, stream=sys.stdout)
    return 1 if any(s == "ERROR" for s, _ in problems) else 0


if __name__ == "__main__":
    sys.exit(_cli())
