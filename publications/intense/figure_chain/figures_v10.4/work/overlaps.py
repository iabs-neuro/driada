"""Label clashes measured on the drawn figure, not by eye.

Every text that is drawn (tick labels inside the view, axis labels, titles, texts of the axes,
legends, figure and subfigure texts) is taken with its box in pixels of the saved canvas. Two
texts clash when their boxes intersect; a text is cut when its box leaves the canvas. The boxes
are those matplotlib uses for layout, so they include the ascent and descent of the font: a clash
here is a clash of the boxes, which is how the earlier figure work counted them. Boxes that
share at most 1.5 pixels (neighbouring tick labels of a heatmap) only touch and are not counted.
The box of an annotation is that of its text, without the arrow.
"""
from collections import Counter

import numpy as np
from matplotlib.text import Text


def _drawn_tick_labels(axis):
    try:
        ticks = axis._update_ticks()  # the ticks matplotlib draws: those inside the view interval
    except Exception:  # pragma: no cover
        ticks = axis.get_major_ticks()
    for t in ticks:
        for lab in (t.label1, t.label2):
            if lab.get_visible() and lab.get_text().strip():
                yield lab


def _axes_name(ax):
    for s in (ax.get_title(), ax.get_ylabel(), ax.get_xlabel(), ax.get_label()):
        if s and s.strip():
            return s.strip().replace("\n", " ")
    return type(ax).__name__


def drawn_texts(fig):
    """[(Text, kind, owner)] of every text drawn on the figure."""
    seen, out = set(), []

    def add(t, kind, owner):
        if t is None or id(t) in seen or not isinstance(t, Text):
            return
        if not (t.get_visible() and t.get_text().strip()):
            return
        seen.add(id(t))
        out.append((t, kind, owner))

    for ax in fig.get_axes():
        if not ax.get_visible():
            continue
        owner = _axes_name(ax)
        if ax.axison:
            for axis, nm in ((ax.xaxis, "x"), (ax.yaxis, "y")):
                if not axis.get_visible():  # e.g. the x axis of a twinx axes is not drawn
                    continue
                for lab in _drawn_tick_labels(axis):
                    add(lab, f"{nm} tick label", owner)
                add(axis.label, f"{nm} axis label", owner)
                add(axis.get_offset_text(), f"{nm} offset", owner)
        for t in (ax.title, ax._left_title, ax._right_title):
            add(t, "title", owner)
        for t in ax.texts:
            add(t, "text", owner)
        leg = ax.get_legend()
        if leg is not None and leg.get_visible():
            for t in leg.get_texts():
                add(t, "legend", owner)
            add(leg.get_title(), "legend title", owner)

    def walk(f, owner):
        for t in f.texts:
            add(t, "figure text", owner)
        for leg in f.legends:
            for t in leg.get_texts():
                add(t, "legend", owner)
        for sf in getattr(f, "subfigs", []):
            walk(sf, "subfigure")

    walk(fig, "figure")
    return out


def _px(bb, height):
    """Box as [row0, row1, col0, col1] of the saved image."""
    return [int(np.floor(height - bb.y1)), int(np.ceil(height - bb.y0)), int(np.floor(bb.x0)), int(np.ceil(bb.x1))]


def text_report(fig, region=None):
    """Clashing text pairs and texts leaving the canvas.

    region: optional [row0, row1, col0, col1]; only texts whose box centre lies inside are kept.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    H, W = fig.bbox.height, fig.bbox.width
    items = []
    for t, kind, owner in drawn_texts(fig):
        bb = Text.get_window_extent(t, r)
        if bb.width <= 0 or bb.height <= 0:
            continue
        box = _px(bb, H)
        if region is not None:
            cy, cx = (box[0] + box[1]) / 2, (box[2] + box[3]) / 2
            if not (region[0] <= cy < region[1] and region[2] <= cx < region[3]):
                continue
        items.append(dict(text=t.get_text(), kind=kind, owner=owner, box=box, bb=bb))
    clashes = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i]["bb"], items[j]["bb"]
            w = min(a.x1, b.x1) - max(a.x0, b.x0)
            h = min(a.y1, b.y1) - max(a.y0, b.y0)
            if w > 1.5 and h > 1.5:
                clashes.append(dict(a=f"{items[i]['kind']} '{items[i]['text']}' ({items[i]['owner']})",
                                    b=f"{items[j]['kind']} '{items[j]['text']}' ({items[j]['owner']})",
                                    overlap_px=[round(float(w), 1), round(float(h), 1)]))
    cut = [dict(text=f"{it['kind']} '{it['text']}' ({it['owner']})", box_rows_cols=it["box"])
           for it in items if it["bb"].x0 < 0 or it["bb"].y0 < 0 or it["bb"].x1 > W or it["bb"].y1 > H]
    return dict(n_texts=len(items), clashes=clashes, cut_by_canvas=cut)


def clash_keys(report):
    return Counter(tuple(sorted((c["a"], c["b"]))) for c in report["clashes"])


def diff_reports(before, after):
    """Clashes present only before (removed) and only after (new)."""
    kb, ka = clash_keys(before), clash_keys(after)
    return dict(removed=[" | ".join(k) for k in (kb - ka)], new=[" | ".join(k) for k in (ka - kb)],
                n_before=sum(kb.values()), n_after=sum(ka.values()))


def text_vs_lines(ax, texts, renderer):
    """For each given Text of ax: the lines of ax whose drawn path meets its box."""
    from matplotlib.path import Path as MPath
    out = []
    for t in texts:
        bb = t.get_window_extent(renderer)
        hits = []
        for ln in ax.lines:
            xy = ln.get_transform().transform(np.column_stack(ln.get_data()))
            if len(xy) > 1 and MPath(xy).intersects_bbox(bb, filled=False):
                hits.append(ln.get_label() or "line")
        out.append(dict(text=t.get_text(), lines_met=hits))
    return out
