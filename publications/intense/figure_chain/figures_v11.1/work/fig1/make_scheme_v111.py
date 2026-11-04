"""Scheme of Fig. 1c with the stage labels spelled out: 'st.1: quick test' / 'st.2: full-scale' ->
'Stage 1: quick test' / 'Stage 2: full-scale'.

usage: python -X utf8 -B figure_chain/figures_v11.1/work/fig1/make_scheme_v111.py
       -> figures_v11.1/work/fig1/fig1c_scheme_v111.png, scheme_check.json under
          [figures] output_dir/figure_chain (read by make_fig1_v111.py)

The scheme is a raster image ([figures] fig1_scheme, assets/fig1c_scheme.png, the file
fig1_delegated.py reads) without an editable source, set in Roboto. The two label lines are erased
and set again in Roboto Regular at the size that reproduces the old lines (fitted below: width and
height of both old lines), on the same baselines, centered on a common axis as before. The new lines
are longer, so their common axis moves right until the wider line starts where the old wider line
started; there is free paper to the right, between the two boxes. Everything else is untouched.
Roboto Regular must be installed where matplotlib finds fonts; it is looked up by its family name.
"""
import functools
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # figure_chain/
import _paths_fig as _F  # noqa: E402  (puts the paper folder on sys.path)

SRC = _F.P.FIG1_SCHEME
OUT = _F.out_dir(HERE) / "fig1c_scheme_v111.png"
OLD = {"st.1: quick test": (1674, 1729, 1, 374), "st.2: full-scale": (1743, 1787, 11, 365)}  # rows, cols
NEW = ("Stage 1: quick test", "Stage 2: full-scale")


@functools.lru_cache(maxsize=None)
def roboto():
    """File of Roboto Regular, the font of the scheme, as matplotlib finds it by family name."""
    from matplotlib import font_manager
    try:
        path = font_manager.findfont(font_manager.FontProperties(family="Roboto", style="normal", weight="regular"),
                                     fallback_to_default=False)
    except ValueError:
        sys.exit("Roboto Regular is not among the fonts matplotlib knows: install the Roboto family "
                 "(and delete the fontlist-*.json cache of matplotlib so that it is found)")
    ft = font_manager.get_font(path)
    if (ft.family_name, ft.style_name) != ("Roboto", "Regular"):
        sys.exit(f"Roboto Regular is needed; matplotlib found {ft.family_name} {ft.style_name} ({path})")
    return Path(path)


def dark_bbox(arr, thr=140):
    ys, xs = np.nonzero(arr < thr)
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def render(text, size, base=200):
    """Text on white with its baseline at row `base`, starting at column 10; returns array, bbox."""
    f = ImageFont.truetype(str(roboto()), size=size)
    c = Image.new("L", (int(size * len(text)) + 40, base + int(size)), 255)
    ImageDraw.Draw(c).text((10, base), text, font=f, fill=0, anchor="ls")
    arr = np.asarray(c).astype(float)
    return arr, dark_bbox(arr)


def main():
    if OUT.exists():
        sys.exit(f"{OUT} exists; not overwriting")
    im = Image.open(SRC)
    assert im.mode == "RGBA" and im.size == (1384, 3017), (im.mode, im.size)
    gray = np.asarray(im.convert("L")).astype(float)
    for text, box in OLD.items():
        assert dark_bbox(gray[box[0] - 2:box[1] + 3, max(box[2] - 2, 0):box[3] + 3]) is not None

    # font size: the one at which both old lines have their measured width and height
    sizes = np.arange(50.0, 65.0, 0.25)
    err = []
    for s in sizes:
        e = 0
        for text, (r0, r1, c0, c1) in OLD.items():
            _, (y0, y1, x0, x1) = render(text, s)
            e += abs((x1 - x0) - (c1 - c0)) + abs((y1 - y0) - (r1 - r0))
        err.append(e)
    size = float(sizes[int(np.argmin(err))])

    # baselines and correlation of the re-rendered old lines with the image (font check)
    check = {}
    baselines = []
    for text, (r0, r1, c0, c1) in OLD.items():
        arr, (y0, y1, x0, x1) = render(text, size)
        base = r1 - (y1 - 200)  # bottom of the dark pixels lies y1 - 200 rows below the baseline
        baselines.append(int(base))
        h, w = r1 - r0 + 1, c1 - c0 + 1
        a = gray[r0:r0 + h, c0:c0 + w]
        b = arr[y0:y0 + h, x0:x0 + w]
        corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.shape == b.shape else None
        check[text] = dict(bbox_rows_cols=[r0, r1, c0, c1], baseline=int(base), corr_with_rerender=corr)

    rgba = np.asarray(im).copy()
    for text, (r0, r1, c0, c1) in OLD.items():
        rgba[r0 - 2:r1 + 3, max(c0 - 2, 0):c1 + 3, :3] = 255
        rgba[r0 - 2:r1 + 3, max(c0 - 2, 0):c1 + 3, 3] = 255
    new = Image.fromarray(rgba, "RGBA")
    d = ImageDraw.Draw(new)
    f = ImageFont.truetype(str(roboto()), size=size)
    widths = [d.textlength(t, font=f) for t in NEW]
    left = min(c0 for _, _, c0, _ in OLD.values())  # where the old wider line started
    axis = left + max(widths) / 2
    for t, base in zip(NEW, baselines):
        d.text((axis, base), t, font=f, fill=(0, 0, 0, 255), anchor="ms")
    new.save(OUT, dpi=im.info.get("dpi", (600, 600)))

    g2 = np.asarray(new.convert("L")).astype(float)
    band = g2[1660:1800, :]
    nb = dark_bbox(band[:, :700])
    changed = np.any(np.asarray(new) != np.asarray(im), axis=2)
    ys, xs = np.nonzero(changed)
    res = dict(source=str(SRC), font=str(roboto()), font_size_px=size, old_lines=check,
               new_lines=list(NEW), new_axis_x=round(axis, 1), new_widths_px=[round(w, 1) for w in widths],
               new_text_bbox_rows_cols=[nb[0] + 1660, nb[1] + 1660, nb[2], nb[3]],
               changed_px_bbox_rows_cols=[int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())],
               size=list(new.size))
    (_F.out_dir(HERE) / "scheme_check.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                                        encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
