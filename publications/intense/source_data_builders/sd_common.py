"""Helpers shared by the Source Data builders of the INTENSE paper.

Each builder re-draws one figure in memory, with the script and the settings that draw it, saves the
re-drawn PNG to <[figures] output_dir>/source_data/render, compares it pixel for pixel with the figure the
figure script wrote (when that file exists), and writes the values of the drawn matplotlib objects into one
workbook per figure with source_data.SourceData. Nothing is recomputed beyond what the figure scripts
compute while drawing.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "figure_chain"))
import _paths_fig as _F  # noqa: E402  (puts the paper folder on sys.path)
import intense_paths as P  # noqa: E402

PAPER = P.PAPER_DIR
CHAIN = _F.CHAIN
ART = _F.D                                     # [v11] figure_artifacts: what the producers of figure_chain wrote
SAVED = _F.SAVED                               # figure_chain/saved_inputs
WORK = CHAIN / "figures_v11.1" / "work"        # the figure scripts
FIG = _F.out_dir(CHAIN / "figures_v11.1")      # the figures they write
OUT = P.SOURCE_DATA_DIR
RENDER = OUT / "render"                        # re-drawn figures and the log of each builder
OUT.mkdir(parents=True, exist_ok=True)


def load_module(name, path):
    """Import a module from its file without putting its folder on sys.path."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SourceData = load_module("source_data", PAPER / "source_data.py").SourceData


def rel(path):
    """Path as written in the provenance of the logs: relative to [v11] figure_artifacts or to the
    figure caches, else the file name."""
    p = Path(path).resolve()
    for root, tag in ((ART, ""), (P.CACHE_DIR, "<[figures] cache_dir>/"), (PAPER, "")):
        try:
            return tag + p.relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            pass
    return p.name


def render_target(name):
    RENDER.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        f = RENDER / f"{name}.{ext}"
        if f.exists():
            f.unlink()          # an earlier render of this builder; the composers refuse to overwrite
    return RENDER


def compare_png(new, ref, boxes=None):
    """Pixel comparison of the re-drawn PNG with the figure the figure script wrote, when it exists."""
    from PIL import Image
    if not Path(ref).exists():
        print(f"[SKIPPED] comparison with the drawn figure: {ref} not found")
        return dict(published=rel(ref), reference_missing=True)
    a, b = np.asarray(Image.open(ref)), np.asarray(Image.open(new))
    res = dict(published=rel(ref), redrawn=rel(new), same_shape=bool(a.shape == b.shape))
    if a.shape == b.shape:
        d = np.any(a != b, axis=2)
        res["identical"] = bool(not d.any())
        res["n_px_differ"] = int(d.sum())
        if d.any():
            ys, xs = np.nonzero(d)
            res["differing_bbox_rows_cols"] = [int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1]
        if boxes:
            res["n_px_differ_by_panel"] = {k: int(d[r0:r1, c0:c1].sum()) for k, (r0, r1, c0, c1) in boxes.items()}
    return res


def px_box(fig, bb, margin=2):
    h = int(round(fig.bbox.height))
    return [max(int(np.floor(h - bb.y1)) - margin, 0), min(int(np.ceil(h - bb.y0)) + margin, h),
            max(int(np.floor(bb.x0)) - margin, 0), int(np.ceil(bb.x1)) + margin]


def subfig_boxes(fig, subfigs):
    """Pixel box (rows, cols) of every subfigure, for the comparison by panel."""
    fig.canvas.draw()
    return {k: px_box(fig, sf.bbox) for k, sf in subfigs.items()}


def spans(ax):
    """Start and end (data units) of every shaded period (axvspan) of an axes, in time order."""
    out = []
    for p in ax.patches:
        if hasattr(p, "get_x") and hasattr(p, "get_width"):
            x0 = float(p.get_x())
            out.append((x0, x0 + float(p.get_width())))
        else:
            xy = np.asarray(p.get_xy(), float)
            out.append((float(xy[:, 0].min()), float(xy[:, 0].max())))
    return sorted(out)


def runs(binary):
    """First frame and last frame + 1 of every run of ones."""
    b = (np.asarray(binary) == 1).astype(np.int8)
    e = np.flatnonzero(np.diff(np.r_[0, b, 0]) != 0)
    return e[0::2], e[1::2]


def data_lines(ax):
    """Lines of an axes that carry data (more than two points), in drawing order."""
    return [ln for ln in ax.get_lines() if len(np.asarray(ln.get_xdata())) > 2]


def write_log(name, res):
    RENDER.mkdir(parents=True, exist_ok=True)
    (RENDER / f"{name}_build.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str),
                                               encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False, default=str))


def assert_equal(a, b, what, atol=0.0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, (what, a.shape, b.shape)
    if atol == 0:
        assert np.array_equal(a, b, equal_nan=True), what
    else:
        assert np.allclose(a, b, atol=atol, rtol=0, equal_nan=True), (what, float(np.nanmax(np.abs(a - b))))


def frame_table(**cols):
    return pd.DataFrame({k: pd.Series(v) for k, v in cols.items()})
