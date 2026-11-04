"""Fig. 1.

usage: python -X utf8 -B figure_chain/figures_v11.1/work/fig1/make_fig1_v111.py
       -> figures_v11.1/fig1.png, fig1.pdf and figures_v11.1/work/fig1/fig1_check.json under
          [figures] output_dir/figure_chain

Drawn from the panel scripts of fig1v8_src/fig1/ with two switches:
  a  YLIM_TOP = 1.2, y ticks unchanged: the grey event ticks at 1.12 stay inside their axes instead of
     hanging over the trace above;
  c  SCHEME_PATH = [figures] fig1_scheme_v111: the scheme with 'Stage 1: quick test', 'Stage 2: full-scale'
     (make_scheme_v111.py wrote it, with the Roboto font).
The density plot of panel d is not reproducible between renders: DRIADA adds unseeded noise of 1e-8 to the
calcium when it builds the neurons, so a few pixels of d differ from one render to the next.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
W104 = HERE.parents[2] / "figures_v10.4" / "work"
sys.path.insert(0, str(HERE.parents[2]))  # figure_chain/
import _paths_fig as _F  # noqa: E402  (puts the paper folder on sys.path)
import intense_paths  # noqa: E402

intense_paths.use_driada()
sys.path.insert(0, str(HERE / "fig1v8_src"))
sys.path.insert(1, str(W104))  # overlaps.py, nfp_fps.py (read only)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.text import Text  # noqa: E402

import fig1.fig1_delegated as comp  # noqa: E402
import fig1.fig1a_calcium_traces as f1a  # noqa: E402
import fig1.fig1b_behavioral_traces as f1b  # noqa: E402
import fig1.fig1c_neuron_feature_pair as f1c  # noqa: E402
import fig1.fig1e_scatter_marginals as f1e  # noqa: E402
import overlaps  # noqa: E402

assert Path(comp.__file__).resolve().is_relative_to(HERE), comp.__file__
V104 = _F.data_dir(W104.parent) / "fig1.png"
NEW_PAIRS = intense_paths.CACHE_DIR / "fig1e_real_pairs.npz"
FPS_TABLE = _F.V.get("fps_table")
EVENTS = _F.data_dir(W104) / "events" / "events_NOF_H01_1D.npz"
OUT = _F.out_dir(HERE.parents[1])                # figures_v11.1

HIST_OUTLINE_LW = 1.0
RIGHT_HIST_HEADROOM = 0.30
MARKER_SCALE = 0.45


def fps_of(session):
    with open(FPS_TABLE, newline="") as f:
        return {r["session"]: float(r["fps"]) for r in csv.DictReader(f)}[session]


def px_box(fig, bb, margin=2):
    h = int(round(fig.bbox.height))
    return [max(int(np.floor(h - bb.y1)) - margin, 0), min(int(np.ceil(h - bb.y0)) + margin, h),
            max(int(np.floor(bb.x0)) - margin, 0), int(np.ceil(bb.x1)) + margin]


def panel_boxes(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    out = {}
    for name, sf in fig._subfigs.items():
        bb = [px_box(fig, sf.bbox)]
        for t in sf.findobj(Text):
            if t.get_visible() and t.get_text():
                bb.append(px_box(fig, t.get_window_extent(r)))
        out[name.lower()] = [min(b[0] for b in bb), max(b[1] for b in bb), min(b[2] for b in bb), max(b[3] for b in bb)]
    return out


def diff_in(a, b, box):
    r0, r1, c0, c1 = box
    return int(np.any(a[r0:r1, c0:c1] != b[r0:r1, c0:c1], axis=2).sum())


def settings():
    """Settings of the panels, with the two switches above."""
    f1a.NEURON_LABEL_IDX = {36: 35, 37: 36, 38: 37}
    f1c.NEURON_LABEL = "neuron 101 fluorescence"
    f1e.LEGEND_COUNTS = False
    ev = np.load(EVENTS)
    rows = dict(zip(ev["rows_older_export"].tolist(), ev["rows_aligned"].tolist()))
    assert rows == {36: 35, 37: 36, 38: 37}, rows
    f1a.FPS = float(ev["fps"])
    assert abs(f1a.FPS - fps_of("NOF_H01_1D")) < 1e-9
    f1a.EVENTS = {k: ev[f"n{r}_true_sp"] for k, r in rows.items()}
    f1a.EXACT_SHADING = True
    f1a.EVENT_TICKS = True
    f1b.FPS, f1b.EXACT_SHADING = fps_of("NOF_H01_3D"), True
    f1c.FPS, f1c.EXACT_SHADING = fps_of("NOF_H01_2D"), True
    f1e.HIST_OUTLINE_LW, f1e.RIGHT_HIST_HEADROOM, f1e.MARKER_SCALE = HIST_OUTLINE_LW, RIGHT_HIST_HEADROOM, MARKER_SCALE
    orig_e = f1e.create_fig1e
    comp.create_fig1e = lambda **kw: orig_e(data_path=NEW_PAIRS, **kw)
    f1a.DEFAULT_LABELS = dict(f1a.DEFAULT_LABELS, events="detected events")
    # v11.1
    f1a.YLIM_TOP = 1.2
    comp.SCHEME_PATH = _F.P.get("figures", "fig1_scheme_v111")   # make_scheme_v111.py
    assert comp.SCHEME_PATH.exists()


def ticks_inside(fig):
    """Panel a: every event tick inside its own axes (y limits) and the y ticks of each axes."""
    out = []
    for ax in fig._subfigs["A"].axes:
        ticks = [ln for ln in ax.lines if ln.get_marker() == "|"]
        y = np.concatenate([np.asarray(ln.get_ydata(), float) for ln in ticks]) if ticks else np.array([])
        lo, hi = ax.get_ylim()
        out.append(dict(ylim=[round(lo, 3), round(hi, 3)], yticks=[float(v) for v in ax.get_yticks()],
                        n_ticks=int(len(y)), all_inside=bool(len(y) and np.all((y >= lo) & (y <= hi)))))
    return out


def main():
    for f in ("fig1.png", "fig1.pdf"):
        if (OUT / f).exists():
            sys.exit(f"{OUT / f} exists; not overwriting")
    settings()
    fig = comp.create_fig1_delegated(output_dir=OUT, output_name="fig1", lowercase=True, save_formats=("png", "pdf"))
    boxes = panel_boxes(fig)
    texts = overlaps.text_report(fig)
    ticks = ticks_inside(fig)
    plt.close(fig)
    f = np.asarray(Image.open(OUT / "fig1.png"))
    a = np.asarray(Image.open(V104)) if _F.reference(V104) else None
    same = a is not None and a.shape == f.shape
    res = dict(output=str(OUT / "fig1.png"), shape=list(f.shape), same_shape_as_v10_4=bool(same),
               px_differ_vs_v10_4_by_panel={k: diff_in(a, f, b) for k, b in boxes.items()} if same else None,
               panel_boxes=boxes, panel_a=ticks, text_clashes=texts["clashes"] if isinstance(texts, dict) else texts)
    (_F.out_dir(HERE) / "fig1_check.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str),
                                                      encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("shape", "same_shape_as_v10_4", "px_differ_vs_v10_4_by_panel", "panel_a",
                                          "text_clashes")}, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
