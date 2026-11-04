"""Fig. 2.

usage (from the paper folder; outputs under [figures] output_dir/figure_chain/figures_v11.1):
       python -X utf8 -B figure_chain/figures_v11.1/work/fig2/panel_b_ticks.py   # first: the two
       python -X utf8 -B figure_chain/figures_v11.1/work/fig2/panel_f_data.py    # inputs of cache/
       python -X utf8 -B figure_chain/figures_v11.1/work/fig2/make_fig2_v111.py control
            # every switch as in v11 -> work/fig2/out/fig2_control.png, must equal figures_v11/fig2.png
       python -X utf8 -B figure_chain/figures_v11.1/work/fig2/make_fig2_v111.py trial NAME [E_WSPACE]
            # v11.1 settings -> work/fig2/out/NAME.png
       python -X utf8 -B figure_chain/figures_v11.1/work/fig2/make_fig2_v111.py final
            # -> fig2.png, fig2.pdf, work/fig2/verify/fig2_v111.json

Everything is figures_v10.4/work/make_fig2_v104.py in mode B (same data, caches and settings) except
four panel modules, copied to src/ with switches, and the data file of panels c, e, f, rebuilt by
panel_f_data.py with the curves of 2f left on
a longer grid (all other arrays copied). The copies replace the originals at import. The two files of
cache/ (fig2b_cbar_ticks.json, fig2_v111_data_B.npz) are written by panel_b_ticks.py and panel_f_data.py;
the data of panel b are saved in figure_chain/saved_inputs (README.md).
  a  title of the frequency axis 'frequency, Hz' (the row frequencies are in Hz at the session rate);
  b  ticks of the colour bar at calcium values 0, 1, 10, 100, placed where the colour of that calcium
     lies (the colour runs along log(calcium + 0.05), rescaled; panel_b_ticks.py);
  d  each activity map normalized to [0, 1] as in Methods, colour scale 0-1;
  e  y-axis titles 'MSE', 'PSNR, dB', 'SSIM', the room between the plots widened so that each title
     clears its tick labels and the plot to its left;
  f  left plot: x-axis to -log10 p = 40, curves on a grid up to 30, drawn while at least 20 cells
     are at the level (rule of v11).
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = Path(__file__).resolve().parent
W104 = HERE.parents[2] / "figures_v10.4" / "work"
sys.path.insert(0, str(W104))
import make_fig2_v104 as m  # noqa: E402  (loads DRIADA 1.2.2 and the v10.4 panel scripts)
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

V11 = m._F.D / "figures_v11" / "fig2.png"
FINAL_DIR = m._F.out_dir(HERE.parents[1])       # figures_v11.1
WORK_OUT = m._F.out_dir(HERE)

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

E_YLABELS = {"MSE": "MSE", "PSNR": "PSNR, dB", "SSIM": "SSIM"}
# room between the plots of 2e (0.55 in v11): with it each y-axis title stands as far from the axes on
# its left as from its own tick labels, 4 pt (verify/fig2_v111.json -> panel_e_geometry; trial renders
# at 0.55 and 0.85 in out/: at 0.55 the titles of PSNR and SSIM run into the plot on their left)
E_WSPACE = 0.96


def load_copy(name):
    spec = importlib.util.spec_from_file_location(f"{name}_v111", HERE / "src" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def substitute():
    """Put the copies of src/ in place of the v10.4 panel modules, for the composer and for m."""
    f2a, f2b, f2d = (load_copy(n) for n in ("fig2a_wavelets_delegated", "fig2b_trajectory_delegated",
                                             "fig2d_activity_maps_delegated"))
    pcef = load_copy("fig2_panels_v8")
    m.f2a, m.f2b, m.f2d, m.pcef = f2a, f2b, f2d, pcef
    m.comp.create_fig2a, m.comp.create_fig2b, m.comp.create_fig2d = f2a.create_fig2a, f2b.create_fig2b, f2d.create_fig2d
    m.comp.create_fig2c, m.comp.create_fig2e, m.comp.create_fig2f = pcef.create_fig2c, pcef.create_fig2e, pcef.create_fig2f


def settings_v11():
    """Settings of make_fig2_v104.py with the event ticks of panel a and the onsets of panel b."""
    m.settings("B")
    m.pcef.E_LABEL_ABOVE = ("MSE", "PSNR", "SSIM")
    m.f2a.EVENT_TICKS = True
    m.f2a.EVENTS_LABEL = "detected events"
    m.f2a.YLABEL_X = m.YLABEL_X
    m.f2b.CACHE_FILE = m._F.SAVED / "fig2b_data_v104_onsets.npz"


def settings_v111(e_wspace=E_WSPACE):
    settings_v11()
    m.f2a.FREQ_LABEL = "frequency, Hz"
    t = json.loads((m._F.data_dir(HERE) / "cache" / "fig2b_cbar_ticks.json").read_text(encoding="utf-8"))
    m.f2b.CBAR_TICKS = (t["ticks_position"], [f"{v:g}" for v in t["ticks_calcium"]])
    m.f2d.NORMALIZE = True
    m.pcef.E_YLABELS = E_YLABELS
    m.pcef.E_WSPACE = e_wspace
    m.pcef.DATA_NPZ = m._F.data_dir(HERE) / "cache" / "fig2_v111_data_B.npz"
    m.pcef.F_XMAX = 40


# ---------------------------------------------------------------- checks on the drawn figure
def label_heights(fig):
    """Bottom of every "p=0.001" label in figure pixels."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    return [round(float(t.get_window_extent(r).y0), 1) for t in fig.findobj(matplotlib.text.Text)
            if t.get_text() == r"$p=0.001$"]


def _bb(b):
    return [round(float(v), 1) for v in (b.x0, b.y0, b.x1, b.y1)]


def find_axes(fig):
    axs = fig.findobj(matplotlib.axes.Axes)
    out = {}
    for ax in axs:
        lab = [ln.get_label() for ln in ax.lines]
        for k in ("MSE", "PSNR", "SSIM"):
            if k in lab:
                out[f"e_{k}"] = ax
        if ax.get_ylabel() == "confirmed (%)":
            out["f_left"] = ax
        if ax.get_ylabel() == "Jaccard index":
            out["f_right"] = ax
        if ax.get_title() in ("Calcium", "Events"):
            out[f"d_{ax.get_title()}"] = ax
        if ax.get_ylabel().startswith("calcium activity"):
            out["b_cbar"] = ax
        if ax.get_xlabel() == "Normalized activity":
            out["d_cbar"] = ax
    return out


def panel_e_geometry(fig, axd):
    """Each y-axis title of 2e against its own tick labels, the plot to its left and the canvas edge."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    out = {}
    order = ["e_MSE", "e_PSNR", "e_SSIM"]
    for i, k in enumerate(order):
        ax = axd[k]
        title = ax.yaxis.label
        tb = title.get_window_extent(r)
        ticks = [t.get_window_extent(r) for t in ax.get_yticklabels() if t.get_visible() and t.get_text()]
        left_tick = min(b.x0 for b in ticks)
        res = dict(title=title.get_text(), title_fontsize=title.get_fontsize(),
                   xlabel_fontsize=ax.xaxis.label.get_fontsize(), title_bbox_px=_bb(tb),
                   room_to_own_tick_labels_px=round(float(left_tick - tb.x1), 1),
                   axes_bbox_px=_bb(ax.get_window_extent(r)))
        if i == 0:
            res["room_to_canvas_left_px"] = round(float(tb.x0), 1)
        else:
            from matplotlib.path import Path as MPath
            left = axd[order[i - 1]]
            res["room_to_axes_box_on_left_px"] = round(float(tb.x0 - left.get_window_extent(r).x1), 1)
            hits = []
            for ln in left.lines:
                xy = left.transData.transform(np.column_stack(ln.get_data()))
                if MPath(xy).intersects_bbox(tb, filled=False):
                    hits.append(f"line {ln.get_label()}")
            for col in left.collections:
                if any(p.transformed(left.transData).intersects_bbox(tb, filled=True) for p in col.get_paths()):
                    hits.append("shaded s.e.m.")
            for t in left.texts + left.get_xticklabels() + left.get_yticklabels() + [left.xaxis.label]:
                if t.get_visible() and t.get_text() and t.get_window_extent(r).overlaps(tb):
                    hits.append(f"text {t.get_text()}")
            res["meets_plot_on_left"] = hits
        out[k] = res
    return out


def panel_checks(fig, axd):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    res = {}
    ax1 = fig._sf_a.axes[0]
    res["a"] = dict(ylabel=ax1.get_ylabel(), yticklabels=[t.get_text() for t in ax1.get_yticklabels()],
                    frequency_of_tick_rows_hz=None)
    sc = m.f2a.DATA["wvt_scales"]
    rows = ax1.get_yticks()
    res["a"]["frequency_of_tick_rows_hz"] = [round(float(m.f2a.FPS / (2 * np.pi * np.exp(
        np.interp(rw, np.arange(len(sc)), np.log(sc))))), 4) for rw in rows]
    res["a"]["fps"] = m.f2a.FPS
    cb = axd["b_cbar"]
    res["b"] = dict(label=cb.get_ylabel(), ticks=[float(v) for v in cb.get_yticks()],
                    ticklabels=[t.get_text() for t in cb.get_yticklabels()], ylim=[float(v) for v in cb.get_ylim()])
    ims = {k: axd[k].images[0] for k in ("d_Calcium", "d_Events")}
    res["d"] = {k: dict(min=float(np.min(im.get_array())), max=float(np.max(im.get_array())),
                        clim=[float(v) for v in im.get_clim()]) for k, im in ims.items()}
    dc = axd["d_cbar"]
    res["d"]["colourbar"] = dict(label=dc.get_xlabel(), ticklabels=[t.get_text() for t in dc.get_xticklabels()])
    fl = axd["f_left"]
    lines = {ln.get_label(): ln for ln in fl.lines if not ln.get_label().startswith("_")}
    fres = dict(xlim=[float(v) for v in fl.get_xlim()], ylim=[float(v) for v in fl.get_ylim()],
                xticklabels=[t.get_text() for t in fl.get_xticklabels()], xlabel=fl.get_xlabel())
    for name, ln in lines.items():
        x, y = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
        key = "red" if name.startswith("INTENSE") else "blue"
        fres[key] = dict(label=name.replace("\n", " "), x_first=float(x[0]), x_last=float(x[-1]), n_points=int(len(x)),
                         y_max=float(np.nanmax(y)),
                         y_at={f"{v:g}": float(y[np.argmin(np.abs(x - v))]) for v in (3, 9.75, 10, 12, 29.5, 30)
                               if x[0] <= v <= x[-1] + 1e-9})
        # the axis must not press the curve to its edge
        fres[key]["room_right_of_last_point_px"] = round(float(
            fl.get_window_extent(r).x1 - fl.transData.transform((x[-1], y[-1]))[0]), 1)
    res["f_left"] = fres
    return res


def diff_regions(a, b, grow=12):
    """Connected regions (x0, y0, x1, y1, n_px) where two images differ (as figures_v11.1/work/was_is.py)."""
    d = np.any(np.abs(a[..., :3].astype(int) - b[..., :3].astype(int)) > 8, axis=2)
    dd = ndimage.binary_dilation(d, iterations=grow)
    lab, n = ndimage.label(dd)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        y, x = sl
        out.append([int(x.start), int(y.start), int(x.stop), int(y.stop), int((d[sl] & (lab[sl] == i)).sum())])
    return out


def compare(png):
    f = np.asarray(Image.open(png))
    if m._F.reference(V11) is None:
        return dict(shape=list(f.shape))
    a = np.asarray(Image.open(V11))
    res = dict(shape=list(f.shape), shape_v11=list(a.shape), same_shape_as_v11=bool(a.shape == f.shape))
    if a.shape == f.shape:
        res["identical_to_v11"] = bool(np.array_equal(a, f))
        res["px_differ_vs_v11_by_panel_box"] = m.diff_boxes(a, f)
        res["diff_regions_x0_y0_x1_y1_npx"] = diff_regions(a, f)
    return res


def run(mode, name=None, e_wspace=E_WSPACE):
    substitute()
    if mode == "control":
        settings_v11()
        out_dir, out_name, formats = WORK_OUT / "out", "fig2_control", ("png",)
    elif mode == "trial":
        settings_v111(e_wspace)
        out_dir, out_name, formats = WORK_OUT / "out", name, ("png",)
    else:
        settings_v111()
        out_dir, out_name, formats = FINAL_DIR, "fig2", ("png", "pdf")
    for ext in formats:
        if (out_dir / f"{out_name}.{ext}").exists():
            sys.exit(f"{out_dir / f'{out_name}.{ext}'} exists; not overwriting")
    out_dir.mkdir(exist_ok=True)
    fig = m.comp.create_fig2_delegated(output_dir=out_dir, output_name=out_name, lowercase=True,
                                       save_formats=formats)
    axd = find_axes(fig)
    res = dict(mode=mode, output=str(out_dir / f"{out_name}.png"), e_wspace=m.pcef.E_WSPACE,
               data_npz=str(m.pcef.DATA_NPZ), label_bottoms_px=label_heights(fig),
               panel_e_label_clashes=m.label_clashes(fig), panel_e_geometry=panel_e_geometry(fig, axd)
               if m.pcef.E_YLABELS else None,
               panel_a_ylabel_geometry=m.ylabel_geometry(fig), panel_a_ticks_on_axis_title=m.axis_title_clashes(fig),
               panels=panel_checks(fig, axd), text_clashes=m.overlaps.text_report(fig))
    plt.close(fig)
    res.update(compare(out_dir / f"{out_name}.png"))
    if mode == "final":
        dpi = Image.open(out_dir / "fig2.png").info.get("dpi")
        res["png_dpi"] = [round(float(v), 2) for v in dpi] if dpi else None
    vname = {"control": "fig2_control.json", "trial": f"{name}.json", "final": "fig2_v111.json"}[mode]
    (WORK_OUT / "verify").mkdir(exist_ok=True)
    (WORK_OUT / "verify" / vname).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "text_clashes"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    mode = sys.argv[1]
    assert mode in ("control", "trial", "final"), mode
    if mode == "trial":
        run(mode, sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else E_WSPACE)
    else:
        run(mode)
