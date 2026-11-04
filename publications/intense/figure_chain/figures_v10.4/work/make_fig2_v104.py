"""Figure 2 of v10.4 (final): copies of the fig2_v10.4 scripts (fig2_src/), classic variants A and B.

usage: python make_fig2_v104.py control_A  # settings of fig2_v10.4 -> must equal fig2_v10.4/fig2_A.png
       python make_fig2_v104.py control_B  # settings of fig2_v10.4 -> must equal fig2_v10.4/fig2_B.png
       python make_fig2_v104.py B          # -> ../fig2.png, ../fig2.pdf   (classic: re-implementation, true clock)
       python make_fig2_v104.py A          # -> ../fig2_A.png, ../fig2_A.pdf (classic: MATLAB file)

Copy of fig2_v10.4/work/make_fig2_v104.py. Kept from fig2_v10.4 (both variants): data of v10.3 for
panel a with frequency labels on their rows, stars and maps at the session rate, maps of 2d smoothed
by one bin as in 2e, "p = 0.001" above the axes in the PSNR and SSIM plots of 2e, 2f without the gray
curves and with the Jaccard axis to 0.5. Added for v10.4, as the author agreed on 29.10:
  a  every detected event is a grey tick above the trace, at the middle of the event, in the middle of
     the gap between the two axes (the ticks of Fig. 1a); the legend entry reads 'detected events';
     both y-axis titles moved left by the same amount so that 'signal, a.u.' clears its tick labels;
  b  one star at the onset (first frame) of every detected event, instead of a star at every tenth
     frame inside the events (fig2_v10.4/fig2b_variant_event_onsets.png; cache fig2b_data_v104_onsets.npz).
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # figure_chain/
import _paths_fig as _F  # noqa: E402  (puts the paper folder on sys.path)
import intense_paths  # noqa: E402

intense_paths.use_driada()
sys.path.insert(0, str(HERE / "fig2_src"))
sys.path.insert(1, str(HERE))
import fig2_v10 as comp  # noqa: E402  (checks that driada is the 1.2.2 snapshot)
import fig2_panels_v8 as pcef  # noqa: E402
import fig2a_wavelets_delegated as f2a  # noqa: E402
import fig2b_trajectory_delegated as f2b  # noqa: E402
import fig2d_activity_maps_delegated as f2d  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import overlaps  # noqa: E402

ART = _F.D
V104 = ART / "fig2_v10.4"
A103 = ART / "figures_v10.3" / "work" / "fig2_out" / "fig2a_data_v103.npz"
OUT = _F.out_dir(HERE.parent)
FREQS = (3, 1, 0.3, 0.1)
# both y-axis titles of panel a, axes fraction (-0.12 before): the right edge of a title sits at
# this position, and at -0.12 'signal, a.u.' reached 16 px into its tick labels; at -0.152 it clears
# them by 35 px, the default label pad of matplotlib (4 pt = 33 px at 600 dpi). Measured on the
# control run, verify/fig2_control_B.json -> panel_a_ylabel_geometry.
YLABEL_X = -0.152
# panel boxes on the 4251 x 5196 canvas (600 dpi), as in fig2_v10.4/work/make_compare_v104.py
BOX = dict(a=(0, 0, 2140, 2060), b=(2080, 0, 4251, 2060), c=(0, 2060, 2095, 3215),
           d=(2080, 2060, 4251, 3215), e=(0, 3400, 2240, 4950), f=(2230, 3400, 4251, 4950))


def label_clashes(fig):
    """For every "p = 0.001" of 2e: does its text box meet a curve, a shaded band or a line of its axes?"""
    from matplotlib.path import Path as MPath
    from matplotlib.text import Text
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    out = []
    for t in fig.findobj(Text):
        if t.get_text() != r"$p=0.001$" or t.axes is None:
            continue
        ax, bb = t.axes, t.get_window_extent(r)
        hits = []
        for ln in ax.lines:
            xy = ax.transData.transform(np.column_stack(ln.get_data()))
            if MPath(xy).intersects_bbox(bb, filled=False):
                hits.append(f"line {ln.get_label() or ln.get_color()}")
        for col in ax.collections:
            for p in col.get_paths():
                if p.transformed(ax.transData).intersects_bbox(bb, filled=True):
                    hits.append("shaded s.e.m.")
                    break
        metric = next((ln.get_label() for ln in ax.lines if not ln.get_label().startswith("_")), "?")
        out.append(dict(metric=metric, clashes=hits))
    return out


def axis_title_clashes(fig):
    """Panel a: tick labels whose box meets the axis title of their axes (scalogram and trace)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    out = {}
    for ax in fig._sf_a.axes[:2]:
        title = ax.yaxis.label
        tb = title.get_window_extent(r)
        hit = [t.get_text() for t in ax.get_yticklabels()
               if t.get_visible() and t.get_text() and t.get_window_extent(r).overlaps(tb)]
        out[title.get_text()] = hit
    return out


def ylabel_geometry(fig):
    """Panel a: horizontal room between each y-axis title and the tick labels of its axes, in pixels."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    out = {}
    for ax in fig._sf_a.axes[:2]:
        tb = ax.yaxis.label.get_window_extent(r)
        ticks = [t.get_window_extent(r) for t in overlaps._drawn_tick_labels(ax.yaxis)]
        left = min(b.x0 for b in ticks)
        ab = ax.get_window_extent(r)
        out[ax.get_ylabel()] = dict(title_x0_px=round(float(tb.x0), 1), title_x1_px=round(float(tb.x1), 1),
                                    leftmost_tick_label_x0_px=round(float(left), 1),
                                    room_px=round(float(left - tb.x1), 1),
                                    axes_x0_px=round(float(ab.x0), 1), axes_width_px=round(float(ab.width), 1),
                                    label_coords_x=float(ax.yaxis.label.get_position()[0]))
    return out


def event_ticks_check(fig, sp, fps):
    """Panel a: ticks as drawn against the events; their room to the time-axis ticks of the scalogram
    above and to the highest point of the trace below; event ticks that touch each other."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ax1, ax2 = fig._sf_a.axes[:2]
    ticks = [ln for ln in ax2.lines if ln.get_marker() == "|"]
    assert len(ticks) == 1, len(ticks)
    ln = ticks[0]
    b = (np.asarray(sp) == 1).astype(np.int8)
    e = np.flatnonzero(np.diff(np.r_[0, b, 0]) != 0)
    want = (e[0::2] + e[1::2]) / 2
    x = np.sort(np.asarray(ln.get_xdata(), float))
    px = ln.get_transform().transform(np.column_stack([np.asarray(ln.get_xdata(), float), ln.get_ydata()]))
    half = ln.get_markersize() / 72 * fig.dpi / 2
    tick_w = ln.get_markeredgewidth() / 72 * fig.dpi
    b1, b2 = ax1.get_window_extent(r), ax2.get_window_extent(r)
    axis_tick = ax1.xaxis.get_major_ticks()[0].tick1line
    axis_tick_end = b1.y0 - axis_tick.get_markersize() / 72 * fig.dpi
    trace = [l for l in ax2.lines if l.get_label() == "Fluorescence signal"][0]
    top_of_trace = float(ax2.transData.transform((0, np.nanmax(trace.get_ydata())))[1])
    xs = np.sort(px[:, 0])
    close = np.flatnonzero(np.diff(xs) < tick_w)
    leg = ax2.get_legend()
    return dict(n_events=int(len(want)), n_ticks=int(len(x)),
                max_abs_tick_minus_event_middle_frames=float(np.max(np.abs(x - want))) if len(x) == len(want) else None,
                event_lengths_s=[round(float(v), 2) for v in (e[1::2] - e[0::2]) / fps],
                tick_height_px=round(float(2 * half), 1), tick_width_px=round(float(tick_w), 1),
                gap_between_axes_px=round(float(b1.y0 - b2.y1), 1),
                scalogram_time_ticks_reach_px=round(float(b1.y0 - axis_tick_end), 1),
                room_above_to_scalogram_time_ticks_px=round(float(axis_tick_end - (px[:, 1].max() + half)), 1),
                room_below_to_top_edge_of_trace_axes_px=round(float(px[:, 1].min() - half - b2.y1), 1),
                room_below_to_highest_point_of_trace_px=round(float(px[:, 1].min() - half - top_of_trace), 1),
                touching_tick_pairs=[dict(seconds=[round(float(x[i]) / fps, 1), round(float(x[i + 1]) / fps, 1)],
                                          centre_distance_px=round(float(xs[i + 1] - xs[i]), 1)) for i in close],
                legend=[t.get_text() for t in leg.get_texts()])


def settings(mode):
    """Settings of fig2_v10.4 for variant mode = 'A' or 'B'."""
    f2b.CBAR_LABEL = "calcium activity (log scale)"
    d = np.load(A103)
    f2a.DATA = {k: d[k] for k in ("sig", "sp", "Wx", "wvt_scales")}
    f2a.FPS = float(d["fps"])
    f2a.FREQ_TICKS = FREQS
    f2b.CACHE_FILE = _F.data_dir(HERE) / "cache" / "fig2b_data_v104.npz"
    f2d.CACHE_FILE = _F.data_dir(HERE) / "cache" / "fig2d_data_v104.npz"
    pcef.DATA_NPZ = _F.data_dir(HERE) / "cache" / f"fig2_v104_data_{mode}.npz"
    pcef.E_LABEL_ABOVE = ("PSNR", "SSIM")
    pcef.F_ONLY_OBSERVED_BASELINE = True
    pcef.F_JACCARD_YMAX = 0.5
    return d


def diff_boxes(a, b):
    out = {}
    for k, (c0, r0, c1, r1) in BOX.items():
        out[k] = int(np.any(a[r0:r1, c0:c1] != b[r0:r1, c0:c1], axis=2).sum())
    return out


def stars_drawn(fig):
    """Panel b: the star markers as drawn (scatter offsets in data units)."""
    sf_b = [sf for sf in fig.subfigs[0].subfigs if sf is not fig._sf_a][0]
    ax = sf_b.axes[0]
    off = [np.asarray(c.get_offsets()) for c in ax.collections if c.get_label() == "events"]
    return np.vstack(off) if off else np.zeros((0, 2))


def main(mode):
    (_F.out_dir(HERE) / "fig2_out").mkdir(exist_ok=True)
    (_F.out_dir(HERE) / "verify").mkdir(exist_ok=True)
    variant = mode[-1]
    assert variant in "AB" and mode in ("control_A", "control_B", "A", "B"), mode
    d = settings(variant)
    ref = V104 / f"fig2_{variant}.png"
    if mode.startswith("control"):
        fig = comp.create_fig2_delegated(output_dir=_F.out_dir(HERE) / "fig2_out", output_name=f"fig2_{mode}", lowercase=True)
        res = dict(reference=str(ref), panel_e_label_clashes=label_clashes(fig),
                   panel_a_ticks_on_axis_title=axis_title_clashes(fig), panel_a_ylabel_geometry=ylabel_geometry(fig),
                   panel_b_stars=int(len(stars_drawn(fig))), text_clashes=overlaps.text_report(fig))
        plt.close(fig)
        a, c = np.asarray(Image.open(ref)), np.asarray(Image.open(_F.out_dir(HERE) / "fig2_out" / f"fig2_{mode}.png"))
        res.update(shape_ref=list(a.shape), shape_control=list(c.shape),
                   identical=bool(a.shape == c.shape and np.array_equal(a, c)),
                   n_px_differ=int(np.any(a != c, axis=2).sum()) if a.shape == c.shape else None)
        (_F.out_dir(HERE) / "verify" / f"fig2_{mode}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                                           encoding="utf-8")
        print(json.dumps({k: v for k, v in res.items() if k != "text_clashes"}, indent=1, ensure_ascii=False))
        return

    name = "fig2" if variant == "B" else "fig2_A"
    for ext in ("png", "pdf"):
        if (OUT / f"{name}.{ext}").exists():
            sys.exit(f"{OUT / f'{name}.{ext}'} exists; not overwriting")
    f2a.EVENT_TICKS = True
    f2a.EVENTS_LABEL = "detected events"
    f2a.YLABEL_X = YLABEL_X
    f2b.CACHE_FILE = _F.data_dir(HERE) / "cache" / "fig2b_data_v104_onsets.npz"
    fig = comp.create_fig2_delegated(output_dir=OUT, output_name=name, lowercase=True, save_formats=("png", "pdf"))
    onsets = np.load(f2b.CACHE_FILE)
    stars = stars_drawn(fig)
    want = np.column_stack([onsets["spike_x"], onsets["spike_y"]])
    res = dict(output=str(OUT / f"{name}.png"), classic=variant, fps_panel_a=f2a.FPS,
               panel_a_event_ticks=event_ticks_check(fig, d["sp"], f2a.FPS),
               panel_a_ticks_on_axis_title=axis_title_clashes(fig), panel_a_ylabel_geometry=ylabel_geometry(fig),
               panel_b_stars=dict(n_drawn=int(len(stars)), n_onsets_in_cache=int(len(want)),
                                  equal_to_cache=bool(stars.shape == want.shape and np.array_equal(stars, want))),
               panel_e_label_clashes=label_clashes(fig))
    texts = overlaps.text_report(fig)
    plt.close(fig)
    before = json.loads((_F.out_dir(HERE) / "verify" / f"fig2_control_{variant}.json").read_text(encoding="utf-8"))["text_clashes"]
    a, f = np.asarray(Image.open(ref)), np.asarray(Image.open(OUT / f"{name}.png"))
    res.update(shape=list(f.shape), px_differ_vs_fig2_v10_4_by_panel=diff_boxes(a, f),
               text_clashes_vs_control=overlaps.diff_reports(before, texts), text_clashes=texts)
    if variant == "B":
        var = np.asarray(Image.open(V104 / "fig2b_variant_event_onsets.png").convert("RGB"))
        c0, r0, c1, r1 = BOX["b"]
        res["panel_b_vs_fig2b_variant_event_onsets_px_differ"] = int(
            np.any(var != np.asarray(Image.open(OUT / f"{name}.png").convert("RGB"))[r0:r1, c0:c1], axis=2).sum())
    (_F.out_dir(HERE) / "verify" / f"fig2_{variant}_final.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                                                encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "text_clashes"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
