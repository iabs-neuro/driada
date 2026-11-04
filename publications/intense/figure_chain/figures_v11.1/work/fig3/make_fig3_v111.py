"""Fig. 3 of v11.1.

Copy of compute_pack_v11.1/scripts/f_make_fig3_candidates.py, the script that drew the candidates for a
new example of panels d, e. That script builds the figure of v11 (the v10.6 chain with DRIADA 1.2.2,
panel b of v11 with H08, day 4, neuron 121 and the fitted density axis) with another example in panels
d, e; through it the v11 example gives figures_v11/fig3.png pixel for pixel
(compute_pack_v11.1/F_fig3/control_v11.json).
Changes of v11.1, all decided by the author on 04.11:
  d, e  candidate 3 of compute_pack_v11.1/F_fig3/candidates.json (written by
        compute_pack_v11.1/scripts/f_candidates.py): H02, day 4, neuron 215, selective for body
        direction and position, both at zero delay; panel d reads the body direction of the session
        (load_body below);
  d     drawn by the copy fig3d_polar_delegated.py of this folder: 0 deg to the right and angles
        counter-clockwise, as in the axes of map e (the polar panel of v11 put 0 deg at the top and
        ran clockwise, which mirrored it against map e);
  f     the state faster than 5 cm/s is labelled 'run' (copy fig3f_heatmap_delegated.py);
  g     symbols of Methods: II(A,X,Y), I(A,X), I(A,Y) (copy fig3g_venn_delegated.py).
Everything else is drawn as in v11. candidates.json is read from where f_candidates.py writes it; the
figures of earlier versions are compared with only when they are there; everything is written under
[figures] output_dir/figure_chain/figures_v11.1.

usage: python -X utf8 -B figure_chain/figures_v11.1/work/fig3/make_fig3_v111.py draft
            # -> work/fig3/draft/fig3.png, work/fig3/verify/fig3_v111_draft.json
       python -X utf8 -B figure_chain/figures_v11.1/work/fig3/make_fig3_v111.py final
            # -> fig3.png, fig3.pdf, work/fig3/verify/fig3_v111_final.json, crops for the sheets
"""
import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import importlib.util  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # figure_chain/
import _paths_fig as _F  # noqa: E402  (puts the paper folder on sys.path)

OUT_DIR = _F.out_dir(HERE.parents[1])          # figures_v11.1
WORK_OUT = _F.out_dir(HERE)
V11_WORK = _F.CHAIN / "figures_v11" / "work"
F_DIR = _F.data_dir(_F.CHAIN / "compute_pack_v11.1" / "F_fig3")
sys.path.insert(0, str(_F.CHAIN / "figures_v10.6" / "work"))
import make_fig3_v106 as v6  # noqa: E402  (loads v10.5, v10.4 and DRIADA 1.2.2)
sys.path.insert(0, str(V11_WORK))
import fig3b_rear_v11 as f3b  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

m = v6.m
V11_FIG = _F.D / "figures_v11" / "fig3.png"
F_CAND3_FIG = F_DIR / "cand3_H02_4D_n215.png"   # the same example as the candidate drawing (mirrored d, old f, g)
TEXT_CONTROL = _F.data_dir(V11_WORK) / "verify" / "fig3_control_v106.json"
PICK_B = dict(mouse="H08", session="4D", neuron_idx=121, delay=60)   # panel b of v11


def load_copy(name):
    """Panel script of this folder, under its own module name (the originals stay imported)."""
    spec = importlib.util.spec_from_file_location(f"{name}_v111", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


F3D, F3F, F3G = (load_copy(n) for n in ("fig3d_polar_delegated", "fig3f_heatmap_delegated",
                                        "fig3g_venn_delegated"))


def example_de():
    """Candidate 3 of compute_pack_v11.1/F_fig3/candidates.json as make_fig3_v104.set_example takes it."""
    c = json.loads((F_DIR / "candidates.json").read_text(encoding="utf-8"))["drawn"][2]
    assert (c["mouse"], c["session"], int(c["neuron_idx"])) == ("H02", "4D", 215), c
    return dict(mouse=c["mouse"], session=c["session"], neuron_idx=int(c["neuron_idx"]), fps=float(c["fps"]),
                delay_hd_frames=int(c["delay_body_frames"]), delay_place_frames=int(c["delay_place_frames"]),
                body_pref_deg_candidate=float(c["body_pref_deg"]))


EX_DE = example_de()


def load_body(aligned_npz=None, neuron_idx=None, optimal_delay=None):
    """load_polar_data of panel d with the body direction of the session."""
    aligned_npz = aligned_npz or F3D._DEFAULT_ALIGNED_NPZ
    neuron_idx = F3D._DEFAULT_NEURON_IDX if neuron_idx is None else neuron_idx
    optimal_delay = F3D._DEFAULT_OPTIMAL_DELAY if optimal_delay is None else optimal_delay
    data = np.load(aligned_npz, allow_pickle=True)
    angles_rad = data["bodydirection"]
    calcium = data["Calcium"][neuron_idx]
    signals = (calcium - calcium.min()) / (calcium.max() - calcium.min())   # MinMaxScaler of the panel
    signals = np.clip(signals, 0, None)
    if optimal_delay > 0:
        signals, angles_rad = signals[optimal_delay:], angles_rad[:-optimal_delay]
    elif optimal_delay < 0:
        signals, angles_rad = signals[:optimal_delay], angles_rad[-optimal_delay:]
    return angles_rad, signals


def set_rear(ex):
    f3b._MOUSE, f3b._SESSION = ex["mouse"], ex["session"]
    f3b._DEFAULT_NEURON_IDX, f3b._DEFAULT_OPTIMAL_DELAY = ex["neuron_idx"], ex["delay"]
    f3b.FPS, f3b.EXACT_SHADING = m.fps_of(f"NOF_{ex['mouse']}_{ex['session']}"), True
    m.comp.create_fig3b = f3b.create_fig3b


def build(out_dir, name, formats):
    """The steps of f_make_fig3_candidates.build with the panel copies of this folder."""
    m.NEW = v6.ALIGNED
    m.f3d, m.f3f = F3D, F3F          # set_example and v103_settings set the defaults of the copies
    m.comp.create_fig3d, m.comp.create_fig3f, m.comp.create_fig3g = (F3D.create_fig3d, F3F.create_fig3f,
                                                                      F3G.create_fig3g)
    m.v103_settings()
    m.f3c.V7_BRACKETS = "stacked"
    m.set_example(EX_DE)
    F3D.load_polar_data = load_body
    f3b.FIT_DENSITY_YLIM = True
    set_rear(PICK_B)
    return m.comp.create_fig3_delegated(output_dir=out_dir, output_name=name, lowercase=True,
                                        save_formats=formats)


def letters_box(fig, names, parts):
    r = fig.canvas.get_renderer()
    for t in fig.texts:
        if t.get_text() in names:
            parts.append(m.px_box(fig, t.get_window_extent(r)))
    return m.union(parts)


def tuning(ang, sig, bins=40):
    """Occupancy-normalised tuning as panel d bins it."""
    a = np.mod(ang + np.pi, 2 * np.pi) - np.pi
    edges = np.linspace(-np.pi, np.pi, bins + 1)
    idx = np.clip(np.digitize(a, edges) - 1, 0, bins - 1)
    return (np.bincount(idx, weights=sig, minlength=bins) / np.maximum(np.bincount(idx, minlength=bins), 1),
            (edges[:-1] + edges[1:]) / 2)


def check_d_e(fig):
    """Panel d as drawn: orientation, bars against the data, preferred direction on screen and in map e."""
    ax_d = fig._subfigs["D"].axes[0]
    ax_e = fig._subfigs["E"].axes[0]
    npz = F3D._DEFAULT_ALIGNED_NPZ
    ang, sig = load_body()
    hist, centres = tuning(ang, sig)
    bars = ax_d.patches
    drawn_h = np.array([p.get_height() for p in bars])
    drawn_c = np.array([p.get_x() + p.get_width() / 2 for p in bars])
    pref = float(np.degrees(np.arctan2((drawn_h * np.sin(drawn_c)).sum(), (drawn_h * np.cos(drawn_c)).sum())))
    mvl = float(np.hypot((drawn_h * np.sin(drawn_c)).sum(), (drawn_h * np.cos(drawn_c)).sum()) / drawn_h.sum())
    r_pref = float(drawn_h.max())
    centre = ax_d.transData.transform([(0.0, 0.0)])[0]
    tip = ax_d.transData.transform([(np.radians(pref), r_pref)])[0]
    screen_pref = float(np.degrees(np.arctan2(tip[1] - centre[1], tip[0] - centre[0])))
    k = int(np.argmax(drawn_h))
    tip_max = ax_d.transData.transform([(drawn_c[k], drawn_h[k])])[0]
    screen_max = float(np.degrees(np.arctan2(tip_max[1] - centre[1], tip_max[0] - centre[0])))
    labels_at = {}
    r = fig.canvas.get_renderer()
    for t in ax_d.get_xticklabels():
        if t.get_text():
            bb = t.get_window_extent(r)
            c = ((bb.x0 + bb.x1) / 2 - centre[0], (bb.y0 + bb.y1) / 2 - centre[1])
            labels_at[t.get_text()] = round(float(np.degrees(np.arctan2(c[1], c[0]))) % 360, 1)
    e0 = ax_e.transData.transform([(0.5, 0.5)])[0]
    e_up = ax_e.transData.transform([(0.5, 0.6)])[0]
    e_right = ax_e.transData.transform([(0.6, 0.5)])[0]
    # Place map as panel e builds it, and its peak
    x, y, s = m.f3e.load_place_data()
    X, Y, Z = m.f3e._build_heatmap(x, y, s)
    iy, ix = np.unravel_index(np.argmax(Z), Z.shape)
    # Direction of the data against the direction of movement on the coordinates of map e
    with np.load(npz, allow_pickle=True) as z:
        bx, by = np.asarray(z["x"], float), np.asarray(z["y"], float)
        body = np.asarray(z["bodydirection"], float)
        cal = np.asarray(z["Calcium"][F3D._DEFAULT_NEURON_IDX], float)
    step = 5
    dx, dy = bx[2 * step:] - bx[:-2 * step], by[2 * step:] - by[:-2 * step]
    mov = np.arctan2(dy, dx)
    disp = np.hypot(dx, dy)
    moving = disp > np.quantile(disp, 0.5)
    dif = np.angle(np.exp(1j * (body[step:-step][moving] - mov[moving])))
    # Inside the field strip along the right wall: mean signal moving up against moving down
    # (both delays of the example are zero, so signal and direction are taken in the same frame)
    assert F3D._DEFAULT_OPTIMAL_DELAY == 0 and m.f3e._DEFAULT_OPTIMAL_DELAY == 0
    xn = (bx - bx.min()) / (bx.max() - bx.min())
    sn = np.clip((cal - cal.min()) / (cal.max() - cal.min()), 0, None)
    strip = xn > 0.75
    b = np.degrees(np.mod(body, 2 * np.pi))
    up, down = strip & (np.abs(b - 90) <= 45), strip & (np.abs(b - 270) <= 45)
    return dict(
        file=str(npz), neuron_row=F3D._DEFAULT_NEURON_IDX, delay_body_frames=F3D._DEFAULT_OPTIMAL_DELAY,
        delay_place_frames=m.f3e._DEFAULT_OPTIMAL_DELAY,
        d_theta_offset_rad=float(ax_d.get_theta_offset()), d_theta_direction=int(ax_d.get_theta_direction()),
        d_tick_label_screen_angle_deg=labels_at,
        d_bars_drawn=int(len(bars)), d_max_abs_diff_drawn_vs_body_tuning=float(np.max(np.abs(drawn_h - hist))),
        d_pref_deg_drawn=pref, d_pref_deg_candidate=EX_DE["body_pref_deg_candidate"], d_mean_vector_length=mvl,
        d_pref_screen_angle_deg=screen_pref, d_longest_bar_centre_deg=float(np.degrees(drawn_c[k])),
        d_longest_bar_screen_angle_deg=screen_max,
        d_rmax=float(ax_d.get_rmax()), d_longest_bar_over_rmax=float(drawn_h.max() / ax_d.get_rmax()),
        e_plus_y_screen_angle_deg=float(np.degrees(np.arctan2(e_up[1] - e0[1], e_up[0] - e0[0]))),
        e_plus_x_screen_angle_deg=float(np.degrees(np.arctan2(e_right[1] - e0[1], e_right[0] - e0[0]))),
        e_map_peak_xy=[float(X[iy, ix]), float(Y[iy, ix])],
        body_minus_movement_direction_deg=dict(
            step_frames=2 * step, n_frames=int(moving.sum()),
            circular_mean=float(np.degrees(np.angle(np.mean(np.exp(1j * dif))))),
            resultant_length=float(np.abs(np.mean(np.exp(1j * dif)))),
            median_abs=float(np.degrees(np.median(np.abs(dif))))),
        right_wall_strip_x_over_0_75=dict(
            frames_up=int(up.sum()), frames_down=int(down.sum()),
            mean_signal_up=float(sn[up].mean()), mean_signal_down=float(sn[down].mean())))


def label_bands(fig):
    """Pixel bands of the 'run' tick labels of panel f: where the longer old label 'locomotion' stood."""
    ax_f = fig._subfigs["F"].axes[0]
    r = fig.canvas.get_renderer()
    ax_bb = m.px_box(fig, ax_f.get_window_extent(r), margin=0)
    H = int(round(fig.bbox.height))
    bands = []
    for t in ax_f.get_yticklabels():
        if t.get_text() == "run":
            b = m.px_box(fig, t.get_window_extent(r))
            bands.append([b[0], b[1], 0, ax_bb[2]])            # the label row, left of the heatmap
    for t in ax_f.get_xticklabels():
        if t.get_text() == "run":
            b = m.px_box(fig, t.get_window_extent(r))
            bands.append([ax_bb[1], H, b[2], b[3]])            # the label column, below the heatmap
    return bands, [t.get_text() for t in ax_f.get_yticklabels()], [t.get_text() for t in ax_f.get_xticklabels()]


def g_texts(fig):
    ax_g = fig._subfigs["G"].axes[0]
    region = [t.get_text() for t in ax_g.texts if t.get_text()]
    legend = [t.get_text() for t in ax_g.get_legend().get_texts()]
    return dict(diagram=region, legend=legend)


def diffs(ref, new, boxes, allowed):
    """Changed pixels by panel and outside the allowed boxes."""
    d = np.any(ref != new, axis=2)
    keep = d.copy()
    for r0, r1, c0, c1 in allowed:
        keep[r0:r1, c0:c1] = False
    ys, xs = np.nonzero(keep)
    return dict(px_total=int(d.sum()),
                px_by_panel={k: int(d[b[0]:b[1], b[2]:b[3]].sum()) for k, b in boxes.items()},
                px_outside_allowed=int(keep.sum()),
                outside_allowed_bbox=[int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1]
                if len(ys) else None)


def run(mode):
    """draft: png only into draft/, to look at before the final files; final: the files of v11.1."""
    out_dir, formats, crops = ((OUT_DIR, ("png", "pdf"), WORK_OUT) if mode == "final" else
                               (WORK_OUT / "draft", ("png",), WORK_OUT / "draft"))
    out_dir.mkdir(exist_ok=True)
    for f in [out_dir / f"fig3.{x}" for x in formats] + [WORK_OUT / "verify" / f"fig3_v111_{mode}.json"]:
        if f.exists():
            sys.exit(f"{f} exists; not overwriting")
    (WORK_OUT / "verify").mkdir(exist_ok=True)
    fig = build(out_dir, "fig3", formats)
    boxes = m.panel_boxes(fig)
    de = letters_box(fig, ("d", "e"), [boxes["D"], boxes["E"]])
    fg = letters_box(fig, ("f", "g"), [boxes["F"], boxes["G"]])
    bands, f_y, f_x = label_bands(fig)
    checks = check_d_e(fig)
    g = g_texts(fig)
    texts = m.overlaps.text_report(fig)
    plt.close(fig)
    new = np.asarray(Image.open(out_dir / "fig3.png"))
    res = dict(output=str(out_dir / "fig3.png"), shape=list(new.shape), example_de=EX_DE, checks_d_e=checks,
               f_tick_labels=dict(y=f_y, x=f_x), f_label_bands=bands, g_texts=g, panel_boxes=boxes,
               de_box=de, fg_box=fg, text_clashes=texts)
    v11 = np.asarray(Image.open(V11_FIG)) if _F.reference(V11_FIG) else None
    if v11 is not None:
        res["vs_v11"] = dict(reference=str(V11_FIG), allowed=["D", "E", "G", "f 'run' label bands"],
                             **diffs(v11, new, boxes, [boxes["D"], boxes["E"], boxes["G"]] + bands))
    if _F.reference(F_CAND3_FIG):
        res["vs_candidate3"] = dict(reference=str(F_CAND3_FIG), allowed=["D", "G", "f 'run' label bands"],
                                    **diffs(np.asarray(Image.open(F_CAND3_FIG)), new, boxes,
                                            [boxes["D"], boxes["G"]] + bands))
    if _F.reference(TEXT_CONTROL):
        before = json.loads(TEXT_CONTROL.read_text(encoding="utf-8"))["text_clashes"]
        res["text_clashes_vs_v106_control"] = m.overlaps.diff_reports(before, texts)
    for tag, img in (("v11", v11), ("v111", new)):
        if img is None:
            continue
        for part, b in (("de", de), ("fg", fg)):
            p = crops / f"crop_{part}_{tag}.png"
            if p.exists():
                sys.exit(f"{p} exists; not overwriting")
            Image.fromarray(img[b[0]:b[1], b[2]:b[3]]).save(p)
    (WORK_OUT / "verify" / f"fig3_v111_{mode}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                                               encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("shape", "checks_d_e", "f_tick_labels", "g_texts", "vs_v11",
                                          "vs_candidate3", "text_clashes_vs_v106_control") if k in res},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    assert sys.argv[1] in ("draft", "final"), sys.argv[1]
    run(sys.argv[1])
