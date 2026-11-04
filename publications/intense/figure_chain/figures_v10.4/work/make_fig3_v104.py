"""Settings and checks of Fig. 3 shared by the later versions (fig3_src/); the figure of the paper is
drawn by figures_v11.1/work/fig3/make_fig3_v111.py, which imports this module through the versions between.

  a, b  the time axis at the session rate and exact shading;
  c     each significance bracket on its own level (starbars, one call per comparison, narrowest
        first; the 'stacked' setting of fig3c_bars_delegated.V7_BRACKETS);
  d, e  set_example() takes the example neuron.
"""
import csv
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
SRC = HERE / "fig3_src"
sys.path.insert(0, str(SRC))
sys.path.insert(1, str(HERE))
import fig3_v10 as comp  # noqa: E402  (checks that driada is the 1.2.2 snapshot)
import fig3a_object_delegated as f3a  # noqa: E402
import fig3b_rear_delegated as f3b  # noqa: E402
import fig3c_bars_delegated as f3c  # noqa: E402
import fig3d_polar_delegated as f3d  # noqa: E402
import fig3e_heatmap_place_delegated as f3e  # noqa: E402
import fig3f_heatmap_delegated as f3f  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.text import Text  # noqa: E402

import overlaps  # noqa: E402

A = _F.D
V103 = A / "figures_v10.3" / "fig3.png"
CAND = A / "figures_v10.3" / "candidates_3de" / "candidates_top4.json"
V2 = intense_paths.NOF_ALIGNED
NEW = intense_paths.CACHE_DIR
FPS_TABLE = _F.V.get("fps_table")
OUT = _F.out_dir(HERE) / "fig3_out"


def fps_of(session):
    with open(FPS_TABLE, newline="") as f:
        return {r["session"]: float(r["fps"]) for r in csv.DictReader(f)}[session]


def px_box(fig, bb, margin=2):
    h = int(round(fig.bbox.height))
    return [max(int(np.floor(h - bb.y1)) - margin, 0), min(int(np.ceil(h - bb.y0)) + margin, h),
            max(int(np.floor(bb.x0)) - margin, 0), int(np.ceil(bb.x1)) + margin]


def union(boxes):
    return [min(b[0] for b in boxes), max(b[1] for b in boxes), min(b[2] for b in boxes), max(b[3] for b in boxes)]


def panel_boxes(fig):
    """Pixel box of every panel: its subfigure plus every text drawn inside it."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = {}
    for name, sf in fig._subfigs.items():
        bb = [px_box(fig, sf.bbox)]
        for t in sf.findobj(Text):
            if t.get_visible() and t.get_text():
                bb.append(px_box(fig, t.get_window_extent(r)))
        boxes[name] = union(bb)
    return boxes


def diff_in(a, b, box):
    r0, r1, c0, c1 = box
    return int(np.any(a[r0:r1, c0:c1] != b[r0:r1, c0:c1], axis=2).sum())


def drawn_c(axes):
    res = []
    for ax in axes:
        res.append(dict(ylabel=ax.get_ylabel(), bar_heights=[float(p.get_height()) for p in ax.patches],
                        stars=[t.get_text() for t in ax.texts], ylim=[float(v) for v in ax.get_ylim()]))
    return res


def stars_check(fig):
    """Panel c: do the star labels meet each other or a bracket line?"""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    out = []
    for ax in fig._panel_c[1]:
        texts = list(ax.texts)
        boxes = [t.get_window_extent(r) for t in texts]
        pairs = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                a, b = boxes[i], boxes[j]
                w, h = min(a.x1, b.x1) - max(a.x0, b.x0), min(a.y1, b.y1) - max(a.y0, b.y0)
                if w > 0 and h > 0:
                    pairs.append([texts[i].get_text(), texts[j].get_text(), round(float(w), 1), round(float(h), 1)])
        lines = overlaps.text_vs_lines(ax, texts, r)
        out.append(dict(axes=ax.get_xlabel(), stars=[t.get_text() for t in texts], star_pairs_overlapping=pairs,
                        star_meets_line=[x for x in lines if x["lines_met"]]))
    return out


def candidate():
    c = json.loads(CAND.read_text(encoding="utf-8"))["top"][1]
    assert (c["mouse"], c["session"], c["neuron_idx"]) == ("H39", "4D", 552), c
    assert (c["delay_hd_frames"], c["delay_place_frames"]) == (15, 0), c
    return c


def set_example(c):
    npz = V2 / f"NOF_{c['mouse']}_{c['session']}_aligned.npz"
    meta = np.load(npz, allow_pickle=True)["_metadata"].item()
    assert abs(float(meta["fps"]) - c["fps"]) < 1e-9, (meta, c["fps"])
    f3d._DEFAULT_ALIGNED_NPZ, f3d._DEFAULT_NEURON_IDX, f3d._DEFAULT_OPTIMAL_DELAY = npz, c["neuron_idx"], c["delay_hd_frames"]
    f3e._DEFAULT_ALIGNED_NPZ, f3e._DEFAULT_NEURON_IDX, f3e._DEFAULT_OPTIMAL_DELAY = npz, c["neuron_idx"], c["delay_place_frames"]
    return npz, float(meta["fps"])


def example_check(fig, c, npz):
    """Panels d, e as drawn against the data of the example, and v1 against v2 for this session."""
    ang, sig = f3d.load_polar_data(npz, c["neuron_idx"], c["delay_hd_frames"])
    v1 = intense_paths.aligned_file(c["mouse"], c["session"])
    ang1, sig1 = f3d.load_polar_data(v1, c["neuron_idx"], c["delay_hd_frames"])
    x, y, s = f3e.load_place_data(npz, c["neuron_idx"], c["delay_place_frames"])
    bins = 40
    a = np.mod(ang + np.pi, 2 * np.pi) - np.pi
    edges = np.linspace(-np.pi, np.pi, bins + 1)
    idx = np.clip(np.digitize(a, edges) - 1, 0, bins - 1)
    hist = np.bincount(idx, weights=sig, minlength=bins) / np.maximum(np.bincount(idx, minlength=bins), 1)
    ax_d = fig._subfigs["D"].axes[0]
    drawn = np.array([p.get_height() for p in ax_d.patches])
    ax_e = fig._subfigs["E"].axes[0]
    levels = getattr(ax_e.collections[0], "levels", None) if ax_e.collections else None
    zmax = float(levels[-1]) if levels is not None else None
    return dict(file=str(npz), neuron_row=c["neuron_idx"], fps=c["fps"],
                delay_hd_frames=c["delay_hd_frames"], delay_place_frames=c["delay_place_frames"],
                n_frames_d=int(len(sig)), n_frames_e=int(len(s)),
                v1_equals_v2=bool(np.array_equal(ang, ang1) and np.array_equal(sig, sig1)),
                d_bars_drawn=int(len(drawn)), d_max_abs_diff_drawn_vs_data=float(np.max(np.abs(drawn - hist))),
                e_top_contour_level=zmax,
                excess_mi_bits=dict(head_direction=c["me_excess_hd"], position=c["me_excess_place"]),
                pval=dict(head_direction=c["pval_hd"], position=c["pval_place"]))


def v103_settings():
    f3a.FPS, f3a.EXACT_SHADING = fps_of("NOF_H02_3D"), True
    f3b.FPS, f3b.EXACT_SHADING = fps_of("NOF_H31_2D"), True
    f3c.V7_BRACKETS, f3c.YLABEL = True, "% selective neurons"
    f3c._PER_SESSION_CSV = NEW / "fig3c_per_session.csv"
    f3f._DEFAULT_MATRIX_NPZ = NEW / "fig3f_matrix.npz"


def main(mode):
    OUT.mkdir(exist_ok=True)
    (_F.out_dir(HERE) / "verify").mkdir(exist_ok=True)
    v103_settings()
    if mode == "control_v103":
        fig = comp.create_fig3_delegated(output_dir=OUT, output_name="fig3_control_v103", lowercase=True)
        texts = overlaps.text_report(fig)
        stars = stars_check(fig)
        plt.close(fig)
        a, c = np.asarray(Image.open(V103)), np.asarray(Image.open(OUT / "fig3_control_v103.png"))
        res = dict(reference=str(V103), identical=bool(a.shape == c.shape and np.array_equal(a, c)),
                   n_px_differ=int(np.any(a != c, axis=2).sum()) if a.shape == c.shape else None,
                   panel_c_stars=stars, text_clashes=texts)
        (_F.out_dir(HERE) / "verify" / "fig3_control_v103.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                                                 encoding="utf-8")
        print(json.dumps({k: res[k] for k in ("identical", "n_px_differ", "panel_c_stars")}, indent=1))
        return

    assert mode == "final", mode
    for f in ("fig3.png", "fig3.pdf"):
        if (_F.out_dir(HERE.parent) / f).exists():
            sys.exit(f"{_F.out_dir(HERE.parent) / f} exists; not overwriting")
    f3c.V7_BRACKETS = "stacked"
    c = candidate()
    npz, fps = set_example(c)
    fig = comp.create_fig3_delegated(output_dir=_F.out_dir(HERE.parent), output_name="fig3", lowercase=True,
                                     save_formats=("png", "pdf"))
    boxes = panel_boxes(fig)
    dc = drawn_c(fig._panel_c[1])
    stars = stars_check(fig)
    ex = example_check(fig, c, npz)
    texts = overlaps.text_report(fig)
    plt.close(fig)
    before = json.loads((_F.out_dir(HERE) / "verify" / "fig3_control_v103.json").read_text(encoding="utf-8"))["text_clashes"]
    a, f = np.asarray(Image.open(V103)), np.asarray(Image.open(_F.out_dir(HERE.parent) / "fig3.png"))
    res = dict(reference=str(V103), output=str(_F.out_dir(HERE.parent) / "fig3.png"), shape=list(f.shape),
               px_differ_vs_v10_3_by_panel={k: diff_in(a, f, b) for k, b in boxes.items()},
               panel_boxes=boxes, panel_c_drawn=dc, panel_c_stars=stars, example_d_e=ex,
               text_clashes_vs_control=overlaps.diff_reports(before, texts), text_clashes=texts)
    (_F.out_dir(HERE) / "verify" / "fig3_final.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("shape", "px_differ_vs_v10_3_by_panel", "panel_c_drawn", "panel_c_stars",
                                          "example_d_e", "text_clashes_vs_control")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
