"""Fig. 4.

usage: python -X utf8 -B figure_chain/figures_v11.1/work/fig4/make_fig4_v111.py
       # -> fig4.png, fig4.pdf, work/fig4/verify/fig4_v111.json under [figures] output_dir/figure_chain/figures_v11.1

The panel scripts are in fig4_v7src/ of this folder; the package is imported before make_fig4_v104, so that
module (figures_v10.4/work) draws with it.

In panel e the points would press against the top of the axes and could not be told apart, so the recall axis
starts higher. The two plots of panel e share the recall axis (only the left one carries its label), and the
lowest recall drawn there is 0.47 (INTENSE at p_skip = 0.8), so both start at 0.4 and end at 1.06; the plots
stay square. Everything else is figures_v10.4/work/make_fig4_v104.py: same data, settings and layout.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
W104 = HERE.parents[2] / "figures_v10.4" / "work"
sys.path.insert(0, str(HERE))
import fig4_v7src  # noqa: E402,F401  (the copy with the fBM label, before make_fig4_v104 imports it)
sys.path.insert(1, str(W104))
import make_fig4_v104 as m  # noqa: E402  (loads the v7 panel scripts of v10.4)
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
V11 = m._F.D / "figures_v11" / "fig4.png"
RECALL_FROM = 0.4
# with the axes ending at 1.02 the many points at recall 1 would lie on the end of the axis; a little
# more room above lets them be seen as points
TOP = 1.06
_pr_panel = m.comp.create_pr_panel


def pr_panel(data, test_type, **kw):
    fig, axs = _pr_panel(data, test_type, **kw)
    if test_type == "d":
        for ax in axs:
            ax.set_ylim(RECALL_FROM, TOP)
            ax.set_yticks(np.arange(RECALL_FROM, 1.0001, 0.2))
            # keep the box square: one unit of recall is drawn longer than one of precision
            ax.set_aspect(1.02 / (TOP - RECALL_FROM), adjustable="box")
    return fig, axs


def main():
    assert Path(m.comp.__file__).resolve().is_relative_to(HERE), m.comp.__file__
    out = m._F.out_dir(HERE.parents[1])
    for f in ("fig4.png", "fig4.pdf"):
        if (out / f).exists():
            sys.exit(f"{out / f} exists; not overwriting")
    (m._F.out_dir(HERE) / "verify").mkdir(exist_ok=True)
    tab = m.load_table()
    c, d = m.arrays(tab, "c"), m.arrays(tab, "d")
    m.pr.PR_SELECT = "slice"
    m.f4a.EXACT_SHADING = True
    m.comp.create_pr_panel = pr_panel
    fig = m.comp.create_fig4_delegated(output_dir=out, save_formats=("png", "pdf"), c_data=c, d_data=d,
                                       output_name="fig4", lowercase=True)
    pr_axes = [ax for ax in fig.get_axes() if ax.get_xlabel() == "precision"]
    lowest = [float(np.asarray(coll.get_offsets())[:, 1].min()) for ax in pr_axes[2:] for coll in ax.collections
              if len(coll.get_offsets())]
    boxes = [[round(v, 4) for v in ax.get_position().bounds] for ax in pr_axes]
    values = m.value_checks(fig, tab)
    plt.close(fig)
    res = dict(output=str(out / "fig4.png"))
    if m._F.reference(V11):
        res = m.compare_png(V11, out / "fig4.png")
        a, b = np.asarray(Image.open(V11)), np.asarray(Image.open(out / "fig4.png"))
        if a.shape == b.shape:
            rows = np.nonzero(np.any(a != b, axis=(1, 2)))[0]
            res["first_differing_row"] = int(rows.min()) if len(rows) else None
    res.update(recall_axis_e=[RECALL_FROM, TOP], lowest_recall_drawn_in_e=min(lowest),
               pr_axes_positions=boxes, values=values)
    (m._F.out_dir(HERE) / "verify" / "fig4_v111.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "values"}, indent=1))
    print("[OK] values max abs diff:", values["max_abs_diff_all"])


if __name__ == "__main__":
    main()
