"""The three MEC figures of the SI for v11.1, on the MEC run with declared x, y ([mec] recompute_dir,
variant new_on).

usage: python -X utf8 -B figure_chain/figures_v11.1/work/mec/make_mec_v111.py control [213]
            # switches off -> work/mec/control/*.png, must equal <recompute_dir>/figures_new_on/*.png
       python -X utf8 -B figure_chain/figures_v11.1/work/mec/make_mec_v111.py final [213]
            # -> si/: cohort with letters a, b; venn as drawn (copied from figures_new_on);
            #    margin with the v10.4 layout fix
       (outputs under [figures] output_dir/figure_chain/figures_v11.1)

The cohort figure counts all analysed sessions, or with the argument 213 the sessions with the authors'
labels, into si_213/ and figures_new_on_213/. The other two figures rest on the labels either way.

mec_src/make_mec_figures.py is a copy of figures_v10.4/work/mec_src/make_mec_figures.py with one more
switch, LOWERCASE_LETTERS, and upper limits of two axes that follow the counts. The margin figure is redrawn
as in figures_v10.4/work/make_mec_v104.py: the layout fix of v10.4 (one x-axis title, legend under the row of
counts) and the pixel box of the unfixed layout, so the figure keeps its size.
Read from [mec] recompute_dir: variants/new_on/{results_all.csv, cell_sets_all.jsonl (with 213: results.csv,
cell_sets.jsonl), arbiter_venn.csv, recall_vs_margin.csv}. figures_new_on/ holds the figures that
make_mec_figures.py draws from these files with its switches off: mec_arbiter_venn.png is the SI figure
itself, the others and mec_numbers.json are compared with.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "mec_src"))
import make_mec_figures as M  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(HERE.parents[2]))  # figure_chain/
import _paths_fig as _F  # noqa: E402  (puts the paper folder on sys.path)

X = _F.P.get("mec", "recompute_dir")
V = X / "variants" / "new_on"
# base: results, cell sets, folder of the figures with the switches off, folder of the SI figures
BASES = {"all": ("results_all.csv", "cell_sets_all.jsonl", "figures_new_on", "si"),
         "213": ("results.csv", "cell_sets.jsonl", "figures_new_on_213", "si_213")}
NAMES = ("mec_cohort_bytype.png", "mec_arbiter_venn.png", "mec_arbiter_margin.png")
WORK_OUT = _F.out_dir(HERE)


def use_base(base):
    global ARGS, NEW_ON, OUT
    results, sets, new_on, si = BASES[base]
    ARGS = dict(results=V / results, sets=V / sets, venn=V / "arbiter_venn.csv", margin=V / "recall_vs_margin.csv")
    NEW_ON = X / new_on
    OUT = _F.out_dir(HERE.parents[1]) / si


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def same_pixels(a, b):
    x, y = np.asarray(Image.open(a)), np.asarray(Image.open(b))
    return dict(shape_a=list(x.shape), shape_b=list(y.shape),
                identical=bool(x.shape == y.shape and np.array_equal(x, y)),
                n_px_differ=int(np.any(x != y, axis=2).sum()) if x.shape == y.shape else None)


def text_clashes(fig):
    """Pairs of drawn texts whose boxes overlap, and the legend against the texts of its axes."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(plt.Text) if t.get_visible() and t.get_text().strip()]
    boxes = [(t, t.get_window_extent(r)) for t in texts]
    out = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if min(a.x1, b.x1) - max(a.x0, b.x0) > 0.5 and min(a.y1, b.y1) - max(a.y0, b.y0) > 0.5:
                out.append(f"'{boxes[i][0].get_text()}' x '{boxes[j][0].get_text()}'")
    return out


def draw_new_on():
    """make_mec_figures.py with its switches off, into figures_new_on/."""
    sys.argv = ["make_mec_figures.py"] + sum(([f"--{k}", str(v)] for k, v in ARGS.items()), []) + ["--out", str(NEW_ON)]
    M.main()


def main(mode, base="all"):
    use_base(base)
    if mode == "control":
        if not (NEW_ON / NAMES[1]).exists():
            draw_new_on()
        out = WORK_OUT / "control"
        if out.exists() and any(out.iterdir()):
            sys.exit(f"{out} is not empty")
        sys.argv = ["make_mec_figures.py"] + sum(([f"--{k}", str(v)] for k, v in ARGS.items()), []) + ["--out", str(out)]
        M.main()
        res = {n: same_pixels(NEW_ON / n, out / n) | dict(md5_equal=md5(NEW_ON / n) == md5(out / n)) for n in NAMES}
        nums_equal = json.loads((NEW_ON / "mec_numbers.json").read_text())["glm"] == \
            json.loads((out / "mec_numbers.json").read_text())["glm"]
        res["mec_numbers_glm_equal"] = nums_equal
        (WORK_OUT / "control.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps(res, indent=1))
        return

    assert mode == "final", mode
    OUT.mkdir(exist_ok=True)
    for n in NAMES:
        if (OUT / n).exists():
            sys.exit(f"{OUT / n} exists; not overwriting")
    # drawn every time: an SI figure is copied from here, so it must not stay behind the numbers of the cohort
    draw_new_on()
    res = {}
    M.FIGS = OUT
    M.LOWERCASE_LETTERS = True
    C, glm, bt, cov, cat, pf = M.mec_counts(ARGS["results"], ARGS["sets"])
    M.fig_mec(C, glm, bt, cov)
    res[NAMES[0]] = dict(letters="a, b", ymax=M.cohort_ymax(C, glm, bt),
                         vs_figures_new_on=same_pixels(NEW_ON / NAMES[0], OUT / NAMES[0]),
                         glm=dict(zip(C, glm)), intense=dict(zip(C, bt)),
                         recovered={c: cov[c] for c in C})
    shutil.copy2(NEW_ON / NAMES[1], OUT / NAMES[1])
    res[NAMES[1]] = dict(source=str(NEW_ON / NAMES[1]), md5_equal=md5(NEW_ON / NAMES[1]) == md5(OUT / NAMES[1]))

    # box of the unfixed layout, as savefig(bbox_inches='tight', pad_inches=0.03) computes it at the saved dpi
    tmp = WORK_OUT / "layout_box"
    tmp.mkdir(exist_ok=True)
    M.FIGS, M.KEEP_MARGIN_FIG = tmp, True
    fig0, axes0 = M.fig_margin(ARGS["margin"])
    fig0.set_dpi(M.DPI)
    fig0.canvas.draw()
    box = fig0.get_tightbbox(fig0.canvas.get_renderer()).padded(M.SAVE["pad_inches"])
    before = text_clashes(fig0)
    plt.close(fig0)
    M.FIGS, M.MARGIN_LAYOUT_FIX = OUT, True
    M.MARGIN_SAVE = dict(bbox_inches=box)
    fig, axes = M.fig_margin(ARGS["margin"])
    fig.set_dpi(M.DPI)
    after = text_clashes(fig)
    plt.close(fig)
    res[NAMES[2]] = dict(layout_fix="v10.4 MARGIN_LAYOUT_FIX", saved_box_inches=[round(v, 4) for v in box.bounds],
                         size=list(Image.open(OUT / NAMES[2]).size), size_figures_new_on=list(Image.open(NEW_ON / NAMES[2]).size),
                         text_clashes_unfixed_layout=before, text_clashes=after)
    name = "final.json" if base == "all" else f"final_{base}.json"
    (WORK_OUT / name).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
