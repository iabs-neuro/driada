"""Data of Fig. 2 for v11.1: the v11 data file with the curves of 2f (left) on a grid up to 30.

usage: python -X utf8 -B figure_chain/figures_v11.1/work/fig2/panel_f_data.py
       ->  figures_v11.1/work/fig2/cache/fig2_v111_data_B.npz, figures_v11.1/work/fig2/verify/panel_f_data.json
           under [figures] output_dir/figure_chain

The curves of the left plot of 2f are computed in fig2_data_v104.py (figures_v10.4/work/fig2_src,
lines 165-172) from the per-cell table that the same run wrote (figures_v10.4/work/cache/fig2_v104_cells_B.csv).
Here the same lines run on that table with the grid up to 30 instead of 12; first they run with the
grid of v11 and must give the curves of the v11 data file exactly. All other arrays of the data file
(panels c, e and the right plot of f) are copied unchanged.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # figure_chain/
import _paths_fig as _F  # noqa: E402  (puts the paper folder on sys.path)

NPZ_V11 = _F.data_dir(_F.CHAIN / "figures_v10.4" / "work") / "cache" / "fig2_v104_data_B.npz"   # pcef.DATA_NPZ of v11
CELLS = _F.data_dir(_F.CHAIN / "figures_v10.4" / "work") / "cache" / "fig2_v104_cells_B.csv"
OUT = _F.out_dir(HERE / "cache") / "fig2_v111_data_B.npz"
VERIFY = _F.out_dir(HERE / "verify")
GRID_END = 30.01
CURVE_KEYS = ("grid", "conf_int", "n_int", "conf_cl", "n_cl")


def curves(d, grid):
    """fig2_data_v104.py, lines 165-172, unchanged except for the grid passed in."""
    lp_int, lp_cl = -np.log10(np.clip(d.p_int, 1e-300, 1)), -np.log10(np.clip(d.p_cl, 1e-300, 1))
    conf_int, n_int, conf_cl, n_cl = [], [], [], []
    for x in grid:
        m = lp_int >= x
        n_int.append(int(m.sum())); conf_int.append(float(d.pc_cl[m].mean()) if m.sum() else np.nan)
        m = lp_cl >= x
        n_cl.append(int(m.sum())); conf_cl.append(float(d.pc_int[m].mean()) if m.sum() else np.nan)
    return dict(grid=grid, conf_int=conf_int, n_int=n_int, conf_cl=conf_cl, n_cl=n_cl)


def at(c, x):
    k = int(np.argmin(np.abs(c["grid"] - x)))
    return dict(grid_x=float(c["grid"][k]), red_pct=round(100 * c["conf_int"][k], 2), red_n=int(c["n_int"][k]),
                blue_pct=round(100 * c["conf_cl"][k], 2), blue_n=int(c["n_cl"][k]))


def main():
    if OUT.exists():
        sys.exit(f"{OUT} exists; not overwriting")
    old = dict(np.load(NPZ_V11))
    d = pd.read_csv(CELLS)
    assert d.pc_int.dtype == bool and d.pc_cl.dtype == bool

    # control: the v11 grid must give the v11 curves bit for bit
    ctl = curves(d, np.round(np.arange(0, 12.01, 0.25), 2))
    same = {k: bool(np.array_equal(np.asarray(ctl[k], float), np.asarray(old[k], float), equal_nan=True))
            for k in CURVE_KEYS}
    assert all(same.values()), same

    new = curves(d, np.round(np.arange(0, GRID_END, 0.25), 2))
    new = {k: np.asarray(v, dtype=old[k].dtype) for k, v in new.items()}
    out = dict(old)
    out.update(new)
    np.savez(OUT, **out)

    # what the figure draws: points with at least 20 cells at the level (fig2_panels_v8.create_fig2f)
    g = new["grid"]
    ok_i, ok_c = new["n_int"] >= 20, new["n_cl"] >= 20
    pc_i, pc_c = d.pc_int.values, d.pc_cl.values
    ri = d[pc_i].sort_values("p_int", kind="mergesort")
    rc = d[pc_c].sort_values("z_cl", ascending=False, kind="mergesort")
    x100_i = float(-np.log10(ri.p_int.iloc[99]))
    x100_c = float(-np.log10(max(float(rc.p_cl.iloc[99]), 1e-300)))
    res = dict(
        source_npz=str(NPZ_V11), source_cells=str(CELLS), output=str(OUT),
        control_v11_grid_reproduces_v11_curves=same,
        other_arrays_copied_unchanged=sorted(k for k in old if k not in CURVE_KEYS),
        grid=[float(g[0]), float(g[-1]), float(g[1] - g[0])], n_grid=int(len(g)),
        red_drawn_to_x=float(g[ok_i].max()), red_last_n=int(new["n_int"][ok_i][-1]),
        red_last_pct=round(100 * float(new["conf_int"][ok_i][-1]), 2),
        blue_drawn_to_x=float(g[ok_c].max()), blue_last_n=int(new["n_cl"][ok_c][-1]),
        blue_last_pct=round(100 * float(new["conf_cl"][ok_c][-1]), 2),
        red_drawn_to_x_in_v11=float(old["grid"][old["n_int"] >= 20].max()),
        max_minuslog10p_intense=float((-np.log10(np.clip(d.p_int, 1e-300, 1))).max()),
        max_minuslog10p_classic=float((-np.log10(np.clip(d.p_cl, 1e-300, 1))).max()),
        top100=dict(intense_x=round(x100_i, 2), intense_confirmed_pct=round(100 * ri.head(100).pc_cl.mean(), 1),
                    classic_x=round(x100_c, 2), classic_confirmed_pct=round(100 * rc.head(100).pc_int.mean(), 1)),
        curve_at={f"{x:g}": at(new, x) for x in (3, 9.75, 9.82, 10, 12, 20, 29.5, 29.64, 29.75, 30)},
        all_cells=dict(intense_confirmed_pct=round(100 * (pc_i & pc_c).sum() / pc_i.sum(), 1),
                       classic_confirmed_pct=round(100 * (pc_i & pc_c).sum() / pc_c.sum(), 1)),
    )
    (VERIFY / "panel_f_data.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
