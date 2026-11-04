"""Agreement of the two place-cell methods (Results 2.2) and the cell counts of Figure 2e (Methods),
from the per-cell table of Figure 2.

Results 2.2, "rising from 60% of the INTENSE place cells to 94% of the 100 most significant ones,
and from 40% to 84% for the classic place cells":
  all_intense_confirmed_pct     INTENSE place cells that the classic method also labels
  top100_intense_confirmed_pct  the same among the 100 INTENSE place cells with the smallest p
  all_classic_confirmed_pct     classic place cells that INTENSE also labels
  top100_classic_confirmed_pct  the same among the 100 classic place cells with the largest z
Cells are ranked with a stable sort (row order of the table among equal values); the ties at rank
100 are reported, since a tie there would make the top 100 depend on the row order.

Methods, Comparison of place-cell populations, "n = 13,191 cells at p < 0.1, 3,370 at p < 10^-3,
1,036 at p < 10^-5 and 347 at p < 10^-7": cells with the p of both methods below the threshold
itself and all three map-similarity metrics finite (fig2e_cells_at_p). The numbers file of
fig2_data_v104.py (fig2E_map_similarity.n_cells_at_p) gives instead the counts at the nearest point
of the 50-point curve grid, which is 1.1e-3 and 9.1e-6 for "1e-3" and "1e-5".

Input: [fig2_numbers] cells, the table fig2_v104_cells_B.csv written by
figure_chain/figures_v10.4/work/fig2_src/fig2_data_v104.py --classic B.
Writes <[results] numbers_dir>/place_cell_agreement.json.
usage: python numbers/place_cell_agreement.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intense_paths as P  # noqa: E402

CELLS = P.get("fig2_numbers", "cells")
THRESHOLDS = (1e-1, 1e-3, 1e-5, 1e-7)


def exact(d, p):
    """Cells counted at one threshold of Figure 2e, and their mean SSIM.

    Parameters
    ----------
    d : pandas.DataFrame
        Per-cell table with the columns p_int, p_cl, mse, psnr and ssim.
    p : float
        Threshold applied to the p-value of both methods.

    Returns
    -------
    n : int
        Cells with both p-values below ``p`` and all three map-similarity metrics finite.
    ssim : float
        Mean SSIM of these cells.
    """
    m = (d.p_int < p) & (d.p_cl < p) & np.isfinite(d[["mse", "psnr", "ssim"]]).all(axis=1)
    return int(m.sum()), float(d.ssim[m].mean())


def main():
    """Write place_cell_agreement.json to [results] numbers_dir and print it."""
    d = pd.read_csv(CELLS)
    pc_i = d["pc_int"].astype(bool).to_numpy()
    pc_c = d["pc_cl"].astype(bool).to_numpy()
    out = {"cells": len(d)}
    out["n_intense_pc"] = int(pc_i.sum())
    out["n_classic_pc"] = int(pc_c.sum())
    out["n_both"] = int((pc_i & pc_c).sum())
    out["all_intense_confirmed_pct"] = round(100 * (pc_i & pc_c).sum() / pc_i.sum(), 1)
    out["all_classic_confirmed_pct"] = round(100 * (pc_i & pc_c).sum() / pc_c.sum(), 1)

    ri = d[pc_i].sort_values("p_int", kind="mergesort")
    rcl = d[pc_c].sort_values("z_cl", ascending=False, kind="mergesort")
    out["top100_intense_confirmed_pct"] = round(100 * ri.head(100)["pc_cl"].astype(bool).mean(), 1)
    out["top100_classic_confirmed_pct"] = round(100 * rcl.head(100)["pc_int"].astype(bool).mean(), 1)
    p100 = float(ri["p_int"].iloc[99])
    out["intense_p_at_rank100"] = p100
    out["intense_ties_at_rank100"] = int((ri["p_int"] == p100).sum())
    out["intense_better_than_rank100"] = int((ri["p_int"] < p100).sum())
    z100 = float(rcl["z_cl"].iloc[99])
    out["classic_z_at_rank100"] = z100
    out["classic_ties_at_rank100"] = int((rcl["z_cl"] == z100).sum())
    out["classic_better_than_rank100"] = int((rcl["z_cl"] > z100).sum())

    counts = {f"{p:g}": exact(d, p) for p in THRESHOLDS}
    out["fig2e_cells_at_p"] = {k: v[0] for k, v in counts.items()}
    out["fig2e_mean_ssim_at_p"] = {k: v[1] for k, v in counts.items()}

    P.NUMBERS_DIR.mkdir(parents=True, exist_ok=True)
    (P.NUMBERS_DIR / "place_cell_agreement.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
