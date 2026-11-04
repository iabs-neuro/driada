# -*- coding: utf-8 -*-
"""Real CA1 example neurons for the GLM section of the Supplementary Information (Supplementary Fig. 4).

Same layout and style as fig_examples in figure_chain/science/intense_paper/extracted/
make_appendix_figures.py, with two changes: the INTENSE side is read from
<[results] db_dir>/NOF_pre_disent.csv (rows = aligned-file rows), and each category shows its neuron with the largest excess MI
rather than the first one in file order: the neuron at the 90th percentile of excess MI
within its category (strong but typical; the maximum tends to pick outliers driven by a
few large transients). Calcium is shifted by the pair's optimal delay
before plotting, since INTENSE detected the pair at that delay.

Two switches are read from the environment (make_glm_examples_v111.py sets them): GLM_EXAMPLES_DB, the
pre-disentanglement database the examples are picked from (default <[results] db_dir>); GLM_EXAMPLES_FPS, a
table of session frame rates (session, fps): when given, the time axis is samples * DS / frame rate of the
session instead of samples / 4.

Each category shows the neuron at the 90th
percentile or, when that neuron comes from a session in which the mouse covered less than 10 m, the
neuron nearest to it in rank from a session in which it covered more: an example of directional or
spatial tuning needs an animal that sampled the directions and the arena. Path lengths are read from
EXPLORE_JSON (path of the tracked position per session). The signal axes read "signal, a.u.", as in
the main figures, since the calcium traces are not normalised to a baseline; speed carries its unit;
panel letters sit above the tick labels.

Rate-map bins with fewer than five samples are drawn gray, so that they are not confused with the
peak of the field, which the "hot" colormap draws white. Panel letters are in lower case, head direction in
degrees, as in the main text; a color scale stands beside each spatial rate map, in the units of the tuning
curves (mean signal, a.u.).

Inputs: the GLM selections of every CA1 neuron ([v11] glm_ca1_per_neuron), the database of
GLM_EXAMPLES_DB, the sessions of [v11] nof_v1, the frame rates of GLM_EXAMPLES_FPS and the path
length of each session ([v11] glm_explore_json, written by figure_chain/explore.py).
"""
import os
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # figure_chain/
import _paths_fig as _F  # noqa: E402
sys.path.insert(0, str(_F.CHAIN / "science" / "intense_paper" / "extracted"))
import make_appendix_figures as M   # house style (rcParams), letter(), colours

ALIGNED = _F.V.get("nof_v1")
DS = 5
EMPTY_BIN_GRAY = "#b3b3b3"   # rate-map bins with fewer than five samples
B2DB = {"position": "place", "head_direction": "headdirection_2d", "speed": "speed"}

DB_CSV = Path(os.environ.get("GLM_EXAMPLES_DB", _F.P.DB_PRE))
FPS_CSV = os.environ.get("GLM_EXAMPLES_FPS")
pn = pd.read_csv(_F.V.get("glm_ca1_per_neuron"))
db = pd.read_csv(DB_CSV)
ok = (db.pval < 0.001) & (db.me_excess > 0.04) & (db.signal_ratio.isna() | (db.signal_ratio > 1))
sel = db[ok & db.feature.isin(B2DB.values())]
key = ["mouse", "session", "neuron_idx"]


EXPLORE_JSON = _F.V.get("glm_explore_json")   # figure_chain/explore.py
MIN_PATH_M = 10.0
import json  # noqa: E402
PATH_M = {r["session"]: r["path_m"] for r in json.loads(EXPLORE_JSON.read_text(encoding="utf-8"))["rows"]}


def pick(block, glm_flag, label):
    s = sel[sel.feature == B2DB[block]].merge(pn[key + [f"glm_{block}"]], on=key)
    s = s[s[f"glm_{block}"] == glm_flag].sort_values("me_excess").reset_index(drop=True)
    k = int(round(0.9 * (len(s) - 1)))
    explored = [i for i in range(len(s)) if PATH_M[f"{s.mouse[i]}_{s.session[i]}"] >= MIN_PATH_M]
    r = s.iloc[min(explored, key=lambda i: (abs(i - k), i))]
    return {"mouse": r.mouse, "session": r.session, "ni": int(r.neuron_idx), "block": block,
            "label": label, "mex": float(r.me_excess), "pval": float(r.pval),
            "delay": int(r.opt_delay), "glm": int(glm_flag)}


picks = [pick("head_direction", 0, "INTENSE-only HD"),
         pick("speed", 0, "INTENSE-only speed"),
         pick("position", 1, "convergent place"),
         pick("position", 0, "INTENSE-only place")]


def load(p):
    d = np.load(ALIGNED / f"NOF_{p['mouse']}_{p['session']}_aligned.npz", allow_pickle=True)
    ca = np.roll(d["Calcium"][p["ni"]].astype(float), -p["delay"])
    return {"resp": ca[::DS], "x": d["x"][::DS], "y": d["y"][::DS],
            "hd": d["headdirection"][::DS], "speed": d["speed"][::DS]}


def tuning_1d(feat, resp, n=20, circular=False):
    if circular:
        feat = np.mod(feat, 2 * np.pi); e = np.linspace(0, 2 * np.pi, n + 1)
    else:
        e = np.quantile(feat, np.linspace(0, 1, n + 1))
        e = np.maximum.accumulate(e + np.arange(len(e)) * 1e-9)
    idx = np.clip(np.searchsorted(e, feat, side="right") - 1, 0, n - 1)
    return 0.5 * (e[:-1] + e[1:]), np.array(
        [resp[idx == k].mean() if np.any(idx == k) else np.nan for k in range(n)])


def main(out):
    fig, axes = plt.subplots(2, 4, figsize=(18 * M.CM, 9 * M.CM))
    for j, p in enumerate(picks):
        b = load(p); resp = b["resp"]
        ok_t = np.isfinite(resp) & np.isfinite(b["x"]) & np.isfinite(b["y"]) & \
            np.isfinite(b["hd"]) & np.isfinite(b["speed"])
        if FPS_CSV:
            fps = pd.read_csv(FPS_CSV).set_index("session").fps[f"NOF_{p['mouse']}_{p['session']}"]
            t = np.arange(len(resp)) * DS / fps
        else:
            t = np.arange(len(resp)) / 4.0
        is_pos = p["block"] == "position"
        fv, circ, name = {"head_direction": (b["hd"], True, "head direction (°)"),
                          "speed": (b["speed"], False, "speed (cm/s)"),
                          "position": (None, False, "position")}[p["block"]]
        ax = axes[0, j]; w = slice(480, min(len(resp), 1680))
        ax.plot(t[w], resp[w], color="k", lw=0.4)
        if not is_pos:
            ax2 = ax.twinx(); ax2.plot(t[w], fv[w], color="#2171b5", lw=0.4, alpha=0.55)
            ax2.set_yticks([]); ax2.spines["right"].set_visible(False)
        ax.set_title(p["label"], fontsize=8); ax.set_xlabel("time (s)")
        if j == 0:
            ax.set_ylabel("signal, a.u.")
        M.letter(ax, "abcd"[j], dx=-0.22, dy=1.30)
        ax = axes[1, j]
        if is_pos:
            nb = 10
            x, y, r = b["x"][ok_t], b["y"][ok_t], resp[ok_t]
            xi = np.clip(((x - x.min()) / (np.ptp(x) + 1e-12) * nb).astype(int), 0, nb - 1)
            yi = np.clip(((y - y.min()) / (np.ptp(y) + 1e-12) * nb).astype(int), 0, nb - 1)
            rate = np.full((nb, nb), np.nan)
            for a in range(nb):
                for c in range(nb):
                    m = (xi == a) & (yi == c)
                    if m.sum() >= 5:
                        rate[c, a] = r[m].mean()
            cmap = plt.get_cmap("hot").copy()
            cmap.set_bad(EMPTY_BIN_GRAY)
            im = ax.imshow(rate, origin="lower", cmap=cmap, aspect="equal")
            ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("spatial rate map")
            cax = make_axes_locatable(ax).append_axes("right", size="6%", pad=0.04)
            cb = fig.colorbar(im, cax=cax)
            cb.outline.set_linewidth(0.5)
            cb.ax.tick_params(labelsize=6, width=0.5, length=2)
            cb.set_label("mean signal, a.u.", fontsize=6, labelpad=2)
        else:
            c, m = tuning_1d(fv[ok_t], resp[ok_t], circular=circ)
            if circ:
                c = np.degrees(c)
                ax.set_xticks([0, 90, 180, 270, 360])
            ax.plot(c, m, "o-", ms=2.5, lw=1.0, color=M.RED); ax.set_xlabel(name)
            if j == 0:
                ax.set_ylabel("mean signal, a.u.")
        verdict = (f"INTENSE: {p['mex']:.2f} bit, p={p['pval']:.0e}\n"
                   f"GLM: {'selected' if p['glm'] else 'not selected'}")
        ax.text(0.03, 1.22, verdict, transform=ax.transAxes, va="top", fontsize=6,
                bbox=dict(boxstyle="round,pad=0.35", fc="#f5f0e6", ec="#999999", lw=0.5))
    fig.subplots_adjust(wspace=0.45, hspace=0.95)
    fig.savefig(out, **M.SAVE); plt.close(fig)
    for p in picks:
        print(p)


if __name__ == "__main__":
    main(sys.argv[1])
