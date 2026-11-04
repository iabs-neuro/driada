"""Candidates for a new example of Fig. 3d,e: a CA1 neuron selective for position and body direction.

Criterion, fixed before looking at the pictures (the rules of figures_v10.3/work/cand_3de.py for the
head-direction example, with body direction in place of head direction, plus the exploration rule of
figures_v11.1/work/glm_examples/glm_src/fig_examples_v111.py):
  1. after disentanglement the neuron keeps both labels, position and body direction, each accepted by
     the criterion of the paper (p < 0.001, excess MI > 0.04 bits, activity ratio NaN or > 1), in the
     post-disentanglement database of the paper ([results] db_dir, conditional MI at the delays of the
     pair);
  2. the mouse covered at least 10 m in the session (path_m of [v11] glm_explore_json,
     figure_chain/explore.py);
  3. the frame rate of the session comes from the CaImAn metadata, and neither optimal delay lies on
     the edge of the delay grid of the session (+-40 frames at ~20 frames/s, +-60 at ~30 frames/s);
  4. strength: score = the smaller of the two excess MIs (the ranking of fig3_data.py); the 12
     strongest eligible neurons are measured;
  5. clarity, measured on the data each panel draws: body-direction tuning by the mean vector length
     of the occupancy-normalised tuning curve of panel d (40 bins, signal shifted by the delay of the
     body-direction pair); place field by the spatial information of the occupancy-normalised map of
     panel e (20 x 20 bins with at least 5 samples, signal shifted by the delay of the position pair),
     in bits per unit of mean signal. The three neurons with the best mean rank of the two clarity
     measures among the 12 are drawn; neurons with no label besides the two are preferred on ties.

Inputs: the databases of [results] db_dir, the frame rates of [v11] fps_table (with their source),
the synchronized sessions of [data] nof_aligned, the path lengths of [v11] glm_explore_json.
make_fig3_v111.py reads candidates.json from where this script writes it.

usage: python -X utf8 -B figure_chain/compute_pack_v11.1/scripts/f_candidates.py
       ->  compute_pack_v11.1/F_fig3/candidates_all.csv, candidates_top12.csv, candidates.json
           under [figures] output_dir/figure_chain
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # figure_chain/
import _paths_fig as _F  # noqa: E402  (puts the paper folder on sys.path)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = _F.out_dir(_F.CHAIN / "compute_pack_v11.1" / "F_fig3")
FPS = _F.V.get("fps_table")
EXPLORE = _F.V.get("glm_explore_json")
PAPER_PRE_DB, PAPER_POST_DB = _F.P.DB_PRE, _F.P.DB_POST
NOF_V2 = _F.P.NOF_ALIGNED
MIN_PATH_M = 10.0
AGG = {"any place", "any object"}
BODY, PLACE = "bodydirection_2d", "place"
N_MEASURED, N_DRAWN = 12, 3


def session_fps_v2(name):
    """Frame rate of a CA1 session ('NOF_H01_1D') from _metadata of its v2 file (pack_common.py of the pack)."""
    with np.load(NOF_V2 / f"{name}_aligned.npz", allow_pickle=True) as d:
        return float(d["_metadata"].item()["fps"])


def write_json(path, obj):
    """pack_common.write_json of the pack."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False,
                               default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                    encoding="utf-8")


def selective(df):
    df = df[~df.feature.isin(AGG)]
    ok = (df.pval < 1e-3) & (df.me_excess > 0.04) & (df.signal_ratio.isna() | (df.signal_ratio > 1))
    return df[ok]


def scaled_signal(cal):
    s = (cal - cal.min()) / (cal.max() - cal.min())
    return np.clip(s, 0, None)


def shift(sig, other, delay):
    """Panels d and e: the signal is delayed by the pair's optimal delay relative to behaviour."""
    if delay > 0:
        return sig[delay:], [o[:-delay] for o in other]
    if delay < 0:
        return sig[:delay], [o[-delay:] for o in other]
    return sig, other


def body_tuning(npz, n, delay, bins=40):
    with np.load(npz, allow_pickle=True) as d:
        ang, cal = d["bodydirection"], d["Calcium"][n].astype(float)
    sig, (ang,) = shift(scaled_signal(cal), [ang], delay)
    a = np.mod(ang + np.pi, 2 * np.pi) - np.pi
    edges = np.linspace(-np.pi, np.pi, bins + 1)
    idx = np.clip(np.digitize(a, edges) - 1, 0, bins - 1)
    occ = np.bincount(idx, minlength=bins)
    hist = np.bincount(idx, weights=sig, minlength=bins) / np.maximum(occ, 1)
    centers = (edges[:-1] + edges[1:]) / 2
    mvl = float(np.abs(np.sum(hist * np.exp(1j * centers))) / np.sum(hist))
    pref = float(np.degrees(np.angle(np.sum(hist * np.exp(1j * centers)))))
    return mvl, pref, float(hist.max() / hist.mean())


def place_information(npz, n, delay, bins=20, min_samples=5):
    with np.load(npz, allow_pickle=True) as d:
        x, y, cal = d["x"].astype(float), d["y"].astype(float), d["Calcium"][n].astype(float)
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    sig, (x, y) = shift(scaled_signal(cal), [x, y], delay)
    ix = np.clip((x * bins).astype(int), 0, bins - 1)
    iy = np.clip((y * bins).astype(int), 0, bins - 1)
    flat = iy * bins + ix
    occ = np.bincount(flat, minlength=bins * bins)
    tot = np.bincount(flat, weights=sig, minlength=bins * bins)
    ok = occ >= min_samples
    rate = tot[ok] / occ[ok]
    p = occ[ok] / occ[ok].sum()
    mean = float(np.sum(p * rate))
    r = rate / mean
    with np.errstate(divide="ignore", invalid="ignore"):
        si = float(np.nansum(p * r * np.log2(np.where(r > 0, r, 1))))
    return si, float(rate.max() / mean)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    post = selective(pd.read_csv(PAPER_POST_DB))
    pre = pd.read_csv(PAPER_PRE_DB)
    fps = pd.read_csv(FPS)
    fps[["mouse", "session"]] = fps.session.str.extract(r"NOF_(H\d+)_(\dD)")
    fps = fps.set_index(["mouse", "session"])
    path_m = {r["session"]: r["path_m"] for r in json.loads(EXPLORE.read_text(encoding="utf-8"))["rows"]}

    edge = pre.groupby(["mouse", "session"]).opt_delay.agg(lambda v: int(np.abs(v).max()))
    grid = {}
    for (m, s), e in edge.items():
        expect = 40 if float(fps.loc[(m, s), "fps"]) < 25 else 60
        if e != expect:
            raise SystemExit(f"{m} {s}: largest |delay| {e}, expected edge {expect}")
        grid[(m, s)] = expect

    key = ["mouse", "session", "neuron_idx"]
    me = post.pivot_table(index=key, columns="feature", values="me_excess")
    dl = post.pivot_table(index=key, columns="feature", values="opt_delay")
    pv = post.pivot_table(index=key, columns="feature", values="pval")
    rows = []
    for k in me.index[me[BODY].notna() & me[PLACE].notna()]:
        m, s, n = k
        f = session_fps_v2(f"NOF_{m}_{s}")
        e = grid[(m, s)]
        d_b, d_p = int(dl.loc[k, BODY]), int(dl.loc[k, PLACE])
        labels = [c for c in me.columns if pd.notna(me.loc[k, c])]
        rows.append(dict(mouse=m, session=s, neuron_idx=int(n), fps=f, fps_source=fps.loc[(m, s), "source"],
                         path_m=path_m[f"{m}_{s}"], delay_edge_frames=e,
                         me_excess_body=float(me.loc[k, BODY]), me_excess_place=float(me.loc[k, PLACE]),
                         score=float(min(me.loc[k, BODY], me.loc[k, PLACE])),
                         pval_body=float(pv.loc[k, BODY]), pval_place=float(pv.loc[k, PLACE]),
                         delay_body_frames=d_b, delay_place_frames=d_p,
                         delay_body_s=round(d_b / f, 3), delay_place_s=round(d_p / f, 3),
                         body_at_edge=abs(d_b) >= e, place_at_edge=abs(d_p) >= e,
                         other_labels=";".join(c for c in labels if c not in (BODY, PLACE)),
                         n_labels=len(labels)))
    cand = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    cand["rank_all"] = np.arange(1, len(cand) + 1)
    cand["explored"] = cand.path_m >= MIN_PATH_M
    cand["eligible"] = cand.explored & (cand.fps_source == "CaImAn fr") & ~cand.body_at_edge & ~cand.place_at_edge
    cand.to_csv(OUT / "candidates_all.csv", index=False)

    top = cand[cand.eligible].head(N_MEASURED).copy()
    meas = []
    for r in top.itertuples():
        npz = NOF_V2 / f"NOF_{r.mouse}_{r.session}_aligned.npz"
        mvl, pref, b_peak = body_tuning(npz, r.neuron_idx, r.delay_body_frames)
        si, p_peak = place_information(npz, r.neuron_idx, r.delay_place_frames)
        meas.append(dict(body_mvl=mvl, body_pref_deg=pref, body_peak_over_mean=b_peak,
                         place_info_bits=si, place_peak_over_mean=p_peak))
    top = pd.concat([top.reset_index(drop=True), pd.DataFrame(meas)], axis=1)
    top["clarity_rank"] = (top.body_mvl.rank(ascending=False) + top.place_info_bits.rank(ascending=False)) / 2
    top = top.sort_values(["clarity_rank", "n_labels", "score"], ascending=[True, True, False]).reset_index(drop=True)
    top.to_csv(OUT / "candidates_top12.csv", index=False)
    drawn = top.head(N_DRAWN)

    # the current example (head direction + position), measured the same way for comparison
    cur = dict(mouse="H08", session="4D", neuron_idx=370, delay_hd_frames=20, delay_place_frames=45)
    npz = NOF_V2 / "NOF_H08_4D_aligned.npz"
    with np.load(npz, allow_pickle=True) as d:
        ang, cal = d["headdirection"], d["Calcium"][370].astype(float)
    sig, (ang,) = shift(scaled_signal(cal), [ang], 20)
    a = np.mod(ang + np.pi, 2 * np.pi) - np.pi
    edges = np.linspace(-np.pi, np.pi, 41)
    idx = np.clip(np.digitize(a, edges) - 1, 0, 39)
    hist = np.bincount(idx, weights=sig, minlength=40) / np.maximum(np.bincount(idx, minlength=40), 1)
    centers = (edges[:-1] + edges[1:]) / 2
    cur["hd_mvl"] = float(np.abs(np.sum(hist * np.exp(1j * centers))) / np.sum(hist))
    cur["place_info_bits"] = place_information(npz, 370, 45)[0]
    cur["path_m"] = path_m["H08_4D"]
    res = dict(source=str(PAPER_POST_DB), n_body_and_place=int(len(cand)),
               n_explored=int(cand.explored.sum()),
               n_eligible=int(cand.eligible.sum()),
               n_eligible_only_body_and_place=int((cand.eligible & (cand.n_labels == 2)).sum()),
               excluded=dict(path_under_10m=int((~cand.explored).sum()),
                             fps_not_caiman=int((cand.explored & (cand.fps_source != "CaImAn fr")).sum()),
                             delay_at_edge=int((cand.explored & (cand.fps_source == "CaImAn fr")
                                                & (cand.body_at_edge | cand.place_at_edge)).sum())),
               drawn=drawn.to_dict("records"), current_example=cur)
    write_json(OUT / "candidates.json", res)
    cols = ["rank_all", "mouse", "session", "neuron_idx", "fps", "path_m", "me_excess_body", "me_excess_place",
            "delay_body_s", "delay_place_s", "other_labels", "body_mvl", "place_info_bits", "clarity_rank"]
    print(top[cols].to_string(index=False))
    print(json.dumps({k: v for k, v in res.items() if k not in ("drawn",)}, indent=1, default=str))


if __name__ == "__main__":
    main()
