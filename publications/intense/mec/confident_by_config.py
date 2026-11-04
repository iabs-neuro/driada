# -*- coding: utf-8 -*-
"""Confidently labelled head-direction cells (margin > 4) found by INTENSE and GLM, by imaging
configuration (microscope column of [mec] fps_table: Zoom* or Bolt) and session order; and, for
the four animals in FOUR, whose object sessions held most of INTENSE's misses, the authors'
labels and the detections in all three sessions of the animal. Also counts the labels with an
impossible mean vector length (mvl > 1).

usage: python -B mec/confident_by_config.py --labels <arbiter_labels.csv> --results <results.csv>
                                            --out <dir>
Writes confident_by_config.csv and four_sessions.csv into <dir>.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intense_paths as P  # noqa: E402

FPS = P.get("mec", "fps_table")
FOUR = ["316933418721b927", "11aad90b527d9584", "9574c34399b1ba85", "2501c7bad5c64d97"]


def main():
    ap = argparse.ArgumentParser()
    for k in ("labels", "results", "out"):
        ap.add_argument(f"--{k}", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(FPS).set_index("session")
    lab = pd.read_csv(a.labels)
    res = pd.read_csv(a.results).set_index("session")
    U = lab[np.isfinite(lab.mvl) & np.isfinite(lab.mvl_95) & np.isfinite(lab.mvl_99)].copy()
    U["m"] = (U.mvl - U.mvl_95) / (U.mvl_99 - U.mvl_95)
    U["order"] = U.session.str.split("_").str[1]
    U["config"] = np.where(U.session.map(meta.microscope).str.startswith("Zoom"), "Zoom", "Bolt")
    U["frames"] = U.session.map(meta.frames)
    S = U[U.m > 4]
    t = S.groupby(["config", "order"]).agg(
        n=("int_hd", "size"), intense=("int_hd", "sum"), glm=("glm_hd", "sum"),
        impossible_mvl=("mvl", lambda v: int((v > 1).sum())), med_frames=("frames", "median"))
    t.to_csv(a.out / "confident_by_config.csv")
    print(t.to_string())
    print(f"confident: {len(S)}, INTENSE misses: {int((S.int_hd == 0).sum())}, "
          f"of them in Bolt sessions: {int(((S.int_hd == 0) & (S.config == 'Bolt')).sum())}; "
          f"labels with mvl > 1: {int((U.mvl > 1).sum())} (all labelled: {int(U[U.mvl > 1].is_hd95.all())})")
    sess = res.assign(order=res.index.str.split("_").str[1]).copy()
    sess["frames"] = sess.index.map(meta.frames)
    print("median session length, frames, by order:",
          sess.groupby("order").frames.median().to_dict())
    rows = []
    L = lab[np.isfinite(lab.mvl) & np.isfinite(lab.mvl_95)]
    for animal in FOUR:
        for s in sorted(x for x in res.index if x.startswith(animal + "_")):
            T = L[L.session == s]
            C = S[S.session == s]
            rows.append({"session": s, "label": meta.loc[s, "label"], "frames": int(meta.loc[s, "frames"]),
                         "cells": len(T), "authors_hd95": int(T.is_hd95.sum()),
                         "confident": len(C), "confident_found_intense": int(C.int_hd.sum()),
                         "mvl_over_1": int((T.mvl > 1).sum()),
                         "intense_hd_session": int(res.loc[s, "int_bytype_head_direction"]),
                         "intense_speed_session": int(res.loc[s, "int_bytype_speed"]),
                         "glm_hd_session": int(res.loc[s, "glm_head_direction"])})
    f = pd.DataFrame(rows)
    f.to_csv(a.out / "four_sessions.csv", index=False)
    print(f.to_string(index=False))


if __name__ == "__main__":
    main()
