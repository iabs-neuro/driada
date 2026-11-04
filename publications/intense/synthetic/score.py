"""Scores of disentanglement on the synthetic sessions against the known truth.

Reads truth/<session>.json and results/<session>.pkl (run_synth.py) of every session
that has both and writes into tables/:

  neurons.csv          one row per neuron x order: truth, labels before and after
                       disentanglement (paper criterion: p < 0.001, excess MI > 0.04 bits,
                       activity ratio > 1), outcome
  outcomes.csv         shares of outcomes per category x order (all SNR levels)
  outcomes_snr.csv     the same per SNR level
  rewrites.csv         what the true label was replaced with
  decisions.csv        every decision of the priority rules, the object rule, the zone
                       rule and conditional MI, with its verdict against the truth
  decisions_summary.csv  counts per order x step x verdict
  summary.md           the tables above in readable form

Outcomes of a neuron with one true variable T (labels after disentanglement F,
labels before it B, both at the paper criterion):
  exact          F == {T}
  exact_merged   place cell, F == {place-Z} with the field centre in zone Z
  extra          T in F together with other labels
  rewritten      T not in F, F not empty (the label went to another variable)
  lost           T in B, F empty
  missed         T not in B (INTENSE did not find it; disentanglement not involved)
A merged label place-Z counts as both place and Z when checking T in F, except
for "exact". Mixed neurons (truth {A, B}): both, one, rewritten, lost, missed
(neither in B). Null neurons: any label is false.

usage: python score.py
"""
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "tables"
PVAL_THR, MI_THR = 0.001, 0.04
ORDERS = ["base", "thrfirst"]


def passes(s):
    if not s:
        return False
    sr, me_excess, pval = s.get("signal_ratio"), s.get("me_excess"), s.get("pval")
    return (pval is not None and pval < PVAL_THR and me_excess is not None
            and me_excess > MI_THR and (sr is None or sr > 1))


def to_source(name, sources):
    name = sources.get(name, name)
    if name.endswith("_2d"):
        name = name[:-3]
    if "-" in name:  # merged place-zone label, place part may carry a derived name
        a, b = name.split("-", 1)
        name = f"{sources.get(a, a)}-{b}"
    return name


def expand(labels):
    out = set()
    for l in labels:
        out.update(l.split("-", 1) if "-" in l else [l])
    return out


def labels_before(res, nid):
    st = res["stats"].get(nid, {})
    return {to_source(f, res["sources"]) for f in res["significant"].get(nid, [])
            if passes(st.get(f))}


def labels_after(res, order, nid):
    st = res["orders"][order]["final_stats"].get(nid, {})
    return {to_source(f, res["sources"]) for f, s in st.items() if passes(s)}


def outcome_single(T, B, F, row):
    if T not in B:
        return "missed"
    if F == {T}:
        return "exact"
    if row["category"].startswith("place") and len(F) == 1:
        (f,) = F
        if f.startswith("place-") and f.split("-", 1)[1] in row.get("field_zones", []):
            return "exact_merged"
    if T in expand(F):
        return "extra"
    return "rewritten" if F else "lost"


def outcome_mixed(T, B, F):
    if not (set(T) & B):
        return "missed"
    got = set(T) & expand(F)
    if len(got) == 2:
        return "both"
    if got:
        return "one"
    return "rewritten" if F else "lost"


def verdict_pair(a, b, r, truth):
    ta, tb = a in truth, b in truth
    if r == 0.5:
        return "correct" if ta and tb else ("kept_spurious" if ta or tb else "irrelevant")
    kept, removed = (a, b) if r == 0 else (b, a)
    tk, tr = kept in truth, removed in truth
    if tr and not tk:
        return "wrong"
    if tr and tk:
        return "wrong_mixed"
    return "correct" if tk else "irrelevant"


def verdict_merge(zone, row):
    cat, truth = row["category"], set(row["truth"])
    if cat.startswith("place_"):
        return "correct" if zone in row.get("field_zones", []) else "wrong"
    if cat == "place+zone" and truth == {"place", zone}:
        return "merged_mixed"
    if "place" not in truth and zone in truth:
        return "wrong"  # a zone cell turned into a place cell with a field in the zone
    return "wrong" if truth & {"place", zone} else "irrelevant"


def score_session(truth, res):
    rows, decisions = [], []
    src = res["sources"]
    for row in truth["neurons"]:
        nid = row["neuron"]
        T = row["truth"]
        B = labels_before(res, nid)
        for order in ORDERS:
            F = labels_after(res, order, nid)
            if row["category"] == "null":
                out = "clean" if not F else "false_label"
            elif len(T) == 1:
                out = outcome_single(T[0], B, F, row)
            else:
                out = outcome_mixed(T, B, F)
            rows.append({"session": truth["session"], "neuron": nid, "category": row["category"],
                         "snr": row["snr"], "order": order, "truth": "+".join(T),
                         "before": ";".join(sorted(B)), "after": ";".join(sorted(F)),
                         "outcome": out, "before_has_all_truth": set(T) <= B})
            # decisions
            pnd = res["orders"][order]["per_neuron_disent"].get(nid)
            tags = res["tags"].get(order, {}).get(nid, {"pairs": {}, "merges": {}, "removed": {}})
            tset = set(T)
            if pnd:
                for (a, b), info in pnd["pairs"].items():
                    sa, sb = to_source(a, src), to_source(b, src)
                    source = info["source"]
                    if source == "pre_filter":
                        step = (tags["pairs"].get((sa, sb)) or tags["pairs"].get((sb, sa)) or ("filter?",))[0]
                    else:
                        step = {"standard": "cmi", "not_significant": "uncorrelated"}.get(source, source)
                    decisions.append({"session": truth["session"], "neuron": nid,
                                      "category": row["category"], "snr": row["snr"], "order": order,
                                      "step": step, "pair": f"{sa}|{sb}", "result": info["result"],
                                      "verdict": verdict_pair(sa, sb, info["result"], tset)})
            for new, (step, (a, b)) in tags["merges"].items():
                zone = to_source(b, src) if to_source(a, src) == "place" else to_source(a, src)
                decisions.append({"session": truth["session"], "neuron": nid, "category": row["category"],
                                  "snr": row["snr"], "order": order, "step": f"{step}_merge",
                                  "pair": f"place|{zone}", "result": "merge",
                                  "verdict": verdict_merge(zone, row)})
            for f, step in tags["removed"].items():
                f = to_source(f, src)
                decisions.append({"session": truth["session"], "neuron": nid, "category": row["category"],
                                  "snr": row["snr"], "order": order, "step": f"{step}_remove",
                                  "pair": f, "result": "remove",
                                  "verdict": "wrong" if f in tset else "correct"})
    return rows, decisions


SINGLE_OUT = ["exact", "exact_merged", "extra", "rewritten", "lost", "missed"]
MIXED_OUT = ["both", "one", "rewritten", "lost", "missed"]
NULL_OUT = ["clean", "false_label"]


def shares(df, keys):
    g = df.groupby(keys + ["outcome"]).size().unstack(fill_value=0)
    n = g.sum(axis=1)
    found = n - g.get("missed", 0)
    out = pd.DataFrame({"n": n, "found_before": found})
    for c in SINGLE_OUT + MIXED_OUT + NULL_OUT:
        if c in g.columns and c not in out.columns:
            out[c] = g[c]
    for c in out.columns[2:]:
        if c not in ("missed", "clean", "false_label"):
            out[f"{c}_pct_of_found"] = (100 * out[c] / out["found_before"].replace(0, np.nan)).round(1)
    return out.reset_index()


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def main():
    OUT.mkdir(exist_ok=True)
    rows, decisions, sessions = [], [], []
    for tpath in sorted((HERE / "truth").glob("*.json")):
        rpath = HERE / "results" / f"{tpath.stem}.pkl"
        if not rpath.exists():
            continue
        truth = json.loads(tpath.read_text(encoding="utf-8"))
        with open(rpath, "rb") as f:
            res = pickle.load(f)
        r, d = score_session(truth, res)
        rows += r
        decisions += d
        sessions.append(tpath.stem)
    df = pd.DataFrame(rows)
    dec = pd.DataFrame(decisions)
    df.to_csv(OUT / "neurons.csv", index=False)
    dec.to_csv(OUT / "decisions.csv", index=False)
    out = shares(df, ["category", "order"])
    out.to_csv(OUT / "outcomes.csv", index=False)
    out_snr = shares(df, ["category", "snr", "order"])
    out_snr.to_csv(OUT / "outcomes_snr.csv", index=False)

    single = df[df.outcome.isin(["rewritten", "extra"])]
    rew = Counter()
    for _, r in single.iterrows():
        for lab in r["after"].split(";"):
            if lab and lab not in r["truth"].split("+"):
                rew[(r["order"], r["category"], r["truth"] if r["category"] in ("zone", "object") else "", lab)] += 1
    rew_df = pd.DataFrame([dict(order=k[0], category=k[1], truth=k[2], replaced_or_added=k[3], n=v)
                           for k, v in rew.most_common()])
    rew_df.to_csv(OUT / "rewrites.csv", index=False)

    dsum = dec.groupby(["order", "step", "verdict"]).size().unstack(fill_value=0).reset_index()
    dsum.to_csv(OUT / "decisions_summary.csv", index=False)
    dcat = dec.groupby(["order", "step", "category", "verdict"]).size().unstack(fill_value=0).reset_index()
    dcat.to_csv(OUT / "decisions_by_category.csv", index=False)

    md = [f"# Disentanglement on synthetic neurons: tables\n\nSessions ({len(sessions)}): {', '.join(sessions)}. "
          f"Neurons: {df[df.order == 'base'].shape[0]}.\n"]
    view_cols = ["category", "order", "n", "found_before"]
    for title, outs in [("One true variable", SINGLE_OUT), ("Mixed", MIXED_OUT)]:
        sub = out[out.category.isin(df[df.outcome.isin(outs[:-1])].category.unique())]
        sub = sub[~sub.category.isin(["null"])]
        cols = view_cols + [c for c in outs if c in sub.columns] + \
            [f"{c}_pct_of_found" for c in outs if f"{c}_pct_of_found" in sub.columns]
        if title == "One true variable":
            sub = sub[~sub.category.str.contains(r"\+")]
        else:
            sub = sub[sub.category.str.contains(r"\+")]
        md += [f"## {title} (shares of those found before disentanglement)\n", md_table(sub[cols]), ""]
    nul = out[out.category == "null"]
    md += ["## Neurons without dependence\n", md_table(nul[[c for c in ["category", "order", "n", "clean", "false_label"]
                                                     if c in nul.columns]]), ""]
    md += ["## Decisions by step\n", md_table(dsum), ""]
    md += ["## What replaced the true label (first 40)\n", md_table(rew_df.head(40)), ""]
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
