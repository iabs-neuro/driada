# -*- coding: utf-8 -*-
"""The numbers of the MEC analysis, from the parts of run_recompute_primary.py and the authors' labels.

Reads <[mec] recompute_dir>/parts, the session files of [mec] sessions_dir,
<[mec] arbiter_dir>/arbiter_labels.csv, [mec] fps_all (imaging configuration of a session, object in the
arena) and [mec] fps_table (its sessions form the second base below).

Four variants of INTENSE: with the delay search and without it, each with no threshold on the information
and with the 0.04 bits of the main text. new_on, the delay search with no threshold, is the setting of
the Supplementary Information. The counts are given on two bases of sessions, whole and by imaging
configuration: all analysed sessions (base_all) and the sessions of [mec] fps_table (base_213). The
comparison with the authors' labels can only use the sessions present in arbiter_labels.csv.

Writes into [mec] recompute_dir:
  numbers_primary.json     all the numbers
  cohort_results_primary.csv, cohort_cell_sets_primary.jsonl, pairs_primary.csv     the parts joined
  variants/new_<variant>/  on the sessions with labels: results.csv, cell_sets.jsonl and what
                           arbiter_join.py, arbiter_stats.py and confident_by_config.py write from them
  variants/new_on/results_all.csv, cell_sets_all.jsonl     the same two on all analysed sessions

arbiter_labels.py joins the authors' scores to the analysed cells, so it needs the cell sets, and this script
needs its labels. While the labels are not there, this script writes variants/new_on/cell_sets.jsonl on the
sessions of [mec] fps_table, the default input of arbiter_labels.py, and stops.

usage: python -X utf8 -B mec/numbers_primary.py
       python -X utf8 -B mec/numbers_primary.py --allow-partial     on the sessions finished so far
"""
import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import intense_paths as P  # noqa: E402
import arbiter_stats as AS  # noqa: E402

OUT = P.get("mec", "recompute_dir")
PARTS = OUT / "parts"
SESS = P.get("mec", "sessions_dir")
LABELS = P.get("mec", "arbiter_dir") / "arbiter_labels.csv"
FPS_TABLE = P.get("mec", "fps_table")
FPS_ALL = P.get("mec", "fps_all")
FEATS = ["head_direction", "speed", "place"]
VARIANTS = ["on", "on_mi04", "off", "off_mi04"]
VARIANT_NAME = {"on": "delay search, no threshold (SI)", "on_mi04": "delay search, 0.04 bits",
                "off": "no delay search, no threshold", "off_mi04": "no delay search, 0.04 bits"}
CFG = ("Bolt", "Zoom")
MARGINS = ("0-0.5", "0.5-1", "1-2", "2-4", "4-inf")


class DownstreamFailed(Exception):
    pass


def fail(msg):
    raise SystemExit(f"[FAILED] {msg}")


def variant_dir(v):
    return OUT / "variants" / f"new_{v}"


def read_jsonl(paths, keep=None, tolerant=False):
    """(session, cell) -> row. A run still writing may leave a cut last line; tolerant skips it."""
    out = {}
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for ln in fh:
                try:
                    d = json.loads(ln)
                except json.JSONDecodeError:
                    if tolerant:
                        continue
                    raise
                if keep is not None and d["session"] not in keep:
                    continue
                key = (d["session"], int(d["cell"]))
                if key in out:
                    fail(f"{key} twice ({p})")
                out[key] = d
    return out


def read_csvs(paths):
    return pd.concat([pd.read_csv(p) for p in paths]) if paths else pd.DataFrame()


def run_python(code, log, cwd):
    """A process of its own for each downstream script: they set up matplotlib when imported, and their
    files must not depend on what ran before them."""
    r = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", code], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=cwd)
    Path(log).write_text(r.stdout + r.stderr, encoding="utf-8")
    if r.returncode:
        raise DownstreamFailed(f"{log}:\n{r.stderr[-2000:]}")


def run_down(script, args, log, cwd, **globs):
    """A downstream script, its code unchanged; globs replace its module-level paths."""
    mod = script[:-3]
    code = (f"import sys\nfrom pathlib import Path\nsys.path.insert(0, {str(HERE)!r})\nimport {mod} as M\n"
            + "".join(f"M.{k} = Path({str(v)!r})\n" for k, v in globs.items())
            + f"sys.argv = {[script] + [str(x) for x in args]!r}\nM.main()\n")
    run_python(code, log, cwd)


# ------------------------------------------------------------------ numbers


def table5(keys, sets_v, glm):
    t5, cov = {}, {}
    for f in FEATS:
        g = sum(f in glm[k] for k in keys)
        i = sum(f in sets_v[k] for k in keys)
        both = sum(f in glm[k] and f in sets_v[k] for k in keys)
        t5[f] = dict(glm=g, intense=i, ratio=round(i / g, 3) if g else None)
        cov[f] = dict(recovered=both, glm=g, pct=round(100 * both / g, 1) if g else None)
    return dict(table5=t5, glm_recovered_by_intense=cov)


def base_numbers(keys, sets_v, glm, config):
    """Counts on one base of sessions, whole and by imaging configuration."""
    out = dict(sessions=len({k[0] for k in keys}), cells=len(keys), **table5(keys, sets_v, glm), by_config={})
    for c in CFG:
        sub = [k for k in keys if config[k[0]] == c]
        out["by_config"][c] = dict(sessions=len({k[0] for k in sub}), cells=len(sub), **table5(sub, sets_v, glm))
    return out


def margin_bin(rm, t, m):
    r = rm[(rm.type == t) & (rm.margin == m)]
    if not len(r):
        return None
    r = r.iloc[0]
    n = int(r.n)
    return dict(n=n, intense=int(round(n * r.recall_int)), glm=int(round(n * r.recall_glm)),
                intense_pct=round(100 * r.recall_int, 1), glm_pct=round(100 * r.recall_glm, 1))


def recall(r):
    return dict(authors=r["authors"], intense=r["intense_found"], glm=r["glm_found"],
                intense_pct=round(100 * r["recall_int"], 1), glm_pct=round(100 * r["recall_glm"], 1))


def recall_block(rn):
    return dict(hd_recall_95=recall(rn["HD@95"]), grid_recall_95=recall(rn["grid@95"]),
                border_recall_95=recall(rn["border@95"]),
                hd_nesting_99=dict(authors_and_glm=rn["HD@99"]["authors_and_glm"],
                                   also_intense=rn["HD@99"]["of_them_also_int"]))


def confident(lab):
    """Cells with margin above 4, as confident_by_config.py selects them."""
    U = lab[np.isfinite(lab.mvl) & np.isfinite(lab.mvl_95) & np.isfinite(lab.mvl_99)]
    return U[(U.mvl - U.mvl_95) / (U.mvl_99 - U.mvl_95) > 4]


def labelled_by_config(lab, config, objects):
    """Recall by configuration; confident cells also by whether the session had an object in the arena."""
    lab = lab.assign(config=lab.session.map(config), has_object=lab.session.map(objects))
    out = {}
    for c in CFG:
        g = lab[lab.config == c]
        S = confident(g)
        out[c] = recall_block(AS.recall_nesting(g))
        out[c]["hd_confident"] = dict(n=len(S), intense=int(S.int_hd.sum()), glm=int(S.glm_hd.sum()),
                                      impossible_mvl=int((S.mvl > 1).sum()))
        for name, flag in (("object", True), ("no_object", False)):
            q = S[S.has_object == flag]
            out[c][f"hd_confident_{name}"] = dict(n=len(q), intense=int(q.int_hd.sum()), glm=int(q.glm_hd.sum()))
    return out


def labelled_numbers(vdir, config, objects):
    """The numbers of a variant that rest on the labels, from what the downstream scripts wrote."""
    st = json.loads((vdir / "arbiter_stats.json").read_text())
    rm = pd.read_csv(vdir / "recall_vs_margin.csv")
    cbc = pd.read_csv(vdir / "confident_by_config.csv")
    zoom, bolt = cbc[cbc.config == "Zoom"], cbc[cbc.config == "Bolt"]
    io = st["intense_only_hd"]
    return dict(
        **recall_block(st["recall_nesting"]),
        hd_borderline_0_05=margin_bin(rm, "HD", "0-0.5"),
        hd_confident_second_config=dict(n=int(zoom.n.sum()), intense=int(zoom.intense.sum()),
                                        glm=int(zoom.glm.sum())),
        hd_confident_first_config=dict(n=int(bolt.n.sum()), intense=int(bolt.intense.sum()),
                                       glm=int(bolt.glm.sum())),
        intense_only_hd=dict(n=io["n"], ratio_median=round(io["ratio_median"], 3),
                             population_ratio_median=round(io["population_ratio_median"], 3),
                             ratio_median_exact=io["ratio_median"],
                             population_ratio_median_exact=io["population_ratio_median"]),
        hd_margin={m: margin_bin(rm, "HD", m) for m in MARGINS},
        grid_margin={m: margin_bin(rm, "grid", m) for m in MARGINS},
        border_margin={m: margin_bin(rm, "border", m) for m in MARGINS},
        by_config=labelled_by_config(pd.read_csv(vdir / "arbiter_labels.csv"), config, objects))


def results_table(sessions, done, keys, glm, sets_v):
    d = done.set_index("session")
    R = pd.DataFrame(dict(session=sessions, n_cells=d.loc[sessions, "n_cells"].astype(int).values,
                          fps=d.loc[sessions, "fps"].values))
    for prefix, src in (("glm", glm), ("int_bytype", sets_v)):
        for f in FEATS:
            cnt = dict.fromkeys(sessions, 0)
            for k in keys:
                cnt[k[0]] += f in src[k]
            R[f"{prefix}_{f}"] = R.session.map(cnt).astype(int)
    return R


def write_sets(path, keys, glm_parts, sets_v):
    with open(path, "w", newline="\n") as fh:
        for k in keys:
            fh.write(json.dumps({"session": k[0], "cell": k[1], "glm": glm_parts[k]["glm"], "int_raw": [],
                                 "int_bytype": sorted(sets_v[k])}) + "\n")


def drop_raw(vdir):
    """run_xyfix.py has no 'raw' arm of INTENSE, and arbiter_join.py needs the field: it is passed empty. The
    zeros it leaves in the outputs are not a result, so they are removed; nothing else reads them."""
    for name, cols, blank in (("arbiter_labels.csv", ("int_raw_place", "int_raw_hd"), False),
                              ("recall_vs_margin.csv", ("recall_raw",), True)):
        p = vdir / name
        with open(p, newline="") as fh:
            rows = list(csv.reader(fh))
        idx = [i for i, c in enumerate(rows[0]) if c in cols]
        with open(p, "w", newline="") as fh:
            w = csv.writer(fh)
            for j, r in enumerate(rows):
                if blank:
                    w.writerow([("" if (i in idx and j) else c) for i, c in enumerate(r)])
                else:
                    w.writerow([c for i, c in enumerate(r) if i not in idx])
    (vdir / "recall_vs_margin.png").unlink(missing_ok=True)


# ------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true", help="count the sessions finished so far")
    a = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"checks": {}}
    C = res["checks"]

    # ---------------- the sessions
    si213 = set(pd.read_csv(FPS_TABLE).session)
    scope = pd.read_csv(FPS_ALL).set_index("session")
    config = {s: "Zoom" if str(v).startswith("Zoom") else "Bolt" for s, v in scope.scope.items()}
    objects = {s: bool(re.search(r"(?i)object|OVC", str(v))) for s, v in scope.timeseries_name.items()}
    files = sorted(f.stem for f in SESS.glob("*.npz"))

    # ---------------- the parts
    done = read_csvs(sorted(PARTS.glob("intense_done_*.csv")))
    gdone = read_csvs(sorted(PARTS.glob("glm_done_*.csv")))
    if done.empty or gdone.empty:
        fail(f"no finished sessions yet in {PARTS}")
    done = done.sort_values("session").reset_index(drop=True)
    done["excluded"] = done.excluded.fillna("")
    gdone["excluded"] = gdone.excluded.fillna("")
    if a.allow_partial:  # the row of a session being written may be cut; such a row is not a finished session
        last = done.columns[-1]
        done = done[(done.excluded != "") | done[last].notna()]
        gdone = gdone[(gdone.excluded != "") | gdone[gdone.columns[-1]].notna()]
    if not done.session.is_unique or not gdone.session.is_unique:
        fail("a session is listed twice in the parts")
    analysed = done[done.excluded == ""]
    n_cells = analysed.set_index("session").n_cells
    g_ok = gdone[(gdone.excluded == "")]
    g_ok = g_ok[g_ok.cells_checked == g_ok.session.map(n_cells)]
    excluded_now = set(done[done.excluded != ""].session)
    included = set(analysed.session) & set(g_ok.session)
    waiting = sorted(set(files) - included - excluded_now)
    if waiting and not a.allow_partial:
        fail(f"{len(waiting)} of {len(files)} sessions have no finished INTENSE and GLM pass, for example {waiting[0]}; "
             "--allow-partial runs on what is there")
    if not included:
        fail("no session has both passes finished yet")
    partial = bool(waiting)
    tolerant = a.allow_partial
    mine = read_jsonl(sorted(PARTS.glob("intense_sets_*.jsonl")), keep=included, tolerant=tolerant)
    pairs = read_csvs(sorted(PARTS.glob("intense_pairs_*.csv")))
    pairs = pairs[pairs.session.isin(included)]
    glm_parts = read_jsonl(sorted(PARTS.glob("glm_sets_*.jsonl")), keep=included, tolerant=tolerant)
    keys_all = sorted(mine)
    if set(keys_all) - set(glm_parts):
        fail(f"{len(set(keys_all) - set(glm_parts))} cells have no GLM result")
    per_session = pd.Series([k[0] for k in keys_all]).value_counts()
    if not per_session.reindex(sorted(included)).astype(int).equals(n_cells.reindex(sorted(included)).astype(int)):
        fail("the cells in the sets differ from n_cells of the finished sessions")
    glm = {k: set(glm_parts[k]["glm"]) for k in keys_all}
    new = {v: {k: set(mine[k][v]) for k in keys_all} for v in VARIANTS}
    keys_213 = [k for k in keys_all if k[0] in si213]
    n_inc = {c: sum(config[s] == c for s in included) for c in CFG}
    n_fil = {c: sum(config[s] == c for s in files) for c in CFG}
    note = (f"{len(included)} of {len(files)} sessions (Bolt {n_inc['Bolt']} of {n_fil['Bolt']}, Zoom {n_inc['Zoom']} of "
            f"{n_fil['Zoom']}), {len(keys_all)} cells")
    print(f"[{'PARTIAL' if partial else 'OK'}] included: {note}", flush=True)

    C["sessions"] = dict(files=len(files), analysed=len(analysed), cells=len(keys_all),
                         excluded=done[done.excluded != ""][["session", "excluded"]].to_dict("records"))
    tested_place = pairs[pairs.feature == "place"].groupby(["session", "arm"]).tested.unique()
    bad = [k for k, v in tested_place.items() if list(v) != ["place_quad"]]
    if bad or len(tested_place) != 2 * len(included):
        fail(f"place not tested as place_quad: {bad[:5]} ({len(tested_place)} session-arms)")
    C["place_tested_as_place_quad"] = dict(session_arms=len(tested_place))

    # ---------------- the parts joined
    with open(OUT / "cohort_cell_sets_primary.jsonl", "w", newline="\n") as fh:
        for k in keys_all:
            fh.write(json.dumps({"session": k[0], "cell": k[1], "glm": glm_parts[k]["glm"],
                                 **{v: sorted(new[v][k]) for v in VARIANTS}}) + "\n")
    pairs.sort_values(["session", "arm", "cell", "feature"]).to_csv(OUT / "pairs_primary.csv", index=False)
    glm_cnt = results_table(sorted(included), done, keys_all, glm, new["on"])[["session"] + [f"glm_{f}" for f in FEATS]]
    cr = done[done.session.isin(included | excluded_now)].merge(glm_cnt, on="session", how="left")
    cr.to_csv(OUT / "cohort_results_primary.csv", index=False)

    if not LABELS.exists():
        vdir = variant_dir("on")
        vdir.mkdir(parents=True, exist_ok=True)
        write_sets(vdir / "cell_sets.jsonl", keys_213, glm_parts, new["on"])
        sys.exit(f"[STOPPED] no labels yet: {LABELS}. Run mec/arbiter_labels.py, which reads "
                 f"{vdir / 'cell_sets.jsonl'}, then this script again")

    # ---------------- the downstream scripts, each variant, on the sessions with labels
    lab_idx = pd.read_csv(LABELS, usecols=["session"])
    lab_sessions = set(lab_idx.session)
    keys_lab = [k for k in keys_all if k[0] in lab_sessions]
    chain_sessions = sorted({k[0] for k in keys_lab})
    label_src = LABELS
    restricted = OUT / "labels_included_sessions.csv"
    restricted.unlink(missing_ok=True)
    if set(chain_sessions) != lab_sessions:
        with open(LABELS, newline="") as fi, open(restricted, "w", newline="") as fo:
            fo.write(fi.readline())
            fo.writelines(ln for ln in fi if ln.split(",", 1)[0] in set(chain_sessions))
        label_src = restricted
    res["variants"] = {}
    chain_failed = None
    for v in VARIANTS:
        vdir = variant_dir(v)
        vdir.mkdir(parents=True, exist_ok=True)
        res["variants"][v] = dict(name=VARIANT_NAME[v], dir=f"variants/new_{v}",
                                  base_all=base_numbers(keys_all, new[v], glm, config),
                                  base_213=base_numbers(keys_213, new[v], glm, config), labelled=None)
    for v in VARIANTS:
        vdir = variant_dir(v)
        write_sets(vdir / "cell_sets.jsonl", keys_lab, glm_parts, new[v])
        results_table(chain_sessions, done, keys_lab, glm, new[v]).to_csv(vdir / "results.csv", index=False)
        try:
            run_down("arbiter_join.py", ["--sets", vdir / "cell_sets.jsonl", "--out", vdir], vdir / "arbiter_join.log", OUT,
                     LBL_SRC=label_src)
            run_down("arbiter_stats.py", ["--labels", vdir / "arbiter_labels.csv", "--out", vdir, "--fps-from",
                                          vdir / "results.csv"], vdir / "arbiter_stats.log", OUT)
            run_down("confident_by_config.py", ["--labels", vdir / "arbiter_labels.csv", "--results", vdir / "results.csv",
                                                "--out", vdir], vdir / "confident_by_config.log", OUT)
            res["variants"][v]["labelled"] = labelled_numbers(vdir, config, objects)
        except DownstreamFailed as e:
            if not partial:
                fail(str(e))
            chain_failed = str(e)
            print(f"[PARTIAL] the comparison with the labels cannot be computed on these sessions yet: {e}", flush=True)
            break
        drop_raw(vdir)
        print(f"[OK] variant {v}", flush=True)
    vdir = variant_dir("on")
    write_sets(vdir / "cell_sets_all.jsonl", keys_all, glm_parts, new["on"])
    results_table(sorted(included), done, keys_all, glm, new["on"]).to_csv(vdir / "results_all.csv", index=False)

    # ---------------- what the text of the SI takes from outside the variants
    res["text_inputs"] = dict(
        object_sessions=dict(base_all=sum(objects[s] for s in included), base_213=sum(objects[s] for s in included & si213)),
        rates={c: sorted({round(float(x), 1) for x in analysed[analysed.session.isin(included)].loc[
            lambda t: t.session.map(config) == c, "fps"]}) for c in CFG},
        added_sessions={})
    for s in scope.index[(scope.in_npz == 1) & (scope.in_si_final == 0)]:
        p = SESS / f"{s}.npz"
        if p.exists():
            with np.load(p) as d:
                res["text_inputs"]["added_sessions"][s] = dict(
                    cells=int(d["calcium"].shape[0]), finite_share=float(np.isfinite(d["calcium"]).mean()))
    res["run"] = dict(partial=partial, included_note=note, sessions_included=len(included), sessions_files=len(files),
                      cells=len(keys_all), sessions_waiting=len(waiting), labelled_sessions=len(chain_sessions),
                      labelled_cells=len(keys_lab), not_labelled=sorted(included - lab_sessions), chain_failed=chain_failed)
    res["seconds"] = round(time.time() - t0)
    (OUT / "numbers_primary.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=lambda o: o.item()),
                                              encoding="utf-8")
    if not chain_failed:
        hd = res["variants"]["on"]["labelled"]["hd_recall_95"]
        print(f"authors' HD cells: INTENSE {hd['intense']}, GLM {hd['glm']} of {hd['authors']} on {len(chain_sessions)} "
              "sessions with labels", flush=True)
    print(f"[{'PARTIAL' if partial else 'OK'}] {note}; {res['seconds']} s; {OUT}", flush=True)


if __name__ == "__main__":
    main()
