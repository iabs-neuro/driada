"""Numbers of Supplementary Section S2, Supplementary Table 6 and their echo in Results ("4.5% of
cases, compared with 11.2%"), each read from an output of the scripts of s2/ with the file and the
key it comes from. Only reads saved tables; nothing is recomputed.

  expected 4.5% (3.7-5.4%), observed 11.2%     data/s3_summary_s7.csv, data/s3_intervals_exact.csv
                                                (s2/11_s3_with_s7.py, s2/15_exact_intervals.py)
  no draw of 2,000 reached 11.2%, p < 0.001;   data/s8_significance.csv, s8_significance_permouse.csv
  all 16 mice, Wilcoxon p = 1.5e-5              (s2/16_significance.py 2000)
  Supplementary Table 6; 335 of 338; 3         data/table6.csv (s2/table6.py)
  640 per category; 640 null neurons unlabeled data/table6_all_categories.csv (s2/table6.py)
  330 of 330 (place + body direction)          data/s7_neurons.csv (s2/8_s7_score.py)
  544 of the 868 (position + body direction)   data/s4_composition.csv (s2/5_composition.py)
  30 neurons with rearing strongest, none      data/ca1_neuron_sessions.csv (s2/3_ca1_table.py)
  multiply selective
  16 sessions; 4, 8, 16, 32 x 0.1 Hz           sessions_16.csv, s2/truth/*.json

Writes <[results] numbers_dir>/s2_numbers.json (number -> value, file relative to [s2] out_dir, key).
usage: python numbers/s2_numbers.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intense_paths as P  # noqa: E402

S2 = P.get("s2", "out_dir")
D = S2 / "data"
# categories of the strongest CA1 label that have synthetic counterparts (CATS of s2/4_calibration.py)
CATS = ["place", "object", "zone", "bodydirection", "headdirection", "locomotion", "speed"]


def row(df, **kw):
    """The single row of a table whose columns have the given values.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to search.
    **kw
        Column names and the values that select the row.

    Returns
    -------
    pandas.Series
        The matching row; AssertionError unless exactly one row matches.
    """
    for k, v in kw.items():
        df = df[df[k] == v]
    assert len(df) == 1, kw
    return df.iloc[0]


def numbers():
    """Every number of the section from the files of [s2] out_dir.

    Returns
    -------
    dict
        Name of the number -> {"value": ..., "file": path relative to [s2] out_dir, "key": where in
        the file}.
    """
    out = {}

    def put(key, value, file, where):
        out[key] = {"value": value, "file": file, "key": where}

    s7s = pd.read_csv(D / "s3_summary_s7.csv")
    r = row(s7s, training="s2+s7", resampling="neuron")
    put("expected_per100", float(r.expected_per100), "data/s3_summary_s7.csv",
        "training=s2+s7, resampling=neuron: expected_per100")
    put("expected_n", float(r.expected_n), "data/s3_summary_s7.csv", "training=s2+s7, resampling=neuron: expected_n")
    put("observed_per100", float(r.observed_per100), "data/s3_summary_s7.csv", "observed_per100")
    ex = pd.read_csv(D / "s3_intervals_exact.csv")
    for variant in ("s2+s7|neuron", "s2+s7|session", "s2|neuron"):
        e = row(ex, file="s3_bootstrap_s7.csv", variant=variant)
        put(f"interval_lo[{variant}]", float(e.expected_lo), "data/s3_intervals_exact.csv",
            f"file=s3_bootstrap_s7.csv, variant={variant}: expected_lo")
        put(f"interval_hi[{variant}]", float(e.expected_hi), "data/s3_intervals_exact.csv",
            f"file=s3_bootstrap_s7.csv, variant={variant}: expected_hi")
    for variant in ("s2+s7|session", "s2|neuron"):
        tr, rs = variant.split("|")
        put(f"expected_per100[{variant}]", float(row(s7s, training=tr, resampling=rs).expected_per100),
            "data/s3_summary_s7.csv", f"training={tr}, resampling={rs}: expected_per100")

    sig = pd.read_csv(D / "s8_significance.csv")
    for test in ("A_main", "B_conservative"):
        s = row(sig, test=test)
        for k in ("draws", "null_median", "null_q975", "null_q999", "null_max", "null_ge_observed", "P_null",
                  "diff_lo", "diff_hi", "diff_min"):
            put(f"{test}.{k}", float(s[k]), "data/s8_significance.csv", f"test={test}: {k}")
    c = row(sig, test="C_per_mouse")
    for k in ("mice", "mice_observed_gt_expected", "wilcoxon_P", "sign_test_P"):
        put(f"C.{k}", float(c[k]), "data/s8_significance.csv", f"test=C_per_mouse: {k}")
    pm = pd.read_csv(D / "s8_significance_permouse.csv")
    below = pm[pm["diff"] <= 0]
    put("C.mice_not_above", ", ".join(f"{m} ({o:.1f} vs {e:.1f} per 100)" for m, o, e in
                                      zip(below.mouse, below.observed_per100, below.expected_per100)) or "none",
        "data/s8_significance_permouse.csv", "rows with diff <= 0: mouse, observed_per100, expected_per100")
    # the mouse closest to equality decides between "exceeded" and "at least" in the text
    close = pm.loc[pm["diff"].idxmin()]
    put("C.closest_mouse", {"mouse": str(close.mouse), "observed": float(close.observed),
                            "expected": float(close.expected)},
        "data/s8_significance_permouse.csv", "row with the smallest diff: mouse, observed, expected")

    t6 = pd.read_csv(D / "table6.csv")
    for _, rr in t6.iterrows():
        for k in ("passed", "alone", "and_others", "one_label_other_variable", "several_labels_true_lost", "no_label",
                  "v11_alone", "v11_and_others"):
            put(f"table6.{rr['category']}.{k}", int(rr[k]), "data/table6.csv", f"category={rr['category']}: {k}")
    allc = pd.read_csv(D / "table6_all_categories.csv")
    allc["category"] = allc.category.fillna("null")   # read_csv reads the category 'null' as NaN
    put("neurons_per_category", sorted({int(v) for v in allc.generated}), "data/table6_all_categories.csv",
        "generated (all categories)")
    nul = row(allc, category="null")
    put("null.generated", int(nul.generated), "data/table6_all_categories.csv", "category=null: generated")
    put("null.any_label", int(nul.null_any_label), "data/table6_all_categories.csv", "category=null: null_any_label")

    s7 = pd.read_csv(D / "s7_neurons.csv")
    mx = s7[s7.category == "place+bodydirection"]
    both = mx[(mx.found_1 == True) & (mx.found_2 == True)]  # noqa: E712
    kept = int((both.has_place_after & both.has_bd_after).sum())
    put("s7_mixed.generated", len(mx), "data/s7_neurons.csv", "category=place+bodydirection: rows")
    put("s7_mixed.both_passed_before", len(both), "data/s7_neurons.csv",
        "category=place+bodydirection, found_1 & found_2: rows")
    put("s7_mixed.both_kept", kept, "data/s7_neurons.csv", "... and has_place_after & has_bd_after")
    put("s7_mixed.selective", int(mx.selective.sum()), "data/s7_neurons.csv", "category=place+bodydirection: selective")
    put("s7_mixed.multi_of_selective", int(mx[mx.selective].multi.sum()), "data/s7_neurons.csv", "... selective & multi")
    pure = s7[(s7.category != "place+bodydirection") & (s7.found_1 == True)]  # noqa: E712
    for name, sel in (("place", pure.category != "bodydirection"), ("bodydirection", pure.category == "bodydirection")):
        put(f"s7_single.{name}.found", int(sel.sum()), "data/s7_neurons.csv", f"single {name} cells with found_1")
        put(f"s7_single.{name}.false_multi", int(pure[sel].multi.sum()), "data/s7_neurons.csv", "... and multi")

    comp = pd.read_csv(D / "s4_composition.csv")
    bdp = row(comp, combo="BD+place")
    put("composition.BD+place.observed", int(bdp.ca1_observed), "data/s4_composition.csv", "combo=BD+place: ca1_observed")
    put("composition.BD+place.expected_false", float(bdp.ca1_expected_false), "data/s4_composition.csv",
        "combo=BD+place: ca1_expected_false")
    put("composition.observed_total", int(comp.ca1_observed.sum()), "data/s4_composition.csv", "sum of ca1_observed")

    ca1 = pd.read_csv(D / "ca1_neuron_sessions.csv")
    an, none = ca1[ca1.category.isin(CATS)], ca1[~ca1.category.isin(CATS)]
    put("ca1.with_counterpart", len(an), "data/ca1_neuron_sessions.csv", "category in CATS: rows")
    put("ca1.with_counterpart_multi_after", int(an.multi_after.sum()), "data/ca1_neuron_sessions.csv",
        "category in CATS: multi_after")
    put("ca1.no_counterpart_by_strongest_label", {str(k): int(v) for k, v in none.max_label.value_counts().items()},
        "data/ca1_neuron_sessions.csv", "category not in CATS: max_label counts")
    put("ca1.no_counterpart_multi_after", int(none.multi_after.sum()), "data/ca1_neuron_sessions.csv",
        "category not in CATS: multi_after")

    sessions = pd.read_csv(S2 / "sessions_16.csv")
    put("sessions", list(sessions.session), "sessions_16.csv", "session")
    put("mice", int(sessions.mouse.nunique()), "sessions_16.csv", "mouse: distinct")
    settings = [json.loads(p.read_text(encoding="utf-8"))["settings"] for p in sorted((S2 / "s2" / "truth").glob("*.json"))]
    for k in ("seed", "baseline_rate_hz", "snr_levels", "per_level"):
        values = {json.dumps(s[k]) for s in settings}
        assert len(values) == 1, (k, values)
        put(f"synthetic.{k}", settings[0][k], "s2/truth/*.json", f"settings.{k} (equal in all {len(settings)} sessions)")
    return out


def main():
    """Write s2_numbers.json to [results] numbers_dir and print every number with its source."""
    out = numbers()
    P.NUMBERS_DIR.mkdir(parents=True, exist_ok=True)
    (P.NUMBERS_DIR / "s2_numbers.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for k, v in out.items():
        print(f"{k}: {v['value']}  [{v['file']}: {v['key']}]")


if __name__ == "__main__":
    main()
