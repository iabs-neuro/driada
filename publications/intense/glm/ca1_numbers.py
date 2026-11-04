"""Supplementary Table 4 and the CA1 numbers of Section S1.4, with and without the delay search of INTENSE.

Counts the columns of Supplementary Table 4 with glm/table4_ca1.py, against the forward-selection GLM
of [v11] glm_ca1_per_neuron, for the paper run with the delay search ([v11] stage2_results ->
table4_paper_with_delay.json, pairs_paper_with_delay.csv) and for the run of
glm/ca1_intense_zero_lag.py without it (<[glm] ca1_intense_dir>/delays_off/results ->
table4_no_delay.json, pairs_no_delay.csv); a count already written is not redone. For every
feature: cells selected by the GLM, by INTENSE with and without the delay search, by both, the
ratio INTENSE / GLM, the share of the GLM cells that INTENSE also selects, the share of the excess
of INTENSE over the GLM (in cells) that remains without the delay search, and the cells significant
without the 0.04-bit threshold on excess information.

Supplementary Table 4: columns GLM (features.<f>.glm), INTENSE (intense_with_delay), INTENSE zero
lag (intense_no_delay) and both (both_with_delay) of <[glm] ca1_intense_dir>/ca1_numbers.json; the
column "GLM, each feature" is perfeature.<f> of ca1_glm_numbers.json (glm/ca1_glm_numbers.py).

usage: python glm/ca1_numbers.py  ->  <[glm] ca1_intense_dir>/ca1_numbers.json
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import intense_paths as P  # noqa: E402
import paths_v11 as V  # noqa: E402
import ca1_common as C  # noqa: E402

FEATS = ["position", "head_direction", "speed"]


def main():
    a = P.get("glm", "ca1_intense_dir")
    on_json = a / "table4_paper_with_delay.json"
    off_json = a / "table4_no_delay.json"
    results = a / "delays_off" / "results"
    n = len(list(results.glob("NOF_*_results.npz")))
    if n != 64:
        raise SystemExit(f"[FAILED] {n} of 64 sessions in {results}")
    for target, res_dir, pairs, label in (
            (on_json, V.get("stage2_results"), a / "pairs_paper_with_delay.csv", "paper run, delay search on"),
            (off_json, results, a / "pairs_no_delay.csv", "delay search off")):
        if not target.exists():
            r = subprocess.run([sys.executable, "-X", "utf8", str(HERE / "table4_ca1.py"), "--results", str(res_dir),
                                "--out", str(target), "--pairs-csv", str(pairs),
                                "--label", label], capture_output=True, text=True)
            if r.returncode:
                raise SystemExit(r.stderr[-2000:])
    on = json.loads(on_json.read_text(encoding="utf-8"))
    off = json.loads(off_json.read_text(encoding="utf-8"))
    res = {"sessions": off["sessions"], "neurons_in_glm_table": off["neurons_in_glm_table"], "features": {}}
    for f in FEATS:
        t_on, t_off = on["table4"][f], off["table4"][f]
        if t_on["glm"] != t_off["glm"]:
            raise SystemExit(f"[FAILED] GLM column differs for {f}")
        g = t_on["glm"]
        res["features"][f] = dict(
            glm=g, intense_with_delay=t_on["intense"], intense_no_delay=t_off["intense"],
            both_with_delay=t_on["both"], both_no_delay=t_off["both"],
            ratio_with_delay=t_on["intense_over_glm"], ratio_no_delay=t_off["intense_over_glm"],
            share_glm_found_with_delay=t_on["share_glm_found_by_intense"],
            share_glm_found_no_delay=t_off["share_glm_found_by_intense"],
            excess_over_glm_retained=round((t_off["intense"] - g) / (t_on["intense"] - g), 3),
            significant_with_delay=on["significant_pairs_by_feature"].get(f),
            significant_no_delay=off["significant_pairs_by_feature"].get(f))
    C.write_json(a / "ca1_numbers.json", res)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
