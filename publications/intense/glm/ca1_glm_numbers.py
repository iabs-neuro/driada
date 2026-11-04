"""Per-neuron GLM selections on CA1 and the GLM numbers of Section S1.4, from the parts of glm/ca1_glm.py.

Joins the parts and writes the two selections in the layout the other scripts read (columns mouse,
session, neuron_idx, glm_position, glm_head_direction, glm_speed; rows sorted by mouse, session and
neuron):
  forward selection, the GLM of Section S1.4 -> [v11] glm_ca1_per_neuron, read by glm/table4_ca1.py,
      numbers/speed_direction_ca1.py and the figure of CA1 example neurons;
  each feature tested on its own -> <[glm] ca1_glm_dir>/glm_perfeature_per_neuron.csv.
<[glm] ca1_glm_dir>/ca1_glm_numbers.json counts the cells each selection selects per feature
(forward: the GLM column of Supplementary Table 4; perfeature: the column "GLM, each feature") and
the cells selected by only one of them, and, where the INTENSE results exist, the columns of
Supplementary Table 4 for the per-feature GLM (glm/table4_ca1.py) against INTENSE with the delay
search of the paper run ([v11] stage2_results) and without it (glm/ca1_intense_zero_lag.py).

--reference CSV compares the forward selection neuron by neuron with a saved per-neuron table of the
same layout and stops if a neuron is missing on either side.

usage: python glm/ca1_glm_numbers.py [--reference CSV]
"""
import argparse
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

import pandas as pd  # noqa: E402

KEY = ["mouse", "session", "neuron_idx"]
FEATS = ["position", "head_direction", "speed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, default=None)
    a = ap.parse_args()
    out = P.get("glm", "ca1_glm_dir")
    parts = sorted((out / "parts").glob("glm_*.csv"))
    b = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    b["session"] = b.session.astype(str)
    b = b.sort_values(KEY).reset_index(drop=True)
    differ = "not compared"
    if a.reference is not None:
        # read before writing: the reference may be the file this script is about to replace
        ref = pd.read_csv(a.reference)
        ref["session"] = ref.session.astype(str)
        m = ref.merge(b, on=KEY, how="outer", indicator=True)
        if (m._merge != "both").any():
            raise SystemExit(f"[FAILED] neurons not matched: {m._merge.value_counts().to_dict()}")
        differ = {c: int((m[f"glm_{c}"] != m[f"fwd_{c}"]).sum()) for c in FEATS}
    fwd = b[KEY + [f"fwd_{c}" for c in FEATS]].rename(columns={f"fwd_{c}": f"glm_{c}" for c in FEATS})
    glm_ca1 = V.get("glm_ca1_per_neuron")
    glm_ca1.parent.mkdir(parents=True, exist_ok=True)
    fwd.to_csv(glm_ca1, index=False)
    pf = b[KEY + [f"pf_{c}" for c in FEATS]].rename(columns={f"pf_{c}": f"glm_{c}" for c in FEATS})
    pf.to_csv(out / "glm_perfeature_per_neuron.csv", index=False)
    res = {"neurons": int(len(b)), "sessions": int(b.groupby(["mouse", "session"]).ngroups),
           "reference": str(a.reference) if a.reference is not None else None,
           "forward_differs_from_reference": differ,
           "forward": {c: int(b[f"fwd_{c}"].sum()) for c in FEATS},
           "perfeature": {c: int(b[f"pf_{c}"].sum()) for c in FEATS},
           "perfeature_not_forward": {c: int(((b[f"pf_{c}"] == 1) & (b[f"fwd_{c}"] == 0)).sum()) for c in FEATS},
           "forward_not_perfeature": {c: int(((b[f"pf_{c}"] == 0) & (b[f"fwd_{c}"] == 1)).sum()) for c in FEATS}}
    runs = {"intense_with_delay": V.get("stage2_results"),
            "intense_no_delay": P.get("glm", "ca1_intense_dir") / "delays_off" / "results"}
    for name, results in runs.items():
        if not results.exists() or len(list(results.glob("*_results.npz"))) != 64:
            res[f"table4_vs_{name}"] = "not available"
            continue
        target = out / f"table4_perfeature_vs_{name}.json"
        r = subprocess.run([sys.executable, "-X", "utf8", str(HERE / "table4_ca1.py"), "--results", str(results),
                            "--out", str(target), "--glm", str(out / "glm_perfeature_per_neuron.csv"),
                            "--label", f"per-feature GLM vs {name}"], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(r.stderr[-2000:])
        res[f"table4_vs_{name}"] = json.loads(target.read_text(encoding="utf-8"))["table4"]
    C.write_json(out / "ca1_glm_numbers.json", res)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
