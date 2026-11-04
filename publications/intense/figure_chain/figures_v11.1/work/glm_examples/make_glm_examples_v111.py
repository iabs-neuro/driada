"""Supplementary Fig. 4: real CA1 example neurons (glm_src/fig_examples_v111.py).

usage: python -X utf8 -B figure_chain/figures_v11.1/work/glm_examples/make_glm_examples_v111.py
       # -> si/glm_examples.png, work/glm_examples/verify/glm_examples_v111.json under
       #    [figures] output_dir/figure_chain/figures_v11.1

This script checks that the four picks are the neurons of the paper and writes the peak of the tuning curve of
panel a in radians and in degrees.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # figure_chain/
import _paths_fig as _F  # noqa: E402  (puts the paper folder on sys.path)

V106 = _F.D / "figures_v10.6" / "si" / "glm_examples.png"
os.environ["GLM_EXAMPLES_DB"] = str(_F.P.DB_PRE)
os.environ["GLM_EXAMPLES_FPS"] = str(_F.V.get("fps_table"))
sys.path.insert(0, str(HERE / "glm_src"))
import fig_examples_v111 as G  # noqa: E402  (picks are made on import)

V106_PICKS = [("H09", "4D", 119), ("H01", "3D", 155), ("H09", "3D", 540), ("H02", "2D", 128)]


def main():
    out = _F.out_dir(HERE.parents[1] / "si") / "glm_examples.png"
    if out.exists():
        sys.exit(f"{out} exists; not overwriting")
    (_F.out_dir(HERE) / "verify").mkdir(exist_ok=True)
    G.main(out)
    picks = [(p["mouse"], p["session"], p["ni"]) for p in G.picks]
    # panel a: tuning curve in degrees against the same curve in radians (the v10.6 computation)
    p = G.picks[0]
    b = G.load(p)
    ok = np.isfinite(b["resp"]) & np.isfinite(b["x"]) & np.isfinite(b["y"]) & np.isfinite(b["hd"]) & np.isfinite(b["speed"])
    c_rad, m_rad = G.tuning_1d(b["hd"][ok], b["resp"][ok], circular=True)
    res = dict(output=str(out), picks=picks, picks_as_v10_6=picks == V106_PICKS,
               panel_a_peak_deg=float(np.degrees(c_rad[np.nanargmax(m_rad)])),
               panel_a_peak_rad=float(c_rad[np.nanargmax(m_rad)]),
               size_v10_6=list(Image.open(V106).size) if _F.reference(V106) else None,
               size=list(Image.open(out).size))
    (_F.out_dir(HERE) / "verify" / "glm_examples_v111.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
