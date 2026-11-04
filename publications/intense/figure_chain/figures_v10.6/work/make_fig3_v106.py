"""Link of Fig. 3 between figures_v10.5/work/make_fig3_v105.py and figures_v11.1/work/fig3/make_fig3_v111.py.

Panel c reads the cache of fig3/fig3_data.py, built on the databases of [results] db_dir: in pairs with a
discrete variable, conditional MI is computed with both behavioral variables at their optimal delays, as for
two continuous variables.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
A = HERE.parents[1]
sys.path.insert(0, str(A / "figures_v10.5" / "work"))
import make_fig3_v105 as v  # noqa: E402  (sets TrueType fonts in PDF, loads the v10.4/v10.3 scripts)
import matplotlib.pyplot as plt  # noqa: E402

m = v.m
ALIGNED = m.intense_paths.CACHE_DIR      # fig3/fig3_data.py on the databases of the configuration
V105 = m._F.D / "figures_v10.5" / "fig3.png"


def main():
    out_dir = m._F.out_dir(HERE.parent)
    for f in ("fig3.png", "fig3.pdf"):
        if (out_dir / f).exists():
            sys.exit(f"{out_dir / f} exists; not overwriting")
    (m._F.out_dir(HERE) / "verify").mkdir(exist_ok=True)
    m.NEW = ALIGNED
    m.v103_settings()
    m.f3c.V7_BRACKETS = "stacked"
    c = v.candidate()
    m.set_example(c)
    fig = m.comp.create_fig3_delegated(output_dir=out_dir, output_name="fig3", lowercase=True,
                                       save_formats=("png", "pdf"))
    boxes = m.panel_boxes(fig)
    plt.close(fig)
    a, f = np.asarray(Image.open(V105)), np.asarray(Image.open(out_dir / "fig3.png"))
    res = dict(cache=str(ALIGNED), per_session_csv=str(m.f3c._PER_SESSION_CSV),
               same_shape_as_v10_5=bool(a.shape == f.shape),
               px_differ_vs_v10_5_by_panel={k: m.diff_in(a, f, b) for k, b in boxes.items()}
               if a.shape == f.shape else None)
    (m._F.out_dir(HERE) / "verify" / "fig3_v106.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
