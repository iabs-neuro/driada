"""Link of Fig. 3 between figures_v10.4/work/make_fig3_v104.py and figures_v11.1/work/fig3/make_fig3_v111.py.

The PDF keeps its text as TrueType (pdf.fonttype 42) so that the journal receives embedded fonts; the PNG does
not depend on this setting.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
W104 = HERE.parents[1] / "figures_v10.4" / "work"
sys.path.insert(0, str(W104))
import make_fig3_v104 as m  # noqa: E402  (loads the 1.2.2 snapshot and the v10.3 panel scripts)
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
V104 = m._F.data_dir(W104.parent) / "fig3.png"


def candidate():
    c = json.loads(m.CAND.read_text(encoding="utf-8"))["top"][0]
    assert (c["mouse"], c["session"], c["neuron_idx"]) == ("H08", "4D", 370), c
    assert (c["delay_hd_frames"], c["delay_place_frames"]) == (20, 45), c
    return c


def main():
    out_dir = m._F.out_dir(HERE.parent)
    for f in ("fig3.png", "fig3.pdf"):
        if (out_dir / f).exists():
            sys.exit(f"{out_dir / f} exists; not overwriting")
    (m._F.out_dir(HERE) / "verify").mkdir(exist_ok=True)
    m.v103_settings()
    m.f3c.V7_BRACKETS = "stacked"
    c = candidate()
    npz, fps = m.set_example(c)
    fig = m.comp.create_fig3_delegated(output_dir=out_dir, output_name="fig3", lowercase=True,
                                       save_formats=("png", "pdf"))
    boxes = m.panel_boxes(fig)
    ex = m.example_check(fig, c, npz)
    texts = m.overlaps.text_report(fig)
    plt.close(fig)
    before = json.loads((m._F.out_dir(W104) / "verify" / "fig3_control_v103.json").read_text(encoding="utf-8"))["text_clashes"]
    a, f = np.asarray(Image.open(V104)), np.asarray(Image.open(out_dir / "fig3.png"))
    res = dict(reference=str(V104), output=str(out_dir / "fig3.png"), shape=list(f.shape),
               same_shape_as_v10_4=bool(a.shape == f.shape),
               px_differ_vs_v10_4_by_panel={k: m.diff_in(a, f, b) for k, b in boxes.items()} if a.shape == f.shape else None,
               example_d_e=ex, text_clashes_vs_v10_3_control=m.overlaps.diff_reports(before, texts))
    (m._F.out_dir(HERE) / "verify" / "fig3_v105.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("shape", "same_shape_as_v10_4", "px_differ_vs_v10_4_by_panel",
                                          "example_d_e", "text_clashes_vs_v10_3_control")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
