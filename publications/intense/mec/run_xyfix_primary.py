# -*- coding: utf-8 -*-
"""mec/run_xyfix.py, unchanged, with a frame rate for every session that has a session file.

[mec] fps_table, the table of rates of the Supplementary Information, does not list the two sessions
whose traces were not available to its analysis, and run_xyfix.py needs a rate for every file. This
wrapper adds them from [mec] fps_all: the same measurement as for the others, the mean rate over the
frame times, rounded as in the table of the Supplementary Information. All other arguments go to
run_xyfix.main().

usage: python -X utf8 -B mec/run_xyfix_primary.py [--npz-dir DIR] --shard k/N --pass intense|glm
                                                  [--out DIR] [run_xyfix.py arguments]
       defaults: [mec] sessions_dir, <[mec] recompute_dir>/parts
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_xyfix as RX  # noqa: E402

FPS_ALL = RX.P.get("mec", "fps_all")


def extend_fps():
    """Give a measured rate to the sessions with a file that the table of rates does not list."""
    t = pd.read_csv(FPS_ALL).set_index("session")
    extra = t[(t.in_npz == 1) & (t.in_si_final == 0)].fps_mean.round(4)
    RX.FPS = pd.concat([RX.FPS, extra[~extra.index.isin(RX.FPS.index)]])


def main():
    extend_fps()
    argv = sys.argv[1:]
    if "--npz-dir" in argv:
        k = argv.index("--npz-dir")
        npz_dir = Path(argv[k + 1])
        del argv[k:k + 2]
        if not npz_dir.is_dir():
            sys.exit(f"refused: --npz-dir {npz_dir} does not exist")
        RX.SESS = npz_dir
    sys.argv = [sys.argv[0]] + argv
    RX.main()


if __name__ == "__main__":
    main()
