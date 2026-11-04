# -*- coding: utf-8 -*-
"""Which signal dataset and channel the authors scored in every session of the dump.

One pass over __angular_rate of [mec] dump_extract with the row parser of extract_authors.py.
The map is what decode_cohort_primary.py needs to keep, from _cell__traces, the rows the
authors' scores were computed from. The same pass checks the two sessions in EXCLUDED, whose
calcium came out entirely NaN when the decode did not select the channel: are the authors' df_f
tuning curves there finite (excluded_sessions_authors.csv)? Needs datajoint 0.12.7 to unpack
the curves.

usage: python -B mec/extract_dataset_map.py
       -> [mec] signal_datasets (authors_signal_dataset.csv) and, in the same folder,
          excluded_sessions_authors.csv
"""
import csv
import mmap
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
sys.path.insert(0, str(PAPER))
sys.path.insert(0, str(HERE))
import intense_paths as P  # noqa: E402
import extract_authors as E  # noqa: E402

OUT = P.get("mec", "signal_datasets")
EXCLUDED = {"4ba725e9db4dce7c_0_e64acc09dc77a53c", "d864f15f14276425_0_c137f5bca9894dfe"}
E_CEN = (np.arange(180) + 0.5) * 2 * np.pi / 180


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    per = defaultdict(lambda: defaultdict(int))
    excl = []
    with open(E.SQL, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        body = E.table_bytes(mm, b"__angular_rate")
        for m in E.ROW_AR.finditer(body):
            g = m.groups()
            stem = f"{g[0].decode()}_{g[1].decode()}_{g[2].decode()}"
            key = (stem, g[3].decode(), g[5].decode(), g[7].decode())   # signal_dataset, channel, tracking
            per[key][g[10].decode()] += 1
            if stem in EXCLUDED and g[10] == b"df_f":
                sm, b = E.blob(g[12]), E.blob(g[14])
                excl.append(dict(session=stem, cell_id=int(g[4]), finite=bool(np.isfinite(b).all()),
                                 binned_sum=float(b.sum()), smoothed_max=float(sm.max()),
                                 mvl=float(np.abs(np.sum(sm * np.exp(1j * E_CEN))) / np.sum(sm))))
    with open(OUT, "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(["session", "signal_dataset", "channel", "tracking_dataset", "rows_df_f", "rows_spikes"])
        for (stem, ds, ch, tr), c in sorted(per.items()):
            w.writerow([stem, ds, ch, tr, c.get("df_f", 0), c.get("spikes", 0)])
    with open(OUT.parent / "excluded_sessions_authors.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=list(excl[0].keys()) if excl else ["session"])
        w.writeheader()
        w.writerows(excl)
    sessions = {k[0] for k in per}
    multi = [s for s in sessions if sum(1 for k in per if k[0] == s) > 1]
    print(f"sessions with authors' curves: {len(sessions)}; with more than one (dataset, channel, tracking): "
          f"{len(multi)}; channels: {sorted({k[2] for k in per})}")
    for s in sorted(EXCLUDED):
        rows = [r for r in excl if r["session"] == s]
        print(f"{s}: {len(rows)} df_f curves, finite {sum(r['finite'] for r in rows)}, "
              f"median binned sum {np.median([r['binned_sum'] for r in rows]) if rows else np.nan:.1f}")


if __name__ == "__main__":
    main()
