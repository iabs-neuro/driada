# -*- coding: utf-8 -*-
"""Selective extractor for the Obenhaus 2022 mysqldump.

The full dump.sql is 101.5 GB / 178 tables, most of it shuffle (permutation-control) tables
the analysis does not read. This carves out only the tables in a list into a small, valid,
loadable SQL file, using the byte offsets of [mec] table_offsets.

Output layout that MySQL can load:
    <header>                       bytes [0 : first_table_offset)   -- SET/charset preamble
    <section for each wanted table> each table's own DROP/CREATE/LOCK/INSERT/UNLOCK
    <footer>                       trailing SET restores from end of file

The 'validate' set is the extract the MEC scripts read ([mec] dump_extract): session metadata,
curated cells, frame times, raw tracking and the authors' tuning curves (0.62 GB).
Read-only on the dump; writes only the extract.

usage: python mec/extract_tables.py [validate|essential]
       validate  -> [mec] dump_extract
       essential -> extract_essential.sql in the same folder (not read by the MEC scripts)
"""
import re, sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))
import intense_paths as P  # noqa: E402

DUMP = P.get("mec", "dump")
OFFS = P.get("mec", "table_offsets")
FILESIZE = DUMP.stat().st_size

# Footer = trailing restore block; grab a generous tail and keep from the last table's end.
FOOTER_BYTES = 4096


def load_offsets():
    rows = []
    for line in OFFS.open(encoding="utf-8", errors="replace"):
        m = re.match(r"(\d+):-- Table structure for table `([^`]+)`", line)
        if m:
            rows.append((int(m.group(1)), m.group(2)))
    rows.sort()
    spans = {}
    for i, (off, name) in enumerate(rows):
        end = rows[i + 1][0] if i + 1 < len(rows) else FILESIZE
        spans[name] = (off, end)
    return rows, spans


def extract(wanted, out_path, verbose=True):
    rows, spans = load_offsets()
    first_off = rows[0][0]
    missing = [w for w in wanted if w not in spans]
    if missing:
        print("!!! not in dump:", missing)
    present = [w for w in wanted if w in spans]
    total = sum(spans[w][1] - spans[w][0] for w in present)
    if verbose:
        print(f"extracting {len(present)} tables, {total/1e9:.2f} GB -> {out_path.name}")
    with DUMP.open("rb") as f, open(out_path, "wb") as out:
        f.seek(0); out.write(f.read(first_off))                 # header
        for w in present:
            a, b = spans[w]
            f.seek(a)
            remaining = b - a
            while remaining > 0:
                chunk = f.read(min(1 << 24, remaining))
                if not chunk:
                    break
                out.write(chunk); remaining -= len(chunk)
            if verbose:
                print(f"  + {w:38s} {(b-a)/1e9:7.3f} GB")
        f.seek(FILESIZE - FOOTER_BYTES)                          # footer (restore SETs)
        tail = f.read()
        cut = tail.rfind(b"UNLOCK TABLES;")
        out.write(tail[cut + len(b"UNLOCK TABLES;"):] if cut != -1 else tail)
    print(f"WROTE {out_path}  ({out_path.stat().st_size/1e9:.2f} GB)")


# The essential set for a first analysis pass (traces + behaviour + cell-type + metadata).
ESSENTIAL = [
    # cell / signal
    "_cell", "_cell__rois", "_cell__traces", "_cell__spikes",
    # behaviour / timing
    "_sync", "_sync__mesc", "_sync__s_i", "_sync__unprocessed", "_sync__ws",
    "_tracking_raw", "_tracking_raw__open_field", "_tracking_raw__linear",
    "_tracking_raw__d_l_c_part", "__tracking__open_field", "__angular_rate",
    # scores / cell-type reference
    "__grid_score", "__ratemap", "__ratemap__fields", "__o_v_map",
    "__b_v_field__fields", "__s_n_r", "__scoremap_f_o_v",
    # inclusion + metadata + params (all tiny)
    "filtered_cells", "filtered_sessions", "session", "session__apparatus",
    "session__arena_object", "session__cue_card", "session__data",
    "session__intervention", "session__session_type", "session_d_l_c",
    "meta_session", "meta_session__note", "meta_session__scope",
    "meta_session__setup", "meta_session__suite2py_ops",
    "setup__sync", "setup__tracking1_d", "setup__tracking2_d",
    "#arena_object", "#arena_object__picture", "#grid_params", "#border_params",
    "#b_v_score_params", "#b_v_score_field_method", "#tracking_params",
    "#signal_tracking_params", "#spike_filter_params", "#session_type",
    "#d_l_c_tracking_processing_method", "#d_l_c_tracking_type",
    "#filtered_cells_params", "#filtered_sessions_params", "#grid_params",
]

# The small tables the MEC scripts read from [mec] dump_extract; the 15 GB of traces stay in the dump.
VALIDATION = [
    "session", "meta_session", "meta_session__scope", "filtered_cells",
    "filtered_sessions", "_sync", "_tracking_raw__open_field", "__angular_rate",
    "#grid_params", "#tracking_params",
]

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "validate"
    target = P.get("mec", "dump_extract")
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "validate":
        extract(VALIDATION, target)
    elif mode == "essential":
        extract(ESSENTIAL, target.parent / "extract_essential.sql")
    else:
        print("usage: extract_tables.py [validate|essential]")
