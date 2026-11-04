# -*- coding: utf-8 -*-
"""Reading the tables of the Obenhaus et al. (2022) dump without a MySQL server.

Helpers shared by decode_cohort_primary.py (calcium of every session) and arbiter_labels.py
(the authors' scores):

  table_span(table)        byte range of a table in DUMP, from the offsets in OFFS
  cols(table)              column names of a table, from its CREATE TABLE
  stream_tuples(s, e)      the INSERT tuples of a byte range, read in chunks, so the 15 GB
                           table of traces is never held in memory
  preload_tracking()       x, y, speed and head angle of every open-field session (extract)
  preload_curated()        the curated cells ('standard' in filtered_cells) of every session

DUMP is the dump ([mec] dump), OFFS its table offsets ([mec] table_offsets), VALID the
extract of small tables ([mec] dump_extract, written by extract_tables.py). A script may
point DUMP and OFFS at a file that holds only some sections of the dump, with an offsets
file for that file.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
sys.path.insert(0, str(PAPER))
sys.path.insert(0, str(HERE))
import intense_paths as P  # noqa: E402
import dj_dump_reader as R  # noqa: E402

DUMP = P.get("mec", "dump")
VALID = P.get("mec", "dump_extract")
OFFS = P.get("mec", "table_offsets")


def table_span(table):
    rows = []
    for line in OFFS.open(encoding="utf-8", errors="replace"):
        m = re.match(r"(\d+):-- Table structure for table `([^`]+)`", line)
        if m: rows.append((int(m.group(1)), m.group(2)))
    rows.sort()
    for i, (off, name) in enumerate(rows):
        if name == table:
            return off, (rows[i + 1][0] if i + 1 < len(rows) else DUMP.stat().st_size)
    raise ValueError(table)


def preload_tracking():
    trk = {}
    for row in R.iter_rows(str(VALID), "_tracking_raw__open_field"):
        key = (row["metasession_name"], str(row["session_order"]), row["session_name"])
        trk[key] = {k: R.unpack_blob(row[k]) for k in ("x_pos", "y_pos", "speed", "head_angle")}
    return trk


def preload_curated():
    cur = {}
    for row in R.iter_rows(str(VALID), "filtered_cells"):
        if str(row.get("param_hash_id_cell")) != "standard":
            continue
        key = (row["metasession_name"], str(row["session_order"]), row["session_name"])
        cur.setdefault(key, set()).add(int(row["cell_id"]))
    return cur


def cols(table):
    s, e = table_span(table)
    with open(DUMP, "rb") as f:
        f.seek(s); body = f.read(min(4000, e - s)).decode("utf-8", "replace")
    m = re.search(r"CREATE TABLE `%s` \((.*?)\n\) ENGINE" % re.escape(table), body, re.S)
    return [cm.group(1) for cm in re.finditer(r"`([^`]+)`\s+[a-z0-9_]+", m.group(1))]


def _txt(x):
    x = x.strip()
    if x.startswith(b"_binary "): x = x[8:].strip()
    return (x[1:-1] if x[:1] == b"'" else x)


def stream_tuples(start, end, chunk=1 << 24):
    """Incrementally yield raw tuple byte-slices across the table byte range."""
    buf = bytearray(); base = 0
    with open(DUMP, "rb") as f:
        f.seek(start); pos = start
        while pos < end or base < len(buf):
            if pos < end:
                data = f.read(min(chunk, end - pos)); pos += len(data); buf += data
            i = base; n = len(buf); progressed = False
            while i < n:
                if buf[i] != 0x28:  # '('
                    i += 1; continue
                j = i + 1; depth = 1; inq = False; ok = False
                while j < n:
                    c = buf[j]
                    if inq:
                        if c == 0x5C: j += 2; continue
                        if c == 0x27: inq = False
                    else:
                        if c == 0x27: inq = True
                        elif c == 0x28: depth += 1
                        elif c == 0x29:
                            depth -= 1
                            if depth == 0: ok = True; break
                    j += 1
                if ok:
                    yield bytes(buf[i + 1:j]); i = j + 1; progressed = True
                else:
                    break                 # incomplete tuple: wait for more data
            # trim consumed prefix
            del buf[:i]; base = 0
            if not progressed and pos >= end:
                break
