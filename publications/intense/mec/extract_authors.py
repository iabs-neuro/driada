# -*- coding: utf-8 -*-
"""Fast reading of the authors' tuning curves (__angular_rate) in the extract of the dump.

The generic reader dj_dump_reader.iter_rows un-escapes every blob byte by byte in Python, which
takes many minutes on the 323 MB __angular_rate table. Here the rows are found with one
byte-level regular expression (ROW_AR) and only the blobs are un-escaped (same escape rules as
dj_dump_reader.unescape) and unpacked with datajoint 0.12.7, the version that wrote the dump.
Used by extract_dataset_map.py; SQL is [mec] dump_extract.
"""
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
sys.path.insert(0, str(PAPER))
sys.path.insert(0, str(HERE))
import intense_paths as P  # noqa: E402
import dj_dump_reader as R  # noqa: E402

SQL = P.get("mec", "dump_extract")

_UNESC = {k: bytes([v]) for k, v in R._UNESC.items()}
_ESC = re.compile(rb"\\(.)", re.S)


def unescape(b):
    return _ESC.sub(lambda m: _UNESC.get(m.group(1)[0], m.group(1)), b)


def blob(field):
    if field == b"NULL":
        return None
    assert field.startswith(b"_binary '") and field.endswith(b"'"), field[:30]
    return np.asarray(R.unpack_blob(unescape(field[9:-1])), dtype=float)


S = rb"'([^']{1,64})'"          # a quoted key without escapes
I = rb"(\d+)"
B = rb"(_binary '[^'\\]*(?:\\.[^'\\]*)*'|NULL)"   # unrolled loop: far faster than an alternation
D = rb"([-0-9.eE+]+|NULL)"
ROW_AR = re.compile(rb"\(" + rb",".join([S, I, S, S, I, S, S, S, S, S, S, S, B, B, B, D, D]) + rb"\)", re.S)


def table_bytes(mm, table):
    a = mm.find(b"-- Table structure for table `%s`" % table)
    b = mm.find(b"-- Table structure for table `", a + 10)
    return mm[a:b if b > 0 else len(mm)]
