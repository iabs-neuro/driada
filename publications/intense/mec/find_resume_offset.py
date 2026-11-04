# -*- coding: utf-8 -*-
"""Byte offset in the _cell__traces section of the dump from which the sessions an interrupted decode
did not reach can be decoded: the start of the INSERT statement that holds the first row of the first
missing metasession. Rows are in primary-key order, so every row of the missing sessions lies after it.
Read-only; prints the offset and the head of that statement.

usage: python -B mec/find_resume_offset.py SECTION_FILE METASESSION
"""
import sys
from pathlib import Path

SEC = Path(sys.argv[1])
FIRST = sys.argv[2].encode()
INSERT = b"INSERT INTO `_cell__traces` VALUES "

size = SEC.stat().st_size
with open(SEC, "rb") as f:
    head = f.read(4096)
    k = head.find(INSERT)
    print("first statement head:", head[k:k + 120] if k >= 0 else "INSERT not in the first 4 KiB")
    # search forward from 80% of the section in 64 MiB blocks with an overlap
    blk, ov = 64 << 20, 256
    pos = int(size * 0.80)
    hit = -1
    while pos < size and hit < 0:
        f.seek(pos)
        data = f.read(blk + ov)
        i = data.find(b"'" + FIRST + b"'")
        if i >= 0:
            hit = pos + i
        pos += blk
    if hit < 0:
        sys.exit(f"[FAILED] {FIRST!r} not found after 80% of the section")
    # the INSERT statement that contains the hit: last statement start before it
    back = max(0, hit - (blk * 4))
    f.seek(back)
    data = f.read(hit - back)
    j = data.rfind(b"\n" + INSERT)
    if j < 0:
        sys.exit("[FAILED] no statement start within 256 MiB before the first row")
    start = back + j + 1
    f.seek(start)
    print(f"first row of {FIRST.decode()} at byte {hit:,} ({hit / size:.4f} of the section)")
    print(f"statement start at byte {start:,}; head: {f.read(120)!r}")
    print(f"RESUME_OFFSET={start}")
