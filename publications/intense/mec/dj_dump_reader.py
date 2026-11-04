# -*- coding: utf-8 -*-
"""Read DataJoint-encoded arrays straight out of the mysqldump — no MySQL server.

The dump writes longblob columns as mysqldump `_binary '...'` escaped-byte string
literals (NOT --hex-blob). The escaping is a fixed, reversible set, and
`datajoint.blob.unpack` acts as a correctness check: if the un-escaping is right, the
bytes decode to a clean numpy array; if wrong, unpack raises or returns garbage shape.

So we parse the INSERT tuples at the byte level, un-escape blob fields, and unpack them
with datajoint 0.12.7 (the exact version that wrote the dump). Only unpack_blob imports
datajoint, so the scripts that read text fields alone run without it.

Public API:
    columns(sql_path, table)         -> [(name, type), ...] from CREATE TABLE
    iter_rows(sql_path, table)       -> yields dict{col: value}; blob cols are raw bytes
    unpack_blob(raw_bytes)           -> numpy array via datajoint.blob.unpack

usage (prints the columns and the first two rows of a table):
    python mec/dj_dump_reader.py [sql file, default [mec] dump_extract] [table]
"""
import re
import numpy as np

# mysqldump (no --hex-blob) emits only these backslash escapes inside string literals.
_UNESC = {0x30: 0x00, 0x6E: 0x0A, 0x72: 0x0D, 0x5A: 0x1A,
          0x27: 0x27, 0x22: 0x22, 0x5C: 0x5C, 0x62: 0x08, 0x74: 0x09}


def unescape(s: bytes) -> bytes:
    out = bytearray(); i = 0; n = len(s)
    while i < n:
        c = s[i]
        if c == 0x5C and i + 1 < n:           # backslash
            nxt = s[i + 1]
            out.append(_UNESC.get(nxt, nxt))  # unknown escape -> literal char (MySQL rule)
            i += 2
        else:
            out.append(c); i += 1
    return bytes(out)


def _read_table_bytes(sql_path, table):
    """Return the raw bytes of `table`'s section (structure + data) from the dump."""
    marker = b"-- Table structure for table `%s`" % table.encode()
    with open(sql_path, "rb") as f:
        data = f.read()          # validation tables are small; for big ones use offsets
    start = data.find(marker)
    if start < 0:
        raise ValueError(f"table {table} not found in {sql_path}")
    nxt = data.find(b"-- Table structure for table `", start + len(marker))
    end = nxt if nxt >= 0 else len(data)
    return data[start:end]


def columns(sql_path, table):
    body = _read_table_bytes(sql_path, table).decode("utf-8", "replace")
    m = re.search(r"CREATE TABLE `%s` \((.*?)\n\) ENGINE" % re.escape(table), body, re.S)
    cols = []
    for line in m.group(1).splitlines():
        line = line.strip()
        cm = re.match(r"`([^`]+)`\s+([a-z0-9_]+)", line)
        if cm:
            cols.append((cm.group(1), cm.group(2)))
    return cols


def _split_tuples(payload: bytes):
    """Yield raw byte-slices of each top-level (...) tuple in an INSERT VALUES payload,
    respecting single-quoted strings with backslash escapes."""
    i, n = 0, len(payload)
    while i < n:
        if payload[i] != 0x28:  # '('
            i += 1; continue
        j = i + 1; depth = 1; inq = False
        while j < n and depth:
            c = payload[j]
            if inq:
                if c == 0x5C:
                    j += 2; continue
                if c == 0x27:
                    inq = False
            else:
                if c == 0x27:
                    inq = True
                elif c == 0x28:
                    depth += 1
                elif c == 0x29:
                    depth -= 1
            j += 1
        yield payload[i + 1:j - 1]
        i = j


def _split_fields(tup: bytes):
    """Split a tuple's bytes into field byte-slices at top-level commas (quote-aware)."""
    fields = []; i = 0; n = len(tup); start = 0; inq = False
    while i < n:
        c = tup[i]
        if inq:
            if c == 0x5C:
                i += 2; continue
            if c == 0x27:
                inq = False
        else:
            if c == 0x27:
                inq = True
            elif c == 0x2C:      # ','
                fields.append(tup[start:i]); start = i + 1
        i += 1
    fields.append(tup[start:])
    return fields


def _decode_field(raw: bytes, is_blob: bool):
    raw = raw.strip()
    if raw == b"NULL":
        return None
    if raw.startswith(b"_binary "):
        raw = raw[len(b"_binary "):].strip()
    if raw[:1] == b"'" and raw[-1:] == b"'":
        inner = unescape(raw[1:-1])
        return inner if is_blob else inner.decode("utf-8", "replace")
    return raw.decode("utf-8", "replace")   # numeric / token


def iter_rows(sql_path, table):
    cols = columns(sql_path, table)
    names = [c for c, _ in cols]
    isblob = [t.endswith("blob") for _, t in cols]
    body = _read_table_bytes(sql_path, table)
    ins = b"INSERT INTO `%s` VALUES " % table.encode()
    pos = 0
    while True:
        k = body.find(ins, pos)
        if k < 0:
            break
        end = body.find(b";\n", k)
        payload = body[k + len(ins): end if end >= 0 else len(body)]
        for tup in _split_tuples(payload):
            vals = _split_fields(tup)
            if len(vals) != len(names):
                continue
            yield {nm: _decode_field(v, b) for nm, v, b in zip(names, vals, isblob)}
        pos = (end + 1) if end >= 0 else len(body)


def unpack_blob(raw: bytes):
    import datajoint.blob as djb
    return djb.unpack(raw)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import intense_paths as P
    sql = sys.argv[1] if len(sys.argv) > 1 else str(P.get("mec", "dump_extract"))
    table = sys.argv[2] if len(sys.argv) > 2 else "_tracking_raw__open_field"
    print("columns:", columns(sql, table))
    for i, row in enumerate(iter_rows(sql, table)):
        arrs = {}
        for k, v in row.items():
            if isinstance(v, (bytes, bytearray)):
                try:
                    a = unpack_blob(v)
                    arrs[k] = (type(a).__name__, getattr(a, "shape", None),
                               getattr(a, "dtype", None))
                except Exception as e:
                    arrs[k] = f"UNPACK FAIL: {type(e).__name__}: {e}"
            else:
                arrs[k] = v
        print(f"\nrow {i}:")
        for k, v in arrs.items():
            print(f"  {k}: {v}")
        if i >= 1:
            break
