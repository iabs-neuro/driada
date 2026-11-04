"""Source Data of Extended Data Fig. 1.

usage: python -X utf8 -B source_data_builders/build_ed1.py
       -> <[figures] output_dir>/source_data/ED_Fig1_source_data.xlsx

ed_calibration/ed_calibration_fps.py draws the figure together with the workbook
<[figures] output_dir>/source_data/ED_calibration_source_data.xlsx. That workbook is taken as it is; only its
title cell (Contents!A1) is set to name the figure as the paper numbers it, as the journal asks the linked
figure to be noted in the file. Every other cell is checked to be unchanged.
"""
import hashlib

import openpyxl

import sd_common as C

BOOK_SRC = C.OUT / "ED_calibration_source_data.xlsx"
BOOK_OUT = C.OUT / "ED_Fig1_source_data.xlsx"
TITLE = ("Source Data for Extended Data Fig. 1 | Calibration of false discoveries on recordings without "
         "neural-behavioral alignment")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def cells(path):
    wb = openpyxl.load_workbook(path)
    return {ws.title: {(c.row, c.column): c.value for row in ws.iter_rows() for c in row if c.value is not None}
            for ws in wb.worksheets}


def main():
    res = {}
    if BOOK_OUT.exists():
        BOOK_OUT.unlink()       # an earlier run of this builder
    wb = openpyxl.load_workbook(BOOK_SRC)
    old_title = wb["Contents"]["A1"].value
    wb["Contents"]["A1"] = TITLE
    wb.save(BOOK_OUT)
    a, b = cells(BOOK_SRC), cells(BOOK_OUT)
    assert list(a) == list(b), (list(a), list(b))
    changed = [(s, k) for s in a for k in set(a[s]) | set(b[s]) if a[s].get(k) != b[s].get(k)]
    assert changed == [("Contents", (1, 1))], changed
    res.update(book_source=BOOK_SRC.name, book_out=BOOK_OUT.name, sha256_book_source=sha(BOOK_SRC),
               sheets=list(b), cells_compared=sum(len(v) for v in a.values()), changed_cells=changed,
               title_was=old_title, title_now=TITLE)
    C.write_log("ed1", res)


if __name__ == "__main__":
    main()
