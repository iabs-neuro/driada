"""Source Data workbooks: one .xlsx per figure, one sheet per panel.

The figure scripts register the numbers they draw while drawing them, so the
workbook holds exactly what is on the page. The first sheet lists every panel
with its description and the files the numbers come from.
"""

from pathlib import Path

import numpy as np
import pandas as pd


class SourceData:
    """Collects the tables behind the panels of one figure."""

    def __init__(self, figure, title):
        self.figure = figure
        self.title = title
        self.sheets = {}   # sheet name -> list of (caption, DataFrame)
        self.notes = []    # (panel, description, source)

    def add(self, panel, df, caption=None):
        """Add a table to the sheet of `panel` (e.g. 'a', 'e-scatter')."""
        if isinstance(df, dict):
            df = pd.DataFrame(df)
        name = f"Fig{self.figure}{panel}"[:31]
        self.sheets.setdefault(name, []).append((caption, df))
        return df

    def note(self, panel, description, source=""):
        self.notes.append((f"{self.figure}{panel}", description, str(source)))

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        index = pd.DataFrame(self.notes, columns=["panel", "what is shown", "computed from"])
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            head = pd.DataFrame({self.title: [
                "Source Data. Each sheet holds the numbers drawn in one panel; "
                "the table below describes them. Neuron indices are rows of the calcium "
                "matrix of the synchronized session files (0-based)."]})
            head.to_excel(xw, sheet_name="Contents", index=False)
            index.to_excel(xw, sheet_name="Contents", index=False, startrow=3)
            for name, blocks in self.sheets.items():
                row = 0
                for caption, df in blocks:
                    if caption:
                        pd.DataFrame({caption: []}).to_excel(xw, sheet_name=name, index=False, startrow=row)
                        row += 1
                    df.to_excel(xw, sheet_name=name, index=False, startrow=row)
                    row += len(df) + 3
        print(f"Source Data: {path} ({len(self.sheets)} sheets)")
        return path


def line_table(ax, xname="x", yname="y"):
    """x, y of every visible data line of an axes, one column pair per line (label in the header)."""
    cols = {}
    for k, ln in enumerate(ax.get_lines()):
        if not ln.get_visible() or ln.get_label().startswith("_child") and len(ln.get_xdata()) < 3:
            continue
        lab = ln.get_label()
        lab = f"line{k}" if lab.startswith("_") else lab
        cols[f"{lab}: {xname}"] = pd.Series(np.asarray(ln.get_xdata(), dtype=float))
        cols[f"{lab}: {yname}"] = pd.Series(np.asarray(ln.get_ydata(), dtype=float))
    return pd.DataFrame(cols)


def runs_of_ones(binary, times):
    """Start and end times of consecutive runs of 1 in a binary series."""
    b = np.asarray(binary) > 0
    d = np.diff(np.r_[0, b.astype(int), 0])
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    t = np.asarray(times)
    return pd.DataFrame({"start (s)": t[starts], "end (s)": t[ends]})
