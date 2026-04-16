"""Mice metadata: download from Google Sheets and table annotation."""

import io
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

SHEETS_ID = '1bnS4P_9tmsv6z6FdRAVuCST-kxKk3QPWQYU-l1clNM8'
SHEET_NAME = 'metadata'
SHEETS_CSV_URL = (
    f'https://docs.google.com/spreadsheets/d/{SHEETS_ID}'
    f'/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}'
)

_CACHE_NAME = 'Mice_info.csv'


def download_mice_info(data_dir):
    """Download metadata sheet from Google Sheets, cache as CSV.

    Returns DataFrame or None on failure.
    """
    cache_path = Path(data_dir).parent / _CACHE_NAME
    try:
        raw = urlopen(SHEETS_CSV_URL, timeout=15).read().decode('utf-8')
        df = pd.read_csv(io.StringIO(raw))
    except Exception as e:
        print(f"Warning: could not download mice metadata: {e}")
        if cache_path.exists():
            print(f"  Using cached {cache_path}")
            df = pd.read_csv(cache_path)
        else:
            return None

    # Drop empty columns
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    df = df.dropna(how='all', axis=1)
    df = df.replace('', None).ffill()

    # Cache for offline use
    df.to_csv(cache_path, index=False)
    return df


def load_mice_info(data_dir):
    """Download mice metadata (falls back to cached CSV)."""
    return download_mice_info(data_dir)


def build_mice_info_dict(mice_info_df, experiment_id, columns):
    """Build {mouse_code: {col: val}} for one experiment."""
    if mice_info_df is None or not columns:
        return {}
    exp = mice_info_df[mice_info_df['Experiment'] == experiment_id]
    return {row['Mouse']: {c: row[c] for c in columns if c in row.index}
            for _, row in exp.iterrows()}


def annotate_mice_table(table, db):
    """Prepend metadata columns to a mice-indexed table (count/fraction/retention)."""
    if not db.mice_metadata_columns or not db._mice_info:
        return table
    for col in reversed(db.mice_metadata_columns):
        table.insert(0, col, [db._mice_info.get(m, {}).get(col, '')
                               for m in table.index])
    return table


def annotate_neuron_table(table, db):
    """Add metadata columns after 'mouse' in a neuron-level table (MI/cross-stats)."""
    if not db.mice_metadata_columns or not db._mice_info or table.empty:
        return table
    pos = table.columns.get_loc('mouse') + 1
    for i, col in enumerate(db.mice_metadata_columns):
        table.insert(pos + i, col,
                     table['mouse'].map(
                         lambda m, c=col: db._mice_info.get(m, {}).get(c, '')))
    return table
