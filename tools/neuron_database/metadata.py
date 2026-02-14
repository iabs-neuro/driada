"""Mice metadata: loading from Mice_info.xlsx and table annotation."""

import pandas as pd
from pathlib import Path


def load_mice_info(data_dir):
    """Load Mice_info.xlsx from parent of data_dir. Returns DataFrame or None."""
    path = Path(data_dir).parent / 'Mice_info.xlsx'
    if not path.exists():
        return None
    df = pd.read_excel(path)
    return df.replace('', None).ffill()


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
