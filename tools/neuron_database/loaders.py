"""CSV parsing and directory loading for NeuronDatabase.

Parses INTENSE stats/significance CSVs (dict-as-string format) and matching
tables into a tidy pandas DataFrame suitable for cross-session analysis.
"""

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd



def parse_matching_csv(path, session_names):
    """Parse a matching table CSV (no header, 0 = absent).

    Parameters
    ----------
    path : str or Path
        Path to the matching CSV file.
    session_names : list[str]
        Ordered session names for column assignment.

    Returns
    -------
    pd.DataFrame
        Index = matched_id (0-based row number),
        columns = session_names,
        values = neuron indices (0 replaced with NaN).
    """
    df = pd.read_csv(path, header=None)
    if len(df.columns) != len(session_names):
        raise ValueError(
            f"Matching table {path} has {len(df.columns)} columns, "
            f"expected {len(session_names)} for sessions {session_names}"
        )
    df.columns = session_names
    df.index.name = 'matched_id'
    df = df.replace(0, np.nan)
    return df


def _parse_dict_cell(cell_str):
    """Parse a dict-as-string CSV cell. Returns dict or None if empty."""
    if pd.isna(cell_str):
        return None
    cell_str = str(cell_str).strip()
    if cell_str == '{}' or cell_str == '':
        return None
    try:
        d = ast.literal_eval(cell_str)
        return d if isinstance(d, dict) and d else None
    except (ValueError, SyntaxError):
        return None


def parse_stats_csv(path):
    """Parse an INTENSE stats CSV.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    dict[int, dict[str, dict]]
        {neuron_idx: {feature: {me, pval, opt_delay, ...}}}
        Only includes non-empty entries.
    """
    df = pd.read_csv(path, index_col=0)
    result = {}
    for neuron_idx in df.index:
        neuron_stats = {}
        for feature in df.columns:
            d = _parse_dict_cell(df.loc[neuron_idx, feature])
            if d is not None:
                neuron_stats[feature] = d
        if neuron_stats:
            result[int(neuron_idx)] = neuron_stats
    return result


def parse_significance_csv(path):
    """Parse an INTENSE significance CSV.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    dict[int, dict[str, bool]]
        {neuron_idx: {feature: is_significant}}
        True if cell dict has 'stage2': True.
    """
    df = pd.read_csv(path, index_col=0)
    result = {}
    for neuron_idx in df.index:
        neuron_sig = {}
        for feature in df.columns:
            d = _parse_dict_cell(df.loc[neuron_idx, feature])
            if d is not None:
                neuron_sig[feature] = bool(d.get('stage2', False))
        if neuron_sig:
            result[int(neuron_idx)] = neuron_sig
    return result


def load_session_from_csvs(stats_path, sig_path):
    """Load one session from stats + significance CSVs.

    Returns
    -------
    list[dict]
        Flat row dicts with keys: neuron_idx, feature, significant, me, pval, opt_delay.
        Only entries where stats dict is non-empty.
    """
    stats = parse_stats_csv(stats_path)
    sig = parse_significance_csv(sig_path)

    records = []
    for neuron_idx, feat_dict in stats.items():
        neuron_sig = sig.get(neuron_idx, {})
        for feature, s in feat_dict.items():
            records.append({
                'neuron_idx': neuron_idx,
                'feature': feature,
                'significant': neuron_sig.get(feature, False),
                'me': s.get('me', np.nan),
                'pval': s.get('pval', np.nan),
                'opt_delay': s.get('opt_delay', np.nan),
            })
    return records


def parse_experiment_filename(filename):
    """Extract (prefix, mouse_id, session) from stats/sig CSV filename.

    Expected pattern: '{prefix}_{mouse}_{session} INTENSE {type}.csv'
    Example: 'NOF_H03_1D INTENSE stats.csv' → ('NOF', 'H03', '1D')

    Returns
    -------
    tuple[str, str, str] or None
        (prefix, mouse_id, session) or None if no match.
    """
    m = re.match(r'^(.+?)_([^_]+)_([^_\s]+)\s+INTENSE\s+(?:stats|significance)\.csv$', filename)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


def _build_inverse_index(matching_df, session):
    """Build neuron_idx → matched_id mapping for one session.

    Parameters
    ----------
    matching_df : pd.DataFrame
        Matching table (index = matched_id, columns = session names).
    session : str
        Session column name.

    Returns
    -------
    dict[int, int]
        {neuron_idx: matched_id}
    """
    col = matching_df[session].dropna()
    # Matching values are 1-based neuron IDs; stats CSV index is 0-based
    return {int(v) - 1: int(idx) for idx, v in col.items()}


def _add_delay_sign(data, fps):
    """Add delay_sign column: -1, 0, or 1 based on opt_delay.

      abs(delay) <= DELAY_BOUNDARY -> 0 (neutral)
      delay < -DELAY_BOUNDARY -> -1 (negative)
      delay > DELAY_BOUNDARY -> 1 (positive)
    """
    boundary = fps * 0.75  # 15 frames at 20 fps
    delay = data['opt_delay']
    data['delay_sign'] = 0
    data.loc[delay < -boundary, 'delay_sign'] = -1
    data.loc[delay > boundary, 'delay_sign'] = 1


def _build_identity_matching(neuron_counts, session_names, mouse_id):
    """Build identity matching for experiments without matching tables.

    Each neuron gets matched_id = neuron_idx (0-based). All neurons are
    assumed present in all sessions.

    Raises ValueError if neuron counts differ across sessions.
    """
    counts = set(neuron_counts.values())
    if len(counts) > 1:
        raise ValueError(
            f"Mouse {mouse_id}: neuron counts differ across sessions "
            f"(identity matching requires equal counts): {neuron_counts}"
        )
    n = counts.pop()
    df = pd.DataFrame(
        {s: np.arange(1, n + 1) for s in session_names},
        index=range(n),
    )
    df.index.name = 'matched_id'
    return df


def load_from_csv_directory(data_dir, session_names,
                            matching_subdir='Matching',
                            tables_subdir='tables_disentangled',
                            experiment_prefix=None,
                            nontrivial_matching=True):
    """Load all matching tables and stats/significance CSVs from a directory.

    Parameters
    ----------
    data_dir : str or Path
        Root data directory.
    session_names : list[str]
        Ordered session names matching matching-table columns.
    matching_subdir : str
        Subdirectory containing matching CSVs.
    tables_subdir : str
        Subdirectory containing INTENSE stats/significance CSVs.
    experiment_prefix : str or None
        Expected filename prefix (e.g., 'NOF'). If None, auto-detected.
    nontrivial_matching : bool
        If True (default), load matching CSVs from matching_subdir.
        If False, build identity matching from INTENSE CSVs (all neurons
        assumed matched across sessions; raises if counts differ).

    Returns
    -------
    matching : dict[str, pd.DataFrame]
        {mouse_id: matching_table_dataframe}
    data : pd.DataFrame
        Tidy DataFrame with columns: mouse, session, matched_id, neuron_idx,
        feature, significant, me, pval, opt_delay.
    """
    data_dir = Path(data_dir)
    tables_dir = data_dir / tables_subdir

    if nontrivial_matching:
        # --- Load matching tables from CSVs ---
        matching_dir = data_dir / matching_subdir
        matching = {}
        for csv_path in sorted(matching_dir.glob('*.csv')):
            name = csv_path.stem
            parts = name.rsplit('_', 1)
            if len(parts) == 2:
                prefix, mouse_id = parts
                if experiment_prefix is not None and prefix != experiment_prefix:
                    continue
                if experiment_prefix is None:
                    experiment_prefix = prefix
            else:
                continue
            matching[mouse_id] = parse_matching_csv(csv_path, session_names)

        if not matching:
            raise FileNotFoundError(
                f"No matching tables found in {matching_dir}"
            )
        print(f"Loaded matching tables for {len(matching)} mice: "
              f"{sorted(matching.keys())}")

        # Build inverse indices
        inverse_indices = {}
        for mouse_id, match_df in matching.items():
            inverse_indices[mouse_id] = {}
            for session in session_names:
                if session in match_df.columns:
                    inverse_indices[mouse_id][session] = _build_inverse_index(
                        match_df, session)
    else:
        matching = None  # built after scanning CSVs
        inverse_indices = None

    # --- Load stats/significance CSVs ---
    all_records = []
    # Track neuron counts per mouse per session (for identity matching)
    mouse_session_neuron_counts = {}

    stats_files = sorted(tables_dir.glob('*INTENSE stats.csv'))

    for stats_path in stats_files:
        parsed = parse_experiment_filename(stats_path.name)
        if parsed is None:
            continue

        prefix, mouse_id, session = parsed
        if experiment_prefix is not None and prefix != experiment_prefix:
            continue
        if experiment_prefix is None:
            experiment_prefix = prefix
        if nontrivial_matching and mouse_id not in matching:
            continue
        if session not in session_names:
            continue

        sig_name = stats_path.name.replace('stats.csv', 'significance.csv')
        sig_path = tables_dir / sig_name
        if not sig_path.exists():
            print(f"  Warning: no significance file for {stats_path.name}")
            continue

        records = load_session_from_csvs(stats_path, sig_path)

        if nontrivial_matching:
            inv = inverse_indices.get(mouse_id, {}).get(session, {})
            for r in records:
                matched_id = inv.get(r['neuron_idx'])
                if matched_id is None:
                    continue
                r['mouse'] = mouse_id
                r['session'] = session
                r['matched_id'] = matched_id
                all_records.append(r)
        else:
            # Identity matching: matched_id = neuron_idx
            max_idx = 0
            for r in records:
                r['mouse'] = mouse_id
                r['session'] = session
                r['matched_id'] = r['neuron_idx']
                max_idx = max(max_idx, r['neuron_idx'])
                all_records.append(r)
            mouse_session_neuron_counts.setdefault(mouse_id, {})[session] = max_idx + 1

    if not all_records:
        raise ValueError("No records loaded from CSV files")

    # --- Build identity matching if needed ---
    if not nontrivial_matching:
        matching = {}
        for mouse_id, counts in mouse_session_neuron_counts.items():
            matching[mouse_id] = _build_identity_matching(
                counts, session_names, mouse_id)
        print(f"Built identity matching for {len(matching)} mice: "
              f"{sorted(matching.keys())}")

    # --- Build tidy DataFrame ---
    data = pd.DataFrame(all_records, columns=[
        'mouse', 'session', 'matched_id', 'neuron_idx',
        'feature', 'significant', 'me', 'pval', 'opt_delay',
    ])

    data['matched_id'] = data['matched_id'].astype(int)
    data['neuron_idx'] = data['neuron_idx'].astype(int)

    # TODO: support per-session fps
    _add_delay_sign(data, fps=20)

    n_mice = data['mouse'].nunique()
    n_sessions = data['session'].nunique()
    n_features = data['feature'].nunique()
    print(f"Loaded {len(data)} records: "
          f"{n_mice} mice, {n_sessions} sessions, {n_features} features")

    return matching, data


# ---------------------------------------------------------------------------
# Parsing helpers for experiment configs
# ---------------------------------------------------------------------------

def _parse_killed_sessions(killed_sessions, experiment_prefix):
    """Parse killed session strings to (mouse, session) tuples.

    'BOWL_F30_3D' with prefix='BOWL' → ('F30', '3D')
    """
    pairs = []
    for ks in killed_sessions:
        parts = ks.split('_')
        if parts[0] == experiment_prefix:
            parts = parts[1:]
        mouse = '_'.join(parts[:-1])
        session = parts[-1]
        pairs.append((mouse, session))
    return pairs


def _parse_excluded_mice(excluded_mice, experiment_prefix):
    """Strip experiment prefix from mouse IDs.

    '3DM_D14' with prefix='3DM' → 'D14'
    """
    prefix = experiment_prefix + '_'
    return [m[len(prefix):] if m.startswith(prefix) else m
            for m in excluded_mice]


def load_experiment(experiment_id, data_dir, config=None):
    """Load an experiment with pre-configured exclusions.

    Parameters
    ----------
    experiment_id : str
        Experiment identifier (e.g., 'NOF', 'BOWL', '3DM').
    data_dir : str or Path
        Root directory containing the experiment data.
    config : ExperimentConfig, optional
        Custom config. If None, uses EXPERIMENT_CONFIGS[experiment_id].

    Returns
    -------
    NeuronDatabase
        Configured database with exclusions applied and aggregate
        features injected.
    """
    from .configs import EXPERIMENT_CONFIGS
    from .database import NeuronDatabase

    if config is None:
        config = EXPERIMENT_CONFIGS[experiment_id]

    from .metadata import load_mice_info, build_mice_info_dict

    matching, data = load_from_csv_directory(
        data_dir, config.sessions,
        matching_subdir=config.matching_subdir,
        tables_subdir=config.tables_subdir,
        experiment_prefix=config.experiment_id,
        nontrivial_matching=config.nontrivial_matching,
    )

    mice_info_dict = {}
    if config.mice_metadata:
        mi_df = load_mice_info(data_dir)
        mice_info_dict = build_mice_info_dict(
            mi_df, config.experiment_id, config.mice_metadata)

    db = NeuronDatabase(config.sessions, matching, data,
                        delay_strategy=config.delay_strategy,
                        sessions_to_match=config.sessions_to_match,
                        mice_metadata_columns=config.mice_metadata,
                        mice_info=mice_info_dict)

    if config.excluded_mice:
        db.exclude_mice(
            _parse_excluded_mice(config.excluded_mice, config.experiment_id))
    if config.killed_sessions:
        db.exclude_mouse_sessions(
            _parse_killed_sessions(config.killed_sessions, config.experiment_id))
    if config.aggregate_features:
        db.inject_aggregate_features(config.aggregate_features)

    return db
