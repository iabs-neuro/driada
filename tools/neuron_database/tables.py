"""Count tables, fraction tables, MI tables, and Excel export for NeuronDatabase.

Builds mice x sessions count/fraction tables and neuron-level MI tables,
with optional filtering to neurons matched across all sessions.
"""

import numpy as np
import pandas as pd

MI_THRESHOLD = 0.04
PVAL_THRESHOLD = 0.001


def _resolve_filter_delay(filter_delay, db):
    """Resolve filter_delay: None means use db.filter_delay."""
    if filter_delay is None:
        return db.filter_delay
    return filter_delay


def apply_significance_filters(df, mi_threshold=MI_THRESHOLD,
                                pval_threshold=PVAL_THRESHOLD,
                                filter_delay=True):
    """Apply standard significance filters to a tidy DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: significant, me, pval, delay_sign.
    mi_threshold : float or None
        Minimum MI (strict >). None to skip.
    pval_threshold : float or None
        Maximum p-value (strict <). None to skip.
    filter_delay : bool
        If True, keep only delay_sign >= 0.

    Returns
    -------
    pd.DataFrame
        Filtered rows.
    """
    mask = df['significant']
    if mi_threshold is not None:
        mask = mask & (df['me'] > mi_threshold)
    if pval_threshold is not None:
        mask = mask & (df['pval'] < pval_threshold)
    if filter_delay:
        mask = mask & (df['delay_sign'] >= 0)
    return df[mask]


def _exclude_aggregates(df, db):
    """Remove aggregate feature rows from a DataFrame."""
    agg = db.aggregate_feature_names
    if agg:
        return df[~df['feature'].isin(agg)]
    return df


def _qualifying_neuron_sessions(db, mi_threshold=MI_THRESHOLD,
                                pval_threshold=PVAL_THRESHOLD,
                                filter_delay=None,
                                min_selectivities=None):
    """Compute qualifying (mouse, matched_id, session) tuples.

    Returns a MultiIndex of neuron-sessions where the neuron has
    >= min_selectivities significant features, or None if no filtering.
    Aggregate features are excluded from the count.
    """
    filter_delay = _resolve_filter_delay(filter_delay, db)
    if min_selectivities is None or min_selectivities <= 1:
        return None
    df = _exclude_aggregates(db.query(), db)
    df = apply_significance_filters(df, mi_threshold, pval_threshold,
                                    filter_delay)
    nsel = df.groupby(['mouse', 'matched_id', 'session']).size()
    return nsel[nsel >= min_selectivities].index


def _apply_nsel_filter(df, qualifying):
    """Keep only rows whose (mouse, matched_id, session) is in qualifying."""
    if qualifying is None:
        return df
    idx = df.set_index(['mouse', 'matched_id', 'session']).index
    return df[idx.isin(qualifying)]


def get_fully_matched_ids(db):
    """Get matched_ids present in all (non-excluded) sessions for each mouse.

    Parameters
    ----------
    db : NeuronDatabase

    Returns
    -------
    dict[str, set[int]]
        {mouse_id: set of matched_ids matched in every session}.
    """
    result = {}
    sessions = db.sessions
    for mouse in db.mice:
        match_df = db.matching[mouse]
        cols = [s for s in sessions if s in match_df.columns]
        if len(cols) < len(sessions):
            result[mouse] = set()
        else:
            mask = match_df[cols].notna().all(axis=1)
            result[mouse] = set(match_df.index[mask])
    return result


def _filter_by_matched_ids(df, matched_ids_per_mouse):
    """Filter DataFrame to rows whose (mouse, matched_id) is in the dict."""
    pairs = []
    for mouse, ids in matched_ids_per_mouse.items():
        for mid in ids:
            pairs.append((mouse, mid))
    if not pairs:
        return df.iloc[:0]
    pair_idx = pd.MultiIndex.from_tuples(pairs)
    mask = df.set_index(['mouse', 'matched_id']).index.isin(pair_idx)
    return df[mask]


def significance_count_table(db, feature, matched_ids_per_mouse=None,
                             mi_threshold=MI_THRESHOLD,
                             pval_threshold=PVAL_THRESHOLD,
                             filter_delay=None,
                             min_selectivities=None):
    """Build a mice x sessions table counting significant neurons.

    Parameters
    ----------
    db : NeuronDatabase
    feature : str
        Feature name.
    matched_ids_per_mouse : dict[str, set[int]], optional
        If provided, only count neurons in these sets (per mouse).
    mi_threshold, pval_threshold, filter_delay, min_selectivities
        Passed to apply_significance_filters. Set to None/False to disable.

    Returns
    -------
    pd.DataFrame
        Index = mouse IDs, columns = session names, values = counts.
    """
    filter_delay = _resolve_filter_delay(filter_delay, db)
    df = db.query(feature=feature)
    df = apply_significance_filters(df, mi_threshold, pval_threshold,
                                    filter_delay)
    qualifying = _qualifying_neuron_sessions(db, mi_threshold, pval_threshold,
                                            filter_delay, min_selectivities)
    df = _apply_nsel_filter(df, qualifying)

    if matched_ids_per_mouse is not None:
        df = _filter_by_matched_ids(df, matched_ids_per_mouse)

    if df.empty:
        table = pd.DataFrame(0, index=db.mice, columns=db.sessions)
    else:
        table = df.groupby(['mouse', 'session']).size().unstack(fill_value=0)
        table = table.reindex(index=db.mice, columns=db.sessions, fill_value=0)

    table.index.name = 'mouse'
    return table.astype(int)


def _selectivity_counts(db, matched_ids_per_mouse=None,
                        mi_threshold=MI_THRESHOLD,
                        pval_threshold=PVAL_THRESHOLD,
                        filter_delay=None,
                        min_selectivities=None):
    """Count selectivities per neuron per (mouse, session).

    Aggregate features are excluded from the count.
    Returns DataFrame with columns: mouse, session, matched_id, n_sel.
    """
    filter_delay = _resolve_filter_delay(filter_delay, db)
    df = _exclude_aggregates(db.query(), db)
    df = apply_significance_filters(df, mi_threshold, pval_threshold,
                                    filter_delay)
    qualifying = _qualifying_neuron_sessions(db, mi_threshold, pval_threshold,
                                            filter_delay, min_selectivities)
    df = _apply_nsel_filter(df, qualifying)

    if matched_ids_per_mouse is not None:
        df = _filter_by_matched_ids(df, matched_ids_per_mouse)

    counts = (df.groupby(['mouse', 'session', 'matched_id'])
              .size().reset_index(name='n_sel'))
    return counts


def _sel_count_to_table(counts, db, n_sel):
    """Build mice x sessions table from selectivity counts.

    n_sel : int
        If positive, count neurons with this many selectivities
        (exact match for 1-2, >= for 3+).
        If 0, count all selective neurons (any n_sel >= 1).
    """
    if n_sel == 0:
        filtered = counts
    else:
        filtered = counts[counts['n_sel'] >= n_sel] if n_sel >= 3 \
            else counts[counts['n_sel'] == n_sel]

    if filtered.empty:
        table = pd.DataFrame(0, index=db.mice, columns=db.sessions)
    else:
        table = (filtered.groupby(['mouse', 'session'])
                 .size().unstack(fill_value=0))
        table = table.reindex(index=db.mice, columns=db.sessions, fill_value=0)

    table.index.name = 'mouse'
    return table.astype(int)


def _neuron_count_table(db, matched_ids_per_mouse=None):
    """Build mice × sessions table of total neuron counts.

    If matched_ids_per_mouse is provided, only count neurons in those sets.
    Otherwise, count all neurons from the matching tables.
    """
    if matched_ids_per_mouse is not None:
        rows = {}
        for mouse in db.mice:
            match_df = db.matching[mouse]
            matched_ids = matched_ids_per_mouse.get(mouse, set())
            counts = {}
            for session in db.sessions:
                if session in match_df.columns:
                    col = match_df.loc[match_df.index.isin(matched_ids), session]
                    counts[session] = int(col.notna().sum())
                else:
                    counts[session] = 0
            rows[mouse] = counts
        table = pd.DataFrame(rows).T
        table = table.reindex(index=db.mice, columns=db.sessions, fill_value=0)
        table.index.name = 'mouse'
        return table.astype(int)
    else:
        return db.n_neurons()


def significance_fraction_table(db, feature, matched_ids_per_mouse=None,
                                mi_threshold=MI_THRESHOLD,
                                pval_threshold=PVAL_THRESHOLD,
                                filter_delay=None,
                                min_selectivities=None):
    """Build mice × sessions fraction table: significant / total neurons.

    Parameters
    ----------
    db : NeuronDatabase
    feature : str
    matched_ids_per_mouse : dict[str, set[int]], optional
    mi_threshold, pval_threshold, filter_delay, min_selectivities
        Passed to apply_significance_filters.

    Returns
    -------
    pd.DataFrame
        Values are fractions (0.0–1.0), NaN where total is 0.
    """
    counts = significance_count_table(db, feature, matched_ids_per_mouse,
                                      mi_threshold, pval_threshold,
                                      filter_delay, min_selectivities)
    totals = _neuron_count_table(db, matched_ids_per_mouse)
    return counts / totals.replace(0, float('nan'))


def significance_fraction_of_sel_table(db, feature, sel_totals,
                                       matched_ids_per_mouse=None,
                                       mi_threshold=MI_THRESHOLD,
                                       pval_threshold=PVAL_THRESHOLD,
                                       filter_delay=None,
                                       min_selectivities=None):
    """Build mice × sessions fraction table: significant for feature / total selective.

    Parameters
    ----------
    db : NeuronDatabase
    feature : str
    sel_totals : pd.DataFrame
        Mice × sessions table of total selective neuron counts (the 'all' composite).
    matched_ids_per_mouse : dict[str, set[int]], optional
    mi_threshold, pval_threshold, filter_delay, min_selectivities
        Passed to apply_significance_filters.

    Returns
    -------
    pd.DataFrame
        Values are fractions (0.0–1.0), NaN where total selective is 0.
    """
    counts = significance_count_table(db, feature, matched_ids_per_mouse,
                                      mi_threshold, pval_threshold,
                                      filter_delay, min_selectivities)
    return counts / sel_totals.replace(0, float('nan'))


def export_count_tables_excel(db, output_path, matched_ids_per_mouse=None,
                              features=None, mi_threshold=MI_THRESHOLD,
                              pval_threshold=PVAL_THRESHOLD,
                              filter_delay=None,
                              min_selectivities=None):
    """Export significance count tables to Excel, one sheet per feature.

    Parameters
    ----------
    db : NeuronDatabase
    output_path : str or Path
        Output .xlsx path.
    matched_ids_per_mouse : dict[str, set[int]], optional
        If provided, only count neurons in these sets.
    features : list[str], optional
        Features to include. If None, uses all features in the database.
    mi_threshold, pval_threshold, filter_delay, min_selectivities
        Passed to apply_significance_filters.
    """
    filter_delay = _resolve_filter_delay(filter_delay, db)
    fkw = dict(mi_threshold=mi_threshold, pval_threshold=pval_threshold,
               filter_delay=filter_delay)
    if features is None:
        features = db.features

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for feature in features:
            table = significance_count_table(
                db, feature, matched_ids_per_mouse=matched_ids_per_mouse,
                min_selectivities=min_selectivities, **fkw)
            # Excel sheet names max 31 chars
            table.to_excel(writer, sheet_name=feature[:31])

        # Composite sheets
        sel = _selectivity_counts(db, matched_ids_per_mouse,
                                  min_selectivities=min_selectivities, **fkw)
        _sel_count_to_table(sel, db, 0).to_excel(writer, sheet_name='all')
        _sel_count_to_table(sel, db, 1).to_excel(writer, sheet_name='sel1')
        _sel_count_to_table(sel, db, 2).to_excel(writer, sheet_name='sel2')
        _sel_count_to_table(sel, db, 3).to_excel(writer, sheet_name='sel3')


def export_fraction_tables_excel(db, output_path, matched_ids_per_mouse=None,
                                 features=None, mi_threshold=MI_THRESHOLD,
                                 pval_threshold=PVAL_THRESHOLD,
                                 filter_delay=None,
                                 min_selectivities=None):
    """Export fraction tables to Excel: significant / total neurons per cell.

    Parameters
    ----------
    db : NeuronDatabase
    output_path : str or Path
    matched_ids_per_mouse : dict[str, set[int]], optional
    features : list[str], optional
    mi_threshold, pval_threshold, filter_delay, min_selectivities
        Passed to apply_significance_filters.
    """
    filter_delay = _resolve_filter_delay(filter_delay, db)
    fkw = dict(mi_threshold=mi_threshold, pval_threshold=pval_threshold,
               filter_delay=filter_delay)
    if features is None:
        features = db.features

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for feature in features:
            table = significance_fraction_table(
                db, feature, matched_ids_per_mouse=matched_ids_per_mouse,
                min_selectivities=min_selectivities, **fkw)
            table.to_excel(writer, sheet_name=feature[:31])

        # Composite sheets
        sel = _selectivity_counts(db, matched_ids_per_mouse,
                                  min_selectivities=min_selectivities, **fkw)
        totals = _neuron_count_table(db, matched_ids_per_mouse)
        for n_sel, name in [(0, 'all'), (1, 'sel1'), (2, 'sel2'), (3, 'sel3')]:
            counts = _sel_count_to_table(sel, db, n_sel)
            frac = counts / totals.replace(0, float('nan'))
            frac.to_excel(writer, sheet_name=name)


def export_fraction_of_sel_tables_excel(db, output_path,
                                        matched_ids_per_mouse=None,
                                        features=None,
                                        mi_threshold=MI_THRESHOLD,
                                        pval_threshold=PVAL_THRESHOLD,
                                        filter_delay=None,
                                        min_selectivities=None):
    """Export fraction-of-selective tables: significant for feature / total selective.

    Parameters
    ----------
    db : NeuronDatabase
    output_path : str or Path
    matched_ids_per_mouse : dict[str, set[int]], optional
    features : list[str], optional
    mi_threshold, pval_threshold, filter_delay, min_selectivities
        Passed to apply_significance_filters.
    """
    filter_delay = _resolve_filter_delay(filter_delay, db)
    fkw = dict(mi_threshold=mi_threshold, pval_threshold=pval_threshold,
               filter_delay=filter_delay)
    if features is None:
        features = db.features

    # Compute total selective neurons once (the 'all' composite)
    sel = _selectivity_counts(db, matched_ids_per_mouse,
                              min_selectivities=min_selectivities, **fkw)
    sel_totals = _sel_count_to_table(sel, db, 0)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for feature in features:
            table = significance_fraction_of_sel_table(
                db, feature, sel_totals,
                matched_ids_per_mouse=matched_ids_per_mouse,
                min_selectivities=min_selectivities, **fkw)
            table.to_excel(writer, sheet_name=feature[:31])

        # Composite sheets: sel1/sel2/sel3 as fraction of all selective
        for n_sel, name in [(1, 'sel1'), (2, 'sel2'), (3, 'sel3')]:
            counts = _sel_count_to_table(sel, db, n_sel)
            frac = counts / sel_totals.replace(0, float('nan'))
            frac.to_excel(writer, sheet_name=name)


# ---------------------------------------------------------------------------
# MI tables (neuron-level)
# ---------------------------------------------------------------------------

def _build_nsel_per_neuron(db, matched_ids_per_mouse=None,
                           mi_threshold=MI_THRESHOLD,
                           pval_threshold=PVAL_THRESHOLD,
                           filter_delay=None,
                           min_selectivities=None):
    """Compute n_sel per (mouse, matched_id, session).

    Returns a pivot DataFrame with index=(mouse, matched_id),
    columns='{session}_nsel'.
    """
    filter_delay = _resolve_filter_delay(filter_delay, db)
    fkw = dict(mi_threshold=mi_threshold, pval_threshold=pval_threshold,
               filter_delay=filter_delay)
    df = _exclude_aggregates(db.query(), db)
    df = apply_significance_filters(df, **fkw)
    qualifying = _qualifying_neuron_sessions(db, mi_threshold, pval_threshold,
                                            filter_delay, min_selectivities)
    df = _apply_nsel_filter(df, qualifying)
    if matched_ids_per_mouse is not None:
        df = _filter_by_matched_ids(df, matched_ids_per_mouse)

    nsel = (df.groupby(['mouse', 'matched_id', 'session'])
            .size().reset_index(name='nsel'))
    pivot = nsel.pivot_table(index=['mouse', 'matched_id'],
                             columns='session', values='nsel',
                             fill_value=0)
    pivot.columns = [f'{s}_nsel' for s in pivot.columns]
    return pivot


def mi_table(db, feature, matched_ids_per_mouse=None,
             mi_threshold=MI_THRESHOLD, pval_threshold=PVAL_THRESHOLD,
             filter_delay=None, min_selectivities=None):
    """Build a neuron-level MI table for a single feature.

    One row per neuron identity that passes the filters for this feature
    in at least one session.

    Parameters
    ----------
    db : NeuronDatabase
    feature : str
    matched_ids_per_mouse : dict[str, set[int]], optional
    mi_threshold, pval_threshold, filter_delay, min_selectivities
        Passed to apply_significance_filters.

    Returns
    -------
    pd.DataFrame
        Columns: sessions (MI values, 0 if absent), {session}_nsel,
        mouse, matched_id.
    """
    filter_delay = _resolve_filter_delay(filter_delay, db)
    fkw = dict(mi_threshold=mi_threshold, pval_threshold=pval_threshold,
               filter_delay=filter_delay)
    sessions = db.sessions

    # Per-feature MI: neurons significant for this feature
    df = apply_significance_filters(db.query(feature=feature), **fkw)
    qualifying = _qualifying_neuron_sessions(db, mi_threshold, pval_threshold,
                                            filter_delay, min_selectivities)
    df = _apply_nsel_filter(df, qualifying)
    if matched_ids_per_mouse is not None:
        df = _filter_by_matched_ids(df, matched_ids_per_mouse)

    if df.empty:
        return pd.DataFrame(columns=sessions +
                            [f'{s}_nsel' for s in sessions] +
                            ['mouse', 'matched_id'])

    # Pivot MI values
    mi_pivot = df.pivot_table(index=['mouse', 'matched_id'],
                              columns='session', values='me',
                              aggfunc='first')
    mi_pivot = mi_pivot.reindex(columns=sessions, fill_value=0).fillna(0)
    mi_pivot = mi_pivot.round(3)

    # Nsel across all features
    nsel_pivot = _build_nsel_per_neuron(db, matched_ids_per_mouse,
                                        min_selectivities=min_selectivities,
                                        **fkw)
    nsel_cols = [f'{s}_nsel' for s in sessions]
    nsel_pivot = nsel_pivot.reindex(columns=nsel_cols, fill_value=0)

    # Join
    result = mi_pivot.join(nsel_pivot, how='left').fillna(0)
    result = result.reset_index()

    # Reorder columns
    col_order = sessions + nsel_cols + ['mouse', 'matched_id']
    result = result[col_order]

    # Ensure nsel columns are int
    for c in nsel_cols:
        result[c] = result[c].astype(int)

    return result


def mi_table_composite(db, n_sel, matched_ids_per_mouse=None,
                       mi_threshold=MI_THRESHOLD,
                       pval_threshold=PVAL_THRESHOLD,
                       filter_delay=None,
                       min_selectivities=None):
    """Build a neuron-level MI table for composite selectivity.

    MI value per session = mean across all features the neuron is
    significant for in that session.

    Parameters
    ----------
    db : NeuronDatabase
    n_sel : int
        0 for 'all' (any selectivity), 1/2 for exact, 3+ for >= 3.
    matched_ids_per_mouse : dict[str, set[int]], optional
    mi_threshold, pval_threshold, filter_delay, min_selectivities
        Passed to apply_significance_filters.

    Returns
    -------
    pd.DataFrame
        Same format as mi_table.
    """
    filter_delay = _resolve_filter_delay(filter_delay, db)
    fkw = dict(mi_threshold=mi_threshold, pval_threshold=pval_threshold,
               filter_delay=filter_delay)
    sessions = db.sessions

    df = _exclude_aggregates(db.query(), db)
    df = apply_significance_filters(df, **fkw)
    qual_nsel = _qualifying_neuron_sessions(db, mi_threshold, pval_threshold,
                                            filter_delay, min_selectivities)
    df = _apply_nsel_filter(df, qual_nsel)
    if matched_ids_per_mouse is not None:
        df = _filter_by_matched_ids(df, matched_ids_per_mouse)

    if df.empty:
        return pd.DataFrame(columns=sessions +
                            [f'{s}_nsel' for s in sessions] +
                            ['mouse', 'matched_id'])

    # Mean MI and nsel per (mouse, matched_id, session)
    per_neuron_session = (df.groupby(['mouse', 'matched_id', 'session'])
                          .agg(mean_mi=('me', 'mean'), nsel=('me', 'size'))
                          .reset_index())

    # Determine which sessions match the selectivity criterion
    if n_sel == 0:
        qualifying = per_neuron_session
    elif n_sel >= 3:
        qualifying = per_neuron_session[per_neuron_session['nsel'] >= n_sel]
    else:
        qualifying = per_neuron_session[per_neuron_session['nsel'] == n_sel]

    if qualifying.empty:
        return pd.DataFrame(columns=sessions +
                            [f'{s}_nsel' for s in sessions] +
                            ['mouse', 'matched_id'])

    # Neuron identities that qualify in at least one session
    qual_ids = qualifying[['mouse', 'matched_id']].drop_duplicates()
    qual_idx = pd.MultiIndex.from_frame(qual_ids)

    # Keep all sessions for qualifying neurons
    all_sessions = per_neuron_session[
        per_neuron_session.set_index(['mouse', 'matched_id']).index.isin(qual_idx)
    ]

    # MI: mean MI only in qualifying sessions, 0 otherwise
    qualifying_key = qualifying.set_index(['mouse', 'matched_id', 'session'])
    mi_pivot = qualifying_key['mean_mi'].unstack(fill_value=0)
    mi_pivot = mi_pivot.reindex(columns=sessions, fill_value=0).fillna(0)
    mi_pivot = mi_pivot.round(3)

    # Nsel: actual nsel for ALL sessions of qualifying neurons
    nsel_pivot = all_sessions.pivot_table(index=['mouse', 'matched_id'],
                                          columns='session', values='nsel',
                                          aggfunc='first')
    nsel_pivot = nsel_pivot.reindex(columns=sessions, fill_value=0).fillna(0)
    nsel_pivot.columns = [f'{s}_nsel' for s in sessions]

    # Join
    result = mi_pivot.join(nsel_pivot, how='left').fillna(0)
    result = result.reset_index()

    nsel_cols = [f'{s}_nsel' for s in sessions]
    col_order = sessions + nsel_cols + ['mouse', 'matched_id']
    result = result[col_order]

    for c in nsel_cols:
        result[c] = result[c].astype(int)

    return result


def export_mi_tables_excel(db, output_path, matched_ids_per_mouse=None,
                           features=None, mi_threshold=MI_THRESHOLD,
                           pval_threshold=PVAL_THRESHOLD,
                           filter_delay=None,
                           min_selectivities=None):
    """Export neuron-level MI tables to Excel, one sheet per feature.

    Parameters
    ----------
    db : NeuronDatabase
    output_path : str or Path
    matched_ids_per_mouse : dict[str, set[int]], optional
    features : list[str], optional
    mi_threshold, pval_threshold, filter_delay, min_selectivities
        Passed to apply_significance_filters.
    """
    filter_delay = _resolve_filter_delay(filter_delay, db)
    fkw = dict(mi_threshold=mi_threshold, pval_threshold=pval_threshold,
               filter_delay=filter_delay)
    if features is None:
        features = db.features

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for feature in features:
            table = mi_table(db, feature, matched_ids_per_mouse,
                             min_selectivities=min_selectivities, **fkw)
            table.to_excel(writer, sheet_name=feature[:31], index=False)

        # Composite sheets
        for n_sel, name in [(0, 'all'), (1, 'sel1'), (2, 'sel2'), (3, 'sel3')]:
            table = mi_table_composite(db, n_sel, matched_ids_per_mouse,
                                       min_selectivities=min_selectivities,
                                       **fkw)
            table.to_excel(writer, sheet_name=name, index=False)


# ---------------------------------------------------------------------------
# Retention tables (neurons selective in >= N sessions)
# ---------------------------------------------------------------------------

def retention_count_table(db, feature, matched_ids_per_mouse=None,
                          mi_threshold=MI_THRESHOLD,
                          pval_threshold=PVAL_THRESHOLD,
                          filter_delay=None,
                          min_selectivities=None):
    """Build a mice × n_sessions retention table for a feature.

    Column k = count of neurons selective for the feature in at least
    k sessions (any combination). Last row is 'total' (sum across mice).

    Parameters
    ----------
    db : NeuronDatabase
    feature : str
    matched_ids_per_mouse : dict[str, set[int]], optional
    mi_threshold, pval_threshold, filter_delay, min_selectivities
        Passed to apply_significance_filters.

    Returns
    -------
    pd.DataFrame
        Index = mouse IDs + 'total', columns = 1..n_sessions.
    """
    filter_delay = _resolve_filter_delay(filter_delay, db)
    df = db.query(feature=feature)
    df = apply_significance_filters(df, mi_threshold, pval_threshold,
                                    filter_delay)
    qualifying = _qualifying_neuron_sessions(db, mi_threshold, pval_threshold,
                                            filter_delay, min_selectivities)
    df = _apply_nsel_filter(df, qualifying)
    if matched_ids_per_mouse is not None:
        df = _filter_by_matched_ids(df, matched_ids_per_mouse)

    n_sessions = len(db.sessions)
    columns = list(range(1, n_sessions + 1))

    if df.empty:
        table = pd.DataFrame(0, index=db.mice, columns=columns)
    else:
        # Count sessions each neuron is significant in
        session_counts = df.groupby(['mouse', 'matched_id'])['session'].nunique()

        rows = {}
        for mouse in db.mice:
            if mouse in session_counts.index.get_level_values(0):
                mc = session_counts.loc[mouse]
                rows[mouse] = [int((mc >= k).sum()) for k in columns]
            else:
                rows[mouse] = [0] * n_sessions
        table = pd.DataFrame.from_dict(rows, orient='index', columns=columns)
        table = table.reindex(index=db.mice, fill_value=0)

    table.index.name = 'mouse'
    table.loc['total'] = table.sum()
    return table.astype(int)


def export_retention_tables_excel(db, output_path,
                                   matched_ids_per_mouse=None,
                                   features=None,
                                   mi_threshold=MI_THRESHOLD,
                                   pval_threshold=PVAL_THRESHOLD,
                                   filter_delay=None,
                                   min_selectivities=None):
    """Export retention tables to Excel, one sheet per feature.

    Parameters
    ----------
    db : NeuronDatabase
    output_path : str or Path
    matched_ids_per_mouse : dict[str, set[int]], optional
    features : list[str], optional
    mi_threshold, pval_threshold, filter_delay, min_selectivities
        Passed to apply_significance_filters.
    """
    filter_delay = _resolve_filter_delay(filter_delay, db)
    fkw = dict(mi_threshold=mi_threshold, pval_threshold=pval_threshold,
               filter_delay=filter_delay)
    if features is None:
        features = db.features

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for feature in features:
            table = retention_count_table(
                db, feature, matched_ids_per_mouse,
                min_selectivities=min_selectivities, **fkw)
            table.to_excel(writer, sheet_name=feature[:31])


# ---------------------------------------------------------------------------
# Cross-stats table (feature names per neuron × session)
# ---------------------------------------------------------------------------

def cross_stats_table(db, min_sessions=1,
                      mi_threshold=MI_THRESHOLD,
                      pval_threshold=PVAL_THRESHOLD,
                      filter_delay=None,
                      min_selectivities=None):
    """Build a cross-session stats table with feature names per cell.

    Replicates the layout of the old INTENSE cross-stats CSVs but with
    plain feature names instead of (feature, delay_sign, MI) triples.

    Each cell contains comma-separated feature names that the neuron is
    significantly selective for, '---' if the neuron is present in that
    session but has no significant features, or empty if the neuron is
    absent from that session.

    Only neurons with at least one significant selectivity across all
    sessions are included.

    Parameters
    ----------
    db : NeuronDatabase
    min_sessions : int
        Minimum number of sessions the neuron must be present in.
    mi_threshold, pval_threshold, filter_delay, min_selectivities
        Passed to apply_significance_filters.

    Returns
    -------
    pd.DataFrame
        Columns: sessions + ['matching_row', 'mouse'].
    """
    sessions = db.sessions

    filter_delay = _resolve_filter_delay(filter_delay, db)
    # Get significant entries (exclude aggregates — cross-stats shows components)
    df = _exclude_aggregates(db.query(), db)
    df = apply_significance_filters(df, mi_threshold, pval_threshold,
                                    filter_delay)
    qualifying = _qualifying_neuron_sessions(db, mi_threshold, pval_threshold,
                                            filter_delay, min_selectivities)
    df = _apply_nsel_filter(df, qualifying)

    # Build lookup: (mouse, matched_id, session) -> sorted feature names
    sig_lookup = {}
    for _, row in df.iterrows():
        key = (row['mouse'], int(row['matched_id']), row['session'])
        sig_lookup.setdefault(key, []).append(row['feature'])

    rows = []
    for mouse in db.mice:
        match_df = db.matching[mouse]
        for matched_id in match_df.index:
            present = [s for s in sessions
                       if s in match_df.columns
                       and pd.notna(match_df.loc[matched_id, s])]

            if len(present) < min_sessions:
                continue

            row_data = {}
            has_selectivity = False

            for s in sessions:
                if s not in present:
                    row_data[s] = ''
                else:
                    feats = sig_lookup.get((mouse, matched_id, s))
                    if feats:
                        row_data[s] = ', '.join(sorted(feats))
                        has_selectivity = True
                    else:
                        row_data[s] = '---'

            if has_selectivity:
                row_data['matching_row'] = matched_id
                row_data['mouse'] = mouse
                rows.append(row_data)

    result = pd.DataFrame(rows, columns=sessions + ['matching_row', 'mouse'])
    return result


def export_cross_stats_csv(db, output_path, min_sessions=1,
                           mi_threshold=MI_THRESHOLD,
                           pval_threshold=PVAL_THRESHOLD,
                           filter_delay=None,
                           min_selectivities=None):
    """Export cross-session stats table to CSV.

    Parameters
    ----------
    db : NeuronDatabase
    output_path : str or Path
    min_sessions : int
    mi_threshold, pval_threshold, filter_delay, min_selectivities
        Passed to apply_significance_filters.

    Returns
    -------
    pd.DataFrame
        The exported table.
    """
    table = cross_stats_table(db, min_sessions, mi_threshold, pval_threshold,
                              filter_delay, min_selectivities)
    table.to_csv(output_path)
    print(f"Exported cross-stats: {len(table)} neurons -> {output_path}")
    return table
