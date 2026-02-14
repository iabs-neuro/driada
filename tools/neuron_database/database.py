"""NeuronDatabase: cross-session querying of INTENSE results.

Holds a tidy DataFrame of neuron-feature statistics across mice and sessions,
linked by matching tables that track neuron identity across sessions.
"""

import numpy as np
import pandas as pd


class NeuronDatabase:
    """Cross-session database for INTENSE selectivity results.

    Parameters
    ----------
    session_names : list[str]
        Ordered session names (e.g., ['1D', '2D', '3D', '4D']).
    matching : dict[str, pd.DataFrame]
        {mouse_id: matching_table}. Each DataFrame has matched_id as index,
        session names as columns, values = session-local neuron indices
        (NaN if absent).
    data : pd.DataFrame
        Tidy DataFrame with columns: mouse, session, matched_id, neuron_idx,
        feature, significant, me, pval, opt_delay.
    """

    def __init__(self, session_names, matching, data,
                 delay_strategy='nonnegative',
                 sessions_to_match=None,
                 mice_metadata_columns=None,
                 mice_info=None):
        self.session_names = list(session_names)
        self.matching = matching
        self._data = data
        self.delay_strategy = delay_strategy
        self.sessions_to_match = (sessions_to_match if sessions_to_match is not None
                                  else [1, len(session_names)])
        self.mice_metadata_columns = mice_metadata_columns or []
        self._mice_info = mice_info or {}
        self._excluded_mice = set()
        self._excluded_sessions = set()
        self._excluded_pairs = set()  # (mouse, session) tuples
        self._aggregate_feature_names = set()

    # --- Exclusion management ---

    def exclude_mice(self, mice):
        """Add mouse IDs to the exclusion set."""
        self._excluded_mice.update(mice)

    def include_mice(self, mice):
        """Remove mouse IDs from the exclusion set."""
        self._excluded_mice -= set(mice)

    def exclude_sessions(self, sessions):
        """Add session names to the exclusion set."""
        self._excluded_sessions.update(sessions)

    def include_sessions(self, sessions):
        """Remove session names from the exclusion set."""
        self._excluded_sessions -= set(sessions)

    def exclude_mouse_sessions(self, pairs):
        """Exclude specific (mouse, session) pairs.

        Parameters
        ----------
        pairs : list[tuple[str, str]]
            E.g., [('H01', '4D'), ('H03', '2D')].
        """
        self._excluded_pairs.update(tuple(p) for p in pairs)

    def include_mouse_sessions(self, pairs):
        """Remove specific (mouse, session) pairs from the exclusion set."""
        self._excluded_pairs -= set(tuple(p) for p in pairs)

    def reset_exclusions(self):
        """Clear all exclusions."""
        self._excluded_mice.clear()
        self._excluded_sessions.clear()
        self._excluded_pairs.clear()

    # --- Delay strategy ---

    @property
    def filter_delay(self):
        """Whether to filter by delay sign (True for 'nonnegative')."""
        return self.delay_strategy == 'nonnegative'

    # --- Aggregate features ---

    @property
    def aggregate_feature_names(self):
        """Set of aggregate feature names (excluded from n_sel counting)."""
        return self._aggregate_feature_names

    def inject_aggregate_features(self, aggregate_features,
                                   mi_threshold=0.04):
        """Add synthetic rows for aggregate features.

        For each neuron-session where ANY constituent feature passes the
        significance filters, adds a row with the aggregate feature name,
        mean MI across passing constituents, and min p-value.

        Parameters
        ----------
        aggregate_features : dict[str, list[str]]
            {aggregate_name: [constituent_feature1, constituent_feature2, ...]}.
        mi_threshold : float
            Minimum MI for a constituent to be included.
        """
        self._aggregate_feature_names.update(aggregate_features.keys())
        new_rows = []
        for agg_name, constituents in aggregate_features.items():
            df = self._data[self._data['feature'].isin(constituents)]
            mask = df['significant'] & (df['me'] > mi_threshold)
            if self.filter_delay:
                mask = mask & (df['delay_sign'] >= 0)
            sig = df[mask]
            if sig.empty:
                continue
            for (mouse, mid, session), grp in sig.groupby(
                    ['mouse', 'matched_id', 'session']):
                new_rows.append({
                    'mouse': mouse,
                    'session': session,
                    'matched_id': int(mid),
                    'neuron_idx': int(grp['neuron_idx'].iloc[0]),
                    'feature': agg_name,
                    'significant': True,
                    'me': grp['me'].mean(),
                    'pval': grp['pval'].min(),
                    'opt_delay': 0,
                    'delay_sign': 0,
                })
        if new_rows:
            self._data = pd.concat(
                [self._data, pd.DataFrame(new_rows)], ignore_index=True)

    # --- Data access ---

    @property
    def data(self):
        """Tidy DataFrame filtered by current exclusions."""
        df = self._data
        if self._excluded_mice:
            df = df[~df['mouse'].isin(self._excluded_mice)]
        if self._excluded_sessions:
            df = df[~df['session'].isin(self._excluded_sessions)]
        if self._excluded_pairs:
            pair_idx = pd.MultiIndex.from_tuples(self._excluded_pairs)
            mask = df.set_index(['mouse', 'session']).index.isin(pair_idx)
            df = df[~mask]
        return df

    # --- Info properties ---

    @property
    def features(self):
        """All unique feature names in the database."""
        return sorted(self.data['feature'].unique())

    @property
    def mice(self):
        """All mouse IDs (respecting exclusions)."""
        all_mice = sorted(self.matching.keys())
        if self._excluded_mice:
            return [m for m in all_mice if m not in self._excluded_mice]
        return all_mice

    @property
    def sessions(self):
        """All session names (respecting exclusions)."""
        if self._excluded_sessions:
            return [s for s in self.session_names
                    if s not in self._excluded_sessions]
        return list(self.session_names)

    def n_neurons(self, mouse=None, session=None):
        """Count of neurons present from matching tables.

        Parameters
        ----------
        mouse : str, optional
            Specific mouse. If None, all mice.
        session : str, optional
            Specific session. If None, all sessions.

        Returns
        -------
        int or pd.DataFrame
            Single count if both mouse and session given.
            mice × sessions DataFrame if neither given.
            Series if only one given.
        """
        if mouse is not None and session is not None:
            match_df = self.matching.get(mouse)
            if match_df is None or session not in match_df.columns:
                return 0
            return int(match_df[session].notna().sum())

        rows = []
        for m in self.mice:
            match_df = self.matching[m]
            counts = {}
            for s in self.sessions:
                if s in match_df.columns:
                    counts[s] = int(match_df[s].notna().sum())
                else:
                    counts[s] = 0
            rows.append(counts)

        result = pd.DataFrame(rows, index=self.mice)
        result.index.name = 'mouse'

        if mouse is not None:
            return result.loc[mouse]
        if session is not None:
            return result[session]
        return result

    # --- Matching ---

    def get_matched_ids(self, spec):
        """Get matched_ids satisfying a matching specification.

        Parameters
        ----------
        spec : int or list[str]
            If int: minimum number of sessions the neuron must be present in.
                1 returns None (no filtering).
            If list[str]: specific session names the neuron must be present in.

        Returns
        -------
        dict[str, set[int]] or None
            {mouse_id: set of matched_ids}, or None if spec is 1.
        """
        if isinstance(spec, int):
            if spec <= 1:
                return None
            result = {}
            sessions = self.sessions
            for mouse in self.mice:
                match_df = self.matching[mouse]
                cols = [s for s in sessions if s in match_df.columns]
                present_count = match_df[cols].notna().sum(axis=1)
                result[mouse] = set(match_df.index[present_count >= spec])
            return result
        else:
            result = {}
            for mouse in self.mice:
                match_df = self.matching[mouse]
                cols = [s for s in spec if s in match_df.columns]
                if len(cols) < len(spec):
                    result[mouse] = set()
                else:
                    mask = match_df[cols].notna().all(axis=1)
                    result[mouse] = set(match_df.index[mask])
            return result

    @staticmethod
    def matching_label(spec):
        """Human-readable label for a matching specification.

        Parameters
        ----------
        spec : int or list[str]

        Returns
        -------
        str
        """
        if isinstance(spec, int):
            return f"at least {spec} sessions"
        return f"sessions {'_'.join(spec)}"

    # --- Querying ---

    def query(self, feature=None, significant=None,
              mi_min=None, mi_max=None,
              pval_max=None,
              delay_positive=None, delay_negative=None,
              mice=None, sessions=None,
              exclude_mice=None, exclude_sessions=None):
        """Filter the tidy DataFrame by any combination of criteria.

        All parameters are optional and AND-combined.

        Parameters
        ----------
        feature : str or list[str], optional
            Filter to specific feature(s).
        significant : bool, optional
            Filter by significance.
        mi_min : float, optional
            Minimum MI (inclusive).
        mi_max : float, optional
            Maximum MI (inclusive).
        pval_max : float, optional
            Maximum p-value (inclusive).
        delay_positive : bool, optional
            If True, require opt_delay > 0.
        delay_negative : bool, optional
            If True, require opt_delay < 0.
        mice : list[str], optional
            Include only these mice (overrides global exclusions for mice).
        sessions : list[str], optional
            Include only these sessions (overrides global exclusions for sessions).
        exclude_mice : list[str], optional
            Additional mice to exclude (combined with global exclusions).
        exclude_sessions : list[str], optional
            Additional sessions to exclude (combined with global exclusions).

        Returns
        -------
        pd.DataFrame
            Filtered copy of the tidy DataFrame.
        """
        df = self._data.copy()

        # --- Mouse/session filtering ---
        if mice is not None:
            df = df[df['mouse'].isin(mice)]
        elif self._excluded_mice:
            df = df[~df['mouse'].isin(self._excluded_mice)]

        if exclude_mice:
            df = df[~df['mouse'].isin(exclude_mice)]

        if sessions is not None:
            df = df[df['session'].isin(sessions)]
        elif self._excluded_sessions:
            df = df[~df['session'].isin(self._excluded_sessions)]

        if exclude_sessions:
            df = df[~df['session'].isin(exclude_sessions)]

        if self._excluded_pairs:
            pair_idx = pd.MultiIndex.from_tuples(self._excluded_pairs)
            mask = df.set_index(['mouse', 'session']).index.isin(pair_idx)
            df = df[~mask]

        # --- Feature filtering ---
        if feature is not None:
            if isinstance(feature, str):
                df = df[df['feature'] == feature]
            else:
                df = df[df['feature'].isin(feature)]

        # --- Metric filtering ---
        if significant is not None:
            df = df[df['significant'] == significant]

        if mi_min is not None:
            df = df[df['me'] >= mi_min]

        if mi_max is not None:
            df = df[df['me'] <= mi_max]

        if pval_max is not None:
            df = df[df['pval'] <= pval_max]

        if delay_positive is True:
            df = df[df['opt_delay'] > 0]

        if delay_negative is True:
            df = df[df['opt_delay'] < 0]

        return df

    # --- Summary ---

    def summary(self):
        """Print a summary of the database contents."""
        df = self.data
        n_mice = len(self.mice)
        n_sessions = len(self.sessions)
        n_features = df['feature'].nunique()
        n_records = len(df)
        n_sig = df['significant'].sum()
        n_neurons_total = sum(
            self.matching[m][s].notna().sum()
            for m in self.mice
            for s in self.sessions
            if s in self.matching[m].columns
        )

        print(f"NeuronDatabase: {n_mice} mice, {n_sessions} sessions")
        print(f"  Records: {n_records} (neuron-session-feature entries)")
        print(f"  Significant: {n_sig} ({100*n_sig/n_records:.1f}%)")
        print(f"  Features: {n_features} - {sorted(df['feature'].unique())}")
        print(f"  Total neuron-sessions: {int(n_neurons_total)}")
        if self._excluded_mice:
            print(f"  Excluded mice: {sorted(self._excluded_mice)}")
        if self._excluded_sessions:
            print(f"  Excluded sessions: {sorted(self._excluded_sessions)}")
        if self._excluded_pairs:
            print(f"  Excluded pairs: {sorted(self._excluded_pairs)}")

    def __repr__(self):
        n = len(self.mice)
        s = len(self.sessions)
        r = len(self.data)
        return f"NeuronDatabase({n} mice, {s} sessions, {r} records)"
