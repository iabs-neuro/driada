"""Filter utilities for disentanglement analysis.

This module provides building blocks for creating custom disentanglement
filters. Filters are composable, population-level functions that run
BEFORE the parallel disentanglement processing.

Filter Protocol
---------------
All filters must follow this signature::

    def my_filter(
        neuron_selectivities,    # dict: {neuron_id: [feat1, feat2, ...]} - MUTATE IN PLACE
        pair_decisions,          # dict: {neuron_id: {(feat_i, feat_j): 0/0.5/1}} - MUTATE
        renames,                 # dict: {neuron_id: {new_name: (old1, old2)}} - MUTATE
        cell_feat_stats,         # Pre-computed MI values: stats[neuron_id][feat] = {'me': MI}
        feat_feat_significance,  # Binary matrix: are features INTENSE-connected?
        feat_names,              # List of all feature names (for matrix indexing)
        **kwargs,                # User-provided extra arguments (thresholds, etc.)
    ):
        '''
        Process ALL neurons at once. Mutates in place for efficiency.

        - neuron_selectivities[nid]: List of features for neuron nid (can remove/add)
        - pair_decisions[nid]: {(feat_i, feat_j): 0/0.5/1} for explicit decisions
          - 0 = feat_i is primary (exclude feat_j)
          - 1 = feat_j is primary (exclude feat_i)
          - 0.5 = keep both (undistinguishable)
        - renames[nid]: {new_name: (old_feat1, old_feat2)} for merged features

        Returns nothing (mutates in place).
        '''
        pass

Key Advantages
--------------
- Filters share one copy of cell_feat_stats, feat_names, etc.
- No exp object passed to parallel workers
- Data-driven filters can be provided with pre-extracted neural data if needed
- Filters are composable: chain multiple filters with compose_filters()

Example Usage
-------------
>>> # Create a priority filter
>>> rules = [('headdirection', 'bodydirection'), ('freezing', 'rest')]
>>> priority_filter = build_priority_filter(rules)
>>>
>>> # Compose multiple filters
>>> combined = compose_filters(priority_filter, my_custom_filter)
>>>
>>> # Use with disentangle_all_selectivities
>>> from driada.intense.disentanglement import disentangle_all_selectivities
>>> results = disentangle_all_selectivities(
...     exp, feat_names,
...     pre_filter_func=combined,
...     filter_kwargs={'my_threshold': 0.5},
... )
"""


def build_priority_filter(priority_rules):
    """Build a population-level filter from declarative priority rules.

    When two features in a rule are both present in a neuron's selectivities,
    the first feature (primary) wins over the second (redundant).

    Parameters
    ----------
    priority_rules : list of tuples
        Each tuple is (primary_feat, redundant_feat).
        If both are present in a neuron's selectivities, primary wins.

    Returns
    -------
    callable
        Population-level filter function compatible with disentangle_all_selectivities.

    Examples
    --------
    >>> rules = [
    ...     ('headdirection', 'bodydirection'),  # headdirection > bodydirection
    ...     ('freezing', 'rest'),                # freezing > rest
    ...     ('locomotion', 'speed'),             # locomotion > speed
    ... ]
    >>> filter_func = build_priority_filter(rules)
    >>>
    >>> # Test the filter
    >>> neuron_sels = {0: ['headdirection', 'bodydirection', 'place'], 1: ['speed', 'place']}
    >>> decisions = {0: {}, 1: {}}
    >>> renames = {0: {}, 1: {}}
    >>> filter_func(neuron_sels, decisions, renames)
    >>> decisions[0]
    {('headdirection', 'bodydirection'): 0}
    >>> decisions[1]  # No decision for neuron 1 (no matching pair)
    {}
    """
    def priority_filter(neuron_selectivities, pair_decisions, renames, **kwargs):
        # Process ALL neurons
        for nid, sels in neuron_selectivities.items():
            for primary, redundant in priority_rules:
                if primary in sels and redundant in sels:
                    # Primary wins (0 = first feature is primary)
                    pair_decisions[nid][(primary, redundant)] = 0

    return priority_filter


def compose_filters(*filters):
    """Chain multiple population-level filters into one.

    Filters are applied in order. Each receives the mutated state
    from previous filters. Later filters can override earlier decisions.

    Parameters
    ----------
    *filters : callable
        Variable number of filter functions to chain together.

    Returns
    -------
    callable
        A single composed filter function that runs all filters in sequence.

    Examples
    --------
    >>> # Create individual filters
    >>> filter1 = build_priority_filter([('a', 'b')])
    >>> filter2 = build_priority_filter([('c', 'd')])
    >>>
    >>> # Compose them
    >>> combined = compose_filters(filter1, filter2)
    >>>
    >>> # Test
    >>> neuron_sels = {0: ['a', 'b', 'c', 'd']}
    >>> decisions = {0: {}}
    >>> renames = {0: {}}
    >>> combined(neuron_sels, decisions, renames)
    >>> sorted(decisions[0].items())
    [(('a', 'b'), 0), (('c', 'd'), 0)]

    >>> # Later filter can override earlier
    >>> def override_filter(neuron_sels, pair_decisions, renames, **kwargs):
    ...     for nid in pair_decisions:
    ...         pair_decisions[nid][('a', 'b')] = 1  # Override: b wins
    >>>
    >>> combined_with_override = compose_filters(filter1, override_filter)
    >>> neuron_sels = {0: ['a', 'b']}
    >>> decisions = {0: {}}
    >>> combined_with_override(neuron_sels, decisions, {0: {}})
    >>> decisions[0][('a', 'b')]
    1
    """
    def composed_filter(neuron_selectivities, pair_decisions, renames, **kwargs):
        for f in filters:
            f(neuron_selectivities, pair_decisions, renames, **kwargs)

    return composed_filter


def build_mi_ratio_filter(feat_pair, mi_ratio_threshold=1.5):
    """Build a filter that decides based on MI ratio between features.

    Compares MI(neuron, feat1) vs MI(neuron, feat2) for each neuron.
    If the ratio exceeds threshold, the feature with higher MI wins.

    Parameters
    ----------
    feat_pair : tuple of str
        Pair of features to compare: (feat1, feat2).
    mi_ratio_threshold : float, optional
        Ratio threshold. If MI(feat1)/MI(feat2) >= threshold, feat1 wins.
        If MI(feat2)/MI(feat1) >= threshold, feat2 wins.
        Otherwise, 0.5 (keep both). Default: 1.5.

    Returns
    -------
    callable
        Population-level filter function.

    Examples
    --------
    >>> filter_func = build_mi_ratio_filter(('place', '3d-place'), mi_ratio_threshold=1.5)
    """
    feat1, feat2 = feat_pair

    def mi_ratio_filter(neuron_selectivities, pair_decisions, renames,
                        cell_feat_stats=None, **kwargs):
        if cell_feat_stats is None:
            return

        for nid, sels in neuron_selectivities.items():
            if feat1 in sels and feat2 in sels:
                # Get MI values from pre-computed stats
                mi1 = cell_feat_stats.get(nid, {}).get(feat1, {}).get('me', 0)
                mi2 = cell_feat_stats.get(nid, {}).get(feat2, {}).get('me', 0)

                if mi2 > 0 and mi1 >= mi_ratio_threshold * mi2:
                    pair_decisions[nid][(feat1, feat2)] = 0  # feat1 wins
                elif mi1 > 0 and mi2 >= mi_ratio_threshold * mi1:
                    pair_decisions[nid][(feat1, feat2)] = 1  # feat2 wins
                # else: no decision (will use standard disentanglement or 0.5)

    return mi_ratio_filter


def build_exclusion_filter(exclusion_map):
    """Build a filter that removes specific features from selectivities.

    When a primary feature is present, removes associated redundant features.

    Parameters
    ----------
    exclusion_map : dict
        Maps primary features to lists of features to exclude.
        Example: {'objects': ['center'], 'object1': ['objects', 'center']}

    Returns
    -------
    callable
        Population-level filter function.

    Examples
    --------
    >>> exclusion_filter = build_exclusion_filter({
    ...     'object1': ['objects', 'center'],
    ...     'object2': ['objects', 'center'],
    ... })
    """
    def exclusion_filter(neuron_selectivities, pair_decisions, renames, **kwargs):
        for nid, sels in neuron_selectivities.items():
            for primary, to_exclude in exclusion_map.items():
                if primary in sels:
                    for feat in to_exclude:
                        if feat in sels:
                            pair_decisions[nid][(primary, feat)] = 0

    return exclusion_filter


def build_merge_filter(merge_rules, correspondence_func=None):
    """Build a filter that merges feature pairs into combined features.

    When both features are present, merges them into a new combined feature.
    Useful for place-corner type merging.

    Parameters
    ----------
    merge_rules : list of tuples
        Each tuple is (feat1, feat2, combined_name).
        If both feat1 and feat2 are present, they're merged.
    correspondence_func : callable or None, optional
        Optional function to check if merging should happen.
        Signature: correspondence_func(nid, feat1, feat2, **kwargs) -> bool
        If None, always merges when both features present.

    Returns
    -------
    callable
        Population-level filter function.

    Examples
    --------
    >>> merge_filter = build_merge_filter([
    ...     ('place', 'corner', 'place-corner'),
    ...     ('place', 'wall', 'place-wall'),
    ... ])
    """
    def merge_filter(neuron_selectivities, pair_decisions, renames, **kwargs):
        for nid, sels in neuron_selectivities.items():
            for feat1, feat2, combined_name in merge_rules:
                if feat1 in sels and feat2 in sels:
                    # Check correspondence if function provided
                    should_merge = True
                    if correspondence_func is not None:
                        should_merge = correspondence_func(nid, feat1, feat2, **kwargs)

                    if should_merge:
                        # Remove original features and add combined
                        sels.remove(feat1)
                        sels.remove(feat2)
                        sels.append(combined_name)
                        renames[nid][combined_name] = (feat1, feat2)
                        break  # Only one merge per neuron per filter pass

    return merge_filter


# =============================================================================
# Example experiment-specific filters (for reference)
# =============================================================================

def example_nof_filter(neuron_selectivities, pair_decisions, renames, **kwargs):
    """Example NOF experiment filter: specific objects > general 'objects' > 'center'.

    This is an example of how to write experiment-specific filters.
    Users should copy and modify this pattern for their experiments.
    """
    specific_objs = {'object1', 'object2', 'object3', 'object4'}

    for nid, sels in neuron_selectivities.items():
        has_specific = bool(specific_objs.intersection(sels))

        if has_specific and 'objects' in sels:
            for obj in specific_objs:
                if obj in sels:
                    pair_decisions[nid][(obj, 'objects')] = 0

        if (has_specific or 'objects' in sels) and 'center' in sels:
            if 'objects' in sels:
                pair_decisions[nid][('objects', 'center')] = 0
            for obj in specific_objs:
                if obj in sels:
                    pair_decisions[nid][(obj, 'center')] = 0


def example_data_driven_filter(neuron_selectivities, pair_decisions, renames,
                                calcium_data=None,
                                feature_data=None,
                                discrete_place_features=None,
                                top_activity_percent=2,
                                correspondence_threshold=0.4,
                                **kwargs):
    """Example data-driven filter for place-discrete correspondence.

    This filter checks if high neural activity corresponds to specific
    discrete feature values, and merges features if correspondence is high.

    Parameters (via filter_kwargs)
    ------------------------------
    calcium_data : dict
        Pre-extracted calcium data: {neuron_id: array}
    feature_data : dict
        Pre-extracted feature data: {feature_name: array}
    discrete_place_features : list
        List of discrete features to check against place
    top_activity_percent : float
        Percentile for high activity (default 2)
    correspondence_threshold : float
        Minimum correspondence to merge (default 0.4)
    """
    import numpy as np

    if discrete_place_features is None or calcium_data is None or feature_data is None:
        return

    def get_high_activity_indices(data, percent):
        threshold = np.percentile(data, 100 - percent)
        return np.where(data >= threshold)[0]

    for nid, sels in neuron_selectivities.items():
        if 'place' not in sels:
            continue

        if nid not in calcium_data:
            continue

        for discr_feat in discrete_place_features:
            if discr_feat not in sels or discr_feat not in feature_data:
                continue

            # Check correspondence using pre-extracted data
            neur_data = calcium_data[nid]
            high_indices = get_high_activity_indices(neur_data, top_activity_percent)
            feat_data = feature_data[discr_feat]
            correspondence = np.mean(feat_data[high_indices])

            if correspondence > correspondence_threshold:
                # Merge into combined feature
                combined_name = f'place-{discr_feat}'
                sels.remove('place')
                sels.remove(discr_feat)
                sels.append(combined_name)
                renames[nid][combined_name] = ('place', discr_feat)
                break  # Only one merge per neuron
