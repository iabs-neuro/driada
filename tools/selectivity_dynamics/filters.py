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

import numpy as np


# =============================================================================
# Filter Building Blocks
# =============================================================================

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


# =============================================================================
# General Priority Rules
# =============================================================================

# General priority rules: first feature wins over second when both present
GENERAL_PRIORITY_RULES = [
    ('headdirection', 'bodydirection'),  # head direction > body direction
    ('freezing', 'rest'),                # freezing > rest
    ('locomotion', 'speed'),             # locomotion > speed
    ('rest', 'speed'),                   # rest > speed
    ('freezing', 'speed'),               # freezing > speed
    ('walk', 'speed'),                   # walk > speed
]


# =============================================================================
# Experiment-Specific Filters
# =============================================================================

def nof_filter(neuron_selectivities, pair_decisions, renames, **kwargs):
    """NOF experiment filter: specific objects beat general categories."""
    specific_objs = {'object1', 'object2', 'object3', 'object4'}

    for nid, sels in neuron_selectivities.items():
        has_specific = bool(specific_objs.intersection(sels))

        # Specific object > 'objects'
        if has_specific and 'objects' in sels:
            for obj in specific_objs:
                if obj in sels:
                    pair_decisions[nid][(obj, 'objects')] = 0

        # Any object feature > 'center'
        if (has_specific or 'objects' in sels) and 'center' in sels:
            if 'objects' in sels:
                pair_decisions[nid][('objects', 'center')] = 0
            for obj in specific_objs:
                if obj in sels:
                    pair_decisions[nid][(obj, 'center')] = 0


def tdm_filter(neuron_selectivities, pair_decisions, renames,
               cell_feat_stats=None, mi_ratio_threshold=1.5, **kwargs):
    """3DM experiment filter: 3d-place vs place based on MI ratio."""
    if cell_feat_stats is None:
        return

    for nid, sels in neuron_selectivities.items():
        if 'place' in sels and '3d-place' in sels:
            mi_2d = cell_feat_stats.get(nid, {}).get('place', {}).get('me', 0)
            mi_3d = cell_feat_stats.get(nid, {}).get('3d-place', {}).get('me', 0)

            if mi_3d >= mi_ratio_threshold * mi_2d:
                pair_decisions[nid][('3d-place', 'place')] = 0
            else:
                pair_decisions[nid][('place', '3d-place')] = 0

        # 3d-place > z (z is a component)
        if 'z' in sels and '3d-place' in sels:
            pair_decisions[nid][('3d-place', 'z')] = 0

        # start_box > 3d-place (discrete trumps continuous place)
        if '3d-place' in sels and 'start_box' in sels:
            pair_decisions[nid][('start_box', '3d-place')] = 0

        # speed > speed_z
        if 'speed' in sels and 'speed_z' in sels:
            pair_decisions[nid][('speed', 'speed_z')] = 0


def _feature_is_loser(feat, sels, pair_decisions):
    """Check if feature is marked as loser against any other feature in sels.

    A feature is a loser if:
    - (other, feat) = 0: other wins over feat
    - (feat, other) = 1: other wins over feat
    """
    for other in sels:
        if other == feat:
            continue
        # Check if (other, feat) = 0 → feat loses to other
        if pair_decisions.get((other, feat)) == 0:
            return True
        # Check if (feat, other) = 1 → feat loses to other
        if pair_decisions.get((feat, other)) == 1:
            return True
    return False


def spatial_filter(neuron_selectivities, pair_decisions, renames,
                   calcium_data=None,
                   feature_data=None,
                   discrete_place_features=None,
                   place_feat_name='place',
                   top_activity_percent=2,
                   correspondence_threshold=0.4,
                   feature_renaming=None,
                   **kwargs):
    """Spatial filter: merge place with discrete zones based on activity correspondence.

    For NOF/LNOF experiments. When a neuron has both place (xy) and a discrete
    spatial feature (corners, walls, center), checks if high neural activity
    corresponds to the discrete feature. If correspondence > threshold, merges
    them into a combined feature (e.g., 'xy-corners').

    Respects pair_decisions from earlier filters: features already marked as
    losers will not be considered for merging.

    Parameters (via filter_kwargs)
    ------------------------------
    calcium_data : dict
        Pre-extracted calcium data: {neuron_id: np.array}
    feature_data : dict
        Pre-extracted feature data: {feature_name: np.array}
    discrete_place_features : list
        Discrete features to check against place. Default: ['corners', 'walls', 'center']
    place_feat_name : str
        Name of continuous place feature. Default: 'xy'
    top_activity_percent : float
        Percentile for high activity detection. Default: 2
    correspondence_threshold : float
        Minimum correspondence to merge features. Default: 0.4
    feature_renaming : dict, optional
        Rename discrete features in merged name: {'corners': 'corner'}
        Results in 'xy-corner' instead of 'xy-corners'
    """
    # No-op if discrete_place_features is empty or None
    if not discrete_place_features:
        return

    if feature_renaming is None:
        feature_renaming = {}

    def get_high_activity_indices(data, percent):
        threshold = np.percentile(data, 100 - percent)
        return np.where(data >= threshold)[0]

    for nid, sels in neuron_selectivities.items():
        # Check if neuron has place and any discrete place feature
        has_place = place_feat_name in sels
        discrete_in_sels = set(discrete_place_features).intersection(sels)

        if not has_place or not discrete_in_sels:
            continue

        # Need calcium and feature data for correspondence check
        if calcium_data is None or feature_data is None:
            # Fallback: just mark as undistinguishable (0.5) for non-loser features
            for discr_feat in discrete_in_sels:
                if not _feature_is_loser(discr_feat, sels, pair_decisions[nid]):
                    pair_decisions[nid][(place_feat_name, discr_feat)] = 0.5
            continue

        if nid not in calcium_data:
            continue

        # Get high activity indices for this neuron
        neur_data = calcium_data[nid]
        high_indices = get_high_activity_indices(neur_data, top_activity_percent)

        # Collect all candidates that pass correspondence threshold
        # Skip features already marked as losers by earlier filters
        candidates = []
        for discr_feat in sorted(discrete_in_sels):  # sorted for deterministic order
            # Skip if this feature is already marked to lose
            if _feature_is_loser(discr_feat, sels, pair_decisions[nid]):
                continue

            if discr_feat not in feature_data:
                continue

            # Calculate correspondence: fraction of high-activity timepoints
            # where the discrete feature is active
            feat_data = feature_data[discr_feat]
            correspondence = np.mean(feat_data[high_indices])

            if correspondence > correspondence_threshold:
                candidates.append((correspondence, discr_feat))

        if candidates:
            # Merge with highest correspondence feature
            _, best_feat = max(candidates)
            renamed = feature_renaming.get(best_feat, best_feat)
            combined_name = f'{place_feat_name}-{renamed}'

            # Remove both original features, add combined
            sels.remove(place_feat_name)
            sels.remove(best_feat)
            sels.append(combined_name)
            renames[nid][combined_name] = (place_feat_name, best_feat)

            # Mark other discrete features as losing to place
            for discr_feat in discrete_in_sels:
                if discr_feat != best_feat and discr_feat in sels:
                    pair_decisions[nid][(place_feat_name, discr_feat)] = 0
        else:
            # No merge candidates - place wins over all discrete features
            for discr_feat in discrete_in_sels:
                if not _feature_is_loser(discr_feat, sels, pair_decisions[nid]):
                    pair_decisions[nid][(place_feat_name, discr_feat)] = 0


def extract_filter_data(exp, discrete_place_features=None):
    """Extract calcium and feature data for spatial_filter.

    Parameters
    ----------
    exp : Experiment
        Experiment object with neurons and features
    discrete_place_features : list, optional
        Features to extract. Default: ['corners', 'walls', 'center']

    Returns
    -------
    dict
        filter_kwargs dict ready to pass to disentanglement
    """
    if discrete_place_features is None:
        discrete_place_features = ['corners', 'walls', 'center']

    # Extract calcium data for all neurons
    calcium_data = {}
    for nid, neuron in enumerate(exp.neurons):
        calcium_data[nid] = neuron.ca.data

    # Extract feature data
    feature_data = {}
    for feat_name in discrete_place_features:
        if hasattr(exp, feat_name):
            feat = getattr(exp, feat_name)
            feature_data[feat_name] = feat.data

    return {
        'calcium_data': calcium_data,
        'feature_data': feature_data,
        'discrete_place_features': discrete_place_features,
    }


# =============================================================================
# Experiment Configurations
# =============================================================================

EXPERIMENT_CONFIGS = {
    'RT': {
        'place_feat_name': 'place',
        'discrete_place_features': [],
        'feature_renaming': {},
        'aggregate_features': {('x', 'y'): 'place'},
        'skip_for_intense': ['x', 'y'],
        'specific_filter': None,
    },
    'NOF': {
        'place_feat_name': 'place',
        'discrete_place_features': ['walls', 'corners', 'center', 'object1', 'object2', 'object3', 'object4', 'objects'],
        'feature_renaming': {'object1': 'object', 'object2': 'object', 'object3': 'object', 'object4': 'object'},
        'aggregate_features': {('x', 'y'): 'place'},
        'skip_for_intense': ['x', 'y'],
        'specific_filter': nof_filter,
    },
    'LNOF': {
        'place_feat_name': 'place',
        'discrete_place_features': ['walls', 'corners', 'center', 'object1', 'object2', 'object3', 'object4', 'objects'],
        'feature_renaming': {'object1': 'object', 'object2': 'object', 'object3': 'object', 'object4': 'object'},
        'aggregate_features': {('x', 'y'): 'place'},
        'skip_for_intense': ['x', 'y'],
        'specific_filter': nof_filter,
    },
    'FOF': {
        'place_feat_name': 'place',
        'discrete_place_features': [],
        'feature_renaming': {},
        'aggregate_features': {('x', 'y'): 'place'},
        'skip_for_intense': ['x', 'y'],
        'specific_filter': None,
    },
    'BOF': {
        'place_feat_name': 'place',
        'discrete_place_features': ['bowlinside', 'objectinside', 'walls', 'corners', 'centermiddle', 'centertrue', 'center'],
        'feature_renaming': {},
        'aggregate_features': {('x', 'y'): 'place'},
        'skip_for_intense': ['x', 'y'],
        'specific_filter': None,
    },
    'BOWL': {
        'place_feat_name': 'place',
        'discrete_place_features': ['bowlinside', 'objectinside', 'walls', 'corners', 'centermiddle', 'centertrue', 'center'],
        'feature_renaming': {},
        'aggregate_features': {('x', 'y'): 'place'},
        'skip_for_intense': ['x', 'y', 'bowl_interaction_any', 'object1_interaction_any', 'object2_interaction_any'],
        'specific_filter': None,
    },
    'MSS': {
        'place_feat_name': 'place',
        'discrete_place_features': [],
        'feature_renaming': {},
        'aggregate_features': {('x', 'y'): 'place'},
        'skip_for_intense': ['x', 'y'],
        'specific_filter': None,
    },
    'HOS': {
        'place_feat_name': 'place',
        'discrete_place_features': [],
        'feature_renaming': {},
        'aggregate_features': {('x', 'y'): 'place'},
        'skip_for_intense': ['x', 'y'],
        'specific_filter': None,
    },
    '3DM': {
        'place_feat_name': '3d-place',
        'discrete_place_features': [f'z_arm_{i}' for i in range(1, 14)] + ['start_box'],
        'feature_renaming': {},
        'aggregate_features': {('x', 'y'): 'place', ('x', 'y', 'z'): '3d-place'},
        'skip_for_intense': ['x', 'y', 'z'],
        'specific_filter': tdm_filter,
    },
}


def get_experiment_config(exp_type):
    """Get config for experiment type.

    Parameters
    ----------
    exp_type : str
        Experiment type identifier (e.g., 'NOF', 'LNOF', '3DM', 'BOF')

    Returns
    -------
    dict
        Configuration dict with keys: place_feat_name, discrete_place_features,
        feature_renaming, aggregate_features, skip_for_intense, specific_filter

    Raises
    ------
    ValueError
        If exp_type is not in EXPERIMENT_CONFIGS
    """
    if exp_type not in EXPERIMENT_CONFIGS:
        raise ValueError(
            f"Unknown experiment type: {exp_type}. "
            f"Known types: {list(EXPERIMENT_CONFIGS.keys())}"
        )
    return EXPERIMENT_CONFIGS[exp_type].copy()


def get_filter_for_experiment(exp_type):
    """Get composed filter for experiment type.

    All experiments use:
    1. general_filter (behavioral priorities)
    2. specific_filter from config (if not None)
    3. spatial_filter (with experiment config)

    Parameters
    ----------
    exp_type : str
        Experiment type identifier (e.g., 'NOF', 'LNOF', '3DM', 'BOF')

    Returns
    -------
    callable
        Composed filter function

    Raises
    ------
    ValueError
        If exp_type is not in EXPERIMENT_CONFIGS
    """
    if exp_type not in EXPERIMENT_CONFIGS:
        raise ValueError(
            f"Unknown experiment type: {exp_type}. "
            f"Known types: {list(EXPERIMENT_CONFIGS.keys())}"
        )

    config = EXPERIMENT_CONFIGS[exp_type]

    # Always start with general priority rules
    general_filter = build_priority_filter(GENERAL_PRIORITY_RULES)
    filters = [general_filter]

    # Add experiment-specific filter from config
    if config['specific_filter'] is not None:
        filters.append(config['specific_filter'])

    # Always add spatial filter (no-op if discrete_place_features is empty)
    filters.append(spatial_filter)

    return compose_filters(*filters)


# =============================================================================
# Example Filters (for reference)
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
