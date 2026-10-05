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
>>> rules = [('bodydirection', 'headdirection'), ('freezing', 'rest')]
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
from scipy.ndimage import gaussian_filter, label, maximum_filter
from scipy.special import digamma
from scipy.stats import norm, rankdata


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
    ...     ('bodydirection', 'headdirection'),  # bodydirection > headdirection
    ...     ('freezing', 'rest'),                # freezing > rest
    ...     ('locomotion', 'speed'),             # locomotion > speed
    ... ]
    >>> filter_func = build_priority_filter(rules)
    >>>
    >>> # Test the filter
    >>> neuron_sels = {0: ['bodydirection', 'headdirection', 'place'], 1: ['speed', 'place']}
    >>> decisions = {0: {}, 1: {}}
    >>> renames = {0: {}, 1: {}}
    >>> filter_func(neuron_sels, decisions, renames)
    >>> decisions[0]
    {('bodydirection', 'headdirection'): 0}
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


# Filter state that a filter may mutate; translated back after the call.
_MUTABLE_FILTER_ARGS = ('neuron_selectivities', 'pair_decisions', 'renames',
                        'per_neuron_disent')
# Read-only filter inputs that are keyed by feature names.
_NAMED_FILTER_ARGS = ('cell_feat_stats', 'feat_names')


def _rename_features(obj, mapping):
    """Recursively rename feature names in dict keys, tuples, lists and strings."""
    if isinstance(obj, str):
        return mapping.get(obj, obj)
    if isinstance(obj, tuple):
        return tuple(_rename_features(o, mapping) for o in obj)
    if isinstance(obj, list):
        return [_rename_features(o, mapping) for o in obj]
    if isinstance(obj, dict):
        return {_rename_features(k, mapping): _rename_features(v, mapping)
                for k, v in obj.items()}
    return obj


def with_source_feature_names(filter_func, exp):
    """Run a filter on source feature names when type-based representations are used.

    With ``representation='by_type'`` INTENSE reports features under derived
    names (``place_quad``, ``speed_quad``, ``headdirection_harm2``, ...). Filters
    are written against the source names (``place``, ``speed``,
    ``headdirection``). The wrapper renames derived features to their sources
    before calling the filter and renames them back in the mutated state
    afterwards, so every rule works unchanged in both modes. Works for pre-
    and post-filters.

    Parameters
    ----------
    filter_func : callable
        Pre-filter or post-filter following the filter protocol of this module.
    exp : Experiment
        Experiment whose derived representations are looked up at call time.

    Returns
    -------
    callable
        Wrapped filter with the same calling convention.

    Examples
    --------
    >>> from types import SimpleNamespace
    >>> exp = SimpleNamespace(_representation_sources={'speed_quad': 'speed'})
    >>> rule = build_priority_filter([('locomotion', 'speed')])
    >>> wrapped = with_source_feature_names(rule, exp)
    >>> sels = {0: ['locomotion', 'speed_quad']}
    >>> decisions = {0: {}}
    >>> wrapped(sels, decisions, {0: {}})
    >>> decisions
    {0: {('locomotion', 'speed_quad'): 0}}
    """
    # Imported here: the package puts the local driada sources on sys.path in
    # analysis.py, which may be imported after this module.
    from driada.intense.representations import get_representation_sources

    def wrapped(*args, **kwargs):
        to_source = get_representation_sources(exp)
        if not to_source:
            return filter_func(*args, **kwargs)
        to_derived = {source: derived for derived, source in to_source.items()}

        new_args = [_rename_features(a, to_source) for a in args]
        new_kwargs = dict(kwargs)
        for key in _MUTABLE_FILTER_ARGS + _NAMED_FILTER_ARGS:
            if key in kwargs:
                new_kwargs[key] = _rename_features(kwargs[key], to_source)

        result = filter_func(*new_args, **new_kwargs)

        # Positional arguments are the mutable state (selectivities, decisions,
        # renames for pre-filters; per-neuron results for post-filters).
        for original, renamed in zip(args, new_args):
            if isinstance(original, dict):
                original.clear()
                original.update(_rename_features(renamed, to_derived))
        for key in _MUTABLE_FILTER_ARGS:
            if key in kwargs and isinstance(kwargs[key], dict):
                kwargs[key].clear()
                kwargs[key].update(_rename_features(new_kwargs[key], to_derived))
        return result

    return wrapped


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
    ('bodydirection', 'headdirection'),      # body direction > head direction
    ('bodydirection_2d', 'headdirection_2d'),  # body direction > head direction (_2d variant)
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
               cell_feat_stats=None, **kwargs):
    """3DM experiment filter: priority rules for 3D maze features."""
    for nid, sels in neuron_selectivities.items():
        # 3d-place > z (z is a component)
        if 'z' in sels and '3d-place' in sels:
            pair_decisions[nid][('3d-place', 'z')] = 0

        # start_box > 3d-place (discrete trumps continuous place)
        if '3d-place' in sels and 'start_box' in sels:
            pair_decisions[nid][('start_box', '3d-place')] = 0

        # speed > speed_z
        if 'speed' in sels and 'speed_z' in sels:
            pair_decisions[nid][('speed', 'speed_z')] = 0

        # 3d-place > all z_arm features
        for i in range(1, 14):
            z_arm = f'z_arm_{i}'
            if '3d-place' in sels and z_arm in sels:
                pair_decisions[nid][('3d-place', z_arm)] = 0


def tdm_post_filter(per_neuron_disent, cell_feat_stats=None, feat_names=None, **kwargs):
    """3DM post-filter: place > 3d-place when disentanglement is undistinguishable.

    When standard disentanglement returns 0.5 for (place, 3d-place), this
    post-filter changes the result to 0 (place wins). This is a principled
    tie-breaker: prefer the simpler 2D model when information theory can't
    distinguish between them.
    """
    for nid, neuron_info in per_neuron_disent.items():
        pairs = neuron_info.get('pairs', {})

        # Check both orderings of the pair
        for pair_key in [('place', '3d-place'), ('3d-place', 'place')]:
            if pair_key in pairs:
                info = pairs[pair_key]
                if info.get('result') == 0.5:
                    # Tie-break: place wins over 3d-place
                    if pair_key == ('place', '3d-place'):
                        info['result'] = 0  # place (first) wins
                    else:
                        info['result'] = 1  # place (second) wins
                    info['source'] = 'post_filter_tiebreak'
                break


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


# Position map used by zone_share_filter: bins per axis, Gaussian smoothing
# width in bins, number of equal-duration map levels.
_MAP_BINS = 20
_MAP_SIGMA = 1.5
_MAP_LEVELS = 8


def _position_bins(position):
    """Index of the map bin visited at every frame.

    Parameters
    ----------
    position : np.ndarray
        Coordinates of shape (2, n_frames).

    Returns
    -------
    np.ndarray
        Flat bin index in [0, _MAP_BINS ** 2) per frame.
    """
    idx = []
    for coord in position:
        lo, hi = coord.min(), coord.max()
        scaled = (coord - lo) / (hi - lo + 1e-9) * _MAP_BINS
        idx.append(np.clip(scaled.astype(int), 0, _MAP_BINS - 1))
    return idx[0] * _MAP_BINS + idx[1]


def _gaussian_entropy_bias(n):
    """Bias term of the Gaussian entropy of a 1D sample of size n."""
    n = np.maximum(np.asarray(n, float), 2)
    return (np.log(2) - np.log(n - 1)) / 2 + digamma((n - 1) / 2) / 2


def _class_conditional_mi(signals, labels, n_classes):
    """Gaussian class-conditional MI between each signal and its own labels.

    Same estimator as ``driada.information.gcmi.mi_model_gd`` for a 1D signal
    with ``biascorrect=True``, computed for all rows at once: the null
    distribution needs it for every circular shift.

    Parameters
    ----------
    signals : np.ndarray
        Copula-normalised signals of shape (n_rows, n_frames).
    labels : np.ndarray
        Integer class labels in [0, n_classes) of shape (n_rows, n_frames).
    n_classes : int
        Number of classes.

    Returns
    -------
    np.ndarray
        MI in bits for every row. Not clipped at zero.
    """
    n_rows, n_frames = signals.shape
    x = signals - signals.mean(1, keepdims=True)
    h_total = (0.5 * np.log(np.einsum('rn,rn->r', x, x) / (n_frames - 1))
               - _gaussian_entropy_bias(n_frames))

    idx = (labels + n_classes * np.arange(n_rows)[:, None]).ravel()
    size = n_rows * n_classes
    shape = (n_rows, n_classes)
    count = np.bincount(idx, minlength=size).reshape(shape).astype(float)
    s1 = np.bincount(idx, weights=x.ravel(), minlength=size).reshape(shape)
    s2 = np.bincount(idx, weights=(x * x).ravel(), minlength=size).reshape(shape)

    # A variance needs two samples; smaller classes get zero weight.
    usable = count >= 2
    var = np.where(usable, (s2 - s1 ** 2 / np.maximum(count, 1)) / np.maximum(count - 1, 1), 1.0)
    h_class = 0.5 * np.log(np.maximum(var, 1e-12)) - _gaussian_entropy_bias(count)
    weight = np.where(usable, count / n_frames, 0.0)
    return (h_total - (h_class * weight).sum(1)) / np.log(2)


def _map_levels(signals, bins):
    """Level of each signal's own activity map at the visited bin, per frame.

    For every row an occupancy-normalised, Gaussian-smoothed map of the signal
    over position bins is built and each frame gets the map value of its bin.
    Frames are then cut into ``_MAP_LEVELS`` groups of equal duration by that
    value, so the levels describe position as seen by this particular signal.

    Parameters
    ----------
    signals : np.ndarray
        Signals of shape (n_rows, n_frames).
    bins : np.ndarray
        Flat position bin per frame (see ``_position_bins``).

    Returns
    -------
    np.ndarray
        Integer levels in [0, _MAP_LEVELS) of shape (n_rows, n_frames).
    """
    n_rows, n_frames = signals.shape
    n_bins = _MAP_BINS * _MAP_BINS
    idx = (bins + n_bins * np.arange(n_rows)[:, None]).ravel()
    sums = np.bincount(idx, weights=signals.ravel(), minlength=n_rows * n_bins)
    occupancy = np.bincount(bins, minlength=n_bins).astype(float)
    num = gaussian_filter(sums.reshape(n_rows, _MAP_BINS, _MAP_BINS), (0, _MAP_SIGMA, _MAP_SIGMA))
    den = gaussian_filter(occupancy.reshape(_MAP_BINS, _MAP_BINS), _MAP_SIGMA)
    frame_values = (num / np.maximum(den, 1e-9)).reshape(n_rows, n_bins)[:, bins]

    # Stable order: frames of one bin share a value and are split by time.
    order = np.argsort(frame_values, axis=1, kind='stable')
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(n_frames)[None, :], axis=1)
    return ranks * _MAP_LEVELS // n_frames


def zone_information_share(calcium, zone, bins, shifts, delay=0):
    """Share of a neuron's position information that is explained by a zone.

    Position is described by the neuron's own activity map (``_map_levels``)
    refined by the zone, so the zone is a function of position and its
    information is a part of the position information. Both quantities are
    corrected by their mean over circular shifts of calcium against behaviour,
    with the map rebuilt for every shift: the estimator bias grows with the
    number of classes and a map fitted to the signal is informative by
    construction.

    Parameters
    ----------
    calcium : np.ndarray
        Calcium trace of the neuron, shape (n_frames,).
    zone : np.ndarray
        Boolean zone indicator, shape (n_frames,).
    bins : np.ndarray
        Flat position bin per frame (see ``_position_bins``).
    shifts : array-like of int
        Circular shifts (frames) forming the null distribution.
    delay : int, optional
        Frames by which calcium lags behaviour. Default: 0.

    Returns
    -------
    share : float
        Corrected zone information divided by corrected position information.
        ``inf`` when only the zone information is positive, ``nan`` when
        neither is.
    excess : float
        Mean calcium inside the zone minus mean calcium outside it.
    """
    calcium = np.roll(calcium, -int(delay))
    x = norm.ppf(rankdata(calcium) / (calcium.size + 1))
    signals = np.vstack([x] + [np.roll(x, s) for s in shifts])
    in_zone = zone.astype(int)

    i_pos = _class_conditional_mi(signals, _map_levels(signals, bins) * 2 + in_zone,
                                  2 * _MAP_LEVELS)
    i_zone = _class_conditional_mi(signals, np.broadcast_to(in_zone, signals.shape), 2)
    i_pos = i_pos[0] - i_pos[1:].mean()
    i_zone = i_zone[0] - i_zone[1:].mean()

    if i_pos > 0:
        share = i_zone / i_pos
    elif i_zone > 0:
        share = np.inf
    else:
        share = np.nan
    return share, calcium[zone].mean() - calcium[~zone].mean()


def zone_share_filter(neuron_selectivities, pair_decisions, renames,
                      calcium_data=None,
                      feature_data=None,
                      position_data=None,
                      discrete_place_features=None,
                      place_feat_name='place',
                      cell_feat_stats=None,
                      fps=20,
                      zone_share_threshold=0.5,
                      n_shifts=100,
                      min_shift_sec=20,
                      shift_seed=0,
                      **kwargs):
    """Spatial filter: place vs discrete zone by the zone's information share.

    For NOF/LNOF-like experiments. When a neuron has both place and a discrete
    spatial feature (corners, walls, objects, ...), the zone wins if it carries
    at least ``zone_share_threshold`` of the neuron's position information
    (see ``zone_information_share``) and the neuron is more active inside the
    zone than outside. Otherwise place wins. Features are never merged, and
    every zone of a neuron is decided independently.

    Respects pair_decisions from earlier filters: zones already marked as
    losers are left untouched.

    Parameters (via filter_kwargs)
    ------------------------------
    calcium_data : dict
        Pre-extracted calcium data: {neuron_id: np.array}
    feature_data : dict
        Pre-extracted feature data: {feature_name: np.array}
    position_data : np.ndarray
        Coordinates of the animal, shape (2, n_frames)
    discrete_place_features : list
        Discrete features to check against place
    place_feat_name : str
        Name of continuous place feature. Default: 'place'
    cell_feat_stats : dict
        INTENSE statistics; ``stats[neuron_id][zone]['opt_delay']`` is the
        calcium delay (frames) used for the neuron-zone pair, 0 if absent
    fps : float
        Sampling rate, frames per second. Default: 20
    zone_share_threshold : float
        Minimum information share for the zone to win. Default: 0.5
    n_shifts : int
        Number of circular shifts in the null distribution. Default: 100
    min_shift_sec : float
        Minimum circular shift, seconds. Default: 20
    shift_seed : int
        Seed of the shift generator; the same seed gives identical decisions.
        Default: 0
    """
    # No-op if discrete_place_features is empty or None
    if not discrete_place_features:
        return

    have_data = calcium_data is not None and feature_data is not None and position_data is not None
    if have_data:
        bins = _position_bins(np.asarray(position_data, float))
        # Shifts are drawn once so that a decision does not depend on which
        # other neurons are processed.
        min_shift = min(int(min_shift_sec * fps), (bins.size - 1) // 2)
        shifts = np.random.default_rng(shift_seed).integers(
            min_shift, bins.size - min_shift, n_shifts)

    for nid, sels in neuron_selectivities.items():
        discrete_in_sels = set(discrete_place_features).intersection(sels)
        if place_feat_name not in sels or not discrete_in_sels:
            continue

        if not have_data:
            # Fallback: just mark as undistinguishable (0.5) for non-loser features
            for discr_feat in discrete_in_sels:
                if not _feature_is_loser(discr_feat, sels, pair_decisions[nid]):
                    pair_decisions[nid][(place_feat_name, discr_feat)] = 0.5
            continue

        if nid not in calcium_data:
            continue

        for discr_feat in sorted(discrete_in_sels):  # sorted for deterministic order
            if _feature_is_loser(discr_feat, sels, pair_decisions[nid]):
                continue

            zone_wins = False
            if discr_feat in feature_data:
                try:
                    delay = cell_feat_stats[nid][discr_feat].get('opt_delay', 0)
                except (KeyError, TypeError):
                    delay = 0
                # Zone indicators can hold fractional values at zone borders
                # after resampling to the imaging frame rate.
                zone = np.asarray(feature_data[discr_feat]) > 0.5
                share, excess = zone_information_share(
                    calcium_data[nid], zone, bins, shifts, delay=delay)
                zone_wins = share >= zone_share_threshold and excess > 0

            # 1 = second feature of the pair (the zone) is primary
            pair_decisions[nid][(place_feat_name, discr_feat)] = 1 if zone_wins else 0


def place_field_map(calcium, bins):
    """Occupancy-normalised, Gaussian-smoothed activity map of a neuron.

    Parameters
    ----------
    calcium : np.ndarray
        Calcium trace of the neuron, shape (n_frames,).
    bins : np.ndarray
        Flat position bin per frame (see ``_position_bins``).

    Returns
    -------
    np.ndarray
        Map of shape (_MAP_BINS, _MAP_BINS); bins that were never visited
        are NaN.
    """
    shape = (_MAP_BINS, _MAP_BINS)
    occupancy = np.bincount(bins, minlength=_MAP_BINS ** 2).astype(float).reshape(shape)
    sums = np.bincount(bins, weights=calcium, minlength=_MAP_BINS ** 2).reshape(shape)
    rate = gaussian_filter(sums, _MAP_SIGMA) / np.maximum(gaussian_filter(occupancy, _MAP_SIGMA), 1e-9)
    return np.where(occupancy > 0, rate, np.nan)


def place_field_in_zone(calcium, zone, bins, delay=0, peak_fraction=0.5,
                        zone_fields_threshold=0.5, peak_tolerance_bins=1):
    """Decide whether the main place field of a neuron lies in a zone.

    Fields are connected regions of the activity map above ``peak_fraction``
    of its peak, the map minimum being the baseline; the main field is the
    one that holds the peak. The neuron is a zone cell when the peak lies in
    the zone and the fields lying in the zone outweigh the fields outside it.
    Fields in the zone are counted together because a zone can be a set of
    separate places (four corners) that one cell covers with several fields.

    Parameters
    ----------
    calcium : np.ndarray
        Calcium trace of the neuron, shape (n_frames,).
    zone : np.ndarray
        Boolean zone indicator, shape (n_frames,).
    bins : np.ndarray
        Flat position bin per frame (see ``_position_bins``).
    delay : int, optional
        Frames by which calcium lags behaviour. Default: 0.
    peak_fraction : float, optional
        Field boundary as a fraction of the peak above baseline. Default: 0.5
    zone_fields_threshold : float, optional
        Minimum share of the fields lying in the zone (their own peak is in
        the zone) in the activity of all fields. Default: 0.5
    peak_tolerance_bins : int, optional
        The peak counts as lying in the zone when a map bin within this
        distance of it belongs to the zone; a zone can be smaller than the
        map resolution needed to place a peak inside it. Default: 1

    Returns
    -------
    in_zone : bool
        True when both conditions hold.
    overlap : float
        Activity-weighted share of the main field that lies in the zone;
        used to choose between several zones of one neuron.
    """
    amap = place_field_map(np.roll(calcium, -int(delay)), bins)
    activity = amap - np.nanmin(amap)
    activity = np.where(np.isnan(activity), 0.0, activity)
    peak = activity.max()
    if peak <= 0:
        return False, 0.0

    fields = activity >= peak_fraction * peak
    labels, n_fields = label(fields, structure=np.ones((3, 3)))
    peak_bin = np.unravel_index(np.argmax(activity), activity.shape)
    main = labels == labels[peak_bin]

    # Share of the time spent in every map bin that was spent inside the zone
    occupancy = np.bincount(bins, minlength=_MAP_BINS ** 2)
    zone_share = (np.bincount(bins, weights=zone.astype(float), minlength=_MAP_BINS ** 2)
                  / np.maximum(occupancy, 1)).reshape(activity.shape)

    size = 2 * peak_tolerance_bins + 1
    near_zone = maximum_filter(zone_share, size=size) > 0.5
    in_zone_fields = 0.0
    for k in range(1, n_fields + 1):
        field = labels == k
        field_peak = np.unravel_index(np.argmax(np.where(field, activity, -1.0)), activity.shape)
        if near_zone[field_peak]:
            in_zone_fields += activity[field].sum()
    zone_fields_share = in_zone_fields / activity[fields].sum()
    overlap = float((activity * zone_share)[main].sum() / activity[main].sum())
    return bool(near_zone[peak_bin] and zone_fields_share > zone_fields_threshold), overlap


def place_field_filter(neuron_selectivities, pair_decisions, renames,
                       calcium_data=None,
                       feature_data=None,
                       position_data=None,
                       discrete_place_features=None,
                       place_feat_name='place',
                       cell_feat_stats=None,
                       feature_renaming=None,
                       peak_fraction=0.5,
                       zone_fields_threshold=0.5,
                       peak_tolerance_bins=1,
                       **kwargs):
    """Spatial filter: a neuron is a zone cell when its main place field lies in the zone.

    For NOF/LNOF-like experiments. When a neuron has both place and a discrete
    spatial feature (corners, walls, objects, ...), its activity map decides
    (see ``place_field_in_zone``). If the main field lies in the zone, place
    and the zone are merged into a combined feature (e.g. 'place-corners');
    otherwise place wins. Of several zones that hold the main field, the one
    covering the largest part of it is merged and the others lose to place.

    Respects pair_decisions from earlier filters: zones already marked as
    losers are not considered.

    Parameters (via filter_kwargs)
    ------------------------------
    calcium_data : dict
        Pre-extracted calcium data: {neuron_id: np.array}
    feature_data : dict
        Pre-extracted feature data: {feature_name: np.array}
    position_data : np.ndarray
        Coordinates of shape (2, n_frames)
    discrete_place_features : list
        Discrete features to check against place
    place_feat_name : str
        Name of continuous place feature. Default: 'place'
    cell_feat_stats : dict
        INTENSE statistics {neuron_id: {feature: {'opt_delay': ...}}}; the
        optimal delay of the (neuron, zone) pair aligns calcium to behaviour
    feature_renaming : dict, optional
        Rename discrete features in the merged name: {'corners': 'corner'}
    peak_fraction, zone_fields_threshold, peak_tolerance_bins
        See ``place_field_in_zone``.
    """
    # No-op if discrete_place_features is empty or None
    if not discrete_place_features:
        return

    if feature_renaming is None:
        feature_renaming = {}

    have_data = calcium_data is not None and feature_data is not None and position_data is not None
    if have_data:
        bins = _position_bins(np.asarray(position_data, float))

    for nid, sels in neuron_selectivities.items():
        discrete_in_sels = set(discrete_place_features).intersection(sels)
        if place_feat_name not in sels or not discrete_in_sels:
            continue

        if not have_data:
            # Fallback: just mark as undistinguishable (0.5) for non-loser features
            for discr_feat in discrete_in_sels:
                if not _feature_is_loser(discr_feat, sels, pair_decisions[nid]):
                    pair_decisions[nid][(place_feat_name, discr_feat)] = 0.5
            continue

        if nid not in calcium_data:
            continue

        candidates = []
        for discr_feat in sorted(discrete_in_sels):  # sorted for deterministic order
            if _feature_is_loser(discr_feat, sels, pair_decisions[nid]):
                continue
            if discr_feat not in feature_data:
                continue
            try:
                delay = cell_feat_stats[nid][discr_feat].get('opt_delay', 0) or 0
            except (KeyError, TypeError):
                delay = 0
            # Zone indicators can hold fractional values at zone borders
            # after resampling to the imaging frame rate.
            zone = np.asarray(feature_data[discr_feat]) > 0.5
            in_zone, overlap = place_field_in_zone(
                calcium_data[nid], zone, bins, delay=delay,
                peak_fraction=peak_fraction,
                zone_fields_threshold=zone_fields_threshold,
                peak_tolerance_bins=peak_tolerance_bins)
            if in_zone:
                candidates.append((overlap, discr_feat))

        if candidates:
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
            # Main field is elsewhere - place wins over all discrete features
            for discr_feat in discrete_in_sels:
                if not _feature_is_loser(discr_feat, sels, pair_decisions[nid]):
                    pair_decisions[nid][(place_feat_name, discr_feat)] = 0


def extract_filter_data(exp, discrete_place_features=None, place_feat_name='place'):
    """Extract calcium, feature and position data for the spatial filters.

    Parameters
    ----------
    exp : Experiment
        Experiment object with neurons and features
    discrete_place_features : list, optional
        Features to extract. Default: ['corners', 'walls', 'center']
    place_feat_name : str, optional
        Name of the aggregated place feature whose first two components are
        the coordinates. Default: 'place'

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

    filter_data = {
        'calcium_data': calcium_data,
        'feature_data': feature_data,
        'discrete_place_features': discrete_place_features,
        'fps': exp.fps,
    }
    if hasattr(exp, place_feat_name):
        filter_data['position_data'] = getattr(exp, place_feat_name).data[:2]
    return filter_data


# =============================================================================
# Experiment Configurations
# =============================================================================

# Declared feature types. Position coordinates are declared linear because
# auto-detection can take a coordinate for an angle, and place then gets no
# type-based representation.
_FEATURE_TYPES = {'bodydirection': 'circular', 'headdirection': 'circular',
                  'x': 'linear', 'y': 'linear'}

EXPERIMENT_CONFIGS = {
    'RT': {
        'place_feat_name': 'place',
        'discrete_place_features': [],
        'feature_renaming': {},
        'aggregate_features': {('x', 'y'): 'place'},
        'skip_for_intense': ['x', 'y'],
        'specific_filter': None,
        'feature_types': _FEATURE_TYPES,
    },
    'NOF': {
        'place_feat_name': 'place',
        'discrete_place_features': ['walls', 'corners', 'center', 'object1', 'object2', 'object3', 'object4', 'objects'],
        'feature_renaming': {},
        'aggregate_features': {('x', 'y'): 'place'},
        'skip_for_intense': ['x', 'y'],
        'specific_filter': nof_filter,
        'feature_types': _FEATURE_TYPES,
    },
    'LNOF': {
        'place_feat_name': 'place',
        'discrete_place_features': ['walls', 'corners', 'center', 'object1', 'object2', 'object3', 'object4', 'objects'],
        'feature_renaming': {},
        'aggregate_features': {('x', 'y'): 'place'},
        'skip_for_intense': ['x', 'y'],
        'specific_filter': nof_filter,
        'feature_types': _FEATURE_TYPES,
    },
    'FOF': {
        'place_feat_name': 'place',
        'discrete_place_features': ['walls', 'centertrue'],
        'feature_renaming': {},
        'aggregate_features': {('x', 'y'): 'place'},
        'skip_for_intense': ['x', 'y', 'centermiddle', 'center'],
        'specific_filter': None,
        'feature_types': _FEATURE_TYPES,
    },
    'BOF': {
        'place_feat_name': 'place',
        'discrete_place_features': ['bowlinside', 'objectinside', 'walls', 'corners', 'centermiddle', 'centertrue', 'center'],
        'feature_renaming': {},
        'aggregate_features': {('x', 'y'): 'place'},
        'skip_for_intense': ['x', 'y'],
        'specific_filter': None,
        'feature_types': _FEATURE_TYPES,
    },
    'BOWL': {
        'place_feat_name': 'place',
        'discrete_place_features': ['bowl_inside', 'walls', 'corners', 'center'],
        'feature_renaming': {},
        'aggregate_features': {('x', 'y'): 'place'},
        'skip_for_intense': ['x', 'y', 'bowl_interaction_any', 'object1_interaction_any', 'object2_interaction_any'],
        'specific_filter': None,
        'feature_types': _FEATURE_TYPES,
    },
    'MSS': {
        'place_feat_name': 'place',
        'discrete_place_features': [],
        'feature_renaming': {},
        'aggregate_features': {('x', 'y'): 'place'},
        'skip_for_intense': ['x', 'y'],
        'specific_filter': None,
        'feature_types': _FEATURE_TYPES,
    },
    'HOS': {
        'place_feat_name': 'place',
        'discrete_place_features': [],
        'feature_renaming': {},
        'aggregate_features': {('x', 'y'): 'place'},
        'skip_for_intense': ['x', 'y'],
        'specific_filter': None,
        'feature_types': _FEATURE_TYPES,
    },
    '3DM': {
        'place_feat_name': '3d-place',
        'discrete_place_features': [],  # Disabled - no spatial pre-filter merging
        'feature_renaming': {},
        'aggregate_features': {('x', 'y'): 'place', ('x', 'y', 'z'): '3d-place'},
        'skip_for_intense': ['x', 'y'],  # Keep z for 3d-place > z rule
        'specific_filter': tdm_filter,
        'post_filter': tdm_post_filter,
        'feature_types': _FEATURE_TYPES,
    },
    'TRACE': {
        'place_feat_name': 'place',
        'discrete_place_features': [],
        'feature_renaming': {},
        'aggregate_features': {},
        'skip_for_intense': [],
        'specific_filter': None,
        'feature_types': _FEATURE_TYPES,
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


# Rules deciding between place and a discrete zone, by --zone-rule value
ZONE_RULES = {
    'place_field': place_field_filter,
    'information_share': zone_share_filter,
    'top_activity': spatial_filter,
}
DEFAULT_ZONE_RULE = 'place_field'


def get_filter_for_experiment(exp_type, zone_rule=DEFAULT_ZONE_RULE):
    """Get composed filter for experiment type.

    All experiments use:
    1. general_filter (behavioral priorities)
    2. specific_filter from config (if not None)
    3. place-vs-zone filter chosen by ``zone_rule`` (with experiment config)

    Parameters
    ----------
    exp_type : str
        Experiment type identifier (e.g., 'NOF', 'LNOF', '3DM', 'BOF')
    zone_rule : {'information_share', 'top_activity'}, optional
        Rule for neurons selective to both place and a discrete zone.
        'information_share' (default): ``zone_share_filter``, the zone wins
        when it carries most of the neuron's position information.
        'top_activity': ``spatial_filter``, place and zone are merged when the
        strongest activity falls into the zone.

    Returns
    -------
    callable
        Composed filter function

    Raises
    ------
    ValueError
        If exp_type is not in EXPERIMENT_CONFIGS or zone_rule is unknown
    """
    if exp_type not in EXPERIMENT_CONFIGS:
        raise ValueError(
            f"Unknown experiment type: {exp_type}. "
            f"Known types: {list(EXPERIMENT_CONFIGS.keys())}"
        )
    if zone_rule not in ZONE_RULES:
        raise ValueError(
            f"Unknown zone rule: {zone_rule}. Known rules: {list(ZONE_RULES.keys())}"
        )

    config = EXPERIMENT_CONFIGS[exp_type]

    # Always start with general priority rules
    general_filter = build_priority_filter(GENERAL_PRIORITY_RULES)
    filters = [general_filter]

    # Add experiment-specific filter from config
    if config['specific_filter'] is not None:
        filters.append(config['specific_filter'])

    # Always add the place-vs-zone filter (no-op if discrete_place_features is empty)
    filters.append(ZONE_RULES[zone_rule])

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
