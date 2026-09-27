"""
Type-based feature representations for INTENSE.

GCMI only detects dependence that is monotone in each component of the feature
it is given. Mutual information itself is invariant to an invertible change of
variables, so the blind spot is closed by changing the representation of the
feature rather than the estimator. The representation is chosen from the
feature type alone (known before the analysis, never from neural responses):

- continuous linear 1D ``x`` -> ``[x, (x - c)^2]``: monotone tuning plus a
  single peak anywhere in the range and symmetric two-peak tuning;
- circular ``theta`` -> ``[cos theta, sin theta, cos 2theta, sin 2theta]``:
  one preferred direction plus axis (pi-periodic) tuning;
- two or three continuous linear components (e.g. place ``(x, y)`` or
  ``(x, y, z)``) -> the components, their centred squares and all pairwise
  products of the centred components (5 or 9 dimensions): any single field of
  any position, width and orientation;
- discrete features are left as they are (the discrete estimator is already
  shape-agnostic).

The centre ``c`` is the median, or the mean when the median coincides with the
minimum or maximum (e.g. a speed that is zero most of the time). The centre
must lie strictly inside the range: otherwise the square is a monotone function
of the variable, has the same ranks and adds nothing after the copula
transform, and the covariance of the representation becomes singular.

The significance machinery (surrogates, null distribution, two stages,
disentanglement) is unchanged; only the feature passed to it differs.
"""

# FUTURE: an omnibus mode could test the by-type representation together with a
# shape-agnostic one (equal-occupancy bins for linear features, angular sectors
# for circular ones, an RBF grid for place) and report p = min(1, 2 * min p).
# That would remove the remaining shape assumptions (e.g. periodic multi-peak
# tuning) at the cost of a doubled computation and a Bonferroni factor of two.

import warnings

import numpy as np

from ..information.circular_transform import (
    detect_circular_period,
    normalize_to_radians,
)
from ..information.info_base import MultiTimeSeries, TimeSeries

REPRESENTATION_MODES = ("raw", "by_type")
QUAD_SUFFIX = "_quad"
HARMONICS_SUFFIX = "_harm2"

# Name of the Experiment attribute mapping each derived feature to its source.
_REGISTRY_ATTR = "_representation_sources"


def _component(data, shuffle_mask, name):
    """Continuous TimeSeries component that keeps the source shuffle mask."""
    return TimeSeries(
        np.asarray(data, dtype=float),
        discrete=False,
        shuffle_mask=shuffle_mask,
        name=name,
    )


def _assemble(comps, name):
    """MultiTimeSeries of representation components.

    A time point where every component is zero is valid here (e.g. a variable
    that sits exactly at the centre of its square term, or the origin of a
    coordinate system), so the zero-column check meant for raw data is off.
    """
    return MultiTimeSeries(comps, name=name, allow_zero_columns=True)


def _square_centre(x):
    """Centre for the square term: median, or mean if the median is at an edge."""
    centre = np.median(x)
    if centre <= x.min() or centre >= x.max():
        centre = np.mean(x)
    return centre


def build_quadratic_1d(ts, name=None):
    """Quadratic representation ``[x, (x - c)^2]`` of a linear feature.

    Parameters
    ----------
    ts : TimeSeries
        Continuous, non-circular feature with at least three distinct values.
    name : str, optional
        Name of the resulting MultiTimeSeries.

    Returns
    -------
    MultiTimeSeries
        Two-dimensional representation. ``c`` is the median of ``x``, or its
        mean when the median equals the minimum or maximum.

    Examples
    --------
    >>> import numpy as np
    >>> from driada.information.info_base import TimeSeries
    >>> x = TimeSeries(np.random.default_rng(0).random(500), discrete=False)
    >>> build_quadratic_1d(x, name='x_quad').data.shape
    (2, 500)
    """
    x = np.asarray(ts.data, dtype=float)
    xc = x - _square_centre(x)
    mask = ts.shuffle_mask
    return _assemble([_component(x, mask, "lin"), _component(xc**2, mask, "sq")], name)


def build_harmonics(ts, n_harmonics=2, name=None):
    """Circular harmonics ``[cos k theta, sin k theta]`` for ``k = 1..n_harmonics``.

    Parameters
    ----------
    ts : TimeSeries
        Circular feature. Its period is taken from the detected type, or
        estimated from the data range if unknown.
    n_harmonics : int, default=2
        Highest harmonic order.
    name : str, optional
        Name of the resulting MultiTimeSeries.

    Returns
    -------
    MultiTimeSeries
        Representation with ``2 * n_harmonics`` dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> from driada.information.info_base import TimeSeries
    >>> th = TimeSeries(np.random.default_rng(0).uniform(0, 2 * np.pi, 500),
    ...                 ts_type='circular')
    >>> build_harmonics(th, name='th_harm2').data.shape
    (4, 500)
    """
    period = None
    if ts.type_info is not None and ts.type_info.is_circular:
        period = ts.type_info.circular_period
    if period is None:
        period = detect_circular_period(ts.data)
    theta = normalize_to_radians(np.asarray(ts.data, dtype=float), period)
    mask = ts.shuffle_mask
    comps = []
    for k in range(1, n_harmonics + 1):
        comps.append(_component(np.cos(k * theta), mask, f"cos{k}"))
        comps.append(_component(np.sin(k * theta), mask, f"sin{k}"))
    return _assemble(comps, name)


def build_quadratic_multi(mts, name=None):
    """Full quadratic representation of a two- or three-dimensional linear feature.

    Parameters
    ----------
    mts : MultiTimeSeries
        Continuous feature with two or three components (e.g. place ``(x, y)``
        or ``(x, y, z)``), each with at least three distinct values.
    name : str, optional
        Name of the resulting MultiTimeSeries.

    Returns
    -------
    MultiTimeSeries
        The components, their centred squares and the pairwise products of the
        centred components: ``[u, v, uc^2, vc^2, uc*vc]`` for two components
        (5 dimensions), and the analogous 9 dimensions for three. Centres are
        chosen as in :func:`build_quadratic_1d`.

    Raises
    ------
    ValueError
        If the feature does not have two or three components.

    Examples
    --------
    >>> import numpy as np
    >>> from driada.information.info_base import MultiTimeSeries
    >>> xy = MultiTimeSeries(np.random.default_rng(0).random((2, 500)), discrete=False)
    >>> build_quadratic_multi(xy, name='place_quad').data.shape
    (5, 500)
    >>> xyz = MultiTimeSeries(np.random.default_rng(1).random((3, 500)), discrete=False)
    >>> build_quadratic_multi(xyz, name='place3d_quad').data.shape
    (9, 500)
    """
    rows = [np.asarray(row, dtype=float) for row in mts.data]
    d = len(rows)
    if d not in (2, 3):
        raise ValueError(f"Expected 2 or 3 components, got {d}")
    centred = [row - _square_centre(row) for row in rows]
    mask = mts.shuffle_mask
    comps = [_component(row, mask, f"lin{i}") for i, row in enumerate(rows)]
    comps += [_component(c**2, mask, f"sq{i}") for i, c in enumerate(centred)]
    comps += [
        _component(centred[i] * centred[j], mask, f"prod{i}{j}")
        for i in range(d)
        for j in range(i + 1, d)
    ]
    return _assemble(comps, name)


def _is_circular(ts):
    return (
        isinstance(ts, TimeSeries)
        and not ts.discrete
        and ts.type_info is not None
        and ts.type_info.is_circular
    )


def _has_three_values(row):
    # With two distinct values the square is a relabelling of the variable
    # itself, so the representation would be singular.
    return np.unique(row).size > 2


def _is_linear_1d(ts):
    return (
        isinstance(ts, TimeSeries)
        and not ts.discrete
        and not _is_circular(ts)
        and _has_three_values(np.asarray(ts.data))
    )


def _is_linear_multi(mts):
    if not isinstance(mts, MultiTimeSeries) or mts.discrete:
        return False
    if mts.data.shape[0] not in (2, 3):
        return False
    components = getattr(mts, "ts_list", None) or []
    if any(_is_circular(ts) for ts in components):
        return False
    return all(_has_three_values(np.asarray(row)) for row in mts.data)


def get_representation_sources(exp):
    """Derived representation features registered in an experiment.

    Parameters
    ----------
    exp : Experiment
        Experiment object.

    Returns
    -------
    dict
        Mapping ``{derived feature name: source feature name}``; empty if
        :func:`substitute_by_type` has not been used on this experiment.
    """
    return getattr(exp, _REGISTRY_ATTR, {})


def restore_source_features(feat_ids, exp):
    """Replace derived representation features with their source features.

    Used when the feature list was taken from the experiment by default, so
    that derived features created by an earlier ``representation='by_type'``
    call are not tested a second time next to their sources.

    Parameters
    ----------
    feat_ids : list
        Feature IDs.
    exp : Experiment
        Experiment object.

    Returns
    -------
    list
        Feature IDs with every derived name replaced by its source name,
        without duplicates and in the original order.
    """
    sources = get_representation_sources(exp)
    restored = []
    for feat_id in feat_ids:
        new_id = sources.get(feat_id, feat_id) if isinstance(feat_id, str) else feat_id
        if new_id not in restored:
            restored.append(new_id)
    return restored


def _registry(exp):
    """Mapping of derived features to their sources, created on first use."""
    sources = getattr(exp, _REGISTRY_ATTR, None)
    if sources is None:
        sources = {}
        setattr(exp, _REGISTRY_ATTR, sources)
    return sources


def _register(exp, name, source, builder):
    """Add a derived feature to the experiment once; reuse it afterwards."""
    sources = _registry(exp)
    if name in exp.dynamic_features:
        if sources.get(name) != source:
            raise ValueError(
                f"Feature '{name}' already exists in the experiment and was not "
                f"created as the representation of '{source}'. Rename it to use "
                "representation='by_type'."
            )
        return name
    with warnings.catch_warnings():
        # The warning targets user features added after cached stats were
        # computed; a derived representation has no stats of its own yet.
        warnings.filterwarnings("ignore", message=".*added after initialization.*")
        exp.add_feature(name, builder())
    sources[name] = source
    return name


def substitute_by_type(feat_ids, exp, verbose=False):
    """Replace features with their type-based representations.

    Derived features are registered in ``exp.dynamic_features`` under
    ``{name}_quad`` (linear with 1, 2 or 3 components) or ``{name}_harm2`` (circular) and
    reused on later calls. Circular features already substituted with their
    ``_2d`` (cos, sin) version are mapped to the harmonic representation of the
    original feature. Discrete features, tuple multifeatures, linear features
    with fewer than three distinct values and features of other shapes are kept
    unchanged.

    Parameters
    ----------
    feat_ids : list
        Feature IDs (strings, or tuples for multifeatures).
    exp : Experiment
        Experiment holding the features.
    verbose : bool, default=False
        If True, print the substitutions.

    Returns
    -------
    tuple
        ``(new_feat_ids, substitutions)`` where ``substitutions`` is a list of
        ``(original, substituted)`` pairs.

    Raises
    ------
    ValueError
        If a feature with the name of a derived representation already exists
        and was not created by this function.

    Examples
    --------
    >>> new_ids, subs = substitute_by_type(['speed', 'place'], exp)  # doctest: +SKIP
    >>> new_ids  # doctest: +SKIP
    ['speed_quad', 'place_quad']
    """
    # Live mapping: names registered earlier in this loop are recognised too.
    derived = _registry(exp)
    new_feat_ids = []
    substituted = []
    for feat_id in feat_ids:
        new_id = feat_id
        if isinstance(feat_id, str) and feat_id not in derived:
            base = feat_id
            if feat_id.endswith("_2d") and _is_circular(
                exp.dynamic_features.get(feat_id[:-3])
            ):
                base = feat_id[:-3]
            ts = exp.dynamic_features.get(base)
            if _is_circular(ts):
                name = base + HARMONICS_SUFFIX
                new_id = _register(
                    exp, name, base, lambda: build_harmonics(ts, 2, name=name)
                )
            elif _is_linear_1d(ts):
                name = base + QUAD_SUFFIX
                new_id = _register(
                    exp, name, base, lambda: build_quadratic_1d(ts, name=name)
                )
            elif _is_linear_multi(ts) and not base.endswith("_2d"):
                name = base + QUAD_SUFFIX
                new_id = _register(
                    exp, name, base, lambda: build_quadratic_multi(ts, name=name)
                )
        if new_id != feat_id:
            substituted.append((feat_id, new_id))
        if new_id not in new_feat_ids:
            new_feat_ids.append(new_id)

    if verbose and substituted:
        print("Features substituted with type-based representations:")
        for orig, sub in substituted:
            print(f"  '{orig}' -> '{sub}'")

    return new_feat_ids, substituted
