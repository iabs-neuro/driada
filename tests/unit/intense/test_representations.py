"""Tests for type-based feature representations in INTENSE."""

import inspect

import numpy as np
import pytest
from scipy.stats import spearmanr

from driada.experiment import Experiment, load_exp_from_aligned_data
from driada.information import MultiTimeSeries, TimeSeries
from driada.information.gcmi import copnorm, mi_gg
from driada.intense.pipelines import compute_cell_feat_significance
from driada.intense.representations import (
    build_harmonics,
    build_quadratic_1d,
    build_quadratic_multi,
    get_representation_sources,
    restore_source_features,
    substitute_by_type,
)


def _smooth_walk(n, tau, rng):
    """Bounded smooth random walk in [0, 1] (AR(1), time constant tau frames)."""
    a = np.exp(-1.0 / tau)
    e = rng.randn(n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = a * x[i - 1] + np.sqrt(1 - a * a) * e[i]
    x -= x.min()
    return x / x.max()


class TestBuilders:
    """Shapes and basic properties of the representation builders."""

    def test_quadratic_1d_not_rank_collinear(self):
        rng = np.random.RandomState(0)
        x = TimeSeries(rng.uniform(0, 10, 2000), discrete=False)
        rep = build_quadratic_1d(x, name="x_quad")
        assert rep.data.shape == (2, 2000)
        # The centred square must carry information the ranks of x do not.
        rho, _ = spearmanr(rep.data[0], rep.data[1])
        assert abs(rho) < 0.5

    def test_quadratic_1d_median_at_minimum(self):
        """A variable at its minimum most of the time (resting speed).

        The median equals the minimum, so a square around it would have the
        same ranks as the variable; the centre must move inside the range.
        """
        rng = np.random.RandomState(5)
        speed = np.r_[np.zeros(6000), rng.exponential(3.0, 4000)]
        rep = build_quadratic_1d(TimeSeries(speed, discrete=False))
        rho, _ = spearmanr(rep.data[0], rep.data[1])
        assert abs(rho) < 0.99
        eig = np.linalg.eigvalsh(np.cov(copnorm(rep.data)))
        assert eig.min() / eig.max() > 1e-6

    def test_quadratic_1d_zeros_at_centre(self):
        """A signed variable that is exactly zero most of the time (vertical speed).

        The median is zero and inside the range, so at every zero both components
        vanish; such time points are valid data.
        """
        rng = np.random.RandomState(9)
        vz = np.r_[np.zeros(6000), rng.randn(4000)]
        rep = build_quadratic_1d(TimeSeries(vz, discrete=False))
        assert rep.data.shape == (2, 10000)
        assert np.sum(np.all(rep.data == 0, axis=0)) == 6000

    def test_harmonics_shape_and_bounds(self):
        rng = np.random.RandomState(1)
        th = TimeSeries(rng.uniform(0, 2 * np.pi, 1500), ts_type="circular")
        rep = build_harmonics(th, n_harmonics=2, name="th_harm2")
        assert rep.data.shape == (4, 1500)
        assert np.all(np.abs(rep.data) <= 1 + 1e-12)

    def test_harmonics_handle_degrees(self):
        rng = np.random.RandomState(2)
        rad = rng.uniform(0, 2 * np.pi, 1500)
        deg = TimeSeries(np.degrees(rad), ts_type="circular")
        rep = build_harmonics(deg, n_harmonics=2)
        np.testing.assert_allclose(rep.data[2], np.cos(2 * rad), atol=1e-9)

    def test_harmonics_invariant_to_angle_origin(self):
        """MI with an axis-tuned signal must not depend on where zero is."""
        rng = np.random.RandomState(3)
        n = 4000
        th = rng.uniform(0, 2 * np.pi, n)
        z = np.cos(2 * (th - 1.0)) + 0.5 * rng.randn(n)
        mis = []
        for origin in (0.0, 2.0):
            ts = TimeSeries((th + origin) % (2 * np.pi), ts_type="circular")
            rep = build_harmonics(ts, 2)
            x = copnorm(rep.data)
            mis.append(mi_gg(x, copnorm(z).ravel(), True, True))
        assert mis[0] > 0.1
        np.testing.assert_allclose(mis[0], mis[1], rtol=0.05)

    def test_quadratic_multi_shape(self):
        rng = np.random.RandomState(4)
        xy = MultiTimeSeries(rng.rand(2, 1000), discrete=False)
        assert build_quadratic_multi(xy).data.shape == (5, 1000)
        xyz = MultiTimeSeries(rng.rand(3, 1000), discrete=False)
        assert build_quadratic_multi(xyz).data.shape == (9, 1000)

    def test_quadratic_multi_3d_well_conditioned(self):
        rng = np.random.RandomState(6)
        xyz = MultiTimeSeries(rng.rand(3, 3000), discrete=False)
        rep = build_quadratic_multi(xyz)
        eig = np.linalg.eigvalsh(np.cov(copnorm(rep.data)))
        assert eig.min() / eig.max() > 1e-3

    def test_quadratic_multi_rejects_other_dimensions(self):
        rng = np.random.RandomState(7)
        four = MultiTimeSeries(rng.rand(4, 500), discrete=False)
        with pytest.raises(ValueError, match="2 or 3 components"):
            build_quadratic_multi(four)


class TestSubstituteByType:
    """Mapping of feature IDs to their type-based representations."""

    @pytest.fixture
    def exp(self):
        rng = np.random.RandomState(42)
        n = 2000
        data = {
            "Calcium": rng.randn(6, n),
            "head_direction": rng.uniform(0, 2 * np.pi, n),
            "speed": rng.uniform(0, 10, n),
            "state": rng.randint(0, 3, n),
        }
        exp = load_exp_from_aligned_data(
            "test", {"animal": "A1"}, data, create_circular_2d=True, verbose=False
        )
        place = MultiTimeSeries(rng.rand(2, n), discrete=False, name="place")
        exp.add_feature("place", place)
        place3d = MultiTimeSeries(rng.rand(3, n), discrete=False, name="3d-place")
        exp.add_feature("3d-place", place3d)
        return exp

    def test_maps_each_type(self, exp):
        feat_ids = ["head_direction", "speed", "state", "place"]
        new_ids, subs = substitute_by_type(feat_ids, exp)
        assert new_ids == ["head_direction_harm2", "speed_quad", "state", "place_quad"]
        assert ("state", "state") not in subs
        assert exp.dynamic_features["head_direction_harm2"].data.shape[0] == 4
        assert exp.dynamic_features["speed_quad"].data.shape[0] == 2
        assert exp.dynamic_features["place_quad"].data.shape[0] == 5

    def test_three_component_feature_maps_to_nine(self, exp):
        new_ids, _ = substitute_by_type(["3d-place"], exp)
        assert new_ids == ["3d-place_quad"]
        assert exp.dynamic_features["3d-place_quad"].data.shape[0] == 9

    def test_circular_2d_maps_to_harmonics(self, exp):
        new_ids, _ = substitute_by_type(["head_direction_2d"], exp)
        assert new_ids == ["head_direction_harm2"]

    def test_idempotent(self, exp):
        first, _ = substitute_by_type(["speed", "place"], exp)
        feature = exp.dynamic_features["speed_quad"]
        second, subs = substitute_by_type(first, exp)
        assert second == first
        assert subs == []
        assert exp.dynamic_features["speed_quad"] is feature

    def test_derived_name_in_same_call_not_expanded_again(self, exp):
        new_ids, _ = substitute_by_type(["speed", "speed_quad"], exp)
        assert new_ids == ["speed_quad"]
        assert "speed_quad_quad" not in exp.dynamic_features

    def test_tuples_pass_through(self, exp):
        new_ids, subs = substitute_by_type([("speed", "state")], exp)
        assert new_ids == [("speed", "state")]
        assert subs == []

    def test_two_valued_continuous_feature_kept(self, exp):
        flag = TimeSeries(np.tile([0.0, 1.0], exp.n_frames // 2), discrete=False)
        exp.add_feature("flag", flag)
        new_ids, subs = substitute_by_type(["flag"], exp)
        assert new_ids == ["flag"]
        assert subs == []

    def test_existing_feature_with_derived_name_raises(self, exp):
        exp.add_feature("speed_quad", np.random.RandomState(0).rand(exp.n_frames))
        with pytest.raises(ValueError, match="already exists"):
            substitute_by_type(["speed"], exp)

    def test_restore_source_features(self, exp):
        substitute_by_type(["speed", "place", "head_direction"], exp)
        assert get_representation_sources(exp) == {
            "speed_quad": "speed",
            "place_quad": "place",
            "head_direction_harm2": "head_direction",
        }
        restored = restore_source_features(
            ["speed", "speed_quad", "place_quad", "head_direction_harm2", "state"], exp
        )
        assert restored == ["speed", "place", "head_direction", "state"]


class TestPipelineByType:
    """End-to-end: by-type representation finds non-monotone tuning."""

    @pytest.fixture(scope="class")
    def central_peak_exp(self):
        rng = np.random.RandomState(7)
        fps, n = 20, 20 * 60 * 10
        x = _smooth_walk(n, 3.0 * fps, rng)
        bump = 0.2 + np.exp(-((x - 0.5) ** 2) / (2 * 0.12**2))
        k = np.exp(-np.arange(6 * fps) / (2.0 * fps))
        calcium = []
        for i, shape in enumerate([bump] * 3 + [np.ones(n)] * 3):
            rate = 1.0 / fps * shape / shape.mean()
            ev = np.random.RandomState(100 + i).poisson(rate).astype(float)
            ca = np.convolve(ev, k)[:n] + 0.05 * np.random.RandomState(200 + i).randn(n)
            calcium.append(np.clip(ca, 0, None))
        return Experiment(
            "central_peak",
            np.vstack(calcium),
            None,
            {},
            {"fps": float(fps)},
            {"xvar": TimeSeries(x, discrete=False)},
            reconstruct_spikes=None,
            verbose=False,
        )

    def _significant(self, exp, representation):
        _, sig, _, _, _ = compute_cell_feat_significance(
            exp,
            mode="two_stage",
            n_shuffles_stage1=100,
            n_shuffles_stage2=1000,
            ds=5,
            pval_thr=0.001,
            multicomp_correction=None,
            representation=representation,
            verbose=False,
            seed=1,
            use_precomputed_stats=False,
            save_computed_stats=False,
        )
        return sig

    def test_by_type_finds_central_peak_and_keeps_nulls(self, central_peak_exp):
        sig = self._significant(central_peak_exp, "by_type")
        found = [bool(sig[c]["xvar_quad"]["stage2"]) for c in range(6)]
        assert all(found[:3])
        assert not any(found[3:])

    def test_invalid_representation_raises(self, central_peak_exp):
        with pytest.raises(ValueError, match="representation must be one of"):
            compute_cell_feat_significance(
                central_peak_exp,
                mode="stage1",
                n_shuffles_stage1=10,
                representation="omnibus",
                verbose=False,
            )


class TestAntiSelectiveByType:
    """Anti-selective removal treats a linear feature alike in both modes."""

    @pytest.fixture(scope="class")
    def monotone_exp(self):
        rng = np.random.RandomState(11)
        fps, n = 20, 20 * 60 * 10
        x = _smooth_walk(n, 3.0 * fps, rng)
        k = np.exp(-np.arange(6 * fps) / (2.0 * fps))
        calcium = []
        # Two neurons fire more at high x, two are suppressed by x.
        for i, shape in enumerate([0.1 + x, 0.1 + x, 1.1 - x, 1.1 - x]):
            rate = 1.0 / fps * shape / shape.mean()
            ev = np.random.RandomState(300 + i).poisson(rate).astype(float)
            ca = np.convolve(ev, k)[:n] + 0.05 * np.random.RandomState(400 + i).randn(n)
            calcium.append(np.clip(ca, 0, None))
        return Experiment(
            "monotone",
            np.vstack(calcium),
            None,
            {},
            {"fps": float(fps)},
            {"xvar": TimeSeries(x, discrete=False)},
            reconstruct_spikes=None,
            verbose=False,
        )

    def _significant(self, exp, representation, remove_anti_selective):
        _, sig, _, _, _ = compute_cell_feat_significance(
            exp,
            mode="two_stage",
            n_shuffles_stage1=100,
            n_shuffles_stage2=1000,
            ds=5,
            pval_thr=0.001,
            multicomp_correction=None,
            representation=representation,
            remove_anti_selective=remove_anti_selective,
            verbose=False,
            seed=1,
            use_precomputed_stats=False,
            save_computed_stats=False,
        )
        name = "xvar_quad" if representation == "by_type" else "xvar"
        return [bool(sig[c][name]["stage2"]) for c in range(4)]

    @pytest.mark.parametrize("representation", ["raw", "by_type"])
    def test_suppressed_neurons_removed(self, monotone_exp, representation):
        # Without the removal all four are detected, so the removal is what differs.
        assert all(self._significant(monotone_exp, representation, False))
        assert self._significant(monotone_exp, representation, True) == [
            True, True, False, False
        ]


class TestDefaultFeatureSet:
    """Derived features must not leak into later runs with the default feature set."""

    def test_raw_after_by_type_tests_sources_only(self):
        rng = np.random.RandomState(8)
        n = 2000
        data = {
            "Calcium": np.abs(rng.randn(3, n)),
            "head_direction": rng.uniform(0, 2 * np.pi, n),
            "speed": rng.uniform(0, 10, n),
            "state": rng.randint(0, 3, n),
        }
        exp = load_exp_from_aligned_data(
            "test", {"animal": "A1"}, data, create_circular_2d=True, verbose=False
        )

        def features(representation):
            stats, *_ = compute_cell_feat_significance(
                exp,
                mode="stage1",
                n_shuffles_stage1=10,
                representation=representation,
                verbose=False,
                use_precomputed_stats=False,
                save_computed_stats=False,
            )
            return set(next(iter(stats.values())).keys())

        raw_before = features("raw")
        by_type = features("by_type")
        raw_after = features("raw")

        assert by_type == {"head_direction_harm2", "speed_quad", "state"}
        assert raw_before == raw_after == {"head_direction_2d", "speed", "state"}


class TestDefaultRepresentation:
    """The type-based representation is the default; 'raw' keeps source features."""

    @pytest.fixture
    def exp(self):
        rng = np.random.RandomState(8)
        n = 2000
        data = {
            "Calcium": np.abs(rng.randn(3, n)),
            "head_direction": rng.uniform(0, 2 * np.pi, n),
            "speed": rng.uniform(0, 10, n),
            "state": rng.randint(0, 3, n),
        }
        return load_exp_from_aligned_data(
            "test", {"animal": "A1"}, data, create_circular_2d=True, verbose=False
        )

    @staticmethod
    def _stats(exp, **kwargs):
        stats, *_ = compute_cell_feat_significance(
            exp,
            mode="stage1",
            n_shuffles_stage1=10,
            verbose=False,
            enable_parallelization=False,
            use_precomputed_stats=False,
            save_computed_stats=False,
            **kwargs,
        )
        return stats

    def test_signature_default_is_by_type(self):
        default = inspect.signature(compute_cell_feat_significance).parameters[
            "representation"
        ].default
        assert default == "by_type"

    def test_default_matches_explicit_by_type(self, exp):
        default = self._stats(exp)
        explicit = self._stats(exp, representation="by_type")

        assert set(default[0]) == {"head_direction_harm2", "speed_quad", "state"}
        assert set(explicit[0]) == set(default[0])
        for cell_id in default:
            for feat_id in default[cell_id]:
                assert default[cell_id][feat_id]["me"] == explicit[cell_id][feat_id]["me"]

    def test_explicit_raw_keeps_source_features(self, exp):
        features_before = set(exp.dynamic_features)
        stats = self._stats(exp, representation="raw")

        assert set(stats[0]) == {"head_direction_2d", "speed", "state"}
        # Nothing is derived or registered in the experiment in raw mode.
        assert set(exp.dynamic_features) == features_before
        assert get_representation_sources(exp) == {}

    def test_raw_values_differ_from_by_type_only_where_substituted(self, exp):
        raw = self._stats(exp, representation="raw")
        by_type = self._stats(exp)

        for cell_id in raw:
            # A discrete feature has no type-based representation.
            assert raw[cell_id]["state"]["me"] == by_type[cell_id]["state"]["me"]
            assert raw[cell_id]["speed"]["me"] != by_type[cell_id]["speed_quad"]["me"]

    @pytest.mark.parametrize(
        "kwargs",
        [{"metric": "spearmanr"}, {"metric": "mi", "mi_estimator": "ksg"}],
        ids=["spearmanr", "ksg"],
    )
    def test_default_leaves_features_raw_without_gcmi(self, exp, kwargs):
        # The derived features are multi-dimensional and meant for GCMI only.
        stats = self._stats(
            exp, feat_bunch=["speed"], find_optimal_delays=False, **kwargs
        )

        assert set(stats[0]) == {"speed"}
        assert get_representation_sources(exp) == {}

    def test_derived_name_is_plotted_through_its_source(self, exp):
        import matplotlib.pyplot as plt

        from driada.intense.visual import plot_neuron_feature_pair

        self._stats(exp)
        fig = plot_neuron_feature_pair(exp, 0, "speed_quad", add_density_plot=False)
        try:
            assert fig.axes[0].get_legend_handles_labels()[1][-1] == "speed"
        finally:
            plt.close(fig)

    def test_feat_feat_default_set_skips_derived_features(self, exp):
        from driada.intense.pipelines import compute_feat_feat_significance

        self._stats(exp)
        assert "speed_quad" in exp.dynamic_features

        *_, feat_ids, _ = compute_feat_feat_significance(
            exp,
            mode="stage1",
            n_shuffles_stage1=10,
            verbose=False,
            enable_parallelization=False,
        )
        assert not set(feat_ids) & set(get_representation_sources(exp))
        assert "speed" in feat_ids
