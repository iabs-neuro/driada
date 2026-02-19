"""Integrity tests for neuron_database tables.

Property-based checks on real NOF data: value bounds, cross-table
consistency, monotonicity, and algebraic relationships.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from neuron_database import (
    load_experiment, get_fully_matched_ids,
    significance_count_table, significance_fraction_table,
    significance_fraction_of_sel_table,
    retention_count_table, retention_enrichment,
    cross_stats_table,
    mi_table, mi_table_composite,
)
from neuron_database.tables import (
    _neuron_count_table, _selectivity_counts, _sel_count_to_table,
    apply_significance_filters, _resolve_filter_delay,
)

DATA_DIR = Path(__file__).parent.parent.parent / "DRIADA data" / "NOF"


@pytest.fixture(scope="module")
def db():
    return load_experiment('NOF', str(DATA_DIR))


@pytest.fixture(scope="module")
def matched(db):
    return db.get_matched_ids(len(db.sessions))


@pytest.fixture(scope="module")
def neuron_counts(db):
    return _neuron_count_table(db)


@pytest.fixture(scope="module")
def sel_counts(db):
    return _selectivity_counts(db)


SPOT_FEATURES = ['place', 'speed', 'walls', 'any object', 'any place']


# ---------------------------------------------------------------------------
# 1. Count tables: value bounds
# ---------------------------------------------------------------------------

class TestCountBounds:

    def test_nonnegative(self, db):
        for feat in SPOT_FEATURES:
            table = significance_count_table(db, feat)
            assert (table >= 0).all().all(), f"{feat}: negative counts"

    def test_bounded_by_neuron_count(self, db, neuron_counts):
        for feat in SPOT_FEATURES:
            table = significance_count_table(db, feat)
            assert (table <= neuron_counts).all().all(), (
                f"{feat}: count exceeds total neurons")

    def test_feature_bounded_by_all_composite(self, db, sel_counts):
        all_table = _sel_count_to_table(sel_counts, db, 0)
        for feat in db.features:
            table = significance_count_table(db, feat)
            assert (table <= all_table).all().all(), (
                f"{feat}: exceeds composite 'all'")


# ---------------------------------------------------------------------------
# 2. Composite decomposition: sel1 + sel2 + sel3 = all
# ---------------------------------------------------------------------------

class TestCompositeDecomposition:

    @pytest.mark.parametrize("matched_key", [None, "matched"])
    def test_sel_decomposition(self, db, matched, matched_key):
        m = matched if matched_key == "matched" else None
        sel = _selectivity_counts(db, m)
        t_all = _sel_count_to_table(sel, db, 0)
        t_s1 = _sel_count_to_table(sel, db, 1)
        t_s2 = _sel_count_to_table(sel, db, 2)
        t_s3 = _sel_count_to_table(sel, db, 3)
        pd.testing.assert_frame_equal(t_all, t_s1 + t_s2 + t_s3,
                                      check_names=False)


# ---------------------------------------------------------------------------
# 3. Fraction tables: algebraic consistency
# ---------------------------------------------------------------------------

class TestFractionConsistency:

    def test_fraction_equals_count_over_total(self, db, neuron_counts):
        for feat in SPOT_FEATURES:
            frac = significance_fraction_table(db, feat)
            count = significance_count_table(db, feat)
            expected = ((count / neuron_counts.replace(0, np.nan)) * 100).round(2)
            pd.testing.assert_frame_equal(frac, expected, check_names=False)

    def test_fraction_range(self, db):
        for feat in SPOT_FEATURES:
            frac = significance_fraction_table(db, feat)
            vals = frac.values[~np.isnan(frac.values)]
            assert (vals >= 0).all() and (vals <= 100).all(), (
                f"{feat}: fraction out of [0, 100]")

    def test_fraction_of_sel_range(self, db, sel_counts):
        sel_totals = _sel_count_to_table(sel_counts, db, 0)
        for feat in SPOT_FEATURES:
            frac = significance_fraction_of_sel_table(db, feat, sel_totals)
            vals = frac.values[~np.isnan(frac.values)]
            assert (vals >= 0).all() and (vals <= 100).all(), (
                f"{feat}: fraction_of_sel out of [0, 100]")


# ---------------------------------------------------------------------------
# 4. Matched ≤ unmatched
# ---------------------------------------------------------------------------

class TestMatchedSubset:

    def test_count_matched_leq_unmatched(self, db, matched):
        for feat in SPOT_FEATURES:
            full = significance_count_table(db, feat)
            part = significance_count_table(db, feat,
                                            matched_ids_per_mouse=matched)
            assert (part <= full).all().all(), (
                f"{feat}: matched count > unmatched")

    def test_composite_matched_leq_unmatched(self, db, matched):
        for n_sel in [0, 1, 2, 3]:
            full = _sel_count_to_table(_selectivity_counts(db), db, n_sel)
            part = _sel_count_to_table(
                _selectivity_counts(db, matched), db, n_sel)
            assert (part <= full).all().all(), (
                f"sel{n_sel}: matched > unmatched")

    def test_retention_matched_leq_unmatched(self, db, matched):
        for feat in SPOT_FEATURES:
            full = retention_count_table(db, feat)
            part = retention_count_table(db, feat,
                                         matched_ids_per_mouse=matched)
            # Compare only mouse rows (not 'total')
            mice = db.mice
            assert (part.loc[mice] <= full.loc[mice]).all().all(), (
                f"{feat}: matched retention > unmatched")


# ---------------------------------------------------------------------------
# 5. Retention tables: monotonicity and bounds
# ---------------------------------------------------------------------------

class TestRetention:

    def test_monotonicity(self, db):
        for feat in db.features:
            table = retention_count_table(db, feat)
            for mouse in db.mice:
                row = table.loc[mouse].values
                for i in range(len(row) - 1):
                    assert row[i] >= row[i + 1], (
                        f"{feat}/{mouse}: col {i+1}={row[i]} < col {i+2}={row[i+1]}")

    def test_total_row(self, db):
        for feat in SPOT_FEATURES:
            table = retention_count_table(db, feat)
            mice = db.mice
            expected_total = table.loc[mice].sum()
            pd.testing.assert_series_equal(
                table.loc['total'], expected_total, check_names=False)

    def test_nonnegative(self, db):
        for feat in SPOT_FEATURES:
            table = retention_count_table(db, feat)
            assert (table >= 0).all().all(), f"{feat}: negative retention"


# ---------------------------------------------------------------------------
# 6. MI tables: value consistency
# ---------------------------------------------------------------------------

class TestMI:

    def test_mi_nonnegative(self, db):
        for feat in SPOT_FEATURES:
            table = mi_table(db, feat)
            if table.empty:
                continue
            for s in db.sessions:
                assert (table[s] >= 0).all(), f"{feat}/{s}: negative MI"

    def test_nsel_nonneg_int(self, db):
        for feat in SPOT_FEATURES:
            table = mi_table(db, feat)
            if table.empty:
                continue
            for s in db.sessions:
                col = f'{s}_nsel'
                assert (table[col] >= 0).all(), f"{feat}/{col}: negative nsel"
                assert (table[col] == table[col].astype(int)).all()

    def test_mi_implies_nsel(self, db):
        for feat in SPOT_FEATURES:
            table = mi_table(db, feat)
            if table.empty:
                continue
            for s in db.sessions:
                has_mi = table[s] > 0
                has_nsel = table[f'{s}_nsel'] >= 1
                assert (has_mi <= has_nsel).all(), (
                    f"{feat}/{s}: MI > 0 but nsel == 0")

    def test_unique_neuron_ids(self, db):
        for feat in SPOT_FEATURES:
            table = mi_table(db, feat)
            if table.empty:
                continue
            pairs = table[['mouse', 'matched_id']].apply(tuple, axis=1)
            assert pairs.is_unique, f"{feat}: duplicate (mouse, matched_id)"

    def test_composite_row_count_monotonicity(self, db):
        t_all = mi_table_composite(db, 0)
        t_s1 = mi_table_composite(db, 1)
        t_s2 = mi_table_composite(db, 2)
        t_s3 = mi_table_composite(db, 3)
        assert len(t_all) >= len(t_s1)
        assert len(t_all) >= len(t_s2)
        assert len(t_all) >= len(t_s3)


# ---------------------------------------------------------------------------
# 7. Cross-stats table: structural integrity
# ---------------------------------------------------------------------------

class TestCrossStats:

    def test_every_row_has_selectivity(self, db):
        table = cross_stats_table(db)
        sessions = db.sessions
        for _, row in table.iterrows():
            has_feat = any(
                row[s] not in ('', '---') for s in sessions
            )
            assert has_feat, (
                f"Row mouse={row['mouse']} mid={row['matching_row']}: "
                f"no selectivity")

    def test_features_sorted(self, db):
        table = cross_stats_table(db)
        sessions = db.sessions
        for _, row in table.iterrows():
            for s in sessions:
                val = row[s]
                if val not in ('', '---'):
                    feats = [f.strip() for f in val.split(',')]
                    assert feats == sorted(feats), (
                        f"Unsorted features: {val}")

    def test_no_aggregate_features(self, db):
        table = cross_stats_table(db)
        agg = db.aggregate_feature_names
        if not agg:
            pytest.skip("No aggregate features")
        sessions = db.sessions
        for _, row in table.iterrows():
            for s in sessions:
                val = row[s]
                if val not in ('', '---'):
                    feats = [f.strip() for f in val.split(',')]
                    for f in feats:
                        assert f not in agg, (
                            f"Aggregate '{f}' in cross-stats cell")

    def test_matching_row_exists(self, db):
        table = cross_stats_table(db)
        for _, row in table.iterrows():
            mouse = row['mouse']
            mid = row['matching_row']
            assert mid in db.matching[mouse].index, (
                f"matched_id {mid} not in matching table for {mouse}")


# ---------------------------------------------------------------------------
# 8. Filter monotonicity
# ---------------------------------------------------------------------------

class TestFilterMonotonicity:

    def test_stricter_mi_fewer_counts(self, db):
        for feat in SPOT_FEATURES:
            loose = significance_count_table(db, feat, mi_threshold=0.04)
            strict = significance_count_table(db, feat, mi_threshold=0.06)
            assert (strict <= loose).all().all(), (
                f"{feat}: stricter MI gave more counts")

    def test_delay_filter_fewer_counts(self, db):
        for feat in SPOT_FEATURES:
            no_filter = significance_count_table(db, feat, filter_delay=False)
            with_filter = significance_count_table(db, feat, filter_delay=True)
            assert (with_filter <= no_filter).all().all(), (
                f"{feat}: delay filter gave more counts")


# ---------------------------------------------------------------------------
# 9. get_matched_ids invariants
# ---------------------------------------------------------------------------

class TestMatchedIds:

    def test_spec_1_returns_none(self, db):
        assert db.get_matched_ids(1) is None

    def test_monotonic_nesting(self, db):
        n = len(db.sessions)
        prev = None
        for k in range(2, n + 1):
            curr = db.get_matched_ids(k)
            if prev is not None:
                for mouse in db.mice:
                    assert curr[mouse] <= prev[mouse], (
                        f"{mouse}: matched({k}) not ⊆ matched({k-1})")
            prev = curr

    def test_full_match_equals_helper(self, db):
        fully = get_fully_matched_ids(db)
        from_method = db.get_matched_ids(len(db.sessions))
        for mouse in db.mice:
            assert from_method[mouse] == fully[mouse], (
                f"{mouse}: get_matched_ids(n) != get_fully_matched_ids")

    def test_named_sessions_equals_int(self, db):
        all_sessions = db.get_matched_ids(db.sessions)
        by_int = db.get_matched_ids(len(db.sessions))
        for mouse in db.mice:
            assert all_sessions[mouse] == by_int[mouse], (
                f"{mouse}: named sessions != int spec")


# ---------------------------------------------------------------------------
# 10. Retention enrichment: consistency with retention tables
# ---------------------------------------------------------------------------

class TestRetentionEnrichment:

    def test_observed_matches_retention(self, db):
        for feat in SPOT_FEATURES:
            ret = retention_count_table(db, feat)
            enr = retention_enrichment(db, feat, n_shuffles=10, seed=0)
            for k in enr.index:
                assert enr.loc[k, 'observed'] == ret.loc['total', k], (
                    f"{feat}/k={k}: enrichment observed != retention total")

    def test_bounds(self, db):
        for feat in SPOT_FEATURES:
            enr = retention_enrichment(db, feat, n_shuffles=10, seed=0)
            assert (enr['observed'] >= 0).all(), f"{feat}: negative observed"
            assert (enr['null_mean'] >= 0).all(), f"{feat}: negative null_mean"
            assert (enr['null_std'] >= 0).all(), f"{feat}: negative null_std"
            vals = enr['enrichment'].dropna().values
            assert (vals >= 0).all(), f"{feat}: negative enrichment"

    def test_pvalue_sanity(self, db):
        for feat in SPOT_FEATURES:
            enr = retention_enrichment(db, feat, n_shuffles=10, seed=0)
            log_p = enr['-log10p'].dropna().values
            assert (log_p >= 0).all(), f"{feat}: negative -log10p"
            z = enr['zscore'].dropna().values
            assert np.all(np.isfinite(z) | np.isnan(z)), (
                f"{feat}: non-finite zscore")

    def test_observed_monotonicity(self, db):
        for feat in SPOT_FEATURES:
            enr = retention_enrichment(db, feat, n_shuffles=10, seed=0)
            obs = enr['observed'].values
            for i in range(len(obs) - 1):
                assert obs[i] >= obs[i + 1], (
                    f"{feat}: observed not monotonic at k={i+1}")


# ---------------------------------------------------------------------------
# 11. Aggregate features: union bounds
# ---------------------------------------------------------------------------

class TestAggregateFeatures:

    def test_aggregate_leq_sum_constituents(self, db):
        from neuron_database.configs import EXPERIMENT_CONFIGS
        config = EXPERIMENT_CONFIGS['NOF']
        place_constituents = ['place'] + config.discrete_place_features
        agg_table = significance_count_table(db, 'any place')
        constituent_sum = sum(
            significance_count_table(db, f) for f in place_constituents
            if f in db.features)
        assert (agg_table <= constituent_sum).all().all(), (
            "any place > sum of constituents")

    def test_aggregate_geq_max_constituent(self, db):
        from neuron_database.configs import EXPERIMENT_CONFIGS
        config = EXPERIMENT_CONFIGS['NOF']
        place_constituents = ['place'] + config.discrete_place_features
        agg_table = significance_count_table(db, 'any place')
        for f in place_constituents:
            if f not in db.features:
                continue
            part = significance_count_table(db, f)
            assert (agg_table >= part).all().all(), (
                f"any place < constituent '{f}'")


# ---------------------------------------------------------------------------
# 12. Database properties
# ---------------------------------------------------------------------------

class TestDatabaseProperties:

    def test_experiment_id(self, db):
        assert db.experiment_id == 'NOF'

    def test_discarded_features_absent(self, db):
        assert 'speed_2d' not in db.features


# ---------------------------------------------------------------------------
# 13. Cross-table consistency: identities linking different table types
# ---------------------------------------------------------------------------

class TestCrossTableConsistency:
    """Verify algebraic identities that must hold across table types."""

    def test_mi_row_count_equals_count_table(self, db):
        """MI table rows with MI>0 per (mouse, session) = count table cell."""
        for feat in SPOT_FEATURES:
            mi_t = mi_table(db, feat)
            count_t = significance_count_table(db, feat)
            for mouse in db.mice:
                rows = mi_t[mi_t['mouse'] == mouse]
                for s in db.sessions:
                    assert (rows[s] > 0).sum() == count_t.loc[mouse, s], (
                        f"{feat}/{mouse}/{s}: MI rows != count")

    def test_retention_sum_equals_count_sum(self, db):
        """Telescoping: sum_k retention[M,k] = sum_S count[M,S]."""
        for feat in SPOT_FEATURES:
            ret = retention_count_table(db, feat)
            count = significance_count_table(db, feat)
            for mouse in db.mice:
                assert ret.loc[mouse].sum() == count.loc[mouse].sum(), (
                    f"{feat}/{mouse}: retention sum != count sum")

    def test_retention_k1_equals_mi_rows_per_mouse(self, db):
        """retention[M,1] = number of MI table rows for mouse M."""
        for feat in SPOT_FEATURES:
            mi_t = mi_table(db, feat)
            ret = retention_count_table(db, feat)
            for mouse in db.mice:
                assert (mi_t['mouse'] == mouse).sum() == ret.loc[mouse, 1], (
                    f"{feat}/{mouse}: MI rows != retention k=1")

    def test_cross_stats_universe_equals_composite_mi(self, db):
        """Cross-stats and composite MI 'all' have the same neuron set."""
        cs = cross_stats_table(db)
        mi_all = mi_table_composite(db, 0)
        assert len(cs) == len(mi_all), (
            f"cross-stats rows ({len(cs)}) != composite MI rows ({len(mi_all)})")

    def test_cross_stats_nsel_matches_composite_mi(self, db):
        """Feature count per cross-stats cell = nsel in composite MI."""
        cs = cross_stats_table(db)
        mi_all = mi_table_composite(db, 0)
        mi_lookup = mi_all.set_index(['mouse', 'matched_id'])
        for _, row in cs.iterrows():
            mi_row = mi_lookup.loc[(row['mouse'], row['matching_row'])]
            for s in db.sessions:
                cell = row[s]
                n_feats = 0 if cell in ('', '---') else len(cell.split(', '))
                assert mi_row[f'{s}_nsel'] == n_feats, (
                    f"{row['mouse']}/mid={row['matching_row']}/{s}: "
                    f"cross-stats has {n_feats} features, nsel={mi_row[f'{s}_nsel']}")

    def test_feature_count_sum_equals_weighted_selectivity(self, db, sel_counts):
        """sum_F count[F][M,S] = sum of nsel over selective neurons in (M,S)."""
        non_agg = [f for f in db.features
                   if f not in db.aggregate_feature_names]
        total_counts = sum(significance_count_table(db, f) for f in non_agg)
        weighted = (sel_counts.groupby(['mouse', 'session'])['n_sel']
                    .sum().unstack(fill_value=0))
        weighted = weighted.reindex(
            index=db.mice, columns=db.sessions, fill_value=0)
        pd.testing.assert_frame_equal(
            total_counts, weighted, check_names=False, check_dtype=False)

    def test_frac_of_sel_sum_geq_100(self, db, sel_counts):
        """Sum of fraction-of-sel across features >= 100 (mean nsel >= 1)."""
        sel_totals = _sel_count_to_table(sel_counts, db, 0)
        non_agg = [f for f in db.features
                   if f not in db.aggregate_feature_names]
        total_frac = sum(
            significance_fraction_of_sel_table(db, f, sel_totals)
            for f in non_agg)
        vals = total_frac.values[~np.isnan(total_frac.values)]
        assert (vals >= 100.0 - 0.1).all(), (
            f"fraction-of-sel sum < 100: min={vals.min():.2f}")

    def test_retention_kmax_bounded_by_fully_matched(self, db):
        """retention[M, k=max] <= fully matched neurons for mouse M."""
        fully = get_fully_matched_ids(db)
        n_sessions = len(db.sessions)
        for feat in db.features:
            ret = retention_count_table(db, feat)
            for mouse in db.mice:
                assert ret.loc[mouse, n_sessions] <= len(fully[mouse]), (
                    f"{feat}/{mouse}: retention k={n_sessions} > fully matched")
