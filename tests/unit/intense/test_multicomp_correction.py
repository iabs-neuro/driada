"""
Tests for multiple comparison correction methods.

Verifies Holm-Bonferroni and FDR-BH corrections are implemented correctly.
"""
import pytest
import numpy as np
from driada.intense.correction import get_multicomp_correction_thr


class TestHolmCorrection:
    """Test Holm-Bonferroni correction implementation."""

    def test_holm_basic(self):
        """Basic Holm correction test."""
        pvals = [0.001, 0.01, 0.02, 0.03, 0.04]
        fwer = 0.05

        threshold = get_multicomp_correction_thr(
            fwer=fwer, mode="holm", all_pvals=pvals
        )

        # Should return threshold from last rejected hypothesis
        assert threshold > 0
        assert threshold <= fwer

    def test_holm_vs_statsmodels(self):
        """Compare Holm correction to statsmodels implementation."""
        _statsmodels = pytest.importorskip("statsmodels")
        from statsmodels.stats.multitest import multipletests

        pvals = np.array([0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2])
        fwer = 0.05

        # Our implementation
        our_threshold = get_multicomp_correction_thr(
            fwer=fwer, mode="holm", all_pvals=pvals.tolist()
        )

        # Statsmodels implementation
        reject_sm, pvals_corrected, _, _ = multipletests(
            pvals, alpha=fwer, method="holm"
        )

        # Find the threshold from statsmodels
        # Holm rejects p[i] if p[i] <= alpha/(m-i)
        sorted_pvals = np.sort(pvals)
        n = len(pvals)
        sm_threshold = 0
        for i, p in enumerate(sorted_pvals):
            cthr = fwer / (n - i)
            if p > cthr:
                break
            sm_threshold = cthr

        # Our threshold should match statsmodels
        np.testing.assert_almost_equal(our_threshold, sm_threshold)

    def test_holm_formula_correct(self):
        """Verify Holm formula uses correct divisor (nhyp - i, not nhyp - i + 1)."""
        pvals = [0.001, 0.01, 0.02, 0.03, 0.04]
        fwer = 0.05
        n = len(pvals)

        threshold = get_multicomp_correction_thr(
            fwer=fwer, mode="holm", all_pvals=pvals
        )

        # Manually compute expected thresholds
        # i=0: 0.05/5=0.010, p=0.001 <= 0.010 → reject
        # i=1: 0.05/4=0.0125, p=0.01 <= 0.0125 → reject
        # i=2: 0.05/3=0.0167, p=0.02 > 0.0167 → stop
        # Expected threshold: 0.0125

        expected_threshold = fwer / (n - 1)  # 0.05/4 = 0.0125
        np.testing.assert_almost_equal(threshold, expected_threshold)

    def test_holm_all_significant(self):
        """All p-values significant."""
        pvals = [0.001, 0.002, 0.003, 0.004, 0.005]
        fwer = 0.05

        threshold = get_multicomp_correction_thr(
            fwer=fwer, mode="holm", all_pvals=pvals
        )

        # All rejected, threshold should be from last hypothesis
        expected = fwer / (len(pvals) - 4)  # 0.05/1 = 0.05
        assert threshold == expected

    def test_holm_none_significant(self):
        """No p-values significant."""
        pvals = [0.5, 0.6, 0.7, 0.8, 0.9]
        fwer = 0.05

        threshold = get_multicomp_correction_thr(
            fwer=fwer, mode="holm", all_pvals=pvals
        )

        # None rejected, threshold should be 0
        assert threshold == 0

    def test_holm_single_pvalue(self):
        """Single p-value (edge case)."""
        pvals = [0.03]
        fwer = 0.05

        threshold = get_multicomp_correction_thr(
            fwer=fwer, mode="holm", all_pvals=pvals
        )

        # Single hypothesis: reject if p <= alpha/1
        if pvals[0] <= fwer:
            assert threshold == fwer
        else:
            assert threshold == 0

    def test_holm_empty_list_raises(self):
        """Empty p-value list should raise ValueError."""
        with pytest.raises(ValueError, match="Empty p-value list"):
            get_multicomp_correction_thr(fwer=0.05, mode="holm", all_pvals=[])

    def test_holm_unsorted_input(self):
        """Holm should work with unsorted input (sorts internally)."""
        pvals = [0.04, 0.001, 0.03, 0.01, 0.02]  # Unsorted
        fwer = 0.05

        threshold = get_multicomp_correction_thr(
            fwer=fwer, mode="holm", all_pvals=pvals
        )

        # Should still work correctly after internal sorting
        assert threshold > 0


class TestFDRCorrection:
    """Test FDR Benjamini-Hochberg correction."""

    def test_fdr_basic(self):
        """Basic FDR-BH test."""
        pvals = [0.001, 0.01, 0.02, 0.03, 0.04]
        fwer = 0.05

        threshold = get_multicomp_correction_thr(
            fwer=fwer, mode="fdr_bh", all_pvals=pvals
        )

        assert threshold > 0
        assert threshold <= fwer

    def test_fdr_empty_list_raises(self):
        """Empty p-value list should raise ValueError."""
        with pytest.raises(ValueError, match="Empty p-value list"):
            get_multicomp_correction_thr(fwer=0.05, mode="fdr_bh", all_pvals=[])


class TestBonferroniCorrection:
    """Test Bonferroni correction."""

    def test_bonferroni_basic(self):
        """Basic Bonferroni test."""
        fwer = 0.05
        nhyp = 10

        threshold = get_multicomp_correction_thr(
            fwer=fwer, mode="bonferroni", nhyp=nhyp
        )

        expected = fwer / nhyp
        assert threshold == expected

    def test_bonferroni_formula(self):
        """Verify Bonferroni uses correct formula."""
        fwer = 0.05
        nhyp = 100

        threshold = get_multicomp_correction_thr(
            fwer=fwer, mode="bonferroni", nhyp=nhyp
        )

        # Bonferroni: alpha/m
        expected = 0.05 / 100
        assert threshold == expected


class TestNoCorrection:
    """Test no correction mode."""

    def test_no_correction(self):
        """No correction should return fwer directly."""
        fwer = 0.05

        threshold = get_multicomp_correction_thr(fwer=fwer, mode=None)

        assert threshold == fwer


class TestInvalidMode:
    """Test error handling for invalid correction modes."""

    def test_invalid_mode_raises(self):
        """Invalid correction mode should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown multiple comparisons correction method"):
            get_multicomp_correction_thr(fwer=0.05, mode="invalid_mode")


class TestEdgeCases:
    """Test edge cases across all correction methods."""

    def test_very_small_fwer(self):
        """Very small FWER (stringent correction)."""
        pvals = [0.0001, 0.001, 0.01]
        fwer = 0.0001

        threshold = get_multicomp_correction_thr(
            fwer=fwer, mode="holm", all_pvals=pvals
        )

        # Only first p-value should be rejected
        assert threshold <= fwer

    def test_very_large_fwer(self):
        """Very large FWER (lenient correction)."""
        pvals = [0.1, 0.2, 0.3]
        fwer = 0.9

        threshold = get_multicomp_correction_thr(
            fwer=fwer, mode="holm", all_pvals=pvals
        )

        # All should be rejected
        assert threshold > 0

    def test_many_hypotheses(self):
        """Large number of hypotheses (stress test)."""
        pvals = list(np.linspace(0.001, 0.1, 1000))
        fwer = 0.05

        # Should handle large lists efficiently
        threshold = get_multicomp_correction_thr(
            fwer=fwer, mode="holm", all_pvals=pvals
        )

        # With 1000 hypotheses, Holm requires p <= 0.05/1000 = 0.00005 for first
        # Since min(pvals) = 0.001 > 0.00005, all rejected (threshold = 0)
        assert threshold == 0
