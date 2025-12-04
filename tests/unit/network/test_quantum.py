"""Tests for quantum graph divergence functions.

These tests verify the mathematical properties of quantum information
measures on graphs - properties that users rely on for network comparison.
"""

import pytest
import numpy as np

from driada.network.quantum import (
    js_divergence,
    get_density_matrix,
    renyi_divergence,
    manual_entropy,
)


class TestJSDivergence:
    """Test quantum Jensen-Shannon divergence properties."""

    def test_identical_graphs_return_zero(self):
        """JS divergence of identical graphs should be zero."""
        # Complete graph K3
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

        result = js_divergence(A, A, t=1.0, return_partial_entropies=False)

        assert result == pytest.approx(0.0, abs=1e-10)

    def test_identical_graphs_with_entropies(self):
        """Identical graphs should have equal partial entropies."""
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

        S_mix, S_A, S_B, jsd = js_divergence(A, A, t=1.0, return_partial_entropies=True)

        assert S_A == pytest.approx(S_B, abs=1e-10)
        assert jsd == pytest.approx(0.0, abs=1e-10)

    def test_symmetry(self):
        """JS divergence should be symmetric: JSD(A,B) == JSD(B,A)."""
        # Two different graphs
        A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])  # Path graph
        B = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])  # Complete graph

        jsd_AB = js_divergence(A, B, t=1.0, return_partial_entropies=False)
        jsd_BA = js_divergence(B, A, t=1.0, return_partial_entropies=False)

        assert jsd_AB == pytest.approx(jsd_BA, abs=1e-10)

    def test_non_negativity(self):
        """JS divergence should always be non-negative."""
        # Various graph pairs
        graphs = [
            np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),  # Path
            np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),  # Complete K3
            np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]]),  # Cycle
        ]

        for i, A in enumerate(graphs):
            for j, B in enumerate(graphs):
                # Pad smaller matrix if needed
                n = max(A.shape[0], B.shape[0])
                A_pad = np.zeros((n, n))
                B_pad = np.zeros((n, n))
                A_pad[:A.shape[0], :A.shape[1]] = A
                B_pad[:B.shape[0], :B.shape[1]] = B

                jsd = js_divergence(A_pad, B_pad, t=1.0, return_partial_entropies=False)
                assert jsd >= 0, f"JSD({i},{j}) = {jsd} is negative"

    def test_different_graphs_positive_divergence(self):
        """Different graphs should have positive JS divergence."""
        # Star graph (central node connected to all)
        star = np.array([
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ])

        # Complete graph K4
        complete = np.array([
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
        ])

        jsd = js_divergence(star, complete, t=1.0, return_partial_entropies=False)

        assert jsd > 0.01, "Star vs complete should have positive divergence"

    def test_temperature_affects_divergence(self):
        """Different temperature values should give different divergences."""
        A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        B = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

        jsd_t1 = js_divergence(A, B, t=0.5, return_partial_entropies=False)
        jsd_t2 = js_divergence(A, B, t=2.0, return_partial_entropies=False)

        # Different temperatures should generally give different divergences
        # (not always, but for most graph pairs)
        assert jsd_t1 != jsd_t2 or (jsd_t1 == 0 and jsd_t2 == 0)


class TestGetDensityMatrix:
    """Test density matrix computation from graphs."""

    def test_trace_normalization(self):
        """Density matrix should have trace = 1."""
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

        rho = get_density_matrix(A, t=1.0)

        assert np.trace(rho) == pytest.approx(1.0, abs=1e-10)

    def test_trace_normalization_various_graphs(self):
        """Trace normalization should hold for various graphs."""
        graphs = [
            np.array([[0, 1], [1, 0]]),  # K2
            np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),  # K3
            np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]),  # Path
        ]

        for A in graphs:
            for t in [0.1, 1.0, 5.0]:
                rho = get_density_matrix(A, t=t)
                assert np.trace(rho) == pytest.approx(1.0, abs=1e-10)

    def test_positive_semi_definite(self):
        """Density matrix should be positive semi-definite (eigenvalues >= 0)."""
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

        rho = get_density_matrix(A, t=1.0)
        eigenvalues = np.linalg.eigvalsh(rho)

        assert np.all(eigenvalues >= -1e-10), "Density matrix must be positive semi-definite"

    def test_hermitian(self):
        """Density matrix should be Hermitian (symmetric for real graphs)."""
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

        rho = get_density_matrix(A, t=1.0)

        assert np.allclose(rho, rho.T), "Density matrix must be symmetric"

    def test_temperature_effect(self):
        """Different temperatures should produce different density matrices."""
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

        rho_t1 = get_density_matrix(A, t=0.5)
        rho_t2 = get_density_matrix(A, t=2.0)

        # Matrices should be different
        assert not np.allclose(rho_t1, rho_t2)

    def test_normalized_laplacian_option(self):
        """Using normalized Laplacian should give different results."""
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

        rho_regular = get_density_matrix(A, t=1.0, norm=0)
        rho_normalized = get_density_matrix(A, t=1.0, norm=1)

        # Both should have trace 1
        assert np.trace(rho_regular) == pytest.approx(1.0, abs=1e-10)
        assert np.trace(rho_normalized) == pytest.approx(1.0, abs=1e-10)

        # But they should be different matrices
        assert not np.allclose(rho_regular, rho_normalized)


class TestRenyiDivergence:
    """Test quantum Rényi divergence properties."""

    def test_non_negativity(self):
        """Rényi divergence should be non-negative."""
        # Create two valid density matrices
        A = get_density_matrix(np.array([[0, 1], [1, 0]]), t=1.0)
        B = get_density_matrix(np.array([[0, 1], [1, 0]]), t=2.0)

        for q in [0.5, 1.0, 2.0]:
            div = renyi_divergence(A, B, q)
            # Note: Due to numerical precision, can be slightly negative
            assert div >= -1e-6, f"Rényi divergence with q={q} should be non-negative"

    def test_identity_returns_zero(self):
        """Rényi divergence of identical matrices should be zero."""
        rho = get_density_matrix(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]), t=1.0)

        for q in [0.5, 1.0, 2.0]:
            div = renyi_divergence(rho, rho, q)
            assert div == pytest.approx(0.0, abs=1e-6), f"D_q(ρ||ρ) should be 0 for q={q}"

    def test_q_must_be_positive(self):
        """q parameter must be positive."""
        rho = np.array([[0.5, 0], [0, 0.5]])

        with pytest.raises(ValueError, match="q must be > 0"):
            renyi_divergence(rho, rho, q=0)

        with pytest.raises(ValueError, match="q must be > 0"):
            renyi_divergence(rho, rho, q=-1)

    def test_q_equals_one_special_case(self):
        """q=1 should compute quantum relative entropy."""
        A = get_density_matrix(np.array([[0, 1], [1, 0]]), t=1.0)
        B = get_density_matrix(np.array([[0, 1], [1, 0]]), t=1.5)

        # At q=1, should use the log formula (quantum relative entropy)
        div = renyi_divergence(A, B, q=1.0)

        # Should be a finite real number
        assert np.isfinite(div)
        assert np.isreal(div)

    def test_different_q_values_give_different_results(self):
        """Different q values should generally give different divergences."""
        A = get_density_matrix(np.array([[0, 1], [1, 0]]), t=1.0)
        B = get_density_matrix(np.array([[0, 1], [1, 0]]), t=2.0)

        div_05 = renyi_divergence(A, B, q=0.5)
        div_1 = renyi_divergence(A, B, q=1.0)
        div_2 = renyi_divergence(A, B, q=2.0)

        # They should be different (for non-identical matrices)
        results = [div_05, div_1, div_2]
        assert len(set(np.round(results, 6))) > 1, "Different q should give different divergences"


class TestManualEntropy:
    """Test Shannon/von Neumann entropy calculation."""

    def test_uniform_distribution(self):
        """Entropy of uniform distribution should be log2(n)."""
        # Uniform over 2 states: entropy = 1 bit
        pr = np.array([0.5, 0.5])
        assert manual_entropy(pr) == pytest.approx(1.0, abs=1e-10)

        # Uniform over 4 states: entropy = 2 bits
        pr = np.array([0.25, 0.25, 0.25, 0.25])
        assert manual_entropy(pr) == pytest.approx(2.0, abs=1e-10)

    def test_deterministic_distribution(self):
        """Entropy of deterministic distribution should be zero."""
        pr = np.array([1.0, 0.0, 0.0])
        assert manual_entropy(pr) == pytest.approx(0.0, abs=1e-10)

    def test_handles_zeros(self):
        """Should handle zero probabilities without errors."""
        pr = np.array([0.5, 0.0, 0.5, 0.0])
        entropy = manual_entropy(pr)

        assert np.isfinite(entropy)
        assert entropy == pytest.approx(1.0, abs=1e-10)  # Effectively 2 equiprobable states

    def test_handles_very_small_values(self):
        """Should handle very small probabilities correctly."""
        pr = np.array([0.5, 0.5, 1e-20])
        entropy = manual_entropy(pr)

        # Should be close to 1 bit (the tiny probability contributes negligibly)
        assert entropy == pytest.approx(1.0, abs=1e-6)

    def test_non_negative(self):
        """Entropy should always be non-negative."""
        # Various distributions
        distributions = [
            np.array([0.5, 0.5]),
            np.array([0.9, 0.1]),
            np.array([0.1, 0.2, 0.3, 0.4]),
        ]

        for pr in distributions:
            assert manual_entropy(pr) >= 0
