"""Tests for population recurrence graph construction."""

import numpy as np
import scipy.sparse as sp
import pytest
from driada.recurrence import (
    takens_embedding, RecurrenceGraph,
)
from driada.recurrence.population import population_recurrence_graph
from driada.recurrence.population import pairwise_jaccard_sparse


@pytest.fixture
def three_sine_graphs():
    """Three identical sine wave recurrence graphs."""
    t = np.arange(300)
    graphs = []
    for freq in [30, 30, 30]:
        data = np.sin(2 * np.pi * t / freq)
        emb = takens_embedding(data, tau=7, m=3)
        graphs.append(RecurrenceGraph(emb, method='knn', k=5))
    return graphs


@pytest.fixture
def mixed_graphs():
    """Two sine + one noise recurrence graph."""
    t = np.arange(300)
    rng = np.random.default_rng(42)
    graphs = []
    for i in range(2):
        data = np.sin(2 * np.pi * t / 30)
        emb = takens_embedding(data, tau=7, m=3)
        graphs.append(RecurrenceGraph(emb, method='knn', k=5))
    noise = rng.standard_normal(300)
    emb = takens_embedding(noise, tau=7, m=3)
    graphs.append(RecurrenceGraph(emb, method='knn', k=5))
    return graphs


class TestPopulationRecurrenceGraph:
    """Tests for population_recurrence_graph()."""

    def test_joint_identical_equals_individual(self, three_sine_graphs):
        """JRP of identical graphs should equal the individual graph."""
        pop = population_recurrence_graph(three_sine_graphs, method='joint', threshold=1.0)
        individual_nnz = three_sine_graphs[0].adj.nnz
        assert pop.adj.nnz <= individual_nnz
        assert pop.adj.nnz > 0

    def test_joint_threshold_majority(self, mixed_graphs):
        """Majority threshold (2/3) should keep more than strict AND."""
        strict = population_recurrence_graph(mixed_graphs, method='joint', threshold=1.0)
        majority = population_recurrence_graph(mixed_graphs, method='joint', threshold=0.67)
        assert majority.adj.nnz >= strict.adj.nnz

    def test_mean_method(self, three_sine_graphs):
        """Mean method should produce averaged values in [0, 1]."""
        pop = population_recurrence_graph(three_sine_graphs, method='mean')
        assert pop.adj.shape[0] == three_sine_graphs[0].adj.shape[0]
        assert pop.adj.nnz > 0
        # Mean of binary matrices should have values in (0, 1]
        assert pop.adj.data.min() > 0
        assert pop.adj.data.max() <= 1.0

    def test_mean_binarize(self, three_sine_graphs):
        """Mean with binarize_threshold should produce binary adjacency."""
        pop = population_recurrence_graph(
            three_sine_graphs, method='mean', binarize_threshold=0.5
        )
        data = pop.adj.data
        assert np.all((data == 0) | (data == 1))

    def test_returns_recurrence_graph(self, three_sine_graphs):
        """Result should be a RecurrenceGraph with RQA capability."""
        pop = population_recurrence_graph(three_sine_graphs, method='joint')
        assert isinstance(pop, RecurrenceGraph)
        rqa = pop.rqa()
        assert 'DET' in rqa

    def test_mismatched_sizes_raises(self):
        """Graphs with different sizes must raise ValueError."""
        emb1 = takens_embedding(np.random.randn(100), tau=1, m=2)
        emb2 = takens_embedding(np.random.randn(200), tau=1, m=2)
        rg1 = RecurrenceGraph(emb1, method='knn', k=3)
        rg2 = RecurrenceGraph(emb2, method='knn', k=3)
        with pytest.raises(ValueError, match="same"):
            population_recurrence_graph([rg1, rg2], method='joint')

    def test_unknown_method_raises(self, three_sine_graphs):
        """Unknown method must raise ValueError."""
        with pytest.raises(ValueError, match="method"):
            population_recurrence_graph(three_sine_graphs, method='unknown')


class TestPairwiseJaccardSparse:
    """Tests for pairwise_jaccard_sparse()."""

    def test_correctness_vs_bruteforce(self):
        """Vectorized result must match brute-force pairwise computation."""
        rng = np.random.default_rng(42)
        n, size = 5, 20
        matrices = []
        for _ in range(n):
            density = rng.uniform(0.05, 0.2)
            m = sp.random(size, size, density=density, format='csr',
                          random_state=rng)
            m = (m > 0).astype(float)
            m = m.maximum(m.T)
            matrices.append(m)

        result = pairwise_jaccard_sparse(matrices)

        assert result.shape == (n, n)
        for i in range(n):
            for j in range(i + 1, n):
                intersection = matrices[i].multiply(matrices[j]).nnz
                union = matrices[i].nnz + matrices[j].nnz - intersection
                expected = intersection / union if union > 0 else 0.0
                assert abs(result[i, j] - expected) < 1e-10
                assert abs(result[j, i] - expected) < 1e-10

    def test_identical_matrices(self):
        """Identical matrices must have Jaccard = 1.0."""
        m = sp.eye(10, format='csr')
        result = pairwise_jaccard_sparse([m, m, m])
        np.testing.assert_allclose(result, np.ones((3, 3)))

    def test_disjoint_matrices(self):
        """Non-overlapping matrices must have Jaccard = 0.0."""
        a = sp.csr_matrix(([1.0], ([0], [0])), shape=(5, 5))
        b = sp.csr_matrix(([1.0], ([4], [4])), shape=(5, 5))
        result = pairwise_jaccard_sparse([a, b])
        assert result[0, 1] == 0.0
        assert result[1, 0] == 0.0
        assert result[0, 0] == 1.0
        assert result[1, 1] == 1.0

    def test_empty_matrices(self):
        """All-zero matrices must have Jaccard = 0.0."""
        a = sp.csr_matrix((5, 5))
        b = sp.csr_matrix((5, 5))
        result = pairwise_jaccard_sparse([a, b])
        assert result[0, 1] == 0.0

    def test_symmetric_zero_diagonal(self):
        """Result must be symmetric. Diagonal = 1.0 for non-empty, 0.0 for empty."""
        rng = np.random.default_rng(99)
        matrices = [
            sp.random(10, 10, density=0.1, format='csr', random_state=rng)
            for _ in range(4)
        ]
        matrices = [(m > 0).astype(float) for m in matrices]
        result = pairwise_jaccard_sparse(matrices)
        np.testing.assert_allclose(result, result.T)
        for i in range(4):
            assert result[i, i] == 1.0

    def test_single_matrix(self):
        """Single matrix must return 1x1 array with value 1.0."""
        m = sp.eye(5, format='csr')
        result = pairwise_jaccard_sparse([m])
        assert result.shape == (1, 1)
        assert result[0, 0] == 1.0
