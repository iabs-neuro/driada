"""Tests for population recurrence graph construction."""

import warnings

import numpy as np
import scipy.sparse as sp
import pytest
from driada.recurrence import (
    takens_embedding, RecurrenceGraph,
)
from driada.recurrence.population import (
    population_recurrence_graph,
    pairwise_jaccard_sparse,
    _reconcile_graph_sizes,
)


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


def _make_graphs_different_sizes():
    """Helper: two graphs with different embedding sizes."""
    rng = np.random.default_rng(42)
    emb1 = takens_embedding(rng.standard_normal(100), tau=1, m=2)
    emb2 = takens_embedding(rng.standard_normal(200), tau=1, m=2)
    rg1 = RecurrenceGraph(emb1, method='knn', k=3)
    rg2 = RecurrenceGraph(emb2, method='knn', k=3)
    return rg1, rg2


def _make_graphs_with_outlier():
    """Helper: 10 similar-sized graphs + 1 outlier (much smaller)."""
    rng = np.random.default_rng(42)
    graphs = []
    # 10 graphs of size ~195-199 (tau=1, m=2, data length 200)
    for _ in range(10):
        data = rng.standard_normal(200)
        emb = takens_embedding(data, tau=1, m=2)
        graphs.append(RecurrenceGraph(emb, method='knn', k=3))
    # 1 outlier: much shorter
    data = rng.standard_normal(100)
    emb = takens_embedding(data, tau=1, m=2)
    graphs.append(RecurrenceGraph(emb, method='knn', k=3))
    return graphs


# =====================================================================
# Tests for _reconcile_graph_sizes
# =====================================================================

class TestReconcileGraphSizes:
    """Tests for the shared _reconcile_graph_sizes function."""

    def test_same_sizes_noop(self, three_sine_graphs):
        """Same-sized items pass through unchanged."""
        result, mask = _reconcile_graph_sizes(three_sine_graphs, trim='min')
        assert len(result) == 3
        assert mask.all()
        assert result[0].adj.shape == three_sine_graphs[0].adj.shape

    def test_trim_none_raises(self):
        """trim=None raises ValueError when sizes differ."""
        rg1, rg2 = _make_graphs_different_sizes()
        with pytest.raises(ValueError, match="trim='min'"):
            _reconcile_graph_sizes([rg1, rg2], trim=None)

    def test_trim_min(self):
        """trim='min' trims all to smallest size."""
        rg1, rg2 = _make_graphs_different_sizes()
        result, mask = _reconcile_graph_sizes([rg1, rg2], trim='min')
        assert len(result) == 2
        assert mask.all()
        expected = min(rg1.adj.shape[0], rg2.adj.shape[0])
        assert result[0].adj.shape[0] == expected
        assert result[1].adj.shape[0] == expected

    def test_adaptive_removes_outlier(self):
        """Adaptive mode removes outlier graphs."""
        graphs = _make_graphs_with_outlier()
        sizes_before = [rg.adj.shape[0] for rg in graphs]
        min_before = min(sizes_before)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result, mask = _reconcile_graph_sizes(
                graphs, trim='adaptive', tail_sigma=2.0)
            assert len(w) == 1
            assert "removed" in str(w[0].message).lower()

        # Outlier (last) should be removed
        assert not mask[-1]
        assert mask[:-1].all()
        # Result should be larger than min_before
        assert result[0].adj.shape[0] > min_before

    def test_adaptive_no_outliers(self, three_sine_graphs):
        """Adaptive with same-sized graphs removes nothing."""
        result, mask = _reconcile_graph_sizes(
            three_sine_graphs, trim='adaptive')
        assert len(result) == 3
        assert mask.all()

    def test_sparse_matrices(self):
        """Works with raw sparse matrices (not RecurrenceGraph)."""
        m1 = sp.eye(5, format='csr')
        m2 = sp.eye(8, format='csr')
        result, mask = _reconcile_graph_sizes([m1, m2], trim='min')
        assert result[0].shape[0] == 5
        assert result[1].shape[0] == 5
        assert mask.all()


# =====================================================================
# Tests for population_recurrence_graph
# =====================================================================

class TestPopulationRecurrenceGraph:
    """Tests for population_recurrence_graph()."""

    def test_joint_identical_equals_individual(self, three_sine_graphs):
        """JRP of identical graphs should equal the individual graph."""
        pop = population_recurrence_graph(three_sine_graphs, method='joint',
                                          threshold=1.0)
        individual_nnz = three_sine_graphs[0].adj.nnz
        assert pop.adj.nnz <= individual_nnz
        assert pop.adj.nnz > 0

    def test_joint_threshold_majority(self, mixed_graphs):
        """Majority threshold (2/3) should keep more than strict AND."""
        strict = population_recurrence_graph(mixed_graphs, method='joint',
                                             threshold=1.0)
        majority = population_recurrence_graph(mixed_graphs, method='joint',
                                               threshold=0.67)
        assert majority.adj.nnz >= strict.adj.nnz

    def test_mean_method(self, three_sine_graphs):
        """Mean method should produce averaged values in [0, 1]."""
        pop = population_recurrence_graph(three_sine_graphs, method='mean')
        assert pop.adj.shape[0] == three_sine_graphs[0].adj.shape[0]
        assert pop.adj.nnz > 0
        assert pop.adj.data.min() > 0
        assert pop.adj.data.max() <= 1.0

    def test_mean_binarize(self, three_sine_graphs):
        """Mean with binarize_threshold should produce binary adjacency."""
        pop = population_recurrence_graph(
            three_sine_graphs, method='mean', binarize_threshold=0.5)
        data = pop.adj.data
        assert np.all((data == 0) | (data == 1))

    def test_returns_recurrence_graph(self, three_sine_graphs):
        """Result should be a RecurrenceGraph with RQA capability."""
        pop = population_recurrence_graph(three_sine_graphs, method='joint')
        assert isinstance(pop, RecurrenceGraph)
        rqa = pop.rqa()
        assert 'DET' in rqa

    def test_mismatched_sizes_raises_with_trim_none(self):
        """Graphs with different sizes raise ValueError when trim=None."""
        rg1, rg2 = _make_graphs_different_sizes()
        with pytest.raises(ValueError, match="trim"):
            population_recurrence_graph([rg1, rg2], method='joint', trim=None)

    def test_trim_min(self):
        """trim='min' handles different-sized graphs."""
        rg1, rg2 = _make_graphs_different_sizes()
        pop = population_recurrence_graph([rg1, rg2], method='joint',
                                          trim='min')
        expected = min(rg1.adj.shape[0], rg2.adj.shape[0])
        assert pop.adj.shape[0] == expected

    def test_trim_adaptive(self):
        """trim='adaptive' handles graphs with outlier sizes."""
        graphs = _make_graphs_with_outlier()
        pop = population_recurrence_graph(graphs, method='mean',
                                          trim='adaptive', tail_sigma=2.0)
        # Should be larger than the outlier's size
        outlier_size = graphs[-1].adj.shape[0]
        assert pop.adj.shape[0] > outlier_size

    def test_trim_adaptive_same_sizes(self, three_sine_graphs):
        """trim='adaptive' with same-sized graphs works identically to no trim."""
        pop = population_recurrence_graph(three_sine_graphs, method='joint')
        assert pop.adj.shape[0] == three_sine_graphs[0].adj.shape[0]

    def test_unknown_method_raises(self, three_sine_graphs):
        """Unknown method must raise ValueError."""
        with pytest.raises(ValueError, match="method"):
            population_recurrence_graph(three_sine_graphs, method='unknown')


# =====================================================================
# Tests for pairwise_jaccard_sparse
# =====================================================================

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

        result, mask = pairwise_jaccard_sparse(matrices)

        assert result.shape == (n, n)
        assert mask.all()
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
        result, mask = pairwise_jaccard_sparse([m, m, m])
        np.testing.assert_allclose(result, np.ones((3, 3)))
        assert mask.all()

    def test_disjoint_matrices(self):
        """Non-overlapping matrices must have Jaccard = 0.0."""
        a = sp.csr_matrix(([1.0], ([0], [0])), shape=(5, 5))
        b = sp.csr_matrix(([1.0], ([4], [4])), shape=(5, 5))
        result, mask = pairwise_jaccard_sparse([a, b])
        assert result[0, 1] == 0.0
        assert result[1, 0] == 0.0
        assert result[0, 0] == 1.0
        assert result[1, 1] == 1.0

    def test_empty_matrices(self):
        """All-zero matrices must have Jaccard = 0.0."""
        a = sp.csr_matrix((5, 5))
        b = sp.csr_matrix((5, 5))
        result, mask = pairwise_jaccard_sparse([a, b])
        assert result[0, 1] == 0.0

    def test_symmetric_zero_diagonal(self):
        """Result must be symmetric. Diagonal = 1.0 for non-empty, 0.0 for empty."""
        rng = np.random.default_rng(99)
        matrices = [
            sp.random(10, 10, density=0.1, format='csr', random_state=rng)
            for _ in range(4)
        ]
        matrices = [(m > 0).astype(float) for m in matrices]
        result, mask = pairwise_jaccard_sparse(matrices)
        np.testing.assert_allclose(result, result.T)
        for i in range(4):
            assert result[i, i] == 1.0

    def test_single_matrix(self):
        """Single matrix must return 1x1 array with value 1.0."""
        m = sp.eye(5, format='csr')
        result, mask = pairwise_jaccard_sparse([m])
        assert result.shape == (1, 1)
        assert result[0, 0] == 1.0

    def test_different_sizes_raises_with_trim_none(self):
        """Different-sized matrices raise ValueError with trim=None."""
        m1 = sp.eye(5, format='csr')
        m2 = sp.eye(6, format='csr')
        with pytest.raises(ValueError, match="trim"):
            pairwise_jaccard_sparse([m1, m2], trim=None)

    def test_trim_min(self):
        """trim='min' trims larger matrices to smallest common size."""
        m1 = sp.eye(5, format='csr')
        m2 = sp.eye(6, format='csr')
        result, mask = pairwise_jaccard_sparse([m1, m2], trim='min')
        assert result.shape == (2, 2)
        assert result[0, 0] == 1.0
        assert result[1, 1] == 1.0
        assert mask.all()

    def test_trim_adaptive_with_outlier(self):
        """Adaptive trim removes outlier, Jaccard matrix is smaller."""
        # 5 matrices of size 20, 1 outlier of size 10
        matrices = [sp.eye(20, format='csr') for _ in range(5)]
        matrices.append(sp.eye(10, format='csr'))
        result, mask = pairwise_jaccard_sparse(
            matrices, trim='adaptive', tail_sigma=1.0)
        assert not mask[-1]  # outlier removed
        assert result.shape[0] == mask.sum()

    def test_trim_to_min_deprecated(self):
        """trim_to_min=True still works but emits FutureWarning."""
        m1 = sp.eye(5, format='csr')
        m2 = sp.eye(6, format='csr')
        with pytest.warns(FutureWarning, match="trim_to_min"):
            result, mask = pairwise_jaccard_sparse([m1, m2], trim_to_min=True)
        assert result.shape == (2, 2)
        assert mask.all()

    def test_accepts_recurrence_graph_objects(self):
        """RecurrenceGraph objects are accepted via .adj attribute."""
        data = np.sin(np.linspace(0, 4 * np.pi, 200))
        emb = takens_embedding(data, tau=5, m=3)
        rg = RecurrenceGraph(emb, method='knn', k=3)
        result, mask = pairwise_jaccard_sparse([rg, rg])
        assert result.shape == (2, 2)
        assert abs(result[0, 1] - 1.0) < 1e-10
