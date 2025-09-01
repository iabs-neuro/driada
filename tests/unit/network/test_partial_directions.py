"""Test partial directions support in Network construction."""

import numpy as np
import scipy.sparse as sp
import networkx as nx
import pytest

from driada.network.net_base import (
    Network,
    calculate_directionality_fraction,
    select_construction_pipeline,
)


class TestPartialDirections:
    """Test suite for partial directions support."""

    def test_calculate_directionality_fraction_undirected(self):
        """Test directionality calculation for fully undirected graph."""
        # Create a symmetric adjacency matrix
        A = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0]
        ])
        
        frac = calculate_directionality_fraction(A)
        assert frac == 0.0, "Symmetric matrix should have directionality 0.0"
        
        # Test with sparse matrix
        A_sparse = sp.csr_matrix(A)
        frac_sparse = calculate_directionality_fraction(A_sparse)
        assert frac_sparse == 0.0, "Sparse symmetric matrix should have directionality 0.0"

    def test_calculate_directionality_fraction_directed(self):
        """Test directionality calculation for fully directed graph."""
        # Create a fully asymmetric adjacency matrix
        A = np.array([
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])
        
        frac = calculate_directionality_fraction(A)
        assert frac == 1.0, "Fully asymmetric matrix should have directionality 1.0"

    def test_calculate_directionality_fraction_partial(self):
        """Test directionality calculation for partially directed graph."""
        # Create a partially directed adjacency matrix
        # Has both symmetric edges (0-1) and asymmetric edges (0->2, 1->3)
        A = np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        
        frac = calculate_directionality_fraction(A)
        # Total edges: 4
        # Symmetric edges: 2 (between 0-1)
        # Directed edges: 2 (0->2, 1->3)
        # Fraction: 2/4 = 0.5
        assert frac == 0.5, f"Partially directed matrix should have directionality 0.5, got {frac}"

    def test_calculate_directionality_fraction_empty(self):
        """Test directionality calculation for empty graph."""
        A = np.zeros((4, 4))
        frac = calculate_directionality_fraction(A)
        assert frac == 0.0, "Empty matrix should have directionality 0.0"

    def test_calculate_directionality_fraction_self_loops(self):
        """Test that self-loops are ignored in directionality calculation."""
        # Matrix with self-loops
        A = np.array([
            [1, 1, 0, 0],  # Self-loop at 0
            [1, 1, 0, 0],  # Self-loop at 1
            [0, 0, 1, 1],  # Self-loop at 2
            [0, 0, 1, 1]   # Self-loop at 3
        ])
        
        frac = calculate_directionality_fraction(A)
        # Should only consider off-diagonal edges
        assert frac == 0.0, "Self-loops should be ignored"

    def test_select_construction_pipeline_with_adj(self):
        """Test pipeline selection with adjacency matrix."""
        # Undirected
        A_undirected = np.array([[0, 1], [1, 0]])
        pipeline, directionality = select_construction_pipeline(A_undirected, None)
        assert pipeline == "adj"
        assert directionality == 0.0
        
        # Directed
        A_directed = np.array([[0, 1], [0, 0]])
        pipeline, directionality = select_construction_pipeline(A_directed, None)
        assert pipeline == "adj"
        assert directionality == 1.0
        
        # Partially directed
        A_partial = np.array([
            [0, 1, 1],
            [1, 0, 0],
            [0, 0, 0]
        ])
        pipeline, directionality = select_construction_pipeline(A_partial, None)
        assert pipeline == "adj"
        assert 0.0 < directionality < 1.0

    def test_select_construction_pipeline_with_graph(self):
        """Test pipeline selection with NetworkX graph."""
        # Undirected graph
        G_undirected = nx.Graph()
        G_undirected.add_edges_from([(0, 1), (1, 2)])
        pipeline, directionality = select_construction_pipeline(None, G_undirected)
        assert pipeline == "graph"
        assert directionality == 0.0
        
        # Directed graph
        G_directed = nx.DiGraph()
        G_directed.add_edges_from([(0, 1), (1, 2)])
        pipeline, directionality = select_construction_pipeline(None, G_directed)
        assert pipeline == "graph"
        assert directionality == 1.0

    def test_network_init_with_partial_directions(self):
        """Test Network initialization with partial directions."""
        # Create a partially directed adjacency matrix
        A = np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])
        
        # Initialize network without specifying directed parameter
        # Set real_world=False to allow fractional directionality
        net = Network(adj=sp.csr_matrix(A), real_world=False)
        
        # Should detect partial directionality
        assert net.directed == True, f"Network should be marked as directed, got {net.directed}"
        
        # Check the calculated directionality fraction
        assert 0.0 < net._calculated_directionality < 1.0, f"Should calculate fractional directionality, got {net._calculated_directionality}"
        
        # Calculate expected value
        # Total edges: 5
        # Symmetric edges: 2 (between 0-1)
        # Directed edges: 3 (0->2, 1->3, 2->3)
        expected = 3.0 / 5.0
        assert abs(net._calculated_directionality - expected) < 1e-10, f"Expected directionality {expected}, got {net._calculated_directionality}"

    def test_network_init_override_directionality(self):
        """Test that user can override calculated directionality."""
        # Create a partially directed adjacency matrix
        A = np.array([
            [0, 1, 1],
            [1, 0, 0],
            [0, 0, 0]
        ])
        
        # Override with user-specified value
        net = Network(adj=sp.csr_matrix(A), directed=0.75, real_world=False)
        assert net.directed == 0.75, "User-specified directionality should override calculation"

    def test_backward_compatibility(self):
        """Test backward compatibility with binary directed detection."""
        # Fully symmetric matrix
        A_sym = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        net = Network(adj=sp.csr_matrix(A_sym))
        assert net.directed == 0.0 or net.directed == False
        
        # Fully asymmetric matrix
        A_asym = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        net = Network(adj=sp.csr_matrix(A_asym))
        assert net.directed == 1.0 or net.directed == True

    def test_real_world_network_constraint(self):
        """Test that real-world networks still require binary directionality."""
        # Partially directed matrix
        A = np.array([
            [0, 1, 1],
            [1, 0, 0],
            [0, 0, 0]
        ])
        
        # Real-world network should convert fractional directionality to boolean
        net = Network(adj=sp.csr_matrix(A), real_world=True)
        # Should be directed (True) since it has some directed edges
        assert net.directed == True
        # But the calculated directionality should still be fractional
        assert 0.0 < net._calculated_directionality < 1.0

    def test_complex_partial_graph(self):
        """Test with a more complex partially directed graph."""
        # Create a graph with:
        # - Some bidirectional edges (symmetric)
        # - Some unidirectional edges (asymmetric)
        # - Different edge weights
        A = np.array([
            [0, 2, 0, 0, 1, 0],
            [2, 0, 3, 0, 0, 0],
            [0, 0, 0, 4, 0, 0],
            [0, 0, 4, 0, 5, 0],
            [0, 0, 0, 0, 0, 6],
            [0, 0, 0, 0, 0, 0]
        ])
        
        net = Network(adj=sp.csr_matrix(A), weighted=True, real_world=False)
        
        # Calculate expected directionality
        # From the output, we have:
        # (0,1)=2, (1,0)=2 - symmetric pair  
        # (0,4)=1 - directed edge (no reciprocal from 4 to 0)
        # (1,2)=3 - directed edge (no reciprocal from 2 to 1)
        # (2,3)=4, (3,2)=4 - symmetric pair
        # (3,4)=5 - directed edge (no reciprocal from 4 to 3)
        # (4,5)=6 - directed edge (no reciprocal from 5 to 4)
        # Total edges: 8
        # Symmetric edges: 4 (two symmetric pairs)
        # Directed edges: 4 (0->4, 1->2, 3->4, 4->5)
        # Fraction: 4/8 = 0.5
        expected = 0.5
        assert abs(net._calculated_directionality - expected) < 1e-10, f"Expected directionality {expected}, got {net._calculated_directionality}"
        assert net.directed == True  # Should be marked as directed
        assert net.weighted == True