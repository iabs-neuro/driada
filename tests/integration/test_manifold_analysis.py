"""Tests for manifold analysis functions.

These tests verify the mathematical properties and correctness of
manifold analysis functions that users rely on.
"""

import pytest
import numpy as np

from driada.integration.manifold_analysis import (
    get_functional_organization,
    compare_embeddings,
)
from driada.experiment.synthetic import generate_circular_manifold_exp


class TestGetFunctionalOrganization:
    """Test get_functional_organization function."""

    @pytest.fixture
    def experiment_with_pca(self):
        """Create experiment with PCA embedding."""
        np.random.seed(42)
        exp = generate_circular_manifold_exp(
            n_neurons=20,
            duration=30,
            fps=10.0,
            kappa=4.0,
            verbose=False,
        )
        exp.create_embedding('pca', n_components=5, verbose=False)
        return exp

    def test_component_importance_sums_to_one(self, experiment_with_pca):
        """Component importance should sum to 1.0 (normalized variance)."""
        org = get_functional_organization(experiment_with_pca, 'pca')

        importance_sum = np.sum(org['component_importance'])
        assert importance_sum == pytest.approx(1.0, abs=1e-10)

    def test_component_importance_non_negative(self, experiment_with_pca):
        """Component importance values should all be non-negative."""
        org = get_functional_organization(experiment_with_pca, 'pca')

        assert np.all(org['component_importance'] >= 0)

    def test_n_components_matches_embedding(self, experiment_with_pca):
        """n_components should match the embedding dimensionality."""
        org = get_functional_organization(experiment_with_pca, 'pca')

        assert org['n_components'] == 5

    def test_n_neurons_used_matches_experiment(self, experiment_with_pca):
        """n_neurons_used should match the experiment's neuron count."""
        org = get_functional_organization(experiment_with_pca, 'pca')

        assert org['n_neurons_used'] == experiment_with_pca.n_cells

    def test_neuron_indices_correct_length(self, experiment_with_pca):
        """neuron_indices should have correct length."""
        org = get_functional_organization(experiment_with_pca, 'pca')

        assert len(org['neuron_indices']) == experiment_with_pca.n_cells

    def test_works_with_tsne_embedding(self):
        """Should work with t-SNE embedding."""
        np.random.seed(42)
        exp = generate_circular_manifold_exp(
            n_neurons=15,
            duration=30,
            fps=10.0,
            kappa=4.0,
            verbose=False,
        )
        exp.create_embedding('tsne', n_components=2, perplexity=5, random_state=42)

        org = get_functional_organization(exp, 'tsne')

        assert org['n_components'] == 2
        assert np.sum(org['component_importance']) == pytest.approx(1.0, abs=1e-6)

    def test_raises_keyerror_for_missing_method(self, experiment_with_pca):
        """Should raise KeyError for non-existent embedding method."""
        with pytest.raises(KeyError):
            get_functional_organization(experiment_with_pca, 'nonexistent_method')

    def test_raises_typeerror_for_invalid_intense_results(self, experiment_with_pca):
        """Should raise TypeError if intense_results is not IntenseResults object."""
        with pytest.raises(TypeError, match="must be an IntenseResults object"):
            get_functional_organization(
                experiment_with_pca, 'pca',
                intense_results={"fake": "data"}
            )


class TestCompareEmbeddings:
    """Test compare_embeddings function."""

    @pytest.fixture
    def experiment_with_multiple_embeddings(self):
        """Create experiment with PCA and t-SNE embeddings."""
        np.random.seed(42)
        exp = generate_circular_manifold_exp(
            n_neurons=20,
            duration=50,
            fps=10.0,
            kappa=4.0,
            verbose=False,
        )
        exp.create_embedding('pca', n_components=3, verbose=False)
        exp.create_embedding('tsne', n_components=2, perplexity=10, random_state=42)
        return exp

    def test_methods_list_contains_valid_methods(self, experiment_with_multiple_embeddings):
        """Result should list valid methods that were compared."""
        comparison = compare_embeddings(
            experiment_with_multiple_embeddings,
            ['pca', 'tsne']
        )

        assert set(comparison['methods']) == {'pca', 'tsne'}

    def test_n_components_per_method(self, experiment_with_multiple_embeddings):
        """n_components should be correct for each method."""
        comparison = compare_embeddings(
            experiment_with_multiple_embeddings,
            ['pca', 'tsne']
        )

        assert comparison['n_components']['pca'] == 3
        assert comparison['n_components']['tsne'] == 2

    def test_handles_single_embedding_gracefully(self):
        """Should handle case with only one valid embedding."""
        np.random.seed(42)
        exp = generate_circular_manifold_exp(
            n_neurons=15,
            duration=30,
            fps=10.0,
            verbose=False,
        )
        exp.create_embedding('pca', n_components=3, verbose=False)

        # Request pca and nonexistent method
        comparison = compare_embeddings(exp, ['pca', 'umap'])

        # Should only have pca
        assert comparison['methods'] == ['pca']
        assert 'pca' in comparison['n_components']

    def test_raises_error_for_empty_method_list(self, experiment_with_multiple_embeddings):
        """Should raise ValueError for empty method list."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compare_embeddings(experiment_with_multiple_embeddings, [])

    def test_raises_error_for_non_list_methods(self, experiment_with_multiple_embeddings):
        """Should raise TypeError if method_names is not a list."""
        with pytest.raises(TypeError, match="must be a list"):
            compare_embeddings(experiment_with_multiple_embeddings, 'pca')

    def test_raises_error_when_no_valid_embeddings(self):
        """Should raise ValueError when no valid embeddings found."""
        np.random.seed(42)
        exp = generate_circular_manifold_exp(
            n_neurons=15,
            duration=30,
            fps=10.0,
            verbose=False,
        )
        # Don't create any embeddings

        with pytest.raises(ValueError, match="No valid embeddings found"):
            compare_embeddings(exp, ['pca', 'tsne'])

    def test_mean_components_per_neuron_without_selectivity(self, experiment_with_multiple_embeddings):
        """Without selectivity data, mean_components_per_neuron should be 0."""
        comparison = compare_embeddings(
            experiment_with_multiple_embeddings,
            ['pca', 'tsne']
        )

        # Without intense_results, these should be 0
        assert comparison['mean_components_per_neuron']['pca'] == 0
        assert comparison['mean_components_per_neuron']['tsne'] == 0

    def test_n_functional_clusters_without_selectivity(self, experiment_with_multiple_embeddings):
        """Without selectivity data, n_functional_clusters should be 0."""
        comparison = compare_embeddings(
            experiment_with_multiple_embeddings,
            ['pca', 'tsne']
        )

        assert comparison['n_functional_clusters']['pca'] == 0
        assert comparison['n_functional_clusters']['tsne'] == 0


class TestOrganizationWithSubsetNeurons:
    """Test organization analysis when embedding uses subset of neurons."""

    def test_neuron_indices_reflect_subset(self):
        """neuron_indices should reflect subset used in embedding."""
        np.random.seed(42)
        exp = generate_circular_manifold_exp(
            n_neurons=30,
            duration=30,
            fps=10.0,
            verbose=False,
        )

        # Create embedding - it should use all neurons by default
        exp.create_embedding('pca', n_components=3, verbose=False)

        org = get_functional_organization(exp, 'pca')

        # By default, should use all neurons
        assert org['n_neurons_used'] == 30
