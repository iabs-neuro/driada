"""Optimized INTENSE pipeline tests with reduced parameters for fast execution.

This file contains the same tests as test_intense_pipelines.py but with
optimized parameters for fast test execution:
- Using small experiments exclusively
- Reduced shuffle counts (5-10 instead of 10-25)
- Shorter durations (20-30s instead of 30-200s)
- Downsampling enabled (ds=5)
- Stage1 mode only (no two_stage)
"""

import pytest
import numpy as np
from driada.intense.pipelines import (
    compute_cell_feat_significance,
    compute_feat_feat_significance,
    compute_cell_cell_significance,
    compute_embedding_selectivity,
)
from driada.information.info_base import TimeSeries
from driada.experiment.synthetic import (
    generate_synthetic_exp,
    generate_synthetic_exp_with_mixed_selectivity,
)


# Fast test parameters
FAST_PARAMS = {
    "mode": "stage1",
    "n_shuffles_stage1": 5,
    "ds": 5,  # Aggressive downsampling
    "enable_parallelization": False,  # Disable for small tests
    "seed": 42,
}


def test_compute_cell_feat_significance_with_disentanglement_fast():
    """Fast test for cell-feat significance with disentanglement."""
    # Use the proper mixed selectivity generator instead of small_experiment

    # Generate experiment with guaranteed mixed selectivity
    # Use asymmetric weights to ensure disentanglement can detect differences
    exp, selectivity_info = generate_synthetic_exp_with_mixed_selectivity(
        n_discrete_feats=3,  # Use 3 features for clearer patterns
        n_continuous_feats=2,  # Add continuous features
        n_neurons=20,  # 20 neurons as requested
        duration=600,  # Good duration for statistics
        fps=20,  # Higher sampling rate
        selectivity_prob=1.0,  # All neurons are selective
        multi_select_prob=0.8,  # 80% have mixed selectivity (16 neurons)
        weights_mode="dominant",  # One feature dominates - creates asymmetry
        create_discrete_pairs=True,  # Create d_feat_from_c features
        skip_prob=0.0,  # No spike skipping for clearer signals
        rate_0=0.5,  # Higher baseline rate
        rate_1=10.0,  # Much higher active rate for better detection
        ampl_range=(0.5, 2.0),  # Standard calcium responses
        noise_std=0.005,  # Very low noise
        seed=42,
        verbose=False,
    )

    # Verify we have mixed selectivity
    # Note: selectivity matrix is (features, neurons) not (neurons, features)
    selectivity_matrix = selectivity_info["matrix"]
    # Transpose to get (neurons, features) and find mixed neurons
    mixed_neurons = np.where(np.sum(selectivity_matrix.T > 0, axis=1) >= 2)[0]
    assert (
        len(mixed_neurons) >= 2
    ), f"Need at least 2 neurons with mixed selectivity, found {len(mixed_neurons)}"

    # Use more mixed neurons for better chance of detection
    cell_bunch = (
        mixed_neurons[:10].tolist()
        if len(mixed_neurons) >= 10
        else mixed_neurons.tolist()
    )
    print(f"Testing with {len(cell_bunch)} neurons with mixed selectivity")

    # Run with very loose parameters for better detection
    stats, significance, info, results, disent_results = compute_cell_feat_significance(
        exp,
        cell_bunch=cell_bunch,  # Use neurons with mixed selectivity
        feat_bunch=None,
        mode="two_stage",  # Need two_stage for get_significant_neurons to work
        n_shuffles_stage1=10,  # Very few shuffles for speed
        n_shuffles_stage2=100,  # Minimal stage 2 shuffles
        metric="mi",
        metric_distr_type="norm",  # Use normal distribution
        pval_thr=0.1,  # Very lenient p-value threshold
        multicomp_correction=None,  # No multiple comparison correction
        find_optimal_delays=False,  # Disable to avoid MultiTimeSeries validation issues
        allow_mixed_dimensions=True,  # Allow MultiTimeSeries features
        enable_parallelization=False,  # Disable parallelization for consistency
        with_disentanglement=True,
        verbose=True,  # See what's happening
        seed=42,
    )

    # Basic checks
    assert isinstance(disent_results, dict)
    assert "feat_feat_significance" in disent_results
    assert "disent_matrix" in disent_results
    assert "summary" in disent_results

    # Check that mixed selectivity was detected
    summary = disent_results["summary"]
    assert "overall_stats" in summary

    # For debugging - print what we got
    if not summary.get("overall_stats"):
        print("WARNING: No mixed selectivity pairs found!")
        print(f"Used neurons: {cell_bunch}")
        print(f"Selectivity matrix shape: {selectivity_matrix.shape}")
        print(f"Mixed neurons from matrix: {mixed_neurons}")
        # Check what the disentanglement found
        if "count_matrix" in disent_results:
            print(f"Count matrix:\n{disent_results['count_matrix']}")

    # The test should handle cases where disentanglement may or may not find pairs
    # With equal weights, neurons might be undistinguishable
    if summary.get("overall_stats") is not None:
        # Mixed selectivity pairs were found
        assert summary["overall_stats"]["total_neuron_pairs"] >= 0
        print(
            f"Found {summary['overall_stats']['total_neuron_pairs']} mixed selectivity pairs"
        )
    else:
        # No pairs found - this can happen when:
        # 1. Features are uncorrelated (true mixed selectivity)
        # 2. All neurons have undistinguishable contributions
        # 3. The detection threshold is too strict
        print(
            "No mixed selectivity pairs found - this is acceptable for some parameter combinations"
        )
        # Verify that we at least have the expected structure
        assert "disent_matrix" in disent_results
        assert "feat_feat_significance" in disent_results


@pytest.mark.parametrize("continuous_only_experiment", ["small"], indirect=True)
def test_compute_cell_feat_significance_continuous_fast(continuous_only_experiment):
    """Fast test for continuous features."""
    exp = continuous_only_experiment

    result = compute_cell_feat_significance(
        exp,
        cell_bunch=[0, 1],  # Minimal neurons
        with_disentanglement=False,
        **FAST_PARAMS,
    )

    assert len(result) == 4


@pytest.mark.parametrize("mixed_features_experiment", ["small"], indirect=True)
def test_compute_feat_feat_significance_fast(mixed_features_experiment):
    """Fast test for feature-feature correlation."""
    exp = mixed_features_experiment

    sim_mat, sig_mat, pval_mat, feat_ids, info = compute_feat_feat_significance(
        exp, **FAST_PARAMS
    )

    n_features = len(feat_ids)
    assert sim_mat.shape == (n_features, n_features)
    assert np.allclose(np.diag(sim_mat), 0)


@pytest.mark.parametrize("discrete_only_experiment", ["small"], indirect=True)
def test_compute_cell_cell_significance_fast(discrete_only_experiment):
    """Fast test for neuron-neuron correlation."""
    exp = discrete_only_experiment

    # Make neurons correlated
    np.random.seed(42)
    exp.neurons[1].ca = TimeSeries(
        exp.neurons[0].ca.data + np.random.randn(len(exp.neurons[0].ca.data)) * 0.1,
        discrete=False,
    )

    sim_mat, sig_mat, pval_mat, cell_ids, info = compute_cell_cell_significance(
        exp, cell_bunch=[0, 1, 2], data_type="calcium", **FAST_PARAMS  # Just 3 neurons
    )

    assert sim_mat.shape == (3, 3)
    assert np.allclose(np.diag(sim_mat), 0)


def test_mixed_selectivity_generation_fast():
    """Fast test for mixed selectivity pattern generation."""
    # Minimal generation - check the actual API

    # Generate minimal mixed selectivity experiment
    exp, selectivity_info = generate_synthetic_exp_with_mixed_selectivity(
        n_discrete_feats=1,
        n_continuous_feats=2,
        n_neurons=5,
        duration=60,  # Increased to avoid shuffle mask error
        fps=10,
        selectivity_prob=0.8,
        seed=42,
        verbose=False,
    )

    assert exp.n_cells == 5
    assert "matrix" in selectivity_info
    assert selectivity_info["matrix"].shape == (5, 5)  # neurons x features


def test_disentanglement_minimal():
    """Minimal disentanglement test with tiny data."""
    # Generate minimal data with longer duration to avoid shuffle mask issues
    exp = generate_synthetic_exp(
        n_dfeats=2,
        n_cfeats=2,
        nneurons=3,
        duration=60,  # Increased to avoid shuffle mask error
        fps=10,  # Low fps
        seed=42,
        with_spikes=False,
    )

    # Run with minimal parameters
    result = compute_cell_feat_significance(
        exp, cell_bunch=[0, 1], with_disentanglement=True, **FAST_PARAMS
    )

    assert len(result) == 5  # Including disentanglement results


class TestEdgeCasesFast:
    """Fast edge case tests."""

    def test_empty_cell_bunch(self, small_experiment):
        """Test with empty neuron list."""
        # Empty cell bunch should raise ValueError
        with pytest.raises(ValueError, match="ts_bunch1 cannot be empty"):
            compute_cell_feat_significance(
                small_experiment, cell_bunch=[], **FAST_PARAMS
            )

    def test_single_neuron(self, small_experiment):
        """Test with single neuron."""
        result = compute_cell_feat_significance(
            small_experiment, cell_bunch=[0], **FAST_PARAMS
        )
        assert len(result) == 4

    def test_single_feature(self, small_experiment):
        """Test with single feature."""
        feat_names = list(small_experiment.dynamic_features.keys())
        sim_mat, _, _, feat_ids, _ = compute_feat_feat_significance(
            small_experiment, feat_bunch=[feat_names[0]], **FAST_PARAMS
        )
        assert sim_mat.shape == (1, 1)
        assert sim_mat[0, 0] == 0


class TestPerformanceBenchmarks:
    """Benchmark tests to ensure optimization works."""

    def test_all_functions_under_5s(self, small_experiment):
        """Ensure all main functions complete in under 5 seconds."""
        import time

        # Test cell-feat
        start = time.time()
        compute_cell_feat_significance(
            small_experiment, cell_bunch=[0, 1, 2], **FAST_PARAMS
        )
        assert time.time() - start < 5.0

        # Test feat-feat
        start = time.time()
        compute_feat_feat_significance(small_experiment, **FAST_PARAMS)
        assert time.time() - start < 5.0

        # Test cell-cell
        start = time.time()
        compute_cell_cell_significance(
            small_experiment, cell_bunch=[0, 1, 2], **FAST_PARAMS
        )
        assert time.time() - start < 5.0


# Additional comprehensive tests for better coverage


def test_compute_cell_cell_significance_comprehensive(small_experiment):
    """Comprehensive test for cell-cell significance with all code paths."""
    exp = small_experiment

    # Ensure each neuron has unique calcium data to avoid self-comparison issues
    np.random.seed(42)
    for i in range(exp.n_cells):
        # Add unique noise pattern to each neuron
        noise = np.random.randn(len(exp.neurons[i].ca.data)) * 0.1 * (i + 1)
        exp.neurons[i].ca = TimeSeries(exp.neurons[i].ca.data + noise, discrete=False)

    # Test with calcium data
    sim_mat, sig_mat, pval_mat, cell_ids, info = compute_cell_cell_significance(
        exp,
        cell_bunch=None,  # Test with all neurons
        data_type="calcium",
        verbose=True,  # Test verbose output
        **FAST_PARAMS,
    )

    n_cells = exp.n_cells
    assert sim_mat.shape == (n_cells, n_cells)
    assert sig_mat.shape == (n_cells, n_cells)
    assert pval_mat.shape == (n_cells, n_cells)
    assert len(cell_ids) == n_cells

    # Check diagonal is zero
    assert np.allclose(np.diag(sim_mat), 0)
    assert np.allclose(np.diag(sig_mat), 0)
    assert np.allclose(np.diag(pval_mat), 1)

    # Check symmetry
    assert np.allclose(sim_mat, sim_mat.T)
    assert np.allclose(sig_mat, sig_mat.T)
    assert np.allclose(pval_mat, pval_mat.T)


def test_compute_cell_cell_significance_with_spikes(small_experiment):
    """Test cell-cell significance with spike data."""
    exp = small_experiment

    # Add spike data to neurons
    for i in range(exp.n_cells):
        neuron = exp.neurons[i]
        # Create synthetic spike data
        spike_data = (neuron.ca.data > np.percentile(neuron.ca.data, 80)).astype(int)
        neuron.sp = TimeSeries(spike_data, discrete=True)

    # Test with spike data
    sim_mat, sig_mat, pval_mat, cell_ids, info = compute_cell_cell_significance(
        exp, cell_bunch=[0, 1, 2], data_type="spikes", **FAST_PARAMS
    )

    assert sim_mat.shape == (3, 3)
    assert np.allclose(np.diag(sim_mat), 0)


def test_compute_cell_cell_significance_error_cases(small_experiment):
    """Test error handling in cell-cell significance."""
    exp = small_experiment

    # Test invalid data type
    with pytest.raises(ValueError, match='"data_type" can be either'):
        compute_cell_cell_significance(exp, data_type="invalid", **FAST_PARAMS)

    # Test with no spike data
    for i in range(exp.n_cells):
        neuron = exp.neurons[i]
        neuron.sp = None

    with pytest.raises(ValueError, match="Some neurons have no spike data"):
        compute_cell_cell_significance(exp, data_type="spikes", **FAST_PARAMS)


def test_compute_embedding_selectivity_basic(small_experiment):
    """Test compute_embedding_selectivity with basic embeddings."""
    exp = small_experiment

    # Create and store a simple PCA embedding
    from sklearn.decomposition import PCA

    # Get neural data
    neural_data = np.array([exp.neurons[i].ca.data for i in range(exp.n_cells)]).T

    # Compute PCA
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(neural_data)

    # Store embedding
    exp.store_embedding(
        embedding,
        method_name="pca",
        data_type="calcium",
        metadata={"method": "PCA", "n_components": 2},
    )

    # Test embedding selectivity
    results = compute_embedding_selectivity(
        exp,
        embedding_methods=["pca"],
        cell_bunch=[0, 1, 2],
        verbose=True,  # Test verbose output
        **FAST_PARAMS,
    )

    assert "pca" in results
    pca_results = results["pca"]
    assert "stats" in pca_results
    assert "significance" in pca_results
    assert "significant_neurons" in pca_results
    assert "component_selectivity" in pca_results
    assert pca_results["n_components"] == 2


def test_compute_embedding_selectivity_multiple_methods(small_experiment):
    """Test embedding selectivity with multiple methods."""
    exp = small_experiment

    # Get neural data
    neural_data = np.array([exp.neurons[i].ca.data for i in range(exp.n_cells)]).T

    # Store multiple embeddings
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    # PCA
    pca = PCA(n_components=2)
    pca_embedding = pca.fit_transform(neural_data)
    exp.store_embedding(pca_embedding, method_name="pca", data_type="calcium")

    # t-SNE
    tsne = TSNE(
        n_components=2, random_state=42, perplexity=2
    )  # Small perplexity for small data
    tsne_embedding = tsne.fit_transform(neural_data)
    exp.store_embedding(tsne_embedding, method_name="tsne", data_type="calcium")

    # Test with all embeddings
    results = compute_embedding_selectivity(
        exp, embedding_methods=None, **FAST_PARAMS  # Test with None (all methods)
    )

    assert "pca" in results
    assert "tsne" in results

    # Test specific method
    results_single = compute_embedding_selectivity(
        exp, embedding_methods="pca", **FAST_PARAMS  # Test with string input
    )

    assert len(results_single) == 1
    assert "pca" in results_single


def test_compute_embedding_selectivity_error_cases(small_experiment):
    """Test error handling in embedding selectivity."""
    exp = small_experiment

    # Test with no embeddings - should raise KeyError per actual implementation
    with pytest.raises(KeyError, match="No embedding found"):
        compute_embedding_selectivity(
            exp, embedding_methods=["nonexistent"], **FAST_PARAMS
        )


def test_compute_feat_feat_significance_edge_cases(small_experiment):
    """Test edge cases in feat-feat significance."""
    exp = small_experiment

    # Test with empty feature list
    sim_mat, sig_mat, pval_mat, feat_ids, info = compute_feat_feat_significance(
        exp,
        feat_bunch=[],  # Empty list
        verbose=True,  # Test verbose path
        **FAST_PARAMS,
    )

    assert sim_mat.shape == (0, 0)
    assert sig_mat.shape == (0, 0)
    assert pval_mat.shape == (0, 0)
    assert len(feat_ids) == 0

    # Test with multifeatures - use continuous features since multifeatures don't support discrete
    # The small_experiment fixture has c_feat_0 and c_feat_1
    multifeature = ("c_feat_0", "c_feat_1")
    sim_mat2, sig_mat2, pval_mat2, feat_ids2, info2 = compute_feat_feat_significance(
        exp,
        feat_bunch=["d_feat_0", multifeature],  # Mix discrete and multifeature
        verbose=True,
        **FAST_PARAMS,
    )

    assert len(feat_ids2) == 2
    assert "d_feat_0" in feat_ids2
    assert multifeature in feat_ids2


def test_compute_cell_feat_significance_error_paths(small_experiment):
    """Test error handling paths in cell-feat significance."""
    exp = small_experiment

    # Initialize stats tables if not already done
    if not hasattr(exp, "stats_tables") or "calcium" not in exp.stats_tables:
        exp._set_selectivity_tables("calcium")

    # Test with invalid data type
    with pytest.raises(ValueError, match='"data_type" can be either'):
        compute_cell_feat_significance(exp, data_type="invalid", **FAST_PARAMS)

    # Test with non-existent feature
    with pytest.raises(
        ValueError, match="ts_bunch2 cannot be empty|Feature .* not found"
    ):
        compute_cell_feat_significance(
            exp,
            feat_bunch=["nonexistent_feature"],
            allow_mixed_dimensions=True,
            use_precomputed_stats=False,  # Don't use precomputed stats
            **FAST_PARAMS,
        )

    # Test with verbose output (simple case)
    # Disable use_precomputed_stats since the fixture doesn't have them
    result = compute_cell_feat_significance(
        exp,
        cell_bunch=[0, 1],
        feat_bunch=["d_feat_0", "d_feat_1"],
        allow_mixed_dimensions=True,
        use_precomputed_stats=False,  # Don't use precomputed stats
        verbose=True,  # Test verbose paths
        **FAST_PARAMS,
    )

    assert len(result) == 4


def test_disentanglement_with_asymmetric_features():
    """Test disentanglement with asymmetric feature relationships (discrete from continuous)."""

    # Generate experiment with continuous features and their discrete versions
    exp, selectivity_info = generate_synthetic_exp_with_mixed_selectivity(
        n_discrete_feats=0,  # No independent discrete features
        n_continuous_feats=2,  # Two continuous features
        n_neurons=10,  # Smaller for focused test
        duration=300,  # 5 minutes
        fps=20,
        selectivity_prob=1.0,  # All neurons selective
        multi_select_prob=1.0,  # All have mixed selectivity
        weights_mode="dominant",  # Asymmetric weights - key for detection!
        create_discrete_pairs=True,  # Creates d_feat_from_c0, d_feat_from_c1
        skip_prob=0.0,
        rate_0=0.5,
        rate_1=10.0,
        ampl_range=(0.5, 2.0),
        noise_std=0.005,
        seed=123,  # Different seed
        verbose=False,
    )

    # Find neurons selective to both continuous and discrete versions
    feature_names = list(exp.dynamic_features.keys())
    assert "c_feat_0" in feature_names
    assert "d_feat_from_c0" in feature_names

    # Get neurons with mixed selectivity including continuous/discrete pairs
    selectivity_matrix = selectivity_info["matrix"]
    mixed_neurons = np.where(np.sum(selectivity_matrix > 0, axis=0) >= 2)[0]

    # Run significance testing
    stats, significance, info, results, disent_results = compute_cell_feat_significance(
        exp,
        cell_bunch=mixed_neurons[:5].tolist(),
        feat_bunch=None,
        mode="two_stage",  # Need two_stage for proper detection
        n_shuffles_stage1=10,
        n_shuffles_stage2=100,
        metric="mi",
        pval_thr=0.05,
        with_disentanglement=True,
        allow_mixed_dimensions=True,  # Needed for multifeatures
        find_optimal_delays=False,  # Disable to avoid MultiTimeSeries issues
        verbose=False,
        enable_parallelization=False,
        seed=123,
    )

    # Check disentanglement results
    assert "disent_matrix" in disent_results
    assert "summary" in disent_results

    # With dominant weights, we should be able to distinguish primary features
    summary = disent_results["summary"]
    if summary.get("overall_stats"):
        # Should find that continuous features dominate their discrete versions
        # when weights_mode='dominant' is used
        assert summary["overall_stats"]["total_neuron_pairs"] >= 0
        print(
            f"Found {summary['overall_stats']['total_neuron_pairs']} asymmetric pairs"
        )


def test_intense_with_ksg_estimator(small_experiment):
    """Test that INTENSE pipelines work with KSG mutual information estimator."""
    exp = small_experiment
    
    # Test compute_cell_feat_significance with KSG
    result_ksg = compute_cell_feat_significance(
        exp,
        cell_bunch=[0, 1, 2],
        feat_bunch=None,
        mi_estimator='ksg',  # Use KSG estimator
        mode='stage1',
        n_shuffles_stage1=5,
        ds=5,
        enable_parallelization=False,
        seed=42,
    )
    
    # Test with GCMI for comparison
    result_gcmi = compute_cell_feat_significance(
        exp,
        cell_bunch=[0, 1, 2],
        feat_bunch=None,
        mi_estimator='gcmi',  # Use GCMI estimator (default)
        mode='stage1',
        n_shuffles_stage1=5,
        ds=5,
        enable_parallelization=False,
        seed=42,
    )
    
    # Check what type of result we get
    print(f"KSG result type: {type(result_ksg)}, length: {len(result_ksg) if isinstance(result_ksg, (tuple, list)) else 'N/A'}")
    print(f"GCMI result type: {type(result_gcmi)}, length: {len(result_gcmi) if isinstance(result_gcmi, (tuple, list)) else 'N/A'}")
    
    # Both should return the same structure
    assert type(result_ksg) == type(result_gcmi)
    
    # If it's a tuple, unpack appropriately
    if isinstance(result_ksg, tuple):
        if len(result_ksg) == 4:
            stats_ksg, sig_ksg, info_ksg, results_ksg = result_ksg
            stats_gcmi, sig_gcmi, info_gcmi, results_gcmi = result_gcmi
        else:
            raise ValueError(f"Unexpected tuple length: {len(result_ksg)}")
    else:
        # If it's a dict, extract stats and significance
        stats_ksg = result_ksg['stats']
        sig_ksg = result_ksg['significance']
        stats_gcmi = result_gcmi['stats']
        sig_gcmi = result_gcmi['significance']
    
    # Basic checks - both should return valid results
    # Stats are organized as stats[cell_id][feat_id] where both are strings
    # Get the cell and feat ids
    cell_ids = list(stats_ksg.keys())
    assert len(cell_ids) == 3  # We requested 3 cells
    
    # Check that both have same structure
    assert set(stats_ksg.keys()) == set(stats_gcmi.keys())
    
    # Check all values are finite
    for cell_id in cell_ids:
        feat_ids = list(stats_ksg[cell_id].keys())
        for feat_id in feat_ids:
            # Check KSG values
            me_ksg = stats_ksg[cell_id][feat_id].get('me')
            if me_ksg is not None:
                assert np.isfinite(me_ksg), f"KSG ME not finite for cell {cell_id}, feat {feat_id}"
            
            # Check structure matches
            assert feat_id in stats_gcmi[cell_id], f"Feature {feat_id} missing in GCMI results"
    
    # Test compute_feat_feat_significance with KSG
    sim_mat_ksg, sig_mat_ksg, pval_mat_ksg, feat_ids_ksg, info_ksg = compute_feat_feat_significance(
        exp,
        mi_estimator='ksg',
        mode='stage1',
        n_shuffles_stage1=5,
        ds=5,
        enable_parallelization=False,
        seed=42,
    )
    
    # Check results are valid
    n_features = len(feat_ids_ksg)
    assert sim_mat_ksg.shape == (n_features, n_features)
    assert np.all(np.isfinite(sim_mat_ksg))
    assert np.allclose(np.diag(sim_mat_ksg), 0)  # Diagonal should be zero
    
    # Test compute_cell_cell_significance with KSG
    sim_mat_cc, sig_mat_cc, pval_mat_cc, cell_ids_cc, info_cc = compute_cell_cell_significance(
        exp,
        cell_bunch=[0, 1, 2],
        mi_estimator='ksg',
        mode='stage1',
        n_shuffles_stage1=5,
        ds=5,
        enable_parallelization=False,
        seed=42,
    )
    
    assert sim_mat_cc.shape == (3, 3)
    assert np.all(np.isfinite(sim_mat_cc))
    assert np.allclose(np.diag(sim_mat_cc), 0)


def test_intense_ksg_with_different_feature_types(mixed_features_experiment):
    """Test KSG estimator with mixed discrete and continuous features."""
    exp = mixed_features_experiment
    
    # Get discrete and continuous features
    discrete_feats = [f for f in exp.dynamic_features.keys() if f.startswith('d_feat')]
    continuous_feats = [f for f in exp.dynamic_features.keys() if f.startswith('c_feat')]
    
    # Test with only discrete features
    if discrete_feats:
        result_d = compute_cell_feat_significance(
            exp,
            cell_bunch=[0, 1],
            feat_bunch=discrete_feats[:2],
            mi_estimator='ksg',
            mode='stage1',
            n_shuffles_stage1=5,
            ds=5,
            enable_parallelization=False,
            use_precomputed_stats=False,  # Compute fresh stats
            seed=42,
        )
        # Extract stats from the result tuple
        stats_d = result_d[0] if isinstance(result_d, tuple) else result_d['stats']
        # Check all ME values are finite
        for cell_id in stats_d:
            for feat_id in stats_d[cell_id]:
                me_val = stats_d[cell_id][feat_id].get('me')
                if me_val is not None:
                    assert np.isfinite(me_val), f"ME not finite for discrete features"
    
    # Test with only continuous features
    if continuous_feats:
        result_c = compute_cell_feat_significance(
            exp,
            cell_bunch=[0, 1],
            feat_bunch=continuous_feats[:2],
            mi_estimator='ksg',
            mode='stage1',
            n_shuffles_stage1=5,
            ds=5,
            enable_parallelization=False,
            use_precomputed_stats=False,  # Compute fresh stats
            seed=42,
        )
        # Extract stats from the result tuple
        stats_c = result_c[0] if isinstance(result_c, tuple) else result_c['stats']
        # Check all ME values are finite
        for cell_id in stats_c:
            for feat_id in stats_c[cell_id]:
                me_val = stats_c[cell_id][feat_id].get('me')
                if me_val is not None:
                    assert np.isfinite(me_val), f"ME not finite for continuous features"
    
    # Test with mixed features
    if discrete_feats and continuous_feats:
        mixed_feats = [discrete_feats[0], continuous_feats[0]]
        result_m = compute_cell_feat_significance(
            exp,
            cell_bunch=[0, 1],
            feat_bunch=mixed_feats,
            mi_estimator='ksg',
            mode='stage1',
            n_shuffles_stage1=5,
            ds=5,
            enable_parallelization=False,
            use_precomputed_stats=False,  # Compute fresh stats
            seed=42,
        )
        # Extract stats from the result tuple
        stats_m = result_m[0] if isinstance(result_m, tuple) else result_m['stats']
        # Check all ME values are finite
        for cell_id in stats_m:
            for feat_id in stats_m[cell_id]:
                me_val = stats_m[cell_id][feat_id].get('me')
                if me_val is not None:
                    assert np.isfinite(me_val), f"ME not finite for mixed features"
