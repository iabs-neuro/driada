"""
Comprehensive tests for mixed selectivity generation.

This module tests that mixed selectivity generation actually produces
detectable mixed selectivity patterns that can be identified by the
analysis pipeline.
"""

import pytest
import numpy as np
from driada.experiment.synthetic import (
    generate_synthetic_exp_with_mixed_selectivity,
    generate_multiselectivity_patterns,
    generate_mixed_selective_signal,
    generate_synthetic_data_mixed_selectivity
)
from driada.intense.pipelines import compute_cell_feat_significance
from driada.intense.disentanglement import disentangle_all_selectivities
from driada.information.info_base import TimeSeries


class TestMultiselectivityPatterns:
    """Test the selectivity pattern generation."""
    
    def test_basic_pattern_generation(self):
        """Test basic selectivity pattern generation."""
        n_neurons = 10
        n_features = 4
        
        # Test with high selectivity probability
        matrix = generate_multiselectivity_patterns(
            n_neurons, n_features,
            selectivity_prob=1.0,  # All neurons selective
            multi_select_prob=0.8,  # Most have mixed selectivity
            weights_mode='equal',
            seed=42
        )
        
        assert matrix.shape == (n_features, n_neurons)
        
        # Check that all neurons are selective
        neurons_with_selectivity = np.sum(np.sum(matrix, axis=0) > 0)
        assert neurons_with_selectivity == n_neurons
        
        # Check that most have mixed selectivity
        mixed_neurons = np.sum(np.sum(matrix > 0, axis=0) >= 2)
        assert mixed_neurons >= int(n_neurons * 0.6)  # Allow some variance
    
    def test_weight_modes(self):
        """Test different weight generation modes."""
        n_neurons = 20
        n_features = 4
        
        # Test equal weights
        matrix_equal = generate_multiselectivity_patterns(
            n_neurons, n_features,
            selectivity_prob=1.0,
            multi_select_prob=1.0,
            weights_mode='equal',
            seed=42
        )
        
        # Check equal weights for mixed selective neurons
        for j in range(n_neurons):
            weights = matrix_equal[:, j]
            nonzero = weights[weights > 0]
            if len(nonzero) > 1:
                # Weights should be equal (1/n_selected)
                assert np.allclose(nonzero, nonzero[0])
        
        # Test dominant weights
        matrix_dominant = generate_multiselectivity_patterns(
            n_neurons, n_features,
            selectivity_prob=1.0,
            multi_select_prob=1.0,
            weights_mode='dominant',
            seed=42
        )
        
        # Check that one feature dominates in MOST mixed neurons
        dominant_count = 0
        mixed_count = 0
        for j in range(n_neurons):
            weights = matrix_dominant[:, j]
            nonzero = weights[weights > 0]
            if len(nonzero) > 1:
                mixed_count += 1
                # One weight should be larger than others
                sorted_weights = np.sort(nonzero)[::-1]
                if len(sorted_weights) >= 2 and sorted_weights[0] > sorted_weights[1] * 1.5:
                    dominant_count += 1
        
        # At least 70% of mixed neurons should show clear dominance
        assert dominant_count >= mixed_count * 0.7, \
            f"Only {dominant_count}/{mixed_count} neurons show dominance"
    
    def test_no_selectivity(self):
        """Test with zero selectivity probability."""
        matrix = generate_multiselectivity_patterns(
            10, 4,
            selectivity_prob=0.0,  # No neurons selective
            seed=42
        )
        
        assert np.all(matrix == 0)


class TestMixedSelectiveSignal:
    """Test the signal generation for mixed selective neurons."""
    
    def test_signal_generation_basic(self):
        """Test basic mixed selective signal generation."""
        duration = 10
        fps = 20
        n_points = int(duration * fps)
        
        # Create test features
        feat1 = np.zeros(n_points)
        feat1[50:70] = 1  # Active period 1
        feat1[120:140] = 1  # Active period 2
        
        feat2 = np.zeros(n_points)
        feat2[60:80] = 1  # Overlaps with feat1
        feat2[150:170] = 1  # Separate period
        
        features = [feat1, feat2]
        weights = [0.6, 0.4]
        
        signal = generate_mixed_selective_signal(
            features, weights, duration, fps,
            rate_0=0.1, rate_1=3.0,  # Higher active rate
            skip_prob=0.0,  # No skipping for test
            noise_std=0.05,  # Low noise
            seed=42
        )
        
        assert len(signal) == n_points
        assert np.max(signal) > np.mean(signal)  # Should have peaks
        
        # Check that signal is elevated during feature activity
        # Period where both features active (60-70)
        both_active = signal[60:70]
        # Period where neither active - choose period far from any activity
        # to avoid calcium decay effects (0-20 is before any activity)
        neither_active = signal[0:20]
        
        assert np.mean(both_active) > np.mean(neither_active)
    
    def test_signal_with_continuous_features(self):
        """Test signal generation with continuous features."""
        duration = 10
        fps = 20
        n_points = int(duration * fps)
        
        # Create continuous features
        feat1 = np.sin(np.linspace(0, 4*np.pi, n_points))
        feat2 = np.cos(np.linspace(0, 3*np.pi, n_points))
        
        features = [feat1, feat2]
        weights = [0.5, 0.5]
        
        signal = generate_mixed_selective_signal(
            features, weights, duration, fps,
            rate_1=2.0,
            noise_std=0.1,
            seed=42
        )
        
        assert len(signal) == n_points
        assert np.std(signal) > 0  # Should have variance


class TestSyntheticDataMixedSelectivity:
    """Test the full data generation pipeline."""
    
    def test_data_generation_with_known_selectivity(self):
        """Test data generation with a known selectivity pattern."""
        n_neurons = 5
        n_features = 3
        duration = 20
        fps = 10
        
        # Create features
        features_dict = {
            'feat1': np.random.choice([0, 1], int(duration * fps)),
            'feat2': np.random.choice([0, 1], int(duration * fps)),
            'feat3': np.random.choice([0, 1], int(duration * fps))
        }
        
        # Create known selectivity pattern
        selectivity_matrix = np.array([
            [1.0, 0.0, 0.5, 0.0, 0.3],  # feat1 selective to neurons 0, 2, 4
            [0.0, 1.0, 0.5, 0.7, 0.3],  # feat2 selective to neurons 1, 2, 3, 4
            [0.0, 0.0, 0.0, 0.3, 0.4],  # feat3 selective to neurons 3, 4
        ])
        
        signals, _ = generate_synthetic_data_mixed_selectivity(
            features_dict, n_neurons, selectivity_matrix,
            duration=duration, sampling_rate=fps,  # Use sampling_rate not fps
            rate_1=3.0,  # High firing rate
            skip_prob=0.0,  # No skipping
            noise_std=0.05,  # Low noise
            seed=42,
            verbose=False
        )
        
        assert signals.shape == (n_neurons, int(duration * fps))
        
        # Verify neurons have different activity patterns
        correlations = []
        for i in range(n_neurons-1):
            corr = np.corrcoef(signals[i], signals[i+1])[0, 1]
            correlations.append(corr)
        
        # Should have diversity in correlations
        assert np.std(correlations) > 0.1


class TestGenerateSyntheticExpWithMixedSelectivity:
    """Test the full experiment generation with mixed selectivity."""
    
    def test_basic_generation(self):
        """Test basic experiment generation."""
        exp, info = generate_synthetic_exp_with_mixed_selectivity(
            n_discrete_feats=2,
            n_continuous_feats=2,
            n_neurons=10,
            duration=60,
            fps=10,
            selectivity_prob=0.8,
            multi_select_prob=0.6,
            seed=42,
            verbose=False
        )
        
        assert exp.n_cells == 10
        assert exp.n_frames == 600
        
        # Check selectivity info
        assert 'matrix' in info
        assert 'feature_names' in info
        assert info['matrix'].shape[1] == 10  # n_neurons
    
    def test_detectability_of_mixed_selectivity(self):
        """Critical test: Verify generated mixed selectivity is detectable."""
        # Generate with parameters designed for strong selectivity
        # Use only discrete features for clearer signals
        exp, info = generate_synthetic_exp_with_mixed_selectivity(
            n_discrete_feats=4,
            n_continuous_feats=0,
            n_neurons=20,
            duration=300,  # Longer for better statistics
            fps=20,
            selectivity_prob=1.0,     # All neurons selective
            multi_select_prob=0.8,    # Most have mixed selectivity
            weights_mode='equal',     # Equal contributions
            rate_0=0.1,              # Low baseline
            rate_1=2.0,              # Max rate for calcium dynamics
            skip_prob=0.0,           # No skipping
            ampl_range=(1.0, 3.0),   # Stronger amplitude
            noise_std=0.005,         # Very low noise
            seed=42,
            verbose=False
        )
        
        # Get neurons with mixed selectivity from ground truth
        selectivity_matrix = info['matrix']
        mixed_neurons = np.where(np.sum(selectivity_matrix > 0, axis=0) >= 2)[0]
        
        assert len(mixed_neurons) >= 10, f"Need at least 10 mixed neurons, found {len(mixed_neurons)}"
        
        # Run INTENSE analysis on mixed neurons
        test_neurons = mixed_neurons[:5].tolist()
        
        # First, run basic cell-feat significance to check if neurons are selective
        stats, significance, _, _, disent_results = compute_cell_feat_significance(
            exp,
            cell_bunch=test_neurons,
            mode='two_stage',  # Use two_stage for proper significance detection
            n_shuffles_stage1=10,  # Fewer shuffles for speed
            n_shuffles_stage2=100,  # Stage 2 shuffles
            metric='mi',
            metric_distr_type='norm',  # Use normal distribution
            pval_thr=0.1,  # Lenient threshold
            multicomp_correction=None,  # No correction for easier detection
            enable_parallelization=False,  # Disable parallelization
            ds=2,  # Downsampling is important for MI calculation
            find_optimal_delays=False,
            allow_mixed_dimensions=True,
            with_disentanglement=True,
            verbose=False,
            seed=42
        )
        
        # Debug: Check which neurons are significantly selective to which features
        print("\n=== SELECTIVITY ANALYSIS ===")
        for neuron_id in test_neurons:
            neuron_idx = neuron_id
            ground_truth_selectivity = selectivity_matrix[:, neuron_idx]
            selective_features = np.where(ground_truth_selectivity > 0)[0]
            feature_names = info['feature_names']
            print(f"\nNeuron {neuron_id}:")
            print(f"  Ground truth: selective to features {[feature_names[i] for i in selective_features]}")
            print(f"  Weights: {ground_truth_selectivity[selective_features]}")
            
            neuron_sig = significance.get(neuron_id, {})
            detected_selective = []
            for feat_name, sig_info in neuron_sig.items():
                # Check if significant in stage2
                if isinstance(sig_info, dict) and sig_info.get('stage2', False):
                    detected_selective.append(feat_name)
                    mi_value = stats.get(neuron_id, {}).get(feat_name, {}).get('me', 0)
                    print(f"  Detected: {feat_name} (MI={mi_value:.4f})")
            
            if not detected_selective:
                print(f"  WARNING: No significant selectivity detected!")
        
        # Check that neurons show significant selectivity to multiple features
        neurons_with_mixed_detected = 0
        for neuron_id in test_neurons:
            neuron_sig = significance.get(neuron_id, {})
            sig_features = [feat for feat, sig_info in neuron_sig.items() 
                           if isinstance(sig_info, dict) and sig_info.get('stage2', False)]
            if len(sig_features) >= 2:
                neurons_with_mixed_detected += 1
        
        # At least some neurons should show mixed selectivity
        assert neurons_with_mixed_detected > 0, \
            f"No neurons with detected mixed selectivity! Need neurons selective to 2+ features."
        
        # Check disentanglement results
        if disent_results and 'summary' in disent_results:
            summary = disent_results['summary']
            if summary.get('overall_stats'):
                total_pairs = summary['overall_stats']['total_neuron_pairs']
                assert total_pairs > 0, "Disentanglement found no mixed selectivity pairs"
            else:
                # Check count matrix directly
                if 'count_matrix' in disent_results:
                    total_pairs = np.sum(disent_results['count_matrix'])
                    assert total_pairs > 0, \
                        f"Count matrix all zeros:\n{disent_results['count_matrix']}"
    
    def test_multifeature_generation(self):
        """Test multifeature generation."""
        exp, info = generate_synthetic_exp_with_mixed_selectivity(
            n_discrete_feats=1,
            n_continuous_feats=4,
            n_neurons=10,
            n_multifeatures=2,
            duration=60,
            fps=10,
            seed=42,
            verbose=False
        )
        
        # Check multifeatures were created
        # Look for keys starting with 'multi' or containing multiple features
        multifeatures = [k for k in exp.dynamic_features.keys() 
                        if (isinstance(k, str) and k.startswith('multi')) or
                           (isinstance(k, tuple) and len(k) > 1)]
        
        assert len(multifeatures) >= 1, f"No multifeatures found. Keys: {list(exp.dynamic_features.keys())}"
    
    def test_parameter_sensitivity(self):
        """Test that parameters actually affect detectability."""
        # Test with weak parameters - all neurons selective but weak signals
        exp_weak, _ = generate_synthetic_exp_with_mixed_selectivity(
            n_discrete_feats=3,
            n_continuous_feats=0,
            n_neurons=20,
            duration=100,
            fps=20,
            selectivity_prob=1.0,   # All neurons selective
            multi_select_prob=0.5,  # Some mixed selectivity
            rate_0=0.8,            # High baseline (close to active rate)
            rate_1=1.2,            # Low active rate (weak modulation)
            noise_std=0.3,         # High noise relative to signal
            seed=42,
            verbose=False
        )
        
        # Test with strong parameters - all neurons selective with strong signals
        exp_strong, _ = generate_synthetic_exp_with_mixed_selectivity(
            n_discrete_feats=3,
            n_continuous_feats=0,
            n_neurons=20,
            duration=100,
            fps=20,
            selectivity_prob=1.0,   # All neurons selective
            multi_select_prob=0.5,  # Same mixed selectivity
            rate_0=0.1,            # Low baseline
            rate_1=3.0,            # High active rate (strong modulation)
            noise_std=0.05,        # Low noise
            seed=42,
            verbose=False
        )
        
        # Compare signal quality metrics
        def compute_signal_quality(exp):
            """Compute signal quality using MAD-based metric."""
            from scipy.stats import median_abs_deviation
            quality_scores = []
            for i in range(exp.n_cells):
                signal = exp.neurons[i].ca.data
                # Calculate MAD (median absolute deviation)
                mad = median_abs_deviation(signal)
                if mad == 0:
                    continue
                # Calculate peak amplitude
                baseline = np.median(signal)
                peak_90 = np.percentile(signal, 90)
                peak_amplitude = peak_90 - baseline
                # Quality is peak amplitude normalized by MAD
                if peak_amplitude > 0:
                    quality = peak_amplitude / mad
                    quality_scores.append(quality)
            return np.mean(quality_scores) if quality_scores else 0
        
        quality_weak = compute_signal_quality(exp_weak)
        quality_strong = compute_signal_quality(exp_strong)
        
        # Strong parameters should produce much better signal quality
        # Given the parameters: weak (rate_0=0.8, rate_1=1.2) vs strong (rate_0=0.1, rate_1=3.0)
        # We expect significantly better quality in the strong case
        assert quality_strong > quality_weak * 1.5, \
            f"Strong params quality ({quality_strong:.2f}) not significantly better than weak ({quality_weak:.2f})"


class TestIntegrationWithAnalysisPipeline:
    """Test that generated data works with the full analysis pipeline."""
    
    def test_full_pipeline_with_mixed_selectivity(self):
        """Test the full pipeline from generation to analysis."""
        # Generate data with known mixed selectivity
        exp, info = generate_synthetic_exp_with_mixed_selectivity(
            n_discrete_feats=3,
            n_continuous_feats=0,
            n_neurons=15,
            duration=200,
            fps=20,
            selectivity_prob=1.0,
            multi_select_prob=0.7,
            weights_mode='equal',
            rate_0=0.1,
            rate_1=4.0,
            skip_prob=0.0,
            noise_std=0.05,
            seed=42,
            verbose=False
        )
        
        # Find neurons with strongest mixed selectivity
        selectivity_matrix = info['matrix']
        n_features_per_neuron = np.sum(selectivity_matrix > 0, axis=0)
        strong_mixed = np.where(n_features_per_neuron >= 2)[0]
        
        # Run full analysis including disentanglement
        test_neurons = strong_mixed[:5].tolist()
        
        stats, sig, _, results, disent_results = compute_cell_feat_significance(
            exp,
            cell_bunch=test_neurons,
            mode='two_stage',  # Need two_stage for disentanglement
            n_shuffles_stage1=10,
            n_shuffles_stage2=50,
            pval_thr=0.05,
            with_disentanglement=True,
            find_optimal_delays=False,
            allow_mixed_dimensions=True,
            ds=2,
            verbose=False,
            seed=42
        )
        
        # Verify disentanglement found mixed selectivity
        assert disent_results is not None
        assert 'summary' in disent_results
        summary = disent_results['summary']
        
        # Should have found some mixed selectivity pairs
        if summary.get('overall_stats'):
            assert summary['overall_stats']['total_neuron_pairs'] > 0, \
                "No mixed selectivity pairs detected in analysis"
        else:
            # If no overall_stats, the count matrix should still exist
            assert 'count_matrix' in disent_results
            assert np.sum(disent_results['count_matrix']) > 0, \
                f"Count matrix all zeros:\n{disent_results['count_matrix']}"


# Performance and edge case tests

def test_generation_performance():
    """Test that generation completes in reasonable time."""
    import time
    
    start = time.time()
    exp, info = generate_synthetic_exp_with_mixed_selectivity(
        n_discrete_feats=5,
        n_continuous_feats=5,
        n_neurons=50,
        duration=100,
        fps=20,
        verbose=False
    )
    elapsed = time.time() - start
    
    assert elapsed < 10.0, f"Generation took too long: {elapsed:.2f}s"
    assert exp.n_cells == 50


def test_edge_cases():
    """Test edge cases in generation."""
    # Many features, few neurons - this is a valid edge case for mixed selectivity
    # Use longer duration to avoid shuffle mask issues
    exp, info = generate_synthetic_exp_with_mixed_selectivity(
        n_discrete_feats=10,
        n_continuous_feats=10,
        n_neurons=3,
        duration=30,  # Increased from 10 to avoid shuffle mask issues
        fps=10,
        verbose=False
    )
    assert exp.n_cells == 3
    
    # Minimal case for mixed selectivity: 2 features, 2 neurons
    exp, info = generate_synthetic_exp_with_mixed_selectivity(
        n_discrete_feats=2,
        n_continuous_feats=0,
        n_neurons=2,
        duration=30,  # Increased from 10 to avoid shuffle mask issues
        fps=10,
        verbose=False
    )
    assert exp.n_cells == 2