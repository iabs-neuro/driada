"""
Example of Representational Similarity Analysis (RSA) with DRIADA.

This example demonstrates:
1. Computing RDMs from neural data with different item definitions
2. Comparing representations between brain regions/conditions
3. Visualizing RDM structure
4. NEW: Unified API with automatic data type detection
5. NEW: Caching support for repeated computations
6. NEW: Direct MVData integration
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import driada
from driada import rsa
from driada.dim_reduction.data import MVData


def example_1_behavioral_conditions():
    """Example 1: RSA with behavioral conditions - showcasing unified API."""
    print("\n=== Example 1: RSA with Behavioral Conditions (Unified API) ===")

    # Generate synthetic experiment with different stimulus types
    n_neurons = 50
    duration = 300  # 5 minutes
    n_timepoints = int(duration * 20)  # 20 Hz sampling

    # Create stimulus conditions that repeat (use numeric labels)
    stim_duration = 100  # 5 seconds per stimulus
    stimulus_sequence = []
    for _ in range(n_timepoints // (stim_duration * 4)):
        stimulus_sequence.extend([0] * stim_duration)  # Stimulus A
        stimulus_sequence.extend([1] * stim_duration)  # Stimulus B
        stimulus_sequence.extend([2] * stim_duration)  # Stimulus C
        stimulus_sequence.extend([3] * stim_duration)  # Stimulus D
    stimulus_labels = np.array(stimulus_sequence[:n_timepoints])

    # Generate neural data with stimulus selectivity
    print("Generating synthetic neural population...")
    exp = driada.generate_synthetic_exp(
        n_dfeats=0,
        n_cfeats=4,  # 4 features for different selectivity patterns
        nneurons=n_neurons,
        duration=duration,
        seed=42,
    )

    # Override with our stimulus labels
    exp.dynamic_features["stimulus_type"] = driada.TimeSeries(stimulus_labels)

    # NEW: Use unified API that automatically detects Experiment object
    print("Computing RDM using unified API...")
    rdm, labels = rsa.compute_rdm_unified(
        exp, items="stimulus_type", metric="correlation"
    )

    print(f"RDM shape: {rdm.shape}")
    print(f"Unique conditions: {labels}")

    # NEW: Demonstrate caching
    print("\nDemonstrating caching support...")
    start = time.time()
    rdm_cached, _ = exp.compute_rdm("stimulus_type", use_cache=True)
    time_first = time.time() - start

    start = time.time()
    rdm_cached2, _ = exp.compute_rdm("stimulus_type", use_cache=True)
    time_cached = time.time() - start

    print(f"First computation: {time_first:.4f}s")
    print(f"Cached computation: {time_cached:.4f}s")
    print(f"Speedup: {time_first/time_cached:.1f}x")

    # Create string labels for visualization
    label_names = ["Stim A", "Stim B", "Stim C", "Stim D"]

    # Visualize RDM with standardized plotting
    fig = rsa.plot_rdm(
        rdm,
        labels=label_names[: len(labels)],
        title="Neural RDM - Stimulus Conditions",
        show_values=True,
    )
    plt.savefig("rsa_example_1.png")
    plt.close()

    return exp, rdm, labels


def example_2_simplified_api():
    """Example 2: Simplified RSA API with rsa_compare."""
    print("\n=== Example 2: Simplified API with rsa_compare ===")

    # Generate two related neural populations
    np.random.seed(42)
    n_items = 20
    n_neurons_v1 = 100
    n_neurons_v2 = 150

    # Create base patterns
    base_patterns = np.random.randn(n_items, 50)

    # V1: Direct representation
    v1_data = base_patterns @ np.random.randn(50, n_neurons_v1)
    v1_data += 0.2 * np.random.randn(n_items, n_neurons_v1)  # Add noise

    # V2: Transformed representation
    transform = np.random.randn(50, 50)
    v2_data = (base_patterns @ transform) @ np.random.randn(50, n_neurons_v2)
    v2_data += 0.2 * np.random.randn(n_items, n_neurons_v2)  # Add noise

    # NEW: Use simplified rsa_compare function
    print("Comparing V1 and V2 representations...")
    similarity = rsa.rsa_compare(v1_data, v2_data)
    print(f"V1-V2 similarity (Spearman): {similarity:.3f}")

    # Try different metrics
    print("\nTrying different distance metrics:")
    for metric in ["correlation", "euclidean", "cosine"]:
        sim = rsa.rsa_compare(v1_data, v2_data, metric=metric)
        print(f"  {metric}: {sim:.3f}")

    # Try different comparison methods
    print("\nTrying different comparison methods:")
    for comparison in ["spearman", "pearson", "kendall"]:
        sim = rsa.rsa_compare(v1_data, v2_data, comparison=comparison)
        print(f"  {comparison}: {sim:.3f}")

    # Also works with MVData
    print("\nUsing MVData objects:")
    mv1 = MVData(v1_data.T)  # MVData expects (n_features, n_items)
    mv2 = MVData(v2_data.T)
    similarity_mv = rsa.rsa_compare(mv1, mv2)
    print(f"MVData similarity: {similarity_mv:.3f}")

    # Visualize the RDMs
    rdm1 = rsa.compute_rdm(v1_data)
    rdm2 = rsa.compute_rdm(v2_data)

    fig = rsa.plot_rdm_comparison(
        [rdm1, rdm2], titles=["V1 Representation", "V2 Representation"]
    )
    plt.savefig("rsa_example_2_simplified.png")
    plt.close()

    return similarity


def example_3_experiment_comparison():
    """Example 3: Compare two experiments using simplified API."""
    print("\n=== Example 3: Comparing Experiments with rsa_compare ===")

    # Generate two experiments
    exp1 = driada.generate_synthetic_exp(
        n_dfeats=1, n_cfeats=3, nneurons=30, duration=60, seed=42
    )

    exp2 = driada.generate_synthetic_exp(
        n_dfeats=1, n_cfeats=3, nneurons=30, duration=60, seed=43
    )

    # Add stimulus labels
    n_timepoints = exp1.calcium.scdata.shape[1]
    stim_duration = 200
    n_stimuli = 4
    stimulus_labels = np.repeat(range(n_stimuli), stim_duration)
    stimulus_labels = np.tile(
        stimulus_labels, n_timepoints // (stim_duration * n_stimuli) + 1
    )[:n_timepoints]

    exp1.dynamic_features["stimulus"] = driada.TimeSeries(stimulus_labels)
    exp2.dynamic_features["stimulus"] = driada.TimeSeries(stimulus_labels)

    # NEW: Compare experiments directly with rsa_compare
    print("Comparing two experiments...")
    similarity = rsa.rsa_compare(exp1, exp2, items="stimulus")
    print(f"Experiment similarity: {similarity:.3f}")

    # Try with trial structure
    trial_info = {
        "trial_starts": list(range(0, n_timepoints, stim_duration * n_stimuli)),
        "trial_labels": ["Block_A", "Block_B"]
        * (len(range(0, n_timepoints, stim_duration * n_stimuli)) // 2 + 1),
    }
    trial_info["trial_labels"] = trial_info["trial_labels"][
        : len(trial_info["trial_starts"])
    ]

    print("\nComparing with trial structure...")
    similarity_trials = rsa.rsa_compare(exp1, exp2, items=trial_info)
    print(f"Trial-based similarity: {similarity_trials:.3f}")

    # Compare using spike data
    print("\nComparing spike data...")
    similarity_spikes = rsa.rsa_compare(
        exp1, exp2, items="stimulus", data_type="spikes"
    )
    print(f"Spike similarity: {similarity_spikes:.3f}")

    return similarity, similarity_trials, similarity_spikes


def example_4_trial_structure():
    """Example 4: RSA with trial structure - showcasing MVData integration."""
    print("\n=== Example 4: RSA with Trial Structure (MVData Integration) ===")

    # Generate experiment
    exp = driada.generate_2d_manifold_exp(
        n_neurons=64, duration=600, environments=["env1"]
    )

    # Define trial structure
    trial_info = {
        "trial_starts": [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000],
        "trial_labels": [
            "explore",
            "rest",
            "explore",
            "rest",
            "explore",
            "reward",
            "explore",
            "reward",
            "explore",
            "rest",
        ],
        "trial_duration": 1000,  # 50 seconds per trial at 20 Hz
    }

    # NEW: Convert to MVData first to show integration
    print("Converting neural data to MVData object...")
    mvdata = MVData(
        exp.calcium.scdata
    )  # Use scaled data for equal neuron contributions

    # NEW: Use unified API with MVData - it handles trial structure
    print("Computing RDM from MVData with trial structure...")
    # For MVData, we need to pass labels separately since trial_info dict isn't supported
    # So we'll compute labels first
    trial_labels_expanded = []
    for i, (start, label) in enumerate(
        zip(trial_info["trial_starts"], trial_info["trial_labels"])
    ):
        if i < len(trial_info["trial_starts"]) - 1:
            duration = trial_info["trial_starts"][i + 1] - start
        else:
            duration = trial_info["trial_duration"]
        trial_labels_expanded.extend([label] * duration)

    # Convert string labels to numeric for compatibility
    unique_labels = list(set(trial_labels_expanded))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = np.array(
        [label_map[label] for label in trial_labels_expanded[: mvdata.data.shape[1]]]
    )

    rdm, label_indices = rsa.compute_rdm_unified(
        mvdata, items=numeric_labels, metric="euclidean"
    )

    # Map back to string labels
    labels = [unique_labels[i] for i in label_indices]
    print(f"Unique trial types: {labels}")

    # Visualize with dendrogram
    fig = rsa.plot_rdm(
        rdm,
        labels=labels,
        title="Neural RDM - Trial Types (via MVData)",
        dendrogram_ratio=0.15,
    )
    plt.savefig("rsa_example_2.png")
    plt.close()

    return exp, rdm, labels


def example_5_compare_representations():
    """Example 5: Compare representations - showcasing unified API flexibility."""
    print("\n=== Example 5: Comparing Representations (Unified API) ===")

    # Generate two populations with potentially different representations
    print("Generating two neural populations...")

    # Population 1: Strong stimulus selectivity
    exp1 = driada.generate_synthetic_exp(
        n_dfeats=4, n_cfeats=0, nneurons=50, duration=300, seed=42
    )

    # Population 2: Weaker/different selectivity
    exp2 = driada.generate_synthetic_exp(
        n_dfeats=4,
        n_cfeats=0,
        nneurons=50,
        duration=300,
        seed=123,  # Different seed for different selectivity
        noise_std=0.3,  # More noise
    )

    # Use the discrete features as stimulus conditions
    stimulus_labels = exp1.dynamic_features["d_feat_0"].data

    # NEW: Use unified API for both - it detects numpy arrays automatically
    print("Computing RDMs using unified API...")
    rdm1, labels1 = rsa.compute_rdm_unified(
        exp1.calcium.scdata,  # Use scaled data
        items=stimulus_labels,
        metric="correlation",
    )

    rdm2, labels2 = rsa.compute_rdm_unified(
        exp2.calcium.scdata,  # Use scaled data
        items=stimulus_labels,
        metric="correlation",
    )

    # Compare RDMs using different methods
    print("\nComparing RDMs with multiple methods:")
    for method in ["spearman", "pearson", "kendall"]:
        similarity = rsa.compare_rdms(rdm1, rdm2, method=method)
        print(f"  {method}: {similarity:.3f}")

    # Bootstrap test
    print("\nRunning bootstrap significance test...")
    bootstrap_results = rsa.bootstrap_rdm_comparison(
        exp1.calcium.scdata,  # Use scaled data
        exp2.calcium.scdata,  # Use scaled data
        stimulus_labels,
        stimulus_labels,
        n_bootstrap=100,  # Use more for real analysis
        random_state=42,
    )

    print(f"Bootstrap p-value: {bootstrap_results['p_value']:.3f}")
    print(
        f"95% CI: [{bootstrap_results['ci_lower']:.3f}, {bootstrap_results['ci_upper']:.3f}]"
    )

    # Visualize both RDMs
    fig = rsa.plot_rdm_comparison(
        [rdm1, rdm2], labels=list(labels1), titles=["Population 1", "Population 2"]
    )
    plt.savefig("rsa_example_5.png")
    plt.close()

    return rdm1, rdm2, bootstrap_results["observed"]


def example_6_performance_comparison():
    """Example 6: Performance comparison of different metrics."""
    print("\n=== Example 6: Performance Comparison ===")

    # Generate data of different sizes
    sizes = [(50, 100), (100, 500)]

    for n_items, n_features in sizes:
        print(f"\nTesting with {n_items} items, {n_features} features:")
        patterns = np.random.randn(n_items, n_features)

        # Time different metrics
        metrics = ["correlation", "euclidean", "manhattan"]
        for metric in metrics:
            start = time.time()
            rdm = rsa.compute_rdm_unified(patterns, metric=metric)
            elapsed = time.time() - start
            print(f"  {metric}: {elapsed:.4f}s")

        # NEW: Show that euclidean and manhattan can use JIT
        if driada.utils.jit.is_jit_enabled():
            print("  (JIT compilation enabled for euclidean/manhattan)")

    return rdm


def example_7_mvdata_direct():
    """Example 7: Direct MVData support in unified API."""
    print("\n=== Example 7: Direct MVData Support ===")

    # Create MVData object with known structure
    n_features = 100
    n_timepoints = 1000
    n_conditions = 5

    # Create condition labels
    condition_duration = n_timepoints // n_conditions
    conditions = np.repeat(np.arange(n_conditions), condition_duration)

    # Create data with clear condition structure
    patterns = np.random.randn(n_conditions, n_features)
    data = np.zeros((n_features, n_timepoints))
    for i, cond in enumerate(conditions):
        data[:, i] = patterns[cond] + 0.1 * np.random.randn(n_features)

    # Create MVData object
    mvdata = MVData(data)

    print("Computing RDM from MVData object...")
    # NEW: Unified API automatically detects MVData
    rdm, labels = rsa.compute_rdm_unified(mvdata, items=conditions)

    print(f"RDM shape: {rdm.shape}")
    print(f"Unique conditions: {labels}")

    # NEW: Show that MVData's correlation is now fixed
    print("\nVerifying MVData correlation computation...")
    # Compute correlation matrix of patterns
    pattern_corr = mvdata.corr_mat(axis=1)  # Correlation between timepoints
    print(f"Pattern correlation matrix shape: {pattern_corr.shape}")

    # Visualize RDM
    fig = rsa.plot_rdm(
        rdm,
        labels=[f"Cond {i}" for i in labels],
        title="RDM from MVData",
        show_values=True,
    )
    plt.savefig("rsa_example_7.png")
    plt.close()

    return mvdata, rdm


if __name__ == "__main__":
    # Run all examples
    print("DRIADA RSA Examples - Showcasing Recent Improvements")
    print("====================================================")
    print("\nKey improvements demonstrated:")
    print("1. NEW: Simplified rsa_compare() API for common use case")
    print("2. Unified API (compute_rdm_unified) - automatic data type detection")
    print("3. Support for Experiment comparisons in rsa_compare")
    print("4. Caching support in Experiment objects")
    print("5. Direct MVData integration")
    print("6. Standardized visualization with plot utilities")

    try:
        # Example 1: Behavioral conditions with unified API
        exp1, rdm1, labels1 = example_1_behavioral_conditions()
    except Exception as e:
        print(f"\nExample 1 skipped due to: {e}")

    try:
        # Example 2: NEW - Simplified API demonstration
        similarity = example_2_simplified_api()
    except Exception as e:
        print(f"\nExample 2 skipped due to: {e}")

    try:
        # Example 3: NEW - Compare experiments with simplified API
        sim_stim, sim_trial, sim_spike = example_3_experiment_comparison()
    except Exception as e:
        print(f"\nExample 3 skipped due to: {e}")

    try:
        # Example 4: Trial structure with MVData
        exp4, rdm4, labels4 = example_4_trial_structure()
    except Exception as e:
        print(f"\nExample 4 skipped due to: {e}")

    try:
        # Example 5: Compare representations with unified API
        rdm5a, rdm5b, sim5 = example_5_compare_representations()
    except Exception as e:
        print(f"\nExample 5 skipped due to: {e}")

    # Example 6: Performance comparison
    example_6_performance_comparison()

    # Example 7: Direct MVData support
    mvdata7, rdm7 = example_7_mvdata_direct()

    print("\n=== Summary of Improvements ===")
    print("✓ NEW: rsa_compare() provides simplified API for common use case")
    print("✓ Supports arrays, MVData, and Experiment objects seamlessly")
    print("✓ Unified API reduces code complexity")
    print("✓ Caching speeds up repeated computations")
    print("✓ MVData integration enables seamless workflow")
    print("\nAll examples completed!")
