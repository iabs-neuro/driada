"""
Example of Representational Similarity Analysis (RSA) with DRIADA.

This example demonstrates:
1. Computing RDMs from stimulus-selective neural populations
2. Comparing representations between brain regions
3. Comparing representations across recording sessions
4. Statistical testing with bootstrap methods
5. Working with MVData objects
"""

import numpy as np
import matplotlib.pyplot as plt
from driada import rsa
from driada.dim_reduction.data import MVData
from driada.experiment.synthetic import generate_tuned_selectivity_exp


def create_stimulus_labels_from_events(exp, event_names):
    """
    Convert multiple binary event features to categorical stimulus labels.

    Parameters
    ----------
    exp : Experiment
        DRIADA experiment with discrete event features
    event_names : list of str
        Event feature names (e.g., ["event_0", "event_1", "event_2", "event_3"])

    Returns
    -------
    labels : np.ndarray
        Categorical labels (0, 1, 2, ...) for each timepoint.
        Timepoints with no event or multiple events are labeled -1.
    """
    n_timepoints = exp.calcium.data.shape[1]
    labels = np.full(n_timepoints, -1, dtype=int)

    # Count active events per timepoint to detect overlaps
    event_count = np.zeros(n_timepoints, dtype=int)
    for event_name in event_names:
        event_data = exp.dynamic_features[event_name].data
        event_count += (event_data > 0).astype(int)

    # Only label timepoints with exactly one active event (no overlap contamination)
    for idx, event_name in enumerate(event_names):
        event_data = exp.dynamic_features[event_name].data
        single_event = (event_data > 0) & (event_count == 1)
        labels[single_event] = idx

    return labels


def example_1_stimulus_conditions():
    """Example 1: RDM from stimulus-selective neurons with category structure."""
    print("\n=== Example 1: RSA with Stimulus Conditions ===")

    # Design a population with two stimulus categories:
    #   Category A: events 0 and 1 share neurons -> similar representations
    #   Category B: events 2 and 3 share neurons -> similar representations
    # Cross-category pairs share no neurons -> dissimilar representations
    # The RDM should reveal this 2x2 block structure.
    population = [
        {"name": "cat_a_shared", "count": 20, "features": ["event_0", "event_1"],
         "combination": "or"},
        {"name": "event_0_specific", "count": 15, "features": ["event_0"]},
        {"name": "event_1_specific", "count": 15, "features": ["event_1"]},
        {"name": "cat_b_shared", "count": 20, "features": ["event_2", "event_3"],
         "combination": "or"},
        {"name": "event_2_specific", "count": 15, "features": ["event_2"]},
        {"name": "event_3_specific", "count": 15, "features": ["event_3"]},
    ]

    # Generate synthetic data with event features as stimuli
    print("Generating stimulus-selective neurons (100 neurons, 4 conditions)...")
    print("  Category A: Stim A & B share 20 neurons")
    print("  Category B: Stim C & D share 20 neurons")
    exp = generate_tuned_selectivity_exp(
        population=population,
        n_discrete_features=4,
        duration=600,
        event_active_fraction=0.08,
        event_avg_duration=1.0,
        baseline_rate=0.05,
        peak_rate=2.0,
        seed=42,
        verbose=False,
        reconstruct_spikes="threshold",
    )

    # Convert binary event features to categorical stimulus labels
    # Use spike data (not calcium) to avoid temporal blurring artifacts
    print("Computing RDM from spike patterns...")
    stimulus_labels = create_stimulus_labels_from_events(
        exp, ["event_0", "event_1", "event_2", "event_3"]
    )

    # Compute RDM across stimulus conditions (euclidean distance on spike rates)
    valid_mask = stimulus_labels >= 0
    rdm, labels = rsa.compute_rdm_unified(
        exp.spikes.data[:, valid_mask], items=stimulus_labels[valid_mask],
        metric="euclidean",
    )

    print(f"RDM shape: {rdm.shape}")
    print(f"Stimulus conditions: {labels}")

    # Visualize RDM
    label_names = ["Stim A", "Stim B", "Stim C", "Stim D"]
    fig = rsa.plot_rdm(
        rdm,
        labels=label_names[: len(labels)],
        title="Neural RDM - Stimulus Conditions",
        show_values=True,
    )
    plt.savefig("examples/rsa/rsa_stimulus_conditions.png")
    plt.close()

    return exp, rdm, labels


def example_2_compare_regions():
    """Example 2: Comparing representations between brain regions."""
    print("\n=== Example 2: Comparing Representations ===")

    # Generate two related neural populations (e.g., V1 and V2)
    np.random.seed(42)
    n_items = 20
    n_neurons_v1 = 100
    n_neurons_v2 = 150

    # Create base patterns that both regions respond to
    base_patterns = np.random.randn(n_items, 50)

    # V1: Direct representation with noise
    v1_data = base_patterns @ np.random.randn(50, n_neurons_v1)
    v1_data += 0.2 * np.random.randn(n_items, n_neurons_v1)

    # V2: Transformed representation with noise
    transform = np.random.randn(50, 50)
    v2_data = (base_patterns @ transform) @ np.random.randn(50, n_neurons_v2)
    v2_data += 0.2 * np.random.randn(n_items, n_neurons_v2)

    # Compare representations directly
    print("Comparing V1 and V2 representations...")
    similarity = rsa.rsa_compare(v1_data, v2_data)
    print(f"V1-V2 similarity (Spearman): {similarity:.3f}")

    # Try different distance metrics
    print("\nTrying different distance metrics:")
    for metric in ["correlation", "euclidean", "cosine"]:
        sim = rsa.rsa_compare(v1_data, v2_data, metric=metric)
        print(f"  {metric}: {sim:.3f}")

    # Try different comparison methods
    print("\nTrying different comparison methods:")
    for comparison in ["spearman", "pearson", "kendall"]:
        sim = rsa.rsa_compare(v1_data, v2_data, comparison=comparison)
        print(f"  {comparison}: {sim:.3f}")

    # Visualize both RDMs
    rdm1 = rsa.compute_rdm(v1_data)
    rdm2 = rsa.compute_rdm(v2_data)

    fig = rsa.plot_rdm_comparison(
        [rdm1, rdm2], titles=["V1 Representation", "V2 Representation"]
    )
    plt.savefig("examples/rsa/rsa_compare_regions.png")
    plt.close()

    return similarity


def example_3_compare_experiments():
    """Example 3: Comparing experiments from different sessions."""
    print("\n=== Example 3: Comparing Experiments ===")

    # Three stimulus categories (A/B, C/D, E/F) with shared within-category neurons
    # Same structure in both sessions - different noise realizations
    population = [
        {"name": "cat_a_shared", "count": 14, "features": ["event_0", "event_1"],
         "combination": "or"},
        {"name": "event_0_only", "count": 10, "features": ["event_0"]},
        {"name": "event_1_only", "count": 10, "features": ["event_1"]},
        {"name": "cat_b_shared", "count": 14, "features": ["event_2", "event_3"],
         "combination": "or"},
        {"name": "event_2_only", "count": 10, "features": ["event_2"]},
        {"name": "event_3_only", "count": 10, "features": ["event_3"]},
        {"name": "cat_c_shared", "count": 12, "features": ["event_4", "event_5"],
         "combination": "or"},
        {"name": "event_4_only", "count": 10, "features": ["event_4"]},
        {"name": "event_5_only", "count": 10, "features": ["event_5"]},
    ]

    event_names = ["event_0", "event_1", "event_2", "event_3", "event_4", "event_5"]

    # Session 1
    print("Generating session 1 (100 neurons, 6 conditions, 3 categories)...")
    exp1 = generate_tuned_selectivity_exp(
        population=population,
        n_discrete_features=6,
        duration=600,
        event_active_fraction=0.08,
        event_avg_duration=1.0,
        baseline_rate=0.05,
        peak_rate=2.0,
        seed=42,
        verbose=False,
        reconstruct_spikes="threshold",
    )

    # Session 2 (different seed = different noise, but same population structure)
    print("Generating session 2 (100 neurons, 6 conditions, 3 categories)...")
    exp2 = generate_tuned_selectivity_exp(
        population=population,
        n_discrete_features=6,
        duration=600,
        event_active_fraction=0.08,
        event_avg_duration=1.0,
        baseline_rate=0.05,
        peak_rate=2.0,
        seed=123,
        verbose=False,
        reconstruct_spikes="threshold",
    )

    # Compare experiments using spike patterns (no temporal blurring)
    print("Comparing two sessions...")
    stim_labels_1 = create_stimulus_labels_from_events(exp1, event_names)
    stim_labels_2 = create_stimulus_labels_from_events(exp2, event_names)

    # Filter to valid timepoints
    valid_1 = stim_labels_1 >= 0
    valid_2 = stim_labels_2 >= 0

    # Compute RDMs from spike data using euclidean distance
    rdm1, labels1 = rsa.compute_rdm_unified(
        exp1.spikes.data[:, valid_1], items=stim_labels_1[valid_1],
        metric="euclidean",
    )
    rdm2, labels2 = rsa.compute_rdm_unified(
        exp2.spikes.data[:, valid_2], items=stim_labels_2[valid_2],
        metric="euclidean",
    )

    similarity = rsa.compare_rdms(rdm1, rdm2, method="spearman")
    print(f"Cross-session RDM similarity: {similarity:.3f}")

    # Visualize the RDMs
    label_names = ["Stim A", "Stim B", "Stim C", "Stim D", "Stim E", "Stim F"]
    fig = rsa.plot_rdm_comparison(
        [rdm1, rdm2],
        labels=label_names[:len(labels1)],
        titles=["Session 1", "Session 2"],
    )
    plt.savefig("examples/rsa/rsa_compare_experiments.png")
    plt.close()

    return similarity


def example_4_bootstrap_testing():
    """Example 4: Statistical testing with bootstrap methods."""
    print("\n=== Example 4: Bootstrap Statistical Testing ===")

    # Same category structure as Example 1: two categories with shared neurons
    population = [
        {"name": "cat_a_shared", "count": 20, "features": ["event_0", "event_1"],
         "combination": "or"},
        {"name": "event_0_specific", "count": 15, "features": ["event_0"]},
        {"name": "event_1_specific", "count": 15, "features": ["event_1"]},
        {"name": "cat_b_shared", "count": 20, "features": ["event_2", "event_3"],
         "combination": "or"},
        {"name": "event_2_specific", "count": 15, "features": ["event_2"]},
        {"name": "event_3_specific", "count": 15, "features": ["event_3"]},
    ]

    event_names = ["event_0", "event_1", "event_2", "event_3"]

    # Generate two populations with same structure but different noise realizations
    print("Generating two neural populations (100 neurons each)...")
    exp1 = generate_tuned_selectivity_exp(
        population=population,
        n_discrete_features=4,
        duration=600,
        event_active_fraction=0.08,
        event_avg_duration=1.0,
        seed=42,
        verbose=False,
        reconstruct_spikes="threshold",
    )

    exp2 = generate_tuned_selectivity_exp(
        population=population,
        n_discrete_features=4,
        duration=600,
        event_active_fraction=0.08,
        event_avg_duration=1.0,
        seed=123,
        verbose=False,
        reconstruct_spikes="threshold",
    )

    # Create stimulus labels from events
    print("Creating stimulus labels...")
    stim_labels_1 = create_stimulus_labels_from_events(exp1, event_names)
    stim_labels_2 = create_stimulus_labels_from_events(exp2, event_names)

    # Compute RDMs from spike data (euclidean distance)
    valid_mask_1 = stim_labels_1 >= 0
    valid_mask_2 = stim_labels_2 >= 0

    rdm1, labels1 = rsa.compute_rdm_unified(
        exp1.spikes.data[:, valid_mask_1],
        items=stim_labels_1[valid_mask_1],
        metric="euclidean",
    )

    rdm2, labels2 = rsa.compute_rdm_unified(
        exp2.spikes.data[:, valid_mask_2],
        items=stim_labels_2[valid_mask_2],
        metric="euclidean",
    )

    # Compare RDMs
    # With few conditions (4), only 6 upper-triangle values exist.
    # Rank-based methods (Spearman, Kendall) are underpowered; Pearson is preferred.
    print("\nComparing RDMs with multiple methods:")
    for method in ["spearman", "pearson", "kendall"]:
        similarity = rsa.compare_rdms(rdm1, rdm2, method=method)
        print(f"  {method}: {similarity:.3f}")

    # Bootstrap significance test (Pearson is more appropriate with few conditions)
    print("\nRunning bootstrap significance test (Pearson)...")
    bootstrap_results = rsa.bootstrap_rdm_comparison(
        exp1.spikes.data[:, valid_mask_1],
        exp2.spikes.data[:, valid_mask_2],
        stim_labels_1[valid_mask_1],
        stim_labels_2[valid_mask_2],
        metric="euclidean",
        comparison_method="pearson",
        n_bootstrap=100,
        random_state=42,
    )

    print(f"Observed similarity: {bootstrap_results['observed']:.3f}")
    print(
        f"95% CI: [{bootstrap_results['ci_lower']:.3f}, {bootstrap_results['ci_upper']:.3f}]"
    )
    print(f"Bootstrap stability p-value: {bootstrap_results['p_value']:.3f}")
    print("  (Tests if observed is extreme relative to bootstrap mean;")
    print("   ~0.5 means stable. CI above 0 confirms reliable similarity.)")

    # Visualize both RDMs
    label_names = ["Stim A", "Stim B", "Stim C", "Stim D"]
    fig = rsa.plot_rdm_comparison(
        [rdm1, rdm2],
        labels=label_names[:len(labels1)],
        titles=["Population 1", "Population 2"],
    )
    plt.savefig("examples/rsa/rsa_bootstrap_testing.png")
    plt.close()

    return rdm1, rdm2, bootstrap_results["observed"]


def example_5_mvdata_integration():
    """Example 5: Working with MVData objects."""
    print("\n=== Example 5: MVData Integration ===")

    # Create MVData object with known structure
    n_features = 100
    n_timepoints = 1000
    n_conditions = 5

    # Create condition labels
    condition_duration = n_timepoints // n_conditions
    conditions = np.repeat(np.arange(n_conditions), condition_duration)

    # Create data with distinct patterns per condition
    patterns = np.random.randn(n_conditions, n_features)
    data = np.zeros((n_features, n_timepoints))
    for i, cond in enumerate(conditions):
        data[:, i] = patterns[cond] + 0.1 * np.random.randn(n_features)

    # Create MVData object
    mvdata = MVData(data)

    print("Computing RDM from MVData object...")
    rdm, labels = rsa.compute_rdm_unified(mvdata, items=conditions)

    print(f"RDM shape: {rdm.shape}")
    print(f"Unique conditions: {labels}")

    # Visualize RDM
    fig = rsa.plot_rdm(
        rdm,
        labels=[f"Cond {i}" for i in labels],
        title="RDM from MVData",
        show_values=True,
    )
    plt.savefig("examples/rsa/rsa_mvdata_integration.png")
    plt.close()

    return mvdata, rdm


if __name__ == "__main__":
    print("DRIADA RSA Examples")
    print("=" * 60)

    try:
        exp1, rdm1, labels1 = example_1_stimulus_conditions()
    except Exception as e:
        print(f"\nExample 1 skipped due to: {e}")

    try:
        similarity = example_2_compare_regions()
    except Exception as e:
        print(f"\nExample 2 skipped due to: {e}")

    try:
        similarity3 = example_3_compare_experiments()
    except Exception as e:
        print(f"\nExample 3 skipped due to: {e}")

    try:
        rdm4a, rdm4b, sim4 = example_4_bootstrap_testing()
    except Exception as e:
        print(f"\nExample 4 skipped due to: {e}")

    try:
        mvdata5, rdm5 = example_5_mvdata_integration()
    except Exception as e:
        print(f"\nExample 5 skipped due to: {e}")

    print("\nAll examples completed!")
