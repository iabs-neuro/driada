#!/usr/bin/env python3
"""
Mixed Selectivity Analysis Example

This example demonstrates advanced disentanglement analysis using DRIADA's INTENSE module:
1. Generate synthetic data with mixed selectivity patterns and MultiTimeSeries features
2. Run standard INTENSE analysis to find significant relationships
3. Apply disentanglement analysis to separate mixed selectivity 
4. Visualize disentanglement results and interpret findings
5. Demonstrate redundancy vs synergy detection

This showcases INTENSE's advanced capabilities for analyzing neurons that respond 
to multiple correlated behavioral variables, including multivariate features.
"""

import sys
import os

# Add the src directory to the path to import driada
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import driada
import matplotlib.pyplot as plt
import numpy as np


def generate_mixed_selectivity_data():
    """Generate synthetic data with known mixed selectivity patterns and MultiTimeSeries."""
    print("\n=== GENERATING MIXED SELECTIVITY DATA ===")
    
    # For demonstration, create a smaller experiment with stronger selectivity
    exp, selectivity_info = driada.experiment.generate_synthetic_exp_with_mixed_selectivity(
        n_discrete_feats=2,      # Fewer features for clearer demonstration
        n_continuous_feats=2,    # Fewer continuous features
        n_neurons=30,            # Smaller population for faster analysis
        n_multifeatures=1,       # One multifeature
        selectivity_prob=1.0,    # All neurons are selective
        multi_select_prob=0.8,   # Most have mixed selectivity
        weights_mode='dominant', # One feature dominates (clearer for disentanglement)
        duration=600,            # 10 minutes
        seed=123,                # Different seed
        fps=20,
        verbose=False,
        create_discrete_pairs=True,  # Create discrete versions for disentanglement demo
        # Stronger signal parameters for better detection
        rate_0=0.01,             # Lower baseline for better dynamic range
        rate_1=3.0,              # Higher active rate for stronger signals
        skip_prob=0.05,          # Less skipping for more consistent signals
        ampl_range=(1.5, 3.5),   # Stronger calcium transients
        decay_time=2,
        noise_std=0.05           # Lower noise for better SNR
    )
    
    print(f"Generated experiment: {exp.n_cells} neurons, {len(exp.dynamic_features)} features, {exp.n_frames/exp.fps:.1f}s recording")
    
    # Debug: Show selectivity matrix info
    if 'matrix' in selectivity_info:
        matrix = selectivity_info['matrix']
        n_selective = np.sum(np.any(matrix > 0, axis=0))
        n_mixed = np.sum(np.sum(matrix > 0, axis=0) > 1)
        print(f"Selectivity matrix: {n_selective} selective neurons, {n_mixed} with mixed selectivity (ground truth)")
        
        # Show which features each neuron is selective to (first 10 neurons)
        print("\nGround truth selectivity (first 10 neurons):")
        for j in range(min(10, matrix.shape[1])):
            selective_features = np.where(matrix[:, j] > 0)[0]
            if len(selective_features) > 0:
                feat_names = [selectivity_info['feature_names'][i] for i in selective_features]
                print(f"  Neuron {j}: {feat_names}")
    
    return exp, selectivity_info


def run_intense_analysis(exp):
    """Run INTENSE analysis to identify significant relationships."""
    print("\n=== RUNNING INTENSE ANALYSIS ===")
    
    # Run comprehensive analysis including multifeatures
    # Get features that should skip delay optimization
    # Check both tuple names (multifeatures) and MultiTimeSeries instances
    skip_delays = []
    for feat_name, feat_data in exp.dynamic_features.items():
        if isinstance(feat_name, tuple):  # Tuple name indicates multifeature
            skip_delays.append(feat_name)
        elif hasattr(feat_data, '__class__') and feat_data.__class__.__name__ == 'MultiTimeSeries':
            skip_delays.append(feat_name)
    
    # Run comprehensive analysis with disentanglement
    results = driada.compute_cell_feat_significance(
        exp,
        mode='two_stage',
        n_shuffles_stage1=50,    # Increased for better statistics
        n_shuffles_stage2=500,   # Increased for more reliable p-values
        allow_mixed_dimensions=True,  # Enable MultiTimeSeries analysis
        skip_delays=skip_delays if skip_delays else None,  # Skip delay optimization for MultiTimeSeries
        verbose=False,
        with_disentanglement=True,  # Enable disentanglement analysis
        multifeature_map=driada.intense.DEFAULT_MULTIFEATURE_MAP,
        metric_distr_type='norm',  # Use normal (Gaussian) distribution for shuffled MI
        pval_thr=0.05  # Slightly less conservative threshold
    )
    
    # Unpack results based on whether disentanglement was included
    if len(results) == 5:
        stats, significance, info, intense_results, disentanglement_results = results
    else:
        stats, significance, info, intense_results = results
        disentanglement_results = None
    
    # Extract significant relationships
    significant_neurons = exp.get_significant_neurons()
    
    # Debug: Check if we're getting any significant relationships at all
    if 'calcium' in significance:
        sig_cal = significance['calcium']
        stage1_passed = np.sum([v.get('stage1', 0) == 1 for v in sig_cal.values() if isinstance(v, dict)])
        stage2_passed = np.sum([v.get('stage2', 0) == 1 for v in sig_cal.values() if isinstance(v, dict)])
        total_pairs = len([v for v in sig_cal.values() if isinstance(v, dict)])
        print(f"\nDEBUG: Stage 1 passed pairs: {stage1_passed} out of {total_pairs}")
        print(f"DEBUG: Stage 2 passed pairs: {stage2_passed} out of {total_pairs}")
        
        # Check p-values for some pairs
        pvals = []
        for k, v in sig_cal.items():
            if isinstance(v, dict) and 'pval' in v and v['pval'] is not None:
                pvals.append(v['pval'])
        if pvals:
            print(f"DEBUG: P-value distribution: min={min(pvals):.4f}, median={np.median(pvals):.4f}, max={max(pvals):.4f}")
            print(f"DEBUG: P-values < 0.05: {np.sum(np.array(pvals) < 0.05)} out of {len(pvals)}")
    
    # Check some MI values
    if hasattr(exp, 'stats_tables') and 'calcium' in exp.stats_tables:
        mi_values = []
        for cell_id in range(min(10, exp.n_cells)):  # Check first 10 neurons
            for feat in list(exp.dynamic_features.keys())[:3]:  # Check first 3 features
                if isinstance(feat, str):
                    pair_stats = exp.get_neuron_feature_pair_stats(cell_id, feat, mode='calcium')
                    if pair_stats and 'me' in pair_stats and pair_stats['me'] is not None:
                        mi_values.append(pair_stats['me'])
        if mi_values:
            print(f"DEBUG: Sample MI values: min={min(mi_values):.4f}, max={max(mi_values):.4f}, mean={np.mean(mi_values):.4f}")
    
    # Count multifeature relationships
    from driada.information.info_base import MultiTimeSeries
    multifeature_count = 0
    for cell_id, features in significant_neurons.items():
        for feat in features:
            if feat in exp.dynamic_features and isinstance(exp.dynamic_features[feat], MultiTimeSeries):
                multifeature_count += 1
    
    # Identify neurons with multiple significant features (mixed selectivity candidates)
    mixed_candidates = {cell_id: features for cell_id, features in significant_neurons.items() 
                       if len(features) > 1}
    
    print(f"INTENSE found {len(significant_neurons)} significant neurons, {len(mixed_candidates)} with mixed selectivity")
    
    # Debug: Show what features neurons are selective to
    if mixed_candidates:
        print("\nMixed selectivity details:")
        for cell_id, features in list(mixed_candidates.items())[:5]:  # Show first 5
            print(f"  Neuron {cell_id}: selective to {features}")
    else:
        # Show single selectivity neurons to debug
        print("\nSingle selectivity neurons found:")
        for i, (cell_id, features) in enumerate(significant_neurons.items()):
            if i < 5:  # Show first 5
                print(f"  Neuron {cell_id}: selective to {features}")
    
    return stats, significance, info, intense_results, significant_neurons, mixed_candidates, disentanglement_results


def analyze_disentanglement(disentanglement_results, mixed_candidates):
    """Process disentanglement results from the pipeline."""
    print("\n=== DISENTANGLEMENT ANALYSIS ===")
    
    if not mixed_candidates:
        print("No mixed selectivity candidates found for disentanglement.")
        return None, None, None
    
    if disentanglement_results is None:
        print("No disentanglement results available.")
        return None, None, None
    
    # Extract results from the pipeline
    disent_matrix = disentanglement_results.get('disent_matrix')
    count_matrix = disentanglement_results.get('count_matrix')
    feat_names = disentanglement_results.get('feature_names', [])
    
    if disent_matrix is None or count_matrix is None:
        print("Disentanglement matrices not found in results.")
        return None, None, None
    
    print(f"Disentanglement analysis completed by pipeline")
    print(f"Matrix shape: {disent_matrix.shape}, Non-zero entries: {np.count_nonzero(count_matrix)}")
    print(f"Feature names analyzed: {feat_names}")
    
    # Show summary if available
    if 'summary' in disentanglement_results:
        summary = disentanglement_results['summary']
        if 'overall_stats' in summary:
            stats = summary['overall_stats']
            print(f"\nOverall statistics:")
            print(f"  Total neuron pairs: {stats.get('total_neuron_pairs', 0)}")
            print(f"  Redundancy rate: {stats.get('redundancy_rate', 0):.1f}%")
            print(f"  True mixed selectivity rate: {stats.get('true_mixed_selectivity_rate', 0):.1f}%")
    
    return disent_matrix, count_matrix, feat_names


def interpret_disentanglement_results(exp, disent_matrix, count_matrix, feat_names, mixed_candidates):
    """Interpret and summarize disentanglement analysis results from matrices."""
    print("\n=== INTERPRETING DISENTANGLEMENT RESULTS ===")
    
    if disent_matrix is None or count_matrix is None:
        return [], [], []
    
    redundancy_cases = []
    synergy_cases = []
    independence_cases = []
    
    # Calculate relative disentanglement matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_disent_matrix = np.divide(disent_matrix, count_matrix) * 100
        rel_disent_matrix[count_matrix == 0] = np.nan
    
    # Extract disentanglement cases based on matrix values
    for i in range(len(feat_names)):
        for j in range(i + 1, len(feat_names)):  # Only upper triangle to avoid duplicates
            if count_matrix[i, j] > 0:  # Only consider pairs with data
                feat1 = feat_names[i]
                feat2 = feat_names[j]
                
                # Get disentanglement score (percentage)
                disent_score = rel_disent_matrix[i, j]
                
                if not np.isnan(disent_score):
                    # Classify based on disentanglement score
                    if disent_score < 30:  # Redundancy: feat2 dominates
                        redundancy_cases.append((f"{feat1}-{feat2}", (feat1, feat2), disent_score/100))
                    elif disent_score > 70:  # Synergy: feat1 dominates  
                        synergy_cases.append((f"{feat1}-{feat2}", (feat1, feat2), disent_score/100))
                    else:  # Independence: balanced
                        independence_cases.append((f"{feat1}-{feat2}", (feat1, feat2), disent_score/100))
    
    # Summary statistics
    total_pairs = len(redundancy_cases) + len(synergy_cases) + len(independence_cases)
    print(f"Found {len(redundancy_cases)} redundancy, {len(independence_cases)} independence, {len(synergy_cases)} synergy cases")
    
    # Show a few examples if available
    if redundancy_cases and len(redundancy_cases) > 0:
        feat1, feat2 = redundancy_cases[0][1]
        print(f"Example redundancy: {feat1} ↔ {feat2}")
    elif synergy_cases and len(synergy_cases) > 0:
        feat1, feat2 = synergy_cases[0][1]
        print(f"Example synergy: {feat1} + {feat2}")
    
    return redundancy_cases, synergy_cases, independence_cases


def create_disentanglement_visualization(exp, disent_matrix, count_matrix, feat_names, mixed_candidates, output_dir):
    """Create comprehensive visualization using production-grade visual.py functions."""
    print("\n=== CREATING DISENTANGLEMENT VISUALIZATIONS ===")
    
    if disent_matrix is None or count_matrix is None or not mixed_candidates:
        print("No disentanglement results to visualize")
        return
    
    try:
        # Create disentanglement heatmap using visual.py
        fig1, ax1 = driada.intense.plot_disentanglement_heatmap(
            disent_matrix, 
            count_matrix, 
            feat_names,
            title="Feature Disentanglement Analysis",
            figsize=(10, 8)
        )
        
        # Save heatmap
        heatmap_path = os.path.join(output_dir, 'disentanglement_heatmap.png')
        fig1.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        
    except Exception as e:
        print(f"Error creating disentanglement heatmap: {str(e)}")
        fig1 = None
    
    try:
        # Create comprehensive summary using visual.py
        fig2 = driada.intense.plot_disentanglement_summary(
            disent_matrix,
            count_matrix, 
            feat_names,
            title_prefix="Mixed Selectivity - ",
            figsize=(14, 10)
        )
        
        # Save summary
        summary_path = os.path.join(output_dir, 'disentanglement_summary.png')
        fig2.savefig(summary_path, dpi=150, bbox_inches='tight')
        
    except Exception as e:
        print(f"Error creating disentanglement summary: {str(e)}")
        fig2 = None
    
    # Create an additional plot showing example neuron-feature relationship
    if mixed_candidates:
        try:
            example_cell = list(mixed_candidates.keys())[0]
            example_features = mixed_candidates[example_cell]
            
            # Try to find a multifeature example
            from driada.information.info_base import MultiTimeSeries
            multifeature_example = None
            for feat in example_features:
                if feat in exp.dynamic_features and isinstance(exp.dynamic_features[feat], MultiTimeSeries):
                    multifeature_example = feat
                    break
            
            example_feature = multifeature_example if multifeature_example else example_features[0]
            
            fig3 = driada.intense.plot_neuron_feature_pair(
                exp, example_cell, example_feature, 
                title=f"Example Mixed Selectivity: Neuron {example_cell} ↔ {example_feature}"
            )
            
            # Save example
            example_path = os.path.join(output_dir, 'example_mixed_selectivity.png')
            fig3.savefig(example_path, dpi=150, bbox_inches='tight')
            
        except Exception as e:
            print(f"Error creating example plot: {str(e)}")
    
    print(f"\nVisualizations saved: disentanglement_heatmap.png, disentanglement_summary.png")
    
    # Show plots if successful
    if fig1 is not None:
        plt.figure(fig1.number)
        plt.show()
    if fig2 is not None:
        plt.figure(fig2.number)
        plt.show()


def main():
    """Run mixed selectivity analysis example."""
    print("=" * 80)
    print("DRIADA INTENSE - Mixed Selectivity Analysis Example")
    print("=" * 80)
    print("\nThis example demonstrates advanced disentanglement analysis for neurons")
    print("that respond to multiple correlated behavioral variables, including")
    print("MultiTimeSeries features that combine multiple signals.")
    
    output_dir = os.path.dirname(__file__)
    
    # Step 1: Generate mixed selectivity data with MultiTimeSeries
    exp, selectivity_info = generate_mixed_selectivity_data()
    
    # Step 2: Run INTENSE analysis with disentanglement
    stats, significance, info, intense_results, significant_neurons, mixed_candidates, disentanglement_results = run_intense_analysis(exp)
    
    # Step 3: Process disentanglement results
    disent_matrix, count_matrix, feat_names = analyze_disentanglement(disentanglement_results, mixed_candidates)
    
    # Step 4: Interpret results
    redundancy_cases, synergy_cases, independence_cases = interpret_disentanglement_results(
        exp, disent_matrix, count_matrix, feat_names, mixed_candidates
    )
    
    # Step 5: Create visualizations
    if disent_matrix is not None:
        create_disentanglement_visualization(
            exp, disent_matrix, count_matrix, feat_names, 
            mixed_candidates, output_dir
        )
    
    # Final summary and interpretation guide
    print("\n" + "=" * 80)
    print("MIXED SELECTIVITY ANALYSIS COMPLETE")
    print("=" * 80)
    
    print(f"\nResults interpretation guide:")
    print(f"• Redundancy (< 0.3): Features provide overlapping information about neuron")
    print(f"• Independence (0.3-0.7): Features provide separate information")
    print(f"• Synergy (> 0.7): Combined features provide more info than sum of parts")
    
    print(f"\nMultiTimeSeries interpretation:")
    print(f"• Spatial MultiTimeSeries (x,y): Neurons encoding 2D position")
    print(f"• Movement MultiTimeSeries (speed,direction): Neurons encoding motion")
    print(f"• These combined features often show higher MI than individual components")
    
    print(f"\nBiological interpretations:")
    print(f"• Redundancy: Multiple correlated behavioral variables (e.g., speed & distance)")
    print(f"• Independence: Neuron encodes multiple independent variables")
    print(f"• Synergy: Neuron requires combination of variables (e.g., place cells need X+Y)")
    
    print(f"\nAnalysis complete: {exp.n_cells} neurons, {len(mixed_candidates)} with mixed selectivity")
    print(f"Results: {len(redundancy_cases)} redundancy, {len(synergy_cases)} synergy cases")


if __name__ == "__main__":
    main()