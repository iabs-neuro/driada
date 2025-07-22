#!/usr/bin/env python3
"""
Compare different MI testing approaches to find the best statistical method.

This script compares the current norm/gamma approach with improved alternatives
that better handle the non-normal nature of MI distributions.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from driada.intense.distribution_investigation import MIDistributionInvestigator
from driada.intense.improved_mi_testing import (
    compare_testing_methods, 
    ImprovedMITesting,
    empirical_p_value
)
from driada.intense.stats import get_mi_distr_pvalue


def analyze_testing_approaches():
    """Run comprehensive comparison of MI testing approaches."""
    
    print("=" * 80)
    print("COMPARING MI TESTING APPROACHES")
    print("=" * 80)
    print()
    
    # Generate test data using the investigation framework
    investigator = MIDistributionInvestigator(random_state=42)
    print("Generating test data...")
    shuffle_data = investigator.generate_test_data(n_scenarios=2, n_shuffles=500)
    
    if not shuffle_data:
        print("ERROR: No data generated")
        return
    
    print(f"Generated {len(shuffle_data)} test cases\n")
    
    # Compare methods on each test case
    comparison_results = []
    
    for i, data in enumerate(shuffle_data):
        print(f"\nTest case {i+1}: {data.neuron_id} - {data.feature_id}")
        print(f"  True MI: {data.true_mi:.4f}")
        print(f"  Shuffle distribution: mean={data.statistical_properties['mean']:.4f}, "
              f"std={data.statistical_properties['std']:.4f}")
        print(f"  Skewness: {data.statistical_properties['skewness']:.3f}")
        
        # Compare all methods
        results = compare_testing_methods(data.true_mi, data.shuffle_values)
        
        print("\n  P-value comparison:")
        for method_name, method_results in results.items():
            p_val = method_results['p_value']
            method_desc = method_results['method']
            print(f"    {method_name:15} p={p_val:8.4f}  ({method_desc})")
        
        comparison_results.append({
            'data': data,
            'results': results
        })
    
    # Analyze differences
    print("\n" + "=" * 80)
    print("ANALYSIS OF DIFFERENCES")
    print("=" * 80)
    
    # Count cases where methods disagree on significance
    disagreements = analyze_disagreements(comparison_results, alpha=0.05)
    
    print(f"\nDisagreement analysis (alpha=0.05):")
    for method1, method2, count, total in disagreements:
        if count > 0:
            print(f"  {method1} vs {method2}: {count}/{total} cases disagree ({100*count/total:.1f}%)")
    
    # Performance comparison
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    
    performance = evaluate_performance(comparison_results)
    
    print("\nDetection performance (based on ground truth from original analysis):")
    for method, metrics in performance.items():
        print(f"\n{method}:")
        print(f"  Sensitivity: {metrics['sensitivity']:.3f}")
        print(f"  Specificity: {metrics['specificity']:.3f}")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
    
    # Create visualizations
    create_comparison_visualizations(comparison_results)
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n1. EMPIRICAL P-VALUES:")
    print("   - Most statistically sound approach")
    print("   - No distributional assumptions")
    print("   - Conservative formula (r+1)/(n+1) prevents p=0")
    print("   - Recommended for n >= 100 shuffles")
    
    print("\n2. ROBUST PARAMETRIC:")
    print("   - Uses gamma when it fits well (KS p > 0.1)")
    print("   - Falls back to empirical when fit is poor")
    print("   - Good balance of power and validity")
    
    print("\n3. ADAPTIVE DISTRIBUTION:")
    print("   - Selects best-fitting distribution automatically")
    print("   - Can be more powerful but less interpretable")
    print("   - Risk of overfitting with small samples")
    
    print("\n4. CURRENT APPROACH ISSUES:")
    print("   - Normal distribution clearly inappropriate")
    print("   - Gamma better but still imperfect for many cases")
    print("   - Both can give misleading p-values")
    
    # Test improved framework
    print("\n" + "=" * 80)
    print("TESTING IMPROVED FRAMEWORK")
    print("=" * 80)
    
    tester = ImprovedMITesting(method='auto')
    
    print("\nTesting auto method selection:")
    for i, data in enumerate(shuffle_data[:3]):  # Test on first 3 cases
        p_val, details = tester.compute_p_value(
            data.true_mi, 
            data.shuffle_values,
            return_details=True
        )
        print(f"\nCase {i+1}:")
        print(f"  P-value: {p_val:.4f}")
        print(f"  Method selected: {details['method']}")


def analyze_disagreements(comparison_results, alpha=0.05):
    """Analyze where different methods disagree on significance."""
    disagreements = []
    
    methods = list(comparison_results[0]['results'].keys())
    
    for i, method1 in enumerate(methods):
        for method2 in methods[i+1:]:
            disagree_count = 0
            total_count = 0
            
            for result in comparison_results:
                p1 = result['results'][method1]['p_value']
                p2 = result['results'][method2]['p_value']
                
                if not (np.isnan(p1) or np.isnan(p2)):
                    total_count += 1
                    # Check if they disagree on significance
                    if (p1 < alpha) != (p2 < alpha):
                        disagree_count += 1
            
            disagreements.append((method1, method2, disagree_count, total_count))
    
    return disagreements


def evaluate_performance(comparison_results):
    """Evaluate detection performance of each method."""
    performance = {}
    
    methods = list(comparison_results[0]['results'].keys())
    
    for method in methods:
        tp = fp = tn = fn = 0
        
        for result in comparison_results:
            data = result['data']
            p_val = result['results'][method]['p_value']
            
            if np.isnan(p_val):
                continue
            
            predicted_sig = p_val < 0.05
            actual_sig = data.is_significant
            
            if predicted_sig and actual_sig:
                tp += 1
            elif predicted_sig and not actual_sig:
                fp += 1
            elif not predicted_sig and actual_sig:
                fn += 1
            else:
                tn += 1
        
        total = tp + fp + tn + fn
        if total > 0:
            performance[method] = {
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'accuracy': (tp + tn) / total
            }
        else:
            performance[method] = {
                'sensitivity': 0,
                'specificity': 0,
                'accuracy': 0
            }
    
    return performance


def create_comparison_visualizations(comparison_results):
    """Create visualizations comparing testing methods."""
    
    # Extract p-values for each method
    methods = list(comparison_results[0]['results'].keys())
    p_values = {method: [] for method in methods}
    
    for result in comparison_results:
        for method in methods:
            p_val = result['results'][method]['p_value']
            if not np.isnan(p_val):
                p_values[method].append(p_val)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Comparison of MI Testing Methods', fontsize=16)
    
    # 1. P-value distributions
    ax1 = axes[0, 0]
    for method in ['current_norm', 'current_gamma', 'empirical', 'robust']:
        if method in p_values and p_values[method]:
            ax1.hist(p_values[method], alpha=0.5, bins=20, label=method, density=True)
    ax1.set_xlabel('P-value')
    ax1.set_ylabel('Density')
    ax1.set_title('P-value Distributions')
    ax1.legend()
    
    # 2. Scatter plot: Empirical vs Current methods
    ax2 = axes[0, 1]
    emp_vals = []
    norm_vals = []
    gamma_vals = []
    
    for result in comparison_results:
        if 'empirical' in result['results'] and 'current_norm' in result['results']:
            emp = result['results']['empirical']['p_value']
            norm = result['results']['current_norm']['p_value']
            gamma = result['results']['current_gamma']['p_value']
            if not any(np.isnan([emp, norm, gamma])):
                emp_vals.append(emp)
                norm_vals.append(norm)
                gamma_vals.append(gamma)
    
    ax2.scatter(emp_vals, norm_vals, alpha=0.6, label='Normal', color='blue')
    ax2.scatter(emp_vals, gamma_vals, alpha=0.6, label='Gamma', color='red')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_xlabel('Empirical p-value')
    ax2.set_ylabel('Parametric p-value')
    ax2.set_title('Empirical vs Parametric P-values')
    ax2.legend()
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # 3. Method agreement heatmap
    ax3 = axes[1, 0]
    
    # Calculate correlation matrix between methods
    method_subset = ['current_norm', 'current_gamma', 'empirical', 'robust', 'adaptive']
    corr_matrix = np.zeros((len(method_subset), len(method_subset)))
    
    for i, method1 in enumerate(method_subset):
        for j, method2 in enumerate(method_subset):
            vals1 = []
            vals2 = []
            for result in comparison_results:
                if method1 in result['results'] and method2 in result['results']:
                    p1 = result['results'][method1]['p_value']
                    p2 = result['results'][method2]['p_value']
                    if not (np.isnan(p1) or np.isnan(p2)):
                        vals1.append(p1)
                        vals2.append(p2)
            
            if vals1:
                corr_matrix[i, j] = np.corrcoef(vals1, vals2)[0, 1]
            else:
                corr_matrix[i, j] = np.nan
    
    im = ax3.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax3.set_xticks(range(len(method_subset)))
    ax3.set_yticks(range(len(method_subset)))
    ax3.set_xticklabels(method_subset, rotation=45, ha='right')
    ax3.set_yticklabels(method_subset)
    ax3.set_title('Method Agreement (Correlation)')
    
    # Add correlation values
    for i in range(len(method_subset)):
        for j in range(len(method_subset)):
            if not np.isnan(corr_matrix[i, j]):
                text = ax3.text(j, i, f'{corr_matrix[i, j]:.2f}',
                               ha='center', va='center', color='black' if abs(corr_matrix[i, j]) < 0.5 else 'white')
    
    # 4. Example shuffle distribution with different fits
    ax4 = axes[1, 1]
    
    # Use first test case as example
    if comparison_results:
        example_data = comparison_results[0]['data']
        shuffle_vals = example_data.shuffle_values
        true_mi = example_data.true_mi
        
        # Plot histogram
        counts, bins, _ = ax4.hist(shuffle_vals, bins=30, density=True, alpha=0.5, color='gray')
        
        # Overlay fitted distributions
        x_range = np.linspace(shuffle_vals.min(), shuffle_vals.max(), 100)
        
        # Fit and plot normal
        from scipy import stats
        norm_params = stats.norm.fit(shuffle_vals)
        ax4.plot(x_range, stats.norm.pdf(x_range, *norm_params), 'b-', label='Normal fit')
        
        # Fit and plot gamma
        if np.all(shuffle_vals > 0):
            gamma_params = stats.gamma.fit(shuffle_vals, floc=0)
            ax4.plot(x_range, stats.gamma.pdf(x_range, *gamma_params), 'r-', label='Gamma fit')
        
        # Mark true MI
        ax4.axvline(true_mi, color='green', linestyle='--', linewidth=2, label=f'True MI = {true_mi:.3f}')
        
        ax4.set_xlabel('MI value')
        ax4.set_ylabel('Density')
        ax4.set_title('Example: Shuffle Distribution with Fits')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig('mi_testing_comparison.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: mi_testing_comparison.png")
    plt.show()


if __name__ == "__main__":
    analyze_testing_approaches()