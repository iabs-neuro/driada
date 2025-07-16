#!/usr/bin/env python3
"""
Investigation of MI Distribution Types for INTENSE Statistical Testing.

This script investigates why metric_distr_type='norm' works better than 'gamma' 
for MI distributions in INTENSE, despite gamma being theoretically more appropriate.

Usage:
    python investigate_mi_distributions.py

Results will be saved to:
    - mi_distribution_investigation_results.png (visualizations)
    - mi_distribution_investigation_report.txt (summary report)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from driada.intense.distribution_investigation import MIDistributionInvestigator


def main():
    """Run the MI distribution investigation."""
    
    print("=" * 80)
    print("INVESTIGATING MI DISTRIBUTION TYPES FOR INTENSE")
    print("=" * 80)
    print()
    
    # Initialize investigator
    investigator = MIDistributionInvestigator(random_state=42)
    
    # Generate test data
    print("Phase 1: Generating test data...")
    shuffle_data = investigator.generate_test_data(
        n_scenarios=2,  # Reduced for initial testing
        n_shuffles=500  # Reduced for faster execution
    )
    
    if not shuffle_data:
        print("ERROR: No shuffle data generated. Exiting.")
        return
    
    print(f"Successfully generated {len(shuffle_data)} shuffle distributions")
    print()
    
    # Analyze distribution fitting
    print("Phase 2: Analyzing distribution fitting...")
    fit_results = investigator.analyze_all_distributions()
    print()
    
    # Generate summary report
    print("Phase 3: Generating summary report...")
    report = investigator.generate_summary_report()
    
    # Save report to file
    report_path = "mi_distribution_investigation_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Report saved to: {report_path}")
    print()
    
    # Create visualizations
    print("Phase 4: Creating visualizations...")
    viz_path = "mi_distribution_investigation_results.png"
    investigator.create_visualizations(save_path=viz_path)
    print()
    
    # Print summary to console
    print("=" * 80)
    print("INVESTIGATION SUMMARY")
    print("=" * 80)
    print(report)
    print()
    
    # Additional analysis: Show specific examples
    print("=" * 80)
    print("DETAILED EXAMPLES")
    print("=" * 80)
    
    # Show examples of distributions where norm vs gamma differ significantly
    significant_differences = []
    for data in shuffle_data:
        p_diff = abs(data.p_value_norm - data.p_value_gamma)
        if p_diff > 0.01:  # Significant difference in p-values
            significant_differences.append((data, p_diff))
    
    # Sort by difference magnitude
    significant_differences.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Found {len(significant_differences)} cases with significant p-value differences")
    print()
    
    # Show top 5 cases
    for i, (data, diff) in enumerate(significant_differences[:5]):
        print(f"Case {i+1}: {data.neuron_id} - {data.feature_id}")
        print(f"  P-value difference: {diff:.4f}")
        print(f"  Normal p-value: {data.p_value_norm:.4f}")
        print(f"  Gamma p-value: {data.p_value_gamma:.4f}")
        print(f"  True MI: {data.true_mi:.4f}")
        print(f"  Shuffle distribution properties:")
        print(f"    Mean: {data.statistical_properties['mean']:.4f}")
        print(f"    Std: {data.statistical_properties['std']:.4f}")
        print(f"    Skewness: {data.statistical_properties['skewness']:.4f}")
        print(f"    Kurtosis: {data.statistical_properties['kurtosis']:.4f}")
        print(f"    Shapiro p-value: {data.statistical_properties['shapiro_pvalue']:.4f}")
        print()
    
    # Analysis of normality
    print("=" * 80)
    print("NORMALITY ANALYSIS")
    print("=" * 80)
    
    # Count distributions that appear normal
    normal_count = 0
    gamma_like_count = 0
    
    for data in shuffle_data:
        shapiro_p = data.statistical_properties['shapiro_pvalue']
        skewness = data.statistical_properties['skewness']
        
        if not np.isnan(shapiro_p) and shapiro_p > 0.05:
            normal_count += 1
        
        # Consider gamma-like if positive skewness > 0.5
        if skewness > 0.5:
            gamma_like_count += 1
    
    print(f"Distributions appearing normal (Shapiro p > 0.05): {normal_count}/{len(shuffle_data)} ({100*normal_count/len(shuffle_data):.1f}%)")
    print(f"Distributions with gamma-like skewness (> 0.5): {gamma_like_count}/{len(shuffle_data)} ({100*gamma_like_count/len(shuffle_data):.1f}%)")
    print()
    
    # Key findings
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    # Compare detection rates
    performance = investigator.compare_detection_performance()
    
    print("1. DETECTION PERFORMANCE:")
    for dist_name, metrics in performance.items():
        print(f"   {dist_name.upper()}: Accuracy = {metrics['accuracy']:.3f}, Sensitivity = {metrics['sensitivity']:.3f}")
    
    print()
    print("2. DISTRIBUTION CHARACTERISTICS:")
    avg_skewness = np.mean([d.statistical_properties['skewness'] for d in shuffle_data])
    avg_kurtosis = np.mean([d.statistical_properties['kurtosis'] for d in shuffle_data])
    print(f"   Average skewness: {avg_skewness:.3f} (gamma expected: > 0)")
    print(f"   Average kurtosis: {avg_kurtosis:.3f} (gamma expected: > 0)")
    
    print()
    print("3. THEORETICAL VS EMPIRICAL:")
    print(f"   Theory predicts: MI values should be gamma-distributed")
    print(f"   Empirical finding: {normal_count}/{len(shuffle_data)} distributions appear normal")
    print(f"   Performance: {'Normal' if performance['norm']['accuracy'] > performance['gamma']['accuracy'] else 'Gamma'} distribution gives better detection")
    
    print()
    print("=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)
    print(f"Full report saved to: {report_path}")
    print(f"Visualizations saved to: {viz_path}")


if __name__ == "__main__":
    main()