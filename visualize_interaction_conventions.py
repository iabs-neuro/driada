#!/usr/bin/env python3
"""
Visualize the two sign conventions for Interaction Information.
Creates plots showing how the conventions differ.
"""

import numpy as np
import matplotlib.pyplot as plt
from demo_interaction_conventions import (
    mutual_information, 
    conditional_mutual_information,
    create_redundancy_example,
    create_synergy_example,
    create_perfect_redundancy_example,
    create_perfect_synergy_example
)


def calculate_ii_values(examples):
    """Calculate II values for all examples under both conventions."""
    results = []
    
    for name, (x, y, z) in examples:
        mi_xy = mutual_information(x, y)
        cmi_xy_given_z = conditional_mutual_information(x, y, z)
        
        # McGill convention
        ii_mcgill = mi_xy - cmi_xy_given_z
        
        # Williams & Beer convention  
        ii_williams = cmi_xy_given_z - mi_xy
        
        results.append({
            'name': name,
            'ii_mcgill': ii_mcgill,
            'ii_williams': ii_williams,
            'type': 'Redundancy' if ii_mcgill > 0 else 'Synergy'
        })
    
    return results


def create_comparison_plot():
    """Create visual comparison of the two conventions."""
    # Generate examples
    examples = [
        ("Perfect\nRedundancy\n(X=Y=Z)", create_perfect_redundancy_example()),
        ("XOR\nRedundancy", create_redundancy_example()),
        ("AND Gate\nSynergy", create_perfect_synergy_example()),
        ("Modular\nSynergy", create_synergy_example()),
    ]
    
    results = calculate_ii_values(examples)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Extract data
    names = [r['name'] for r in results]
    mcgill_values = [r['ii_mcgill'] for r in results]
    williams_values = [r['ii_williams'] for r in results]
    types = [r['type'] for r in results]
    
    x_pos = np.arange(len(names))
    
    # McGill convention plot
    colors1 = ['#2ecc71' if v > 0 else '#e74c3c' for v in mcgill_values]
    bars1 = ax1.bar(x_pos, mcgill_values, color=colors1, alpha=0.8, edgecolor='black')
    ax1.set_title('McGill Convention: II = I(X;Y) - I(X;Y|Z)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Interaction Information', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, mcgill_values)):
        height = bar.get_height()
        label = f'{val:.3f}\n({types[i]})'
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.02 if height > 0 else height - 0.08,
                 label, ha='center', va='bottom' if height > 0 else 'top', fontsize=10)
    
    # Add convention explanation
    ax1.text(0.02, 0.98, 'Green (Positive) = Redundancy\nRed (Negative) = Synergy', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Williams & Beer convention plot
    colors2 = ['#e74c3c' if v < 0 else '#2ecc71' for v in williams_values]
    bars2 = ax2.bar(x_pos, williams_values, color=colors2, alpha=0.8, edgecolor='black')
    ax2.set_title('Williams & Beer Convention: II = I(X;Y|Z) - I(X;Y)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Interaction Information', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, williams_values)):
        height = bar.get_height()
        label = f'{val:.3f}\n({types[i]})'
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.02 if height > 0 else height - 0.08,
                 label, ha='center', va='bottom' if height > 0 else 'top', fontsize=10)
    
    # Add convention explanation
    ax2.text(0.02, 0.98, 'Red (Negative) = Redundancy\nGreen (Positive) = Synergy', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('interaction_conventions_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('interaction_conventions_comparison.pdf', bbox_inches='tight')
    plt.show()


def create_sign_reference_table():
    """Create a reference table showing sign interpretations."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data
    columns = ['Information Type', 'McGill Convention\nII = I(X;Y) - I(X;Y|Z)', 
               'Williams & Beer Convention\nII = I(X;Y|Z) - I(X;Y)']
    
    data = [
        ['Redundancy\n(X and Y share info about Z)', 
         'Positive (+)', 
         'Negative (-)'],
        ['Synergy\n(X and Y together give more info)', 
         'Negative (-)', 
         'Positive (+)'],
        ['Interpretation', 
         'Positive → Redundancy\nNegative → Synergy', 
         'Negative → Redundancy\nPositive → Synergy']
    ]
    
    # Create table
    table = ax.table(cellText=data, colLabels=columns, loc='center', 
                     cellLoc='center', colWidths=[0.35, 0.325, 0.325])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Color header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color cells based on content
    for i in range(1, len(data) + 1):
        table[(i, 0)].set_facecolor('#ecf0f1')
        table[(i, 0)].set_text_props(weight='bold')
        
        if i <= 2:  # Redundancy and Synergy rows
            # McGill column
            if 'Positive' in data[i-1][1]:
                table[(i, 1)].set_facecolor('#d5f4e6')
            else:
                table[(i, 1)].set_facecolor('#f8d7da')
            
            # Williams & Beer column
            if 'Positive' in data[i-1][2]:
                table[(i, 2)].set_facecolor('#d5f4e6')
            else:
                table[(i, 2)].set_facecolor('#f8d7da')
    
    plt.title('Interaction Information Sign Conventions Reference', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig('interaction_sign_reference.png', dpi=300, bbox_inches='tight')
    plt.savefig('interaction_sign_reference.pdf', bbox_inches='tight')
    plt.show()


def main():
    """Create all visualizations."""
    print("Creating visualization of interaction information conventions...")
    
    # Create comparison plot
    create_comparison_plot()
    
    # Create reference table
    create_sign_reference_table()
    
    print("\nVisualizations saved as:")
    print("  - interaction_conventions_comparison.png/pdf")
    print("  - interaction_sign_reference.png/pdf")


if __name__ == "__main__":
    main()