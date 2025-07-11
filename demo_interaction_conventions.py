#!/usr/bin/env python3
"""
Demonstration of the two sign conventions for Interaction Information.

This script shows how McGill convention and Williams & Beer convention
differ in their signs for redundancy and synergy cases.
"""

import numpy as np
from scipy.stats import entropy


def mutual_information(x, y, base=2):
    """Calculate mutual information I(X;Y)."""
    # Create joint distribution
    xy_counts = {}
    for xi, yi in zip(x, y):
        key = (xi, yi)
        xy_counts[key] = xy_counts.get(key, 0) + 1
    
    # Convert to probabilities
    n = len(x)
    p_xy = np.array(list(xy_counts.values())) / n
    
    # Marginal distributions
    x_counts = {}
    y_counts = {}
    for xi, yi in zip(x, y):
        x_counts[xi] = x_counts.get(xi, 0) + 1
        y_counts[yi] = y_counts.get(yi, 0) + 1
    
    p_x = np.array(list(x_counts.values())) / n
    p_y = np.array(list(y_counts.values())) / n
    
    # Calculate MI
    h_x = entropy(p_x, base=base)
    h_y = entropy(p_y, base=base)
    h_xy = entropy(p_xy, base=base)
    
    return h_x + h_y - h_xy


def conditional_mutual_information(x, y, z, base=2):
    """Calculate conditional mutual information I(X;Y|Z)."""
    # Create joint distributions
    xyz_counts = {}
    xz_counts = {}
    yz_counts = {}
    z_counts = {}
    
    for xi, yi, zi in zip(x, y, z):
        xyz_counts[(xi, yi, zi)] = xyz_counts.get((xi, yi, zi), 0) + 1
        xz_counts[(xi, zi)] = xz_counts.get((xi, zi), 0) + 1
        yz_counts[(yi, zi)] = yz_counts.get((yi, zi), 0) + 1
        z_counts[zi] = z_counts.get(zi, 0) + 1
    
    n = len(x)
    
    # Calculate CMI by summing over z values
    cmi = 0.0
    for zi in z_counts:
        p_z = z_counts[zi] / n
        
        # Get conditional distributions
        x_given_z = []
        y_given_z = []
        for i, z_val in enumerate(z):
            if z_val == zi:
                x_given_z.append(x[i])
                y_given_z.append(y[i])
        
        if len(x_given_z) > 0:
            mi_given_z = mutual_information(x_given_z, y_given_z, base=base)
            cmi += p_z * mi_given_z
    
    return cmi


def create_redundancy_example():
    """Create example where Z = XOR(X, Y) - redundancy case."""
    np.random.seed(42)
    n = 1000
    
    # Binary random variables
    x = np.random.randint(0, 2, n)
    y = np.random.randint(0, 2, n)
    z = np.logical_xor(x, y).astype(int)
    
    return x, y, z


def create_synergy_example():
    """Create example where X and Y are independent but together determine Z."""
    np.random.seed(42)
    n = 1000
    
    # Independent binary variables
    x = np.random.randint(0, 2, n)
    y = np.random.randint(0, 2, n)
    # Z is deterministic function of both X and Y
    z = (x + y) % 2  # Another form of XOR
    
    return x, y, z


def analyze_case(x, y, z, case_name):
    """Analyze interaction information under both conventions."""
    print(f"\n{'='*60}")
    print(f"{case_name}")
    print(f"{'='*60}")
    
    # Calculate mutual informations
    mi_xy = mutual_information(x, y)
    mi_xz = mutual_information(x, z)
    mi_yz = mutual_information(y, z)
    cmi_xy_given_z = conditional_mutual_information(x, y, z)
    
    # McGill convention: II = I(X;Y) - I(X;Y|Z)
    ii_mcgill = mi_xy - cmi_xy_given_z
    
    # Williams & Beer convention: II = I(X;Y|Z) - I(X;Y)
    ii_williams = cmi_xy_given_z - mi_xy
    
    print(f"\nMutual Information values:")
    print(f"  I(X;Y) = {mi_xy:.4f}")
    print(f"  I(X;Z) = {mi_xz:.4f}")
    print(f"  I(Y;Z) = {mi_yz:.4f}")
    print(f"  I(X;Y|Z) = {cmi_xy_given_z:.4f}")
    
    print(f"\nInteraction Information:")
    print(f"  McGill convention (II = I(X;Y) - I(X;Y|Z)):")
    print(f"    II = {mi_xy:.4f} - {cmi_xy_given_z:.4f} = {ii_mcgill:.4f}")
    print(f"    Sign: {'POSITIVE' if ii_mcgill > 0 else 'NEGATIVE'} → {'REDUNDANCY' if ii_mcgill > 0 else 'SYNERGY'}")
    
    print(f"\n  Williams & Beer convention (II = I(X;Y|Z) - I(X;Y)):")
    print(f"    II = {cmi_xy_given_z:.4f} - {mi_xy:.4f} = {ii_williams:.4f}")
    print(f"    Sign: {'POSITIVE' if ii_williams > 0 else 'NEGATIVE'} → {'SYNERGY' if ii_williams > 0 else 'REDUNDANCY'}")
    
    print(f"\nInterpretation:")
    if abs(ii_mcgill) < 0.01:
        print("  Near zero interaction - neither strong redundancy nor synergy")
    elif ii_mcgill > 0:
        print("  McGill: Positive → Redundancy (X and Y share information about Z)")
        print("  W&B: Negative → Redundancy (same interpretation, opposite sign)")
    else:
        print("  McGill: Negative → Synergy (X and Y together provide more info about Z)")
        print("  W&B: Positive → Synergy (same interpretation, opposite sign)")


def create_perfect_redundancy_example():
    """Create example where X = Y (perfect redundancy)."""
    np.random.seed(42)
    n = 1000
    
    # X and Y are identical
    x = np.random.randint(0, 2, n)
    y = x.copy()
    z = x.copy()  # Z is also identical
    
    return x, y, z


def create_perfect_synergy_example():
    """Create example of perfect synergy - AND gate."""
    np.random.seed(42)
    n = 1000
    
    # Independent X and Y
    x = np.random.randint(0, 2, n)
    y = np.random.randint(0, 2, n)
    z = np.logical_and(x, y).astype(int)  # Z = X AND Y
    
    return x, y, z


def main():
    """Run demonstration of both conventions."""
    print("INTERACTION INFORMATION SIGN CONVENTIONS DEMONSTRATION")
    print("=====================================================")
    
    print("\nBackground:")
    print("-----------")
    print("Interaction Information (II) measures the amount of information")
    print("that is shared between X and Y about Z.")
    print("")
    print("Two main conventions exist:")
    print("1. McGill convention: II = I(X;Y) - I(X;Y|Z)")
    print("   - Positive for redundancy, negative for synergy")
    print("2. Williams & Beer convention: II = I(X;Y|Z) - I(X;Y)")
    print("   - Negative for redundancy, positive for synergy")
    
    # Test different cases
    cases = [
        ("PERFECT REDUNDANCY: X = Y = Z", create_perfect_redundancy_example()),
        ("REDUNDANCY: Z = XOR(X, Y)", create_redundancy_example()),
        ("PERFECT SYNERGY: Z = X AND Y", create_perfect_synergy_example()),
        ("SYNERGY: Z = (X + Y) mod 2", create_synergy_example()),
    ]
    
    for case_name, (x, y, z) in cases:
        analyze_case(x, y, z, case_name)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nKey takeaway:")
    print("- Both conventions measure the same concept")
    print("- They have OPPOSITE signs")
    print("- McGill: positive = redundancy, negative = synergy")
    print("- Williams & Beer: negative = redundancy, positive = synergy")
    print("\nCurrent INTENSE implementation uses McGill convention.")
    print("Tests may expect Williams & Beer convention.")


if __name__ == "__main__":
    main()