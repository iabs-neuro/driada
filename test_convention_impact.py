#!/usr/bin/env python3
"""
Test script showing how sign conventions impact INTENSE implementation.
Demonstrates the current McGill convention vs expected Williams & Beer convention.
"""

import numpy as np
import torch
import sys
sys.path.insert(0, 'src')
from driada.intense.stats import compute_interaction_information_matrix


def test_redundancy_case():
    """Test redundancy case with both conventions."""
    print("\n" + "="*60)
    print("REDUNDANCY TEST CASE: Identical neurons")
    print("="*60)
    
    # Create redundant neurons (all identical)
    np.random.seed(42)
    n_samples = 1000
    n_neurons = 3
    
    # All neurons have identical activity
    base_activity = np.random.rand(n_samples, 1)
    X = np.tile(base_activity, (1, n_neurons))
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Create DiT instance
    dit = DriadaInformationTheory(verbose=False)
    
    # Compute interaction information matrix
    ii_matrix = dit.compute_interaction_information_matrix(X_tensor)
    
    print(f"\nData shape: {X.shape}")
    print(f"All neurons are identical (perfect redundancy)")
    
    print(f"\nInteraction Information Matrix (McGill convention):")
    print(ii_matrix.numpy())
    
    # Get a specific interaction value
    ii_value = ii_matrix[0, 1].item()  # Interaction between neuron 0 and 1
    
    print(f"\nInteraction between neurons 0 and 1:")
    print(f"  McGill convention value: {ii_value:.4f}")
    print(f"  Sign: {'POSITIVE' if ii_value > 0 else 'NEGATIVE'}")
    print(f"  Interpretation: {'REDUNDANCY' if ii_value > 0 else 'SYNERGY'}")
    
    print(f"\nIf using Williams & Beer convention:")
    print(f"  W&B value would be: {-ii_value:.4f}")
    print(f"  Sign: {'POSITIVE' if -ii_value > 0 else 'NEGATIVE'}")
    print(f"  Interpretation: {'SYNERGY' if -ii_value > 0 else 'REDUNDANCY'}")
    
    return ii_value


def test_synergy_case():
    """Test synergy case with both conventions."""
    print("\n" + "="*60)
    print("SYNERGY TEST CASE: XOR-like relationship")
    print("="*60)
    
    # Create synergistic neurons
    np.random.seed(42)
    n_samples = 1000
    
    # Two independent neurons
    X1 = np.random.randint(0, 2, (n_samples, 1))
    X2 = np.random.randint(0, 2, (n_samples, 1))
    # Third neuron is XOR of first two (synergistic)
    X3 = np.logical_xor(X1, X2).astype(float)
    
    X = np.hstack([X1, X2, X3])
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Create DiT instance
    dit = DriadaInformationTheory(verbose=False)
    
    # Compute interaction information matrix
    ii_matrix = dit.compute_interaction_information_matrix(X_tensor)
    
    print(f"\nData shape: {X.shape}")
    print(f"Neuron 2 = XOR(Neuron 0, Neuron 1)")
    
    print(f"\nInteraction Information Matrix (McGill convention):")
    print(ii_matrix.numpy())
    
    # Get interaction between first two neurons
    ii_value = ii_matrix[0, 1].item()
    
    print(f"\nInteraction between neurons 0 and 1:")
    print(f"  McGill convention value: {ii_value:.4f}")
    print(f"  Sign: {'POSITIVE' if ii_value > 0 else 'NEGATIVE'}")
    print(f"  Interpretation: {'REDUNDANCY' if ii_value > 0 else 'SYNERGY'}")
    
    print(f"\nIf using Williams & Beer convention:")
    print(f"  W&B value would be: {-ii_value:.4f}")
    print(f"  Sign: {'POSITIVE' if -ii_value > 0 else 'NEGATIVE'}")
    print(f"  Interpretation: {'SYNERGY' if -ii_value > 0 else 'REDUNDANCY'}")
    
    return ii_value


def test_mixed_case():
    """Test mixed redundancy and synergy."""
    print("\n" + "="*60)
    print("MIXED TEST CASE: Some redundant, some synergistic")
    print("="*60)
    
    np.random.seed(42)
    n_samples = 1000
    
    # Create mixed relationships
    # Neurons 0 and 1 are identical (redundant)
    X0 = np.random.rand(n_samples, 1)
    X1 = X0.copy()
    
    # Neurons 2 and 3 are independent
    X2 = np.random.rand(n_samples, 1)
    X3 = np.random.rand(n_samples, 1)
    
    # Neuron 4 is a synergistic combination
    X4 = (X2 + X3) / 2 + 0.1 * np.random.randn(n_samples, 1)
    
    X = np.hstack([X0, X1, X2, X3, X4])
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Create DiT instance
    dit = DriadaInformationTheory(verbose=False)
    
    # Compute interaction information matrix
    ii_matrix = dit.compute_interaction_information_matrix(X_tensor)
    
    print(f"\nData shape: {X.shape}")
    print(f"Neurons 0,1: Identical (redundant)")
    print(f"Neurons 2,3: Independent")
    print(f"Neuron 4: Combination of 2,3 (synergistic)")
    
    print(f"\nInteraction Information Matrix (McGill convention):")
    print(ii_matrix.numpy())
    
    # Analyze specific interactions
    print(f"\nSpecific interactions:")
    
    # Redundant pair
    ii_01 = ii_matrix[0, 1].item()
    print(f"\nNeurons 0-1 (redundant pair):")
    print(f"  McGill: {ii_01:.4f} ({'POSITIVE' if ii_01 > 0 else 'NEGATIVE'} → {'REDUNDANCY' if ii_01 > 0 else 'SYNERGY'})")
    print(f"  W&B would be: {-ii_01:.4f} ({'POSITIVE' if -ii_01 > 0 else 'NEGATIVE'} → {'SYNERGY' if -ii_01 > 0 else 'REDUNDANCY'})")
    
    # Independent pair
    ii_23 = ii_matrix[2, 3].item()
    print(f"\nNeurons 2-3 (independent pair):")
    print(f"  McGill: {ii_23:.4f} ({'near zero' if abs(ii_23) < 0.01 else 'non-zero'})")
    print(f"  W&B would be: {-ii_23:.4f}")


def demonstrate_convention_conversion():
    """Show how to convert between conventions."""
    print("\n" + "="*60)
    print("CONVENTION CONVERSION")
    print("="*60)
    
    print("\nTo convert from McGill to Williams & Beer convention:")
    print("  II_williams_beer = -II_mcgill")
    print("\nTo convert from Williams & Beer to McGill convention:")
    print("  II_mcgill = -II_williams_beer")
    
    print("\nExample values:")
    mcgill_values = [0.5, -0.3, 0.8, -0.6]
    
    for mcgill in mcgill_values:
        wb = -mcgill
        print(f"\n  McGill: {mcgill:6.2f} ({'redundancy' if mcgill > 0 else 'synergy'})")
        print(f"  W&B:    {wb:6.2f} ({'synergy' if wb > 0 else 'redundancy'})")


def main():
    """Run all demonstration tests."""
    print("INTERACTION INFORMATION CONVENTION IMPACT ON INTENSE")
    print("===================================================")
    
    print("\nCurrent INTENSE implementation uses McGill convention:")
    print("  II = I(X;Y) - I(X;Y|Z)")
    print("  Positive values indicate redundancy")
    print("  Negative values indicate synergy")
    
    print("\nTests may expect Williams & Beer convention:")
    print("  II = I(X;Y|Z) - I(X;Y)")
    print("  Negative values indicate redundancy")
    print("  Positive values indicate synergy")
    
    # Run test cases
    test_redundancy_case()
    test_synergy_case()
    test_mixed_case()
    demonstrate_convention_conversion()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nThe sign difference is systematic:")
    print("- Every McGill value has opposite sign in Williams & Beer")
    print("- The magnitude remains the same")
    print("- Only the interpretation of positive/negative changes")
    print("\nTo fix failing tests expecting W&B convention:")
    print("- Either negate the expected values in tests")
    print("- Or add a convention parameter to the implementation")


if __name__ == "__main__":
    main()