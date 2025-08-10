#!/usr/bin/env python
"""Debug weight generation."""

import numpy as np
from driada.experiment.synthetic.mixed_selectivity import generate_multiselectivity_patterns

# Test dominant weights
matrix_dominant = generate_multiselectivity_patterns(
    20, 4,
    selectivity_prob=1.0,
    multi_select_prob=1.0,
    weights_mode='dominant',
    seed=42
)

print("Checking dominant weights:")
for j in range(20):
    weights = matrix_dominant[:, j]
    nonzero = weights[weights > 0]
    if len(nonzero) > 1:
        sorted_weights = np.sort(nonzero)[::-1]
        ratio = sorted_weights[0] / sorted_weights[1] if len(sorted_weights) >= 2 else 0
        print(f"Neuron {j}: weights={nonzero}, ratio={ratio:.2f}")