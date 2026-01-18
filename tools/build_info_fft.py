"""Script to build info_fft.py by extracting FFT functions from info_base.py and entropy.py"""

# Read source files
with open('src/driada/information/info_base.py', 'r', encoding='utf-8') as f:
    info_base_lines = f.readlines()

with open('src/driada/information/entropy.py', 'r', encoding='utf-8') as f:
    entropy_lines = f.readlines()

# Define extraction ranges (1-indexed to 0-indexed)
info_base_extractions = [
    (1724, 1823, 'compute_mi_batch_fft'),
    (1824, 1867, 'compute_mi_gd_fft'),
    (1872, 2045, 'compute_mi_mts_fft'),
    (2046, 2105, '_compute_joint_entropy_3x3_mts'),
    (2106, 2187, '_compute_joint_entropy_4x4_mts'),
    (2188, 2255, '_compute_joint_entropy_mts_mts_block'),
    (2256, 2441, 'compute_mi_mts_mts_fft'),
    (2442, 2669, 'compute_mi_mts_discrete_fft'),
]

entropy_extractions = [
    (377, 501, 'mi_cd_fft'),
]

# Build the new file
header = '''"""FFT-accelerated mutual information computation for INTENSE.

This module contains all FFT-based MI estimators optimized for computing
mutual information across many shifts efficiently. These functions provide
10-1000x speedup over loop-based approaches for INTENSE shuffle testing.

Key optimizations implemented:
- rfft/irfft for real inputs (50% faster than fft/ifft)
- Memory-efficient shift extraction (100-1000x less memory)
- Unified bias correction using scipy.special.psi
- Dimension-specific regularization thresholds

Functions
---------
Univariate:
    compute_mi_batch_fft : MI between two 1D continuous variables at multiple shifts
    mi_cd_fft : MI between 1D continuous and discrete variables at multiple shifts
    compute_mi_gd_fft : Wrapper for mi_cd_fft (continuous-discrete MI)

Multivariate:
    compute_mi_mts_fft : MI between 1D and multi-dimensional (d≤3) continuous variables
    compute_mi_mts_mts_fft : MI between two multi-dimensional (d1,d2≤3) continuous variables
    compute_mi_mts_discrete_fft : MI between multi-dimensional continuous and discrete variables

Internal helpers (not for direct use):
    _compute_joint_entropy_3x3_mts : 3×3 determinant-based entropy
    _compute_joint_entropy_4x4_mts : 4×4 determinant-based entropy
    _compute_joint_entropy_mts_mts_block : Block determinant for MTS-MTS MI

Author: DRIADA Development Team
"""

import numpy as np
import warnings
from scipy.special import psi

from .gcmi import regularized_cholesky
from .info_utils import py_fast_digamma, py_fast_digamma_arr
from .info_fft_utils import (
    REG_VARIANCE_THRESHOLD,
    REG_DET_2D_THRESHOLD,
    REG_DET_3D_THRESHOLD,
)

ln2 = np.log(2)

'''

# Extract functions from info_base.py
extracted_code = [header]

for start_1idx, end_1idx, name in info_base_extractions:
    start = start_1idx - 1
    end = end_1idx
    extracted_code.append(''.join(info_base_lines[start:end]))
    extracted_code.append('\n\n')

# Extract mi_cd_fft from entropy.py
for start_1idx, end_1idx, name in entropy_extractions:
    start = start_1idx - 1
    end = end_1idx
    extracted_code.append(''.join(entropy_lines[start:end]))
    extracted_code.append('\n')

# Write the new file
with open('src/driada/information/info_fft.py', 'w', encoding='utf-8') as f:
    f.write(''.join(extracted_code))

print(f"Created info_fft.py with {len(extracted_code)} sections")
print(f"Total lines: {sum(code.count(chr(10)) for code in extracted_code)}")
