"""
Information theory functions for DRIADA.

This module provides various information-theoretic measures including
mutual information, entropy, and conditional information measures.
"""

# Core information theory functions
from .info_base import (
    TimeSeries,
    MultiTimeSeries,
    get_mi,
    get_1d_mi,
    get_multi_mi,
    get_sim,
    get_tdmi,
    conditional_mi,
    interaction_information,
)

# Gaussian Copula MI functions
from .gcmi import (
    copnorm,
    ctransform,
    mi_gg,
    ent_g,
    gcmi_cc,
    gccmi_ccd,
    cmi_ggg,
    mi_model_gd,
)

# KSG estimator functions
from .ksg import (
    nonparam_entropy_c,
    nonparam_mi_cc,
    nonparam_mi_cd,
    nonparam_mi_dc,
)

# Entropy functions
from .entropy import (
    entropy_d,
    probs_to_entropy,
    joint_entropy_dd,
    joint_entropy_cd,
    joint_entropy_cdd,
    conditional_entropy_cd,
    conditional_entropy_cdd,
)

__all__ = [
    # Core classes and functions
    "TimeSeries",
    "MultiTimeSeries",
    "get_mi",
    "get_1d_mi",
    "get_multi_mi",
    "get_sim",
    "get_tdmi",
    "conditional_mi",
    "interaction_information",
    # GCMI
    "copnorm",
    "ctransform",
    "mi_gg",
    "ent_g",
    "gcmi_cc",
    "gccmi_ccd",
    "cmi_ggg",
    "mi_model_gd",
    # KSG
    "nonparam_entropy_c",
    "nonparam_mi_cc",
    "nonparam_mi_cd",
    "nonparam_mi_dc",
    # Entropy
    "entropy_d",
    "probs_to_entropy",
    "joint_entropy_dd",
    "joint_entropy_cd",
    "joint_entropy_cdd",
    "conditional_entropy_cd",
    "conditional_entropy_cdd",
]
