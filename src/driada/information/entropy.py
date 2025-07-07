"""
Entropy calculation functions for discrete, continuous, and mixed variable types.

This module provides various entropy calculation methods including:
- Discrete entropy
- Joint entropy for discrete and mixed variables
- Conditional entropy for different variable type combinations
"""

import numpy as np
import scipy.stats
from .gcmi import ent_g


def entropy_d(x):
    """Calculate entropy for a discrete variable.
    
    Parameters
    ----------
    x : array-like
        Discrete variable values.
        
    Returns
    -------
    float
        Entropy in bits.
    """
    unique_x, counts_x = np.unique(x, return_counts=True)
    p_x = counts_x / len(x)
    H_x = probs_to_entropy(p_x)
    return H_x


def probs_to_entropy(p):
    """Calculate entropy for a discrete probability distribution.
    
    Parameters
    ----------
    p : array-like
        Probability distribution (must sum to 1).
        
    Returns
    -------
    float
        Entropy in bits.
    """
    return -np.sum(p * np.log2(p + 1e-10))  # Add small value to avoid log(0)


def joint_entropy_dd(x, y):
    """Calculate joint entropy for two discrete variables.
    
    Parameters
    ----------
    x : array-like
        First discrete variable.
    y : array-like
        Second discrete variable.
        
    Returns
    -------
    float
        Joint entropy H(X,Y) in bits.
    """
    joint_prob = np.histogram2d(x, y, bins=[np.unique(x).size, np.unique(y).size], density=True)[0]
    joint_prob /= np.sum(joint_prob)  # Normalize
    return probs_to_entropy(joint_prob.flatten())


def conditional_entropy_cdd(z, x, y, k=5):
    """Calculate conditional differential entropy for a continuous variable given two discrete variables.
    
    Computes H(Z|X,Y) where Z is continuous and X,Y are discrete.
    
    Parameters
    ----------
    z : array-like
        Continuous variable.
    x : array-like
        First discrete variable.
    y : array-like
        Second discrete variable.
    k : int, optional
        Number of neighbors for entropy estimation (used as threshold). Default: 5.
        
    Returns
    -------
    float
        Conditional entropy H(Z|X,Y).
    """
    unique_x = np.unique(x)
    unique_y = np.unique(y)

    h_conditional = 0
    for ux in unique_x:
        for uy in unique_y:
            # Filter z based on x and y
            filtered_z = z[(x == ux) & (y == uy)]
            if len(filtered_z) > k:
                # if n points is less than number of neighbors, result will be meaningless
                h_conditional += ent_g(filtered_z.reshape(1, -1)) * (len(filtered_z) / len(z))

    return h_conditional


def conditional_entropy_cd(z, x, k=5):
    """Calculate conditional differential entropy for a continuous variable given a discrete variable.
    
    Computes H(Z|X) where Z is continuous and X is discrete.
    
    Parameters
    ----------
    z : array-like
        Continuous variable.
    x : array-like
        Discrete variable.
    k : int, optional
        Number of neighbors for entropy estimation (used as threshold). Default: 5.
        
    Returns
    -------
    float
        Conditional entropy H(Z|X).
    """
    unique_x = np.unique(x)
    h_conditional = 0

    for ux in unique_x:
        # Filter z based on x
        filtered_z = z[x == ux]
        if len(filtered_z) > k:
            # if n points is less than number of neighbors, result will be meaningless
            h_conditional += ent_g(filtered_z.reshape(1, -1)) * (len(filtered_z) / len(z))

    return h_conditional


def joint_entropy_cdd(x, y, z, k=5):
    """Calculate joint entropy for two discrete and one continuous variable.
    
    Computes H(X,Y,Z) where X,Y are discrete and Z is continuous.
    Uses the chain rule: H(X,Y,Z) = H(X,Y) + H(Z|X,Y)
    
    Parameters
    ----------
    x : array-like
        First discrete variable.
    y : array-like
        Second discrete variable.
    z : array-like
        Continuous variable.
    k : int, optional
        Number of neighbors for entropy estimation. Default: 5.
        
    Returns
    -------
    float
        Joint entropy H(X,Y,Z).
    """
    H_xy = joint_entropy_dd(x, y)
    H_z_given_xy = conditional_entropy_cdd(z, x, y, k=k)
    H_xyz = H_xy + H_z_given_xy
    return H_xyz


def joint_entropy_cd(x, z, k=5):
    """Calculate joint entropy for one discrete and one continuous variable.
    
    Computes H(X,Z) where X is discrete and Z is continuous.
    Uses the chain rule: H(X,Z) = H(X) + H(Z|X)
    
    Parameters
    ----------
    x : array-like
        Discrete variable.
    z : array-like
        Continuous variable.
    k : int, optional
        Number of neighbors for entropy estimation. Default: 5.
        
    Returns
    -------
    float
        Joint entropy H(X,Z).
    """
    H_x = entropy_d(x)
    H_z_given_x = conditional_entropy_cd(z, x, k=k)
    H_xz = H_x + H_z_given_x
    return H_xz