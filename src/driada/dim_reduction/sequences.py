"""
Dimensionality Reduction Sequences for DRIADA.

This module provides functionality for performing sequential dimensionality
reduction, where the output of one reduction step becomes the input for the next.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from .data import MVData
from .embedding import Embedding


def dr_sequence(
    data: MVData,
    steps: List[Union[Tuple[str, Dict[str, Any]], str]],
    logger: Optional[logging.Logger] = None
) -> Embedding:
    """Perform sequential dimensionality reduction.
    
    Applies multiple dimensionality reduction steps in sequence, where each
    step operates on the output of the previous step.
    
    Parameters
    ----------
    data : MVData
        Initial high-dimensional data
    steps : List[Union[Tuple[str, Dict], str]]
        List of reduction steps. Each step can be:
        - A tuple of (method_name, parameters_dict)
        - A string method name (uses default parameters)
    logger : logging.Logger, optional
        Logger for tracking progress
        
    Returns
    -------
    Embedding
        Final embedding after all reduction steps
        
    Examples
    --------
    >>> # Simple two-step reduction
    >>> embedding = dr_sequence(
    ...     mvdata,
    ...     steps=[
    ...         ('pca', {'dim': 50}),
    ...         ('umap', {'dim': 2, 'n_neighbors': 30})
    ...     ]
    ... )
    
    >>> # Using default parameters
    >>> embedding = dr_sequence(
    ...     mvdata,
    ...     steps=['pca', 'tsne']
    ... )
    
    >>> # Three-step reduction with mixed format
    >>> embedding = dr_sequence(
    ...     mvdata,
    ...     steps=[
    ...         ('pca', {'dim': 100}),
    ...         'lle',  # Use defaults
    ...         ('umap', {'dim': 2, 'min_dist': 0.1})
    ...     ]
    ... )
    """
    if not steps:
        raise ValueError("At least one reduction step must be provided")
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    current_data = data
    
    for i, step in enumerate(steps):
        # Parse step format
        if isinstance(step, str):
            method = step
            params = {}
        elif isinstance(step, tuple) and len(step) == 2:
            method, params = step
        else:
            raise ValueError(
                f"Invalid step format: {step}. "
                "Expected method name string or (method, params) tuple."
            )
        
        # Log progress
        logger.info(
            f"Step {i+1}/{len(steps)}: {method} "
            f"from dim {current_data.n_dim} to dim {params.get('dim', 2)}"
        )
        
        # Apply reduction
        embedding = current_data.get_embedding(method=method, **params)
        
        # Convert to MVData for next step (if not last)
        if i < len(steps) - 1:
            current_data = embedding.to_mvdata()
    
    return embedding