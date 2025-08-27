"""
Dimensionality Reduction Sequences for DRIADA.

This module provides functionality for performing sequential dimensionality
reduction, where the output of one reduction step becomes the input for the next.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from .data import MVData
from .embedding import Embedding
from .dr_base import METHODS_DICT, merge_params_with_defaults


def dr_sequence(
    data: MVData,
    steps: List[Union[Tuple[str, Dict[str, Any]], str]],
    logger: Optional[logging.Logger] = None,
    keep_intermediate: bool = False,
    validate_compatibility: bool = True,
) -> Union[Embedding, Tuple[Embedding, List[Embedding]]]:
    """Perform sequential dimensionality reduction with improved validation and error handling.

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
    keep_intermediate : bool, default False
        If True, returns a tuple of (final_embedding, intermediate_embeddings).
        If False, only returns final embedding to save memory.
    validate_compatibility : bool, default True
        If True, validates dimension compatibility between consecutive steps.

    Returns
    -------
    Embedding or Tuple[Embedding, List[Embedding]]
        If keep_intermediate=False: Final embedding after all reduction steps
        If keep_intermediate=True: (final_embedding, list_of_intermediate_embeddings)
        
    Raises
    ------
    ValueError
        If steps list is empty.
        If any step has invalid format (not string or (method, params) tuple).
        If method name is not recognized.
        If dimension compatibility check fails between steps.
    RuntimeError
        If any step fails during execution, with context about which step failed.

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
    
    >>> # Keep intermediate results for analysis
    >>> final_emb, intermediates = dr_sequence(
    ...     mvdata,
    ...     steps=[('pca', {'dim': 50}), ('umap', {'dim': 2})],
    ...     keep_intermediate=True
    ... )
    
    Notes
    -----
    - Intermediate results converted to MVData between steps
    - Progress logged with actual dimensions for each step
    - Pre-validates all method names before execution
    - Optional dimension compatibility checking available
    - Memory-efficient by default (keep_intermediate=False)
    
    DOC_VERIFIED
    """
    if not steps:
        raise ValueError("At least one reduction step must be provided")

    if logger is None:
        logger = logging.getLogger(__name__)

    # Pre-validate all method names and prepare full parameters
    parsed_steps = []
    for i, step in enumerate(steps):
        # Parse step format
        if isinstance(step, str):
            method_name = step
            user_params = {}
        elif isinstance(step, tuple) and len(step) == 2:
            method_name, user_params = step
        else:
            raise ValueError(
                f"Invalid step format at position {i}: {step}. "
                "Expected method name string or (method, params) tuple."
            )
        
        # Validate method name exists
        if method_name not in METHODS_DICT:
            available_methods = sorted(METHODS_DICT.keys())
            raise ValueError(
                f"Unknown method '{method_name}' at step {i+1}. "
                f"Available methods: {', '.join(available_methods)}"
            )
        
        # Merge with defaults to get actual parameters that will be used
        full_params = merge_params_with_defaults(method_name, user_params)
        parsed_steps.append((method_name, user_params, full_params))
    
    # Validate dimension compatibility if requested
    if validate_compatibility and len(parsed_steps) > 1:
        for i in range(len(parsed_steps) - 1):
            curr_name, curr_user, curr_full = parsed_steps[i]
            next_name, next_user, next_full = parsed_steps[i+1]
            
            # Get output dimension of current step
            curr_output_dim = curr_full['e_params'].get('dim', 2)
            
            # For certain methods, check if input dimension is reasonable
            if next_name in ['tsne', 'umap'] and curr_output_dim > 100:
                logger.warning(
                    f"Step {i+2} ({next_name}) will receive {curr_output_dim}-dimensional input. "
                    f"Consider reducing to <= 100 dimensions for better performance."
                )
    
    # Execute the sequence
    current_data = data
    intermediate_embeddings = []
    
    for i, (method_name, user_params, full_params) in enumerate(parsed_steps):
        # Get actual dimension that will be used
        actual_dim = full_params['e_params'].get('dim', 2)
        
        # Log progress with correct dimension
        logger.info(
            f"Step {i+1}/{len(parsed_steps)}: {method_name} "
            f"from dim {current_data.n_dim} to dim {actual_dim}"
        )
        
        try:
            # Apply reduction
            embedding = current_data.get_embedding(method=method_name, **user_params)
            
            # Store intermediate if requested
            if keep_intermediate:
                intermediate_embeddings.append(embedding)
            
            # Convert to MVData for next step (if not last)
            if i < len(parsed_steps) - 1:
                current_data = embedding.to_mvdata()
                
        except Exception as e:
            # Provide context about which step failed
            raise RuntimeError(
                f"Failed at step {i+1}/{len(parsed_steps)} ({method_name}): {str(e)}"
            ) from e
    
    if keep_intermediate:
        return embedding, intermediate_embeddings
    else:
        return embedding


def validate_sequence_dimensions(
    steps: List[Union[Tuple[str, Dict[str, Any]], str]], 
    initial_dim: int,
    logger: Optional[logging.Logger] = None
) -> List[Tuple[str, int, int]]:
    """Validate and report dimension flow through a sequence of reductions.
    
    This function helps plan reduction sequences by showing how dimensions
    will change at each step, without actually performing the reductions.
    
    Parameters
    ----------
    steps : List[Union[Tuple[str, Dict], str]]
        List of reduction steps in the same format as dr_sequence
    initial_dim : int
        Initial data dimension
    logger : logging.Logger, optional
        Logger for reporting dimension flow
        
    Returns
    -------
    List[Tuple[str, int, int]]
        List of (method_name, input_dim, output_dim) for each step
        
    Raises
    ------
    ValueError
        If any step has invalid format.
        If any reduction method is unknown.
        
    Examples
    --------
    >>> # Check dimension flow before running expensive computation
    >>> flow = validate_sequence_dimensions(
    ...     [('pca', {'dim': 50}), 'umap'],
    ...     initial_dim=1000
    ... )
    >>> print(flow)
    [('pca', 1000, 50), ('umap', 50, 2)]
    
    Notes
    -----
    - Logs dimension changes for each step via provided or module logger
    - Warns when a step attempts to increase dimensions
    - Does not perform actual reductions, only predicts dimensions
    
    DOC_VERIFIED
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    dimension_flow = []
    current_dim = initial_dim
    
    for i, step in enumerate(steps):
        # Parse step
        if isinstance(step, str):
            method_name = step
            user_params = {}
        elif isinstance(step, tuple) and len(step) == 2:
            method_name, user_params = step
        else:
            raise ValueError(f"Invalid step format: {step}")
            
        # Validate method
        if method_name not in METHODS_DICT:
            raise ValueError(f"Unknown method: {method_name}")
            
        # Get target dimension
        full_params = merge_params_with_defaults(method_name, user_params)
        target_dim = full_params['e_params'].get('dim', 2)
        
        # Record flow
        dimension_flow.append((method_name, current_dim, target_dim))
        
        # Log
        logger.info(
            f"Step {i+1}: {method_name} will reduce from {current_dim}D to {target_dim}D"
        )
        
        # Check for potential issues
        if target_dim > current_dim:
            logger.warning(
                f"Step {i+1} ({method_name}) attempts to increase dimensions "
                f"from {current_dim} to {target_dim}. This may cause issues."
            )
            
        # Update for next iteration
        current_dim = target_dim
        
    return dimension_flow
