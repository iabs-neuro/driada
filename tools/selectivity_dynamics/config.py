"""Configuration constants for INTENSE selectivity analysis.

Contains:
- skip_for_intense: Features to exclude from analysis (aggregated instead)
- aggregate_features: Feature aggregation mapping
- DEFAULT_CONFIG: Default INTENSE parameters
"""

# Features to skip from INTENSE analysis (will be aggregated instead)
skip_for_intense = ['x', 'y', 'Reconstructions']

# Aggregation mapping: {(tuple_of_features): 'new_aggregated_name'}
# These features are combined into MultiTimeSeries during experiment construction
# The component features (e.g., 'x', 'y') are consumed and replaced by the combined name ('xy')
aggregate_features = {
    ('x', 'y'): 'place',
}

# Default INTENSE configuration
DEFAULT_CONFIG = {
    'n_shuffles_stage1': 100,
    'n_shuffles_stage2': 10000,
    'pval_thr': 0.001,            # Strict threshold
    'multicomp_correction': None,  # No correction
    'ds': 5,                       # Downsampling for speed
}
