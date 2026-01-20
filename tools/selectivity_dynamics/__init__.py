"""Selectivity dynamics analysis package.

This package provides tools for running INTENSE analysis with disentanglement
on neural experiment data.

Main Entry Points
-----------------
- run_intense_analysis: Run INTENSE with disentanglement
- load_experiment_from_npz: Load experiment from NPZ file
- get_filter_for_experiment: Get composed filter for experiment type
- save_all_results: Save all results to structured output

Configuration
-------------
- EXPERIMENT_CONFIGS: Per-experiment configuration (aggregate_features, skip_for_intense, etc.)
- get_experiment_config: Get config for experiment type
- DEFAULT_CONFIG: Default INTENSE parameters

Filters
-------
- GENERAL_PRIORITY_RULES: General behavioral priority rules
- build_priority_filter: Create filter from priority rules
- compose_filters: Chain multiple filters
- get_filter_for_experiment: Get composed filter for experiment type
- nof_filter, tdm_filter, spatial_filter: Experiment-specific filters

Example Usage
-------------
>>> from selectivity_dynamics import (
...     load_experiment_from_npz,
...     run_intense_analysis,
...     get_experiment_config,
...     get_filter_for_experiment,
...     save_all_results,
...     DEFAULT_CONFIG,
... )
>>>
>>> # Get experiment config
>>> config = get_experiment_config('NOF')
>>>
>>> # Load experiment with config-based aggregation
>>> exp = load_experiment_from_npz('data.npz', agg_features=config['aggregate_features'])
>>>
>>> # Get filter
>>> pre_filter = get_filter_for_experiment('NOF')
>>>
>>> # Run analysis
>>> stats, significance, info, results, disent_results, timings = run_intense_analysis(
...     exp, DEFAULT_CONFIG, config['skip_for_intense'],
...     pre_filter_func=pre_filter,
... )
>>>
>>> # Save results
>>> save_all_results('exp_name', exp, stats, significance, info, results, disent_results, 'output/')
"""

# Configuration
from .config import skip_for_intense, aggregate_features, DEFAULT_CONFIG

# Filters
from .filters import (
    GENERAL_PRIORITY_RULES,
    build_priority_filter,
    compose_filters,
    build_mi_ratio_filter,
    build_exclusion_filter,
    nof_filter,
    tdm_filter,
    spatial_filter,
    extract_filter_data,
    EXPERIMENT_CONFIGS,
    get_experiment_config,
    get_filter_for_experiment,
)

# Loader
from .loader import (
    load_experiment_from_npz,
    build_feature_list,
    get_skip_delays,
)

# Analysis
from .analysis import (
    run_intense_analysis,
    print_results,
    build_disentangled_stats,
)

# Summary
from .summary import (
    compute_summary_metrics,
    print_per_file_summary,
    print_batch_summary,
    save_batch_summary_csv,
    load_batch_summary_csv,
    format_metric,
)

# Export
from .export import (
    save_all_results,
    save_results,
    get_exp_name,
    filter_stats_by_stage2,
)

# Plots
from .plots import plot_disentanglement


__all__ = [
    # Config
    'skip_for_intense',
    'aggregate_features',
    'DEFAULT_CONFIG',
    # Filters
    'GENERAL_PRIORITY_RULES',
    'build_priority_filter',
    'compose_filters',
    'build_mi_ratio_filter',
    'build_exclusion_filter',
    'nof_filter',
    'tdm_filter',
    'spatial_filter',
    'extract_filter_data',
    'EXPERIMENT_CONFIGS',
    'get_experiment_config',
    'get_filter_for_experiment',
    # Loader
    'load_experiment_from_npz',
    'build_feature_list',
    'get_skip_delays',
    # Analysis
    'run_intense_analysis',
    'print_results',
    'build_disentangled_stats',
    # Summary
    'compute_summary_metrics',
    'print_per_file_summary',
    'print_batch_summary',
    'save_batch_summary_csv',
    'load_batch_summary_csv',
    'format_metric',
    # Export
    'save_all_results',
    'save_results',
    'get_exp_name',
    'filter_stats_by_stage2',
    # Plots
    'plot_disentanglement',
]
