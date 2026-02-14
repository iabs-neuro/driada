"""NeuronDatabase: cross-session INTENSE analysis.

Provides a tidy DataFrame interface for querying neuron selectivity
across mice and sessions, linked by matching tables.

Usage
-----
>>> from neuron_database import load_experiment
>>> db = load_experiment('NOF', 'DRIADA data/NOF')
>>> db.summary()
"""

from .database import NeuronDatabase
from .configs import (ExperimentConfig, EXPERIMENT_CONFIGS, DELAY_STRATEGY,
                      DISCARDED_FEATURES, MI_THRESHOLD, PVAL_THRESHOLD)
from .loaders import load_from_csv_directory, load_experiment
from .tables import (apply_significance_filters,
                     get_fully_matched_ids, significance_count_table,
                     significance_fraction_table,
                     significance_fraction_of_sel_table,
                     export_count_tables_excel,
                     export_fraction_tables_excel,
                     export_fraction_of_sel_tables_excel,
                     mi_table, mi_table_composite,
                     export_mi_tables_excel,
                     retention_count_table,
                     retention_enrichment,
                     export_retention_tables_excel,
                     export_retention_enrichment_excel,
                     cross_stats_table, export_cross_stats_csv,
                     export_all)

__all__ = ['NeuronDatabase', 'ExperimentConfig', 'EXPERIMENT_CONFIGS',
           'DELAY_STRATEGY', 'DISCARDED_FEATURES',
           'MI_THRESHOLD', 'PVAL_THRESHOLD',
           'load_from_csv_directory', 'load_experiment',
           'apply_significance_filters',
           'get_fully_matched_ids', 'significance_count_table',
           'significance_fraction_table',
           'significance_fraction_of_sel_table',
           'export_count_tables_excel',
           'export_fraction_tables_excel',
           'export_fraction_of_sel_tables_excel',
           'mi_table', 'mi_table_composite',
           'export_mi_tables_excel',
           'retention_count_table',
           'retention_enrichment',
           'export_retention_tables_excel',
           'export_retention_enrichment_excel',
           'cross_stats_table', 'export_cross_stats_csv',
           'export_all']
