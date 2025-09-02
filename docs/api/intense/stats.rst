INTENSE Statistics
==================

.. automodule:: driada.intense.stats

Statistical tools for INTENSE analysis including distribution fitting and p-value computation.

Function Groups
---------------

**Distribution Functions**
   .. autofunction:: get_lognormal_p
   .. autofunction:: get_gamma_p
   .. autofunction:: get_distribution_function
   .. autofunction:: chebyshev_ineq

**P-value Computation**
   .. autofunction:: get_mi_distr_pvalue
   .. autofunction:: get_all_nonempty_pvals

**Data Filtering and Validation**
   .. autofunction:: get_mask
   .. autofunction:: stats_not_empty
   .. autofunction:: criterion1
   .. autofunction:: criterion2

**Result Aggregation**
   .. autofunction:: get_table_of_stats
   .. autofunction:: merge_stage_stats
   .. autofunction:: merge_stage_significance