INTENSE Core Implementation
===========================

.. automodule:: driada.intense.intense_base

Low-level computation functions for INTENSE analysis.

Classes
-------

.. autoclass:: IntenseResults
   :members:

Function Groups
---------------

**Mutual Information Computation**
   .. autofunction:: get_calcium_feature_me_profile
   .. autofunction:: scan_pairs
   .. autofunction:: scan_pairs_parallel
   .. autofunction:: scan_pairs_router

**Statistical Analysis**
   .. autofunction:: compute_me_stats

**I/O**
   .. autofunction:: driada.intense.io.save_results
   .. autofunction:: driada.intense.io.load_results