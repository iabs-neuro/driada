Spatial Analysis Utilities
==========================

.. automodule:: driada.utils.spatial
   :no-members:
   :noindex:

Utilities for spatial data visualization and evaluation metrics.

.. note::
   **Place cell detection**: Use :mod:`driada.intense` (MI-based analysis with
   shuffle-based significance testing) for principled place cell detection.

   ``extract_place_fields()`` is **EXPERIMENTAL** and uses arbitrary thresholds.
   It's provided for quick visualization only, not scientific analysis.

.. warning::
   **Removed in v1.0**: ``analyze_spatial_coding()`` and ``filter_by_speed()``
   have been removed. Use INTENSE for place cell detection and analysis.

Visualization Utilities
-----------------------

Core functions for creating spatial visualizations:

.. autofunction:: driada.utils.spatial.compute_occupancy_map

   Compute time spent in each spatial bin. Use for trajectory occupancy maps.

.. autofunction:: driada.utils.spatial.compute_rate_map

   Compute firing rate as a function of position. Use for place field visualization.

Spatial Information Metrics
---------------------------

Evaluation metrics for spatial coding:

.. autofunction:: driada.utils.spatial.compute_spatial_information_rate

   Skaggs et al. (1993) spatial information metric in bits/spike.
   Measures how much information about position is conveyed by each spike.

.. autofunction:: driada.utils.spatial.compute_spatial_information

   Mutual information between neural activity and position.
   Returns MI for X, Y, and total 2D position.

.. autofunction:: driada.utils.spatial.compute_spatial_decoding_accuracy

   ML-based position decoding from neural activity.
   Uses Random Forest regression to predict position from population activity.

Metrics Wrapper
---------------

.. autofunction:: driada.utils.spatial.compute_spatial_metrics

   Compute multiple spatial metrics (decoding and information).
   Place field detection removed - use INTENSE instead.

Experimental Functions
----------------------

.. warning::
   The following function is **EXPERIMENTAL** and uses arbitrary thresholds.
   For scientific analysis, use INTENSE for principled place cell detection.

.. autofunction:: driada.utils.spatial.extract_place_fields

   **EXPERIMENTAL**: Threshold-based place field detection for visualization only.
   Uses arbitrary parameters (min_peak_rate, min_field_size, peak_to_mean_ratio).
   For quantitative analysis, use ``compute_cell_feat_significance()`` from INTENSE.