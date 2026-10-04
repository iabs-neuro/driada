Feature Representations
=======================

.. automodule:: driada.intense.representations
   :no-members:

Type-based representations used by ``compute_cell_feat_significance`` by default
(``representation='by_type'``). Results are reported under the derived feature names
(``speed_quad``, ``headdirection_harm2``); pass ``representation='raw'`` to analyse the
features as they are and keep their names.

Functions
---------

.. autofunction:: substitute_by_type
.. autofunction:: restore_source_features
.. autofunction:: get_representation_sources
.. autofunction:: build_quadratic_1d
.. autofunction:: build_harmonics
.. autofunction:: build_quadratic_multi
