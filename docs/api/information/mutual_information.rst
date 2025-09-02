Mutual Information Functions
============================

.. currentmodule:: driada.information

This module contains various mutual information estimation methods and related functions.

Core MI Functions
-----------------

.. autofunction:: get_mi
.. autofunction:: get_1d_mi
.. autofunction:: get_multi_mi
.. autofunction:: conditional_mi
.. autofunction:: interaction_information

Time-Delayed MI
---------------

.. autofunction:: get_tdmi

Similarity Measures
-------------------

.. autofunction:: get_sim

Gaussian Copula MI (GCMI)
-------------------------

Fast parametric mutual information estimation using Gaussian copulas.

.. autofunction:: mi_gg
.. autofunction:: gcmi_cc
.. autofunction:: gccmi_ccd
.. autofunction:: cmi_ggg
.. autofunction:: mi_model_gd

KSG Estimators
--------------

Non-parametric mutual information estimation using k-nearest neighbors.

.. autofunction:: nonparam_mi_cc
.. autofunction:: nonparam_mi_cd
.. autofunction:: nonparam_mi_dc