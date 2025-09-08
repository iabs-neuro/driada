Information Estimators
======================

.. currentmodule:: driada.information

This module provides different estimators for computing information-theoretic quantities.

Gaussian Copula Estimators
--------------------------

The Gaussian Copula Mutual Information (GCMI) method provides fast parametric estimation.

.. autofunction:: driada.information.gcmi.copnorm
.. autofunction:: driada.information.gcmi.ctransform
.. autofunction:: driada.information.gcmi.gcmi_cc
.. autofunction:: driada.information.gcmi.ent_g

KSG Non-parametric Estimators
------------------------------

K-nearest neighbor based estimators for non-parametric entropy and MI estimation.

.. autofunction:: nonparam_entropy_c

Utility Functions
-----------------

.. automodule:: driada.information.gcmi_jit_utils
   :members:
   :undoc-members: