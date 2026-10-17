Higher-Order Information Measures
=================================

.. currentmodule:: driada.information

This module measures shared information among three or more variables of a
:class:`~driada.information.info_base.MultiTimeSeries`, for example the activity of a
group of neurons. Values are in bits and use the Gaussian copula estimator.

Functions
---------

.. autofunction:: driada.information.higher_order.total_correlation
.. autofunction:: driada.information.higher_order.dual_total_correlation
.. autofunction:: driada.information.higher_order.o_information

Gaussian Copula Estimators
--------------------------

Array-level versions that take copula-normalised data of shape
``(n_variables, n_samples)``.

.. autofunction:: driada.information.gcmi.tc_gg
.. autofunction:: driada.information.gcmi.dtc_gg
.. autofunction:: driada.information.gcmi.o_info_gg

Usage Example
-------------

.. code-block:: python

   import numpy as np
   from driada.information import (
       MultiTimeSeries, total_correlation, dual_total_correlation, o_information
   )

   rng = np.random.default_rng(0)
   source = rng.standard_normal(2000)
   # Three noisy copies of one source: redundancy-dominated
   data = np.vstack([source + 0.5 * rng.standard_normal(2000) for _ in range(3)])
   mts = MultiTimeSeries(data, discrete=False)

   tc = total_correlation(mts)
   dtc = dual_total_correlation(mts)
   omega = o_information(mts)  # TC - DTC; > 0 redundancy, < 0 synergy
   print(f"TC = {tc:.3f} bits, DTC = {dtc:.3f} bits, O-information = {omega:.3f} bits")

Theory
------

For variables :math:`X_1, \dots, X_n`:

.. math::

   TC = \sum_i H(X_i) - H(X_1, \dots, X_n)

.. math::

   DTC = H(X_1, \dots, X_n) - \sum_i H(X_i \mid X_{-i})

.. math::

   \Omega = TC - DTC

O-information needs at least three variables; for two it is identically zero.
