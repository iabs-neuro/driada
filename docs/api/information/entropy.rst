Entropy Estimation
==================

.. currentmodule:: driada.information.entropy

This module provides functions for estimating entropy of discrete and continuous random variables.

Functions
---------

.. autofunction:: driada.information.entropy.joint_entropy_dd
.. autofunction:: driada.information.entropy.conditional_entropy_cdd
.. autofunction:: driada.information.entropy.conditional_entropy_cd

Usage Examples
--------------


Joint Entropy
^^^^^^^^^^^^^

.. code-block:: python

   from driada.information.entropy import joint_entropy_dd
   import numpy as np
   
   # Joint entropy of two variables
   x = np.random.randint(0, 4, 1000)
   y = np.random.randint(0, 3, 1000)
   
   H_xy = joint_entropy_dd(x, y)
   print(f"H(X,Y) = {H_xy:.3f} bits")

Conditional Entropy
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.information.entropy import conditional_entropy_cdd
   import numpy as np
   
   # H(Z|X,Y) - uncertainty in continuous Z given discrete X,Y
   x = np.random.randint(0, 3, 1000)  # Discrete
   y = np.random.randint(0, 2, 1000)  # Discrete
   z = np.random.randn(1000) + x  # Continuous, depends on X
   
   H_z_given_xy = conditional_entropy_cdd(z, x, y)
   print(f"H(Z|X,Y) = {H_z_given_xy:.3f} bits")


Theory
------

**Shannon Entropy**:

For discrete random variable X:

.. math::

   H(X) = -\sum_{i} p(x_i) \log_2 p(x_i)

For continuous random variable X:

.. math::

   h(X) = -\int p(x) \log_2 p(x) dx

**Conditional Entropy**:

.. math::

   H(X|Y) = H(X,Y) - H(Y)

**Mutual Information**:

.. math::

   I(X;Y) = H(X) + H(Y) - H(X,Y)