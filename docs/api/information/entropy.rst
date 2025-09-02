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
   
   # Joint entropy of two variables
   x = np.random.randint(0, 4, 1000)
   y = np.random.randint(0, 3, 1000)
   
   H_xy = joint_entropy_dd(x, y)
   print(f"H(X,Y) = {H_xy:.3f} bits")

Conditional Entropy
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.information.entropy import conditional_entropy_cdd
   
   # H(Y|X) - uncertainty in Y given X
   x = np.random.randint(0, 3, 1000)
   y = x + np.random.randint(0, 2, 1000)  # Y depends on X
   
   H_y_given_x = conditional_entropy_cdd(y, x)
   print(f"H(Y|X) = {H_y_given_x:.3f} bits")


Advanced Features
-----------------


Block Entropy
^^^^^^^^^^^^^

.. code-block:: python

   from driada.information.entropy import block_entropy
   
   # Entropy of sequences
   sequence = np.random.randint(0, 2, 1000)
   
   # Single symbol entropy
   H1 = block_entropy(sequence, block_size=1)
   
   # Pair entropy
   H2 = block_entropy(sequence, block_size=2)
   
   # Entropy rate
   h = H2 - H1
   print(f"Entropy rate: {h:.3f} bits/symbol")

Cross Entropy
^^^^^^^^^^^^^

.. code-block:: python

   from driada.information.entropy import cross_entropy
   
   # Cross entropy for classification
   true_labels = np.array([0, 1, 1, 0, 1])
   predictions = np.array([
       [0.9, 0.1],  # Correct
       [0.2, 0.8],  # Correct
       [0.3, 0.7],  # Correct
       [0.6, 0.4],  # Wrong
       [0.1, 0.9],  # Correct
   ])
   
   ce = cross_entropy(true_labels, predictions)
   print(f"Cross entropy: {ce:.3f} bits")

Performance Optimization
------------------------


Batch Processing
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Compute entropy for multiple time series efficiently
   from concurrent.futures import ProcessPoolExecutor
   
   # Example of batch processing pattern
   # (actual entropy functions should be imported as shown above)

Theory
------

**Shannon Entropy**:

For discrete variable X with probability distribution p(x):

.. math::

   H(X) = -\sum_{x} p(x) \log_2 p(x)

**Differential Entropy**:

For continuous variable X with probability density f(x):

.. math::

   h(X) = -\int_{-\infty}^{\infty} f(x) \log_2 f(x) dx

**Properties**:

- Non-negative for discrete variables
- Can be negative for continuous variables
- Maximum when distribution is uniform
- Invariant under reordering
- Additive for independent variables

Best Practices
--------------

1. **Sample Size**: Ensure sufficient samples (>100 per state)
2. **Bias Correction**: Use for small samples or many states
3. **Continuous Data**: Consider discretization vs k-NN estimators
4. **Numerical Stability**: Use log-sum-exp trick for small probabilities
