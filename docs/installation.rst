Installation
============

Requirements
------------

DRIADA requires Python 3.9 or higher.

Installing from PyPI
--------------------

The easiest way to install DRIADA is using pip:

.. code-block:: bash

   pip install driada

For GPU support (PyTorch-based methods):

.. code-block:: bash

   pip install driada[gpu]

For Maximum Variance Unfolding (MVU) support:

.. code-block:: bash

   pip install driada[mvu]

For all optional dependencies:

.. code-block:: bash

   pip install driada[all]

Installing from Source
----------------------

To install the latest development version:

.. code-block:: bash

   git clone https://github.com/iabs-neuro/driada.git
   cd driada
   pip install -e .

Verifying Installation
----------------------

You can verify your installation by running:

.. code-block:: python

   import driada
   print(driada.__version__)

Dependencies
------------

Core dependencies include:

- numpy >= 1.24.3
- scipy >= 1.11.4
- scikit-learn >= 1.3.0
- pandas >= 2.1.4
- matplotlib >= 3.7.0
- networkx >= 3.1
- numba >= 0.59.0
- umap-learn

Optional dependencies:

- **GPU support**: torch, torchvision
- **MVU support**: cvxpy >= 1.4.2