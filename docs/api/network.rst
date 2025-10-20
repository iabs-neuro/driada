Network Analysis Module
=======================

.. automodule:: driada.network
   :no-members:
   :noindex:

Tools for analyzing functional networks in neural data, including graph-based analysis,
spectral methods, and network visualization.

.. note::
   As of version 0.5.1, all major functions are now exported at the module level.
   You can use ``from driada.network import get_giant_cc_from_graph`` etc.

Module Components
-----------------

.. toctree::
   :maxdepth: 1

   network/core
   network/graph_utils
   network/matrix_utils
   network/spectral
   network/quantum
   network/randomization
   network/visualization

Quick Links
-----------

**Core Class**
   * :class:`~driada.network.net_base.Network` - Main network analysis class
   * :doc:`network/core` - Network construction and basic properties

**Graph Operations**
   * :doc:`network/graph_utils` - Component extraction, cleaning
   * :doc:`network/matrix_utils` - Adjacency matrix operations

**Advanced Analysis**
   * :doc:`network/spectral` - Spectral entropy and analysis
   * :doc:`network/quantum` - Quantum-inspired network methods
   * :doc:`network/randomization` - Network randomization algorithms

**Visualization**
   * :doc:`network/visualization` - Network plotting utilities

Usage Example
-------------

.. code-block:: python

   from driada.network import Network
   from driada.network.graph_utils import get_giant_cc_from_graph
   import numpy as np
   import scipy.sparse as sp
   
   # Create example adjacency matrix
   n_nodes = 20
   adjacency_matrix = sp.random(n_nodes, n_nodes, density=0.1, format='csr')
   adjacency_matrix = adjacency_matrix + adjacency_matrix.T  # Make symmetric
   
   # Create network from adjacency matrix
   net = Network(adj=adjacency_matrix, preprocessing='giant_cc')
   
   # Analyze network properties
   print(f"Nodes: {net.n}, Edges: {net.graph.number_of_edges()}")
   print(f"Mean degree: {net.deg.mean():.2f}")