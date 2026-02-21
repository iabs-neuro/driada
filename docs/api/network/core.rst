Network Core
============

.. automodule:: driada.network.net_base

General-purpose graph analysis class. Accepts any scipy sparse adjacency matrix
or NetworkX graph — connectomes, correlation networks, functional connectivity,
or any other graph. Also serves as the base class for
:class:`~driada.dim_reduction.graph.ProximityGraph`, so graph-based DR methods
produce objects with the full Network analysis toolkit.

Network Class
-------------

.. autoclass:: driada.network.net_base.Network
   :members:
   :special-members: __init__
   :inherited-members:

Validation Functions
--------------------

The module also contains various validation functions used internally by the Network class.