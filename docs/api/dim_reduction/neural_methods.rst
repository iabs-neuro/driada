Neural Network Methods
======================

Neural network-based dimensionality reduction methods including autoencoders.

Standard Architectures
----------------------

.. autoclass:: driada.dim_reduction.neural.AE
   :members:
   :special-members: __init__

.. autoclass:: driada.dim_reduction.neural.VAE
   :members:
   :special-members: __init__

Building Blocks
---------------

.. autoclass:: driada.dim_reduction.neural.Encoder
   :members:
   :special-members: __init__

.. autoclass:: driada.dim_reduction.neural.VAEEncoder
   :members:
   :special-members: __init__

.. autoclass:: driada.dim_reduction.neural.Decoder
   :members:
   :special-members: __init__

Flexible Architectures
----------------------

.. autoclass:: driada.dim_reduction.flexible_ae.FlexibleAutoencoderBase
   :members:
   :special-members: __init__

.. autoclass:: driada.dim_reduction.flexible_ae.ModularAutoencoder
   :members:
   :special-members: __init__

.. autoclass:: driada.dim_reduction.flexible_ae.FlexibleVAE
   :members:
   :special-members: __init__

Loss Functions
--------------

.. autoclass:: driada.dim_reduction.losses.AELoss
   :members:

.. autoclass:: driada.dim_reduction.losses.ReconstructionLoss
   :members:

.. autoclass:: driada.dim_reduction.losses.BetaVAELoss
   :members:

.. autoclass:: driada.dim_reduction.losses.CorrelationLoss
   :members:

.. autoclass:: driada.dim_reduction.losses.OrthogonalityLoss
   :members:

.. autoclass:: driada.dim_reduction.losses.SparsityLoss
   :members:

See :class:`~driada.dim_reduction.losses.LossRegistry` for the complete list of available losses.

Data Handling
-------------

.. autoclass:: driada.dim_reduction.neural.NeuroDataset
   :members:
   :special-members: __init__