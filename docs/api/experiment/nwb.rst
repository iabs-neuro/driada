NWB Input and Output
====================

.. currentmodule:: driada.experiment.nwb

This module saves an :class:`~driada.experiment.exp_base.Experiment` to a
`Neurodata Without Borders <https://www.nwb.org/>`_ (NWB) file and loads it back.
Calcium traces, neuron centers, the tracked position, the other dynamic features
and DRIADA metadata are stored in the file. The functions are imported from
``driada.experiment.nwb``.

The experiment must have ``x`` and ``y`` dynamic features: they are written as
the animal position. The file is named after ``exp.metadata['session_name']``.

Functions
---------

.. autofunction:: driada.experiment.nwb.save_exp_to_nwb
.. autofunction:: driada.experiment.nwb.load_exp_from_nwb

Usage Example
-------------

.. code-block:: python

   from driada.experiment.synthetic import generate_tuned_selectivity_exp
   from driada.experiment.nwb import save_exp_to_nwb, load_exp_from_nwb

   population = [
       {"name": "x_cells", "count": 5, "features": ["x"]},
       {"name": "y_cells", "count": 5, "features": ["y"]},
   ]
   exp = generate_tuned_selectivity_exp(population=population, duration=60,
                                        seed=0, verbose=False)
   exp.metadata = exp.metadata or {}
   exp.metadata["session_name"] = "MOUSE1_S01"

   # Writes ./MOUSE1_S01.nwb
   save_exp_to_nwb(exp, output_path=".", session_description="Synthetic session")

   exp_loaded = load_exp_from_nwb("MOUSE1_S01.nwb")
   print(exp_loaded.n_cells, exp_loaded.n_frames)
