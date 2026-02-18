#!/usr/bin/env python
"""
Loading Real Data into DRIADA
=============================

This example shows how to load your own neural recording data into
a DRIADA Experiment object. Once loaded, you can use any DRIADA analysis
(INTENSE, dimensionality reduction, RSA, network analysis, etc.).

For analysis examples, see:
  - intense_basic_usage   -- single-neuron selectivity detection
  - full_intense_pipeline -- complete INTENSE workflow
  - compare_dr_methods    -- dimensionality reduction comparison
  - rsa/rsa_example       -- representational similarity analysis
"""

import os
import numpy as np
from driada.experiment import (
    load_exp_from_aligned_data,
    save_exp_to_pickle,
    load_exp_from_pickle,
)

OUTPUT_DIR = "examples/data_loading"


def main():
    print("=" * 60)
    print("Loading real data into DRIADA")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Prepare your data as numpy arrays
    # ------------------------------------------------------------------
    #
    # The most common starting point: you have numpy arrays from your
    # recording pipeline (Suite2P, CaImAn, DeepLabCut, custom, etc.).
    #
    # For this demo we load them from a bundled .npz file, but in your
    # code you would already have them as variables:
    #
    #   calcium = ...          # (n_neurons, n_frames) fluorescence traces
    #   speed = ...            # (n_frames,) animal speed
    #   head_direction = ...   # (n_frames,) head direction in radians
    #   trial_type = ...       # (n_frames,) integer trial labels
    #
    print("\n1. Loading arrays...")

    npz_path = "examples/example_data/sample_recording.npz"
    raw = dict(np.load(npz_path, allow_pickle=True))
    for key, val in sorted(raw.items()):
        if hasattr(val, "shape") and val.ndim > 0:
            print(f"   {key}: shape={val.shape}, dtype={val.dtype}")

    # ------------------------------------------------------------------
    # 2. Build the data dict
    # ------------------------------------------------------------------
    #
    # Pack everything into a single dict. Rules:
    #
    #   'calcium'  -- REQUIRED. Shape (n_neurons, n_frames).
    #   'spikes'   -- optional. Same shape as calcium (deconvolved spikes).
    #   other keys -- become dynamic features: time-varying behavioral
    #                 variables (1D or 2D arrays with n_frames samples).
    #                 Each one is stored as a TimeSeries in the Experiment.
    #
    print("\n2. Building data dict...")

    data = {
        # --- neural activity (required) --------------------------------
        "calcium": raw["calcium"],           # (50, 10000)
        # "spikes": my_spikes_array,         # optional, same shape as calcium
        # --- dynamic features: behavioral variables (one per timepoint) -
        "x_pos": raw["x_pos"],               # continuous
        "y_pos": raw["y_pos"],               # continuous
        "speed": raw["speed"],               # continuous
        "head_direction": raw["head_direction"],  # circular (radians)
        "trial_type": raw["trial_type"],     # discrete labels
    }

    # ------------------------------------------------------------------
    # 3. Override auto-detected feature types (optional)
    # ------------------------------------------------------------------
    #
    # DRIADA auto-detects whether each feature is continuous or discrete.
    # Use feature_types to correct or refine the detection.
    #
    # Valid type strings:
    #   continuous / linear   -- default for float arrays
    #   circular / phase      -- for angular variables (radians)
    #   discrete              -- generic discrete
    #   categorical / cat     -- unordered categories
    #   binary / spike        -- 0/1 variables
    #   count                 -- non-negative integers
    #
    feature_types = {
        "head_direction": "circular",   # auto-detection may miss this
        "trial_type": "categorical",    # refine from generic discrete
    }

    # ------------------------------------------------------------------
    # 4. Aggregate multi-component features (optional)
    # ------------------------------------------------------------------
    #
    # If you have related 1D features that form a single variable
    # (e.g. x_pos + y_pos = 2D position), you can tell DRIADA to
    # combine them into a MultiTimeSeries. The individual features
    # are kept as well -- aggregation only adds the combined version.
    #
    aggregate_features = {
        ("x_pos", "y_pos"): "position_2d",
    }

    # ------------------------------------------------------------------
    # 5. Build the Experiment
    # ------------------------------------------------------------------
    #
    # static_features are scalar constants for the whole recording
    # (frame rate, calcium kinetics, etc.) -- as opposed to dynamic
    # features which have one value per timepoint.
    #
    print("\n3. Building Experiment...")

    exp = load_exp_from_aligned_data(
        data_source="MyLab",
        exp_params={"name": "demo_recording"},
        data=data,
        feature_types=feature_types,
        aggregate_features=aggregate_features,
        static_features={"fps": 30.0},
        # create_circular_2d=True is the default: for every circular
        # feature (here head_direction), DRIADA auto-creates a _2d
        # version as (cos, sin). This is important because MI estimators
        # (GCMI, KSG) work on the real line -- a raw angle wraps around
        # at 0/2pi, breaking the distance metric. The (cos, sin) encoding
        # maps the circle onto R^2 where Euclidean distance is meaningful.
        create_circular_2d=True,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 6. Inspect the result
    # ------------------------------------------------------------------
    print("\n4. Inspecting Experiment...")
    print(f"   Neurons:     {exp.n_cells}")
    print(f"   Timepoints:  {exp.n_frames}")
    print(f"   FPS:         {exp.static_features.get('fps', 'unknown')}")
    print(f"   Calcium:     {exp.calcium.data.shape}")

    # Note the auto-generated features in the list below:
    #   - position_2d:        from aggregate_features (x_pos + y_pos)
    #   - head_direction_2d:  from create_circular_2d (cos + sin encoding)
    print("\n   Dynamic features (time-varying behavioral variables):")
    for name, ts in sorted(exp.dynamic_features.items()):
        ti = getattr(ts, "type_info", None)
        if ti and hasattr(ti, "primary_type"):
            dtype_str = f"{ti.primary_type}/{ti.subtype}"
            if ti.is_circular:
                dtype_str += " (circular)"
        else:
            dtype_str = "discrete" if ts.discrete else "continuous"
        shape = ts.data.shape
        print(f"     {name:25s}  shape={str(shape):15s}  type={dtype_str}")

    # ------------------------------------------------------------------
    # 7. Working with TimeSeries and MultiTimeSeries
    # ------------------------------------------------------------------
    #
    # Each dynamic feature is stored as one of two classes:
    #
    #   TimeSeries       -- a single 1D variable (e.g. speed)
    #   MultiTimeSeries  -- multiple aligned 1D variables stacked into
    #                       a 2D array (e.g. position_2d = [x, y])
    #
    # Key attributes on both:
    #   .data                -- raw numpy array (1D or 2D)
    #   .discrete            -- True if discrete, False if continuous
    #   .type_info           -- rich type metadata (subtype, circularity)
    #   .copula_normal_data  -- GCMI-ready transform (continuous only)
    #   .int_data            -- integer-coded values (discrete only)
    #
    # MultiTimeSeries additionally has:
    #   .ts_list             -- list of component TimeSeries objects
    #   .n_dim               -- number of components (rows)
    #
    print("\n5. Working with TimeSeries objects...")

    # Features are accessible as attributes: exp.speed, exp.position_2d, etc.
    # This is equivalent to exp.dynamic_features["speed"].
    speed_ts = exp.speed
    print(f"   speed.data.shape:   {speed_ts.data.shape}")
    print(f"   speed.discrete:     {speed_ts.discrete}")
    print(f"   speed.type_info:    {speed_ts.type_info.primary_type}"
          f"/{speed_ts.type_info.subtype}")
    print(f"   speed has copula:   {speed_ts.copula_normal_data is not None}")

    # Access a 2D feature (MultiTimeSeries)
    pos_mts = exp.position_2d
    print(f"\n   position_2d.data.shape: {pos_mts.data.shape}")
    print(f"   position_2d.n_dim:      {pos_mts.n_dim}  (x and y)")
    # Individual components are full TimeSeries objects:
    print(f"   position_2d.ts_list[0]: {pos_mts.ts_list[0].name}"
          f"  shape={pos_mts.ts_list[0].data.shape}")

    # Discrete feature
    trial_ts = exp.trial_type
    print(f"\n   trial_type.discrete:  {trial_ts.discrete}")
    print(f"   trial_type.int_data:  {trial_ts.int_data[:8]}...")
    print(f"   trial_type has copula: {trial_ts.copula_normal_data is not None}")

    # ------------------------------------------------------------------
    # 8. Neural data: MultiTimeSeries and Neuron objects
    # ------------------------------------------------------------------
    #
    # Neural activity is stored in two complementary ways:
    #
    #   exp.calcium  -- MultiTimeSeries (n_neurons, n_frames), convenient
    #                   for population-level analysis (DR, RSA, decoding)
    #   exp.spikes   -- MultiTimeSeries of spike trains (if provided)
    #   exp.neurons  -- list of Neuron objects, one per cell, for
    #                   single-neuron analysis (see neuron_basic_usage.py)
    #
    print("\n6. Neural data...")

    # Population-level: full calcium matrix as MultiTimeSeries
    print(f"   exp.calcium:        {type(exp.calcium).__name__}"
          f"  shape={exp.calcium.data.shape}")
    has_spikes = exp.spikes is not None and exp.spikes.data.any()
    print(f"   exp.spikes:         {'available' if has_spikes else 'not provided'}")

    # Single-neuron level: list of Neuron objects
    neuron = exp.neurons[0]
    print(f"\n   exp.neurons:        {len(exp.neurons)} Neuron objects")
    print(f"   neuron.cell_id:     {neuron.cell_id}")
    print(f"   neuron.ca:          {type(neuron.ca).__name__}"
          f"  shape={neuron.ca.data.shape}")
    print(f"   neuron.sp:          "
          f"{'shape=' + str(neuron.sp.data.shape) if neuron.sp else 'None (no spikes provided)'}")
    print(f"   neuron.fps:         {neuron.fps}")
    # See neuron_basic_usage.py for spike reconstruction, event
    # detection, kinetics optimization, and other Neuron methods.

    # ------------------------------------------------------------------
    # 9. Save and reload (pickle roundtrip)
    # ------------------------------------------------------------------
    print("\n7. Save/load roundtrip...")

    pkl_path = os.path.join(OUTPUT_DIR, "demo_experiment.pkl")
    save_exp_to_pickle(exp, pkl_path, verbose=False)
    file_size_mb = os.path.getsize(pkl_path) / 1024 / 1024
    print(f"   Saved:  {pkl_path} ({file_size_mb:.1f} MB)")

    exp_loaded = load_exp_from_pickle(pkl_path, verbose=False)
    print(f"   Loaded: {exp_loaded.n_cells} neurons, {exp_loaded.n_frames} frames")

    # Verify roundtrip
    assert exp_loaded.n_cells == exp.n_cells
    assert exp_loaded.n_frames == exp.n_frames
    assert np.allclose(exp_loaded.calcium.data, exp.calcium.data)
    print("   Roundtrip verified -- data matches.")

    # Clean up
    os.remove(pkl_path)
    print(f"   Cleaned up {pkl_path}")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[OK] Data loading complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Run INTENSE to detect selective neurons")
    print("  - Apply dimensionality reduction with MVData")
    print("  - Compute RSA between conditions")
    print("  - See other examples/ for complete workflows")


if __name__ == "__main__":
    main()
