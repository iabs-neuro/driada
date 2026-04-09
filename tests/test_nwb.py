import json
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pynwb import NWBFile, NWBHDF5IO
from pynwb.behavior import Position
from pynwb.ophys import (
    Fluorescence,
    ImageSegmentation,
    OpticalChannel,
    RoiResponseSeries,
)

from driada.experiment.synthetic.generators import generate_tuned_selectivity_exp
from driada.experiment.nwb import save_exp_to_nwb, load_exp_from_nwb


class FeatureMock:
    def __init__(self, data):
        self.data = data


def test_nwb_roundtrip():
    population = [
        {"name": "hd_cells", "count": 5, "features": ["head_direction"]},
        {"name": "speed_cells", "count": 5, "features": ["speed"]},
    ]

    exp_original = generate_tuned_selectivity_exp(
        population=population,
        duration=60,
        fps=20,
        seed=42,
        verbose=False
    )

    n_frames = exp_original.calcium.shape[1]
    t = np.linspace(0, 2 * np.pi, n_frames)

    exp_original.dynamic_features['x'] = FeatureMock(100 + 50 * np.cos(t))
    exp_original.dynamic_features['y'] = FeatureMock(100 + 50 * np.sin(t))

    if exp_original.metadata is None:
        exp_original.metadata = {}
    exp_original.metadata['session_name'] = "TEST_MOUSE_S01"

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir)

        save_exp_to_nwb(
            exp=exp_original,
            output_path=output_path,
            session_description="Integration Test",
            verbose=False
        )

        nwb_files = list(output_path.glob("*.nwb"))
        assert len(nwb_files) > 0
        nwb_file = nwb_files[0]

        exp_loaded = load_exp_from_nwb(
            path=nwb_file,
            verbose=False
        )

        assert exp_loaded.n_cells == exp_original.n_cells
        assert exp_loaded.calcium.shape == exp_original.calcium.shape

        orig_keys = set(exp_original.dynamic_features.keys())
        load_keys = set(exp_loaded.dynamic_features.keys())
        assert orig_keys.issubset(load_keys) or orig_keys == load_keys

        np.testing.assert_allclose(
            exp_original.calcium.data,
            exp_loaded.calcium.data,
            atol=1e-6
        )

        for key in ['x', 'y']:
            np.testing.assert_allclose(
                exp_original.dynamic_features[key].data,
                exp_loaded.dynamic_features[key].data,
                atol=1e-6
            )

        assert exp_loaded.metadata.get('session_name') == "TEST_MOUSE_S01"


def test_load_nwb_without_driada_notes(tmp_path):
    """A generic NWB file (notes without the DRIADA 'animal_id:' marker)
    must load without raising UnboundLocalError from extra_metadata."""
    n_cells = 3
    n_frames = 2000
    fps = 20.0
    rng = np.random.default_rng(0)

    nwb = NWBFile(
        session_description="External file",
        identifier="test-external",
        session_start_time=datetime.now(timezone.utc),
        session_id="external_session",
        notes="free-form experimenter notes with no DRIADA markers",
    )

    device = nwb.create_device(name="dev")
    optical_channel = OpticalChannel(
        name="oc", description="d", emission_lambda=525.0
    )
    imaging_plane = nwb.create_imaging_plane(
        name="ip",
        optical_channel=optical_channel,
        device=device,
        description="d",
        excitation_lambda=470.0,
        indicator="GCaMP6s",
        location="CA1",
    )

    ophys_mod = nwb.create_processing_module("ophys", "desc")
    img_seg = ImageSegmentation()
    ophys_mod.add(img_seg)
    ps = img_seg.create_plane_segmentation(
        name="PlaneSegmentation",
        description="ROIs",
        imaging_plane=imaging_plane,
    )
    for _ in range(n_cells):
        ps.add_roi(pixel_mask=[(0, 0, 1.0)])
    rt_region = ps.create_roi_table_region(
        description="All", region=list(range(n_cells))
    )

    calcium_series = RoiResponseSeries(
        name="Calcium",
        data=rng.standard_normal((n_frames, n_cells)).astype(np.float32),
        rois=rt_region,
        unit="df/f",
        rate=fps,
    )
    ophys_mod.add(Fluorescence(roi_response_series=[calcium_series]))

    beh_mod = nwb.create_processing_module("behavior", "desc")
    pos = Position()
    pos.create_spatial_series(
        name="SpatialSeries",
        data=rng.standard_normal((n_frames, 2)).astype(np.float32),
        reference_frame="none",
        rate=fps,
        unit="pixels",
    )
    beh_mod.add(pos)

    out = tmp_path / "external.nwb"
    with NWBHDF5IO(str(out), "w") as io:
        io.write(nwb)

    # Must not raise
    exp = load_exp_from_nwb(out, verbose=False)
    assert exp.n_cells == n_cells


def _make_hd_experiment():
    """Build a small synthetic experiment with both head_direction and head_direction_2d."""
    exp = generate_tuned_selectivity_exp(
        population=[{"name": "hd", "count": 3, "features": ["head_direction"]}],
        duration=60,
        fps=20,
        seed=0,
        verbose=False,
    )
    n_frames = exp.calcium.shape[1]
    t = np.linspace(0, 2 * np.pi, n_frames)
    exp.dynamic_features["x"] = FeatureMock(100 + 50 * np.cos(t))
    exp.dynamic_features["y"] = FeatureMock(100 + 50 * np.sin(t))
    if exp.metadata is None:
        exp.metadata = {}
    return exp


def test_save_nwb_drops_2d_feature_with_parent_silently(tmp_path):
    """When a _2d feature has its 1-d parent, dropping it must emit no warning."""
    exp = _make_hd_experiment()
    assert "head_direction" in exp.dynamic_features
    assert "head_direction_2d" in exp.dynamic_features
    exp.metadata["session_name"] = "DROPPARENT_MOUSE_S01"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        save_exp_to_nwb(exp=exp, output_path=tmp_path, verbose=False)

    loss_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "Dropping derived feature" in str(w.message)
    ]
    assert loss_warnings == []


def test_save_nwb_warns_for_orphan_2d_feature(tmp_path):
    """A _2d feature whose 1-d parent was removed must raise UserWarning on save."""
    exp = _make_hd_experiment()
    assert "head_direction" in exp.dynamic_features
    assert "head_direction_2d" in exp.dynamic_features
    del exp.dynamic_features["head_direction"]
    exp.metadata["session_name"] = "ORPHAN_MOUSE_S01"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        save_exp_to_nwb(exp=exp, output_path=tmp_path, verbose=False)

    loss_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "head_direction_2d" in str(w.message)
        and "head_direction" in str(w.message)
    ]
    assert len(loss_warnings) == 1, (
        f"expected 1 UserWarning, got {len(loss_warnings)}: "
        f"{[str(w.message) for w in caught]}"
    )


def test_save_nwb_centers_are_explicit_columns(tmp_path):
    """Centers must be stored as explicit center_y/center_x columns,
    not as single-pixel pixel_mask placeholders."""
    exp = _make_hd_experiment()
    exp.metadata["session_name"] = "NOF_H01_1D"
    exp.metadata["metrics_df"] = {
        "center": [[10.0, 20.0], [30.5, 40.25], [1.0, 2.0]],
        "snr": [1.1, 2.2, 3.3],
    }

    save_exp_to_nwb(exp=exp, output_path=tmp_path, verbose=False)

    nwb_file = next(tmp_path.glob("*.nwb"))
    with NWBHDF5IO(str(nwb_file), mode="r") as io:
        nwb = io.read()
        ps = nwb.processing["ophys"].data_interfaces["ImageSegmentation"]["PlaneSegmentation"]

        assert "center_y" in ps.colnames
        assert "center_x" in ps.colnames
        assert ps["center_y"].data[:].tolist() == [10.0, 30.5, 1.0]
        assert ps["center_x"].data[:].tolist() == [20.0, 40.25, 2.0]
        assert "snr" in ps.colnames

    exp_loaded = load_exp_from_nwb(nwb_file, verbose=False)
    recovered = exp_loaded.metadata["metrics_df"]["center"]
    assert recovered == [[10.0, 20.0], [30.5, 40.25], [1.0, 2.0]]


def test_save_nwb_metadata_in_scratch_not_notes(tmp_path):
    """DRIADA metadata must be stored in NWB scratch, leaving notes clean."""
    exp = _make_hd_experiment()
    exp.metadata["session_name"] = "NOF_H01_1D"
    exp.metadata["cnmf_params"] = {"gSig": [4, 4], "p": 1}
    exp.metadata["autoinspection_stats"] = {"n_total": 42, "n_good": 40}

    save_exp_to_nwb(exp=exp, output_path=tmp_path, verbose=False)

    nwb_file = next(tmp_path.glob("*.nwb"))
    with NWBHDF5IO(str(nwb_file), mode="r") as io:
        nwb = io.read()
        # notes must contain only animal_id, no JSON blob.
        assert nwb.notes == "animal_id: H01"
        assert "DRIADA METADATA" not in (nwb.notes or "")
        # scratch must contain the JSON-encoded metadata.
        assert "driada_metadata" in nwb.scratch
        raw = nwb.scratch["driada_metadata"].data
        if hasattr(raw, "item"):
            raw = raw.item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        parsed = json.loads(raw)
        assert parsed["cnmf_params"] == {"gSig": [4, 4], "p": 1}
        assert parsed["autoinspection_stats"] == {"n_total": 42, "n_good": 40}

    exp_loaded = load_exp_from_nwb(nwb_file, verbose=False)
    assert exp_loaded.metadata["cnmf_params"] == {"gSig": [4, 4], "p": 1}
    assert exp_loaded.metadata["autoinspection_stats"] == {"n_total": 42, "n_good": 40}


def test_save_nwb_animal_id_from_iabs_name(tmp_path):
    """animal_id must be extracted via parse_iabs_filename, not by ad-hoc split."""
    exp = _make_hd_experiment()
    exp.metadata["session_name"] = "NOF_H01_1D"

    save_exp_to_nwb(exp=exp, output_path=tmp_path, verbose=False)

    nwb_file = next(tmp_path.glob("*.nwb"))
    with NWBHDF5IO(str(nwb_file), mode="r") as io:
        nwb = io.read()
        first_line = (nwb.notes or "").split("\n")[0]
        assert first_line == "animal_id: H01"


if __name__ == "__main__":
    test_nwb_roundtrip()