import tempfile
from pathlib import Path
import numpy as np

from driada.experiment.synthetic.generators import generate_tuned_selectivity_exp
from driada.experiment.nwb import save_exp_to_nwb, load_exp_from_nwb


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

    class FeatureMock:
        def __init__(self, data):
            self.data = data

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


if __name__ == "__main__":
    test_nwb_roundtrip()