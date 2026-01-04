"""
Comprehensive tests for exp_build module.

Tests cover:
- load_exp_from_aligned_data with various inputs
- load_experiment with different scenarios
- save_exp_to_pickle and load_exp_from_pickle
- Edge cases and error handling
"""

import numpy as np
import pytest
import os
import tempfile
import shutil
from unittest.mock import patch
import pickle

from driada.experiment.exp_build import (
    load_exp_from_aligned_data,
    load_experiment,
    save_exp_to_pickle,
    load_exp_from_pickle,
)
from driada.experiment.exp_base import Experiment


# Mock experiment class for pickle tests
class MockExperiment:
    def __init__(self):
        self.signature = "Test Experiment"
        self.n_cells = 10
        self.n_frames = 100
        self.data = None


class TestLoadExpFromAlignedData:
    """Test load_exp_from_aligned_data function."""

    @pytest.fixture
    def basic_data(self):
        """Basic valid data dictionary."""
        np.random.seed(42)
        return {
            "calcium": np.random.rand(
                10, 500
            ),  # 10 neurons, 500 timepoints (25 sec at 20fps)
            "spikes": np.random.randint(0, 2, (10, 500)),
            "position": np.random.rand(500),
            "speed": np.random.rand(500),
            "direction": np.random.randint(0, 4, 500),  # discrete with fewer values
        }

    @pytest.fixture
    def exp_params(self):
        """Basic experiment parameters."""
        return {"track": "HT", "animal_id": "mouse1", "session": "day1"}

    def test_basic_loading(self, basic_data, exp_params):
        """Test basic experiment loading."""
        with patch(
            "driada.experiment.exp_build.construct_session_name",
            return_value="test_exp",
        ):
            exp = load_exp_from_aligned_data(
                "IABS", exp_params, basic_data, verbose=False
            )

        assert isinstance(exp, Experiment)
        assert exp.n_cells == 10
        assert exp.n_frames == 500
        assert len(exp.neurons) == 10
        assert hasattr(exp, "position")
        assert hasattr(exp, "speed")
        assert hasattr(exp, "direction")

    def test_input_validation(self, exp_params):
        """Test input validation."""
        # Test non-dict data
        with pytest.raises(TypeError, match="data must be a dictionary"):
            load_exp_from_aligned_data("IABS", exp_params, "not a dict")

        # Test empty data
        with pytest.raises(ValueError, match="data dictionary cannot be empty"):
            load_exp_from_aligned_data("IABS", exp_params, {})

        # Test non-dict exp_params
        with pytest.raises(TypeError, match="exp_params must be a dictionary"):
            load_exp_from_aligned_data("IABS", "not a dict", {"calcium": np.array([])})

    def test_missing_calcium(self, exp_params):
        """Test error when calcium data is missing."""
        data = {"spikes": np.array([[1, 0]]), "position": np.array([1, 2])}

        with pytest.raises(ValueError, match="No calcium data found"):
            load_exp_from_aligned_data("IABS", exp_params, data)

    def test_case_insensitive_keys(self, exp_params):
        """Test case-insensitive key handling for calcium/spikes."""
        data = {
            "Calcium": np.random.rand(5, 500),  # Capital C
            "SPIKES": np.random.randint(0, 2, (5, 500)),  # All caps
            "position": np.random.rand(500),
        }

        with patch(
            "driada.experiment.exp_build.construct_session_name", return_value="test"
        ):
            exp = load_exp_from_aligned_data("IABS", exp_params, data, verbose=False)

        assert exp.n_cells == 5
        assert exp.n_frames == 500

    def test_garbage_feature_removal(self, exp_params):
        """Test removal of constant and all-NaN features."""
        data = {
            "calcium": np.random.rand(3, 500),
            "constant_feature": np.ones(500),  # All same value
            "nan_feature": np.full(500, np.nan),  # All NaN
            "good_feature": np.array([1, 2, 3] * 166 + [1, 2]),  # Varying values
            "mixed_constant": np.array([1] * 499 + [np.nan]),  # Constant except NaN
        }

        with patch(
            "driada.experiment.exp_build.construct_session_name", return_value="test"
        ):
            exp = load_exp_from_aligned_data("IABS", exp_params, data, verbose=False)

        # Only good_feature should remain
        assert hasattr(exp, "good_feature")
        assert not hasattr(exp, "constant_feature")
        assert not hasattr(exp, "nan_feature")
        assert not hasattr(exp, "mixed_constant")

    def test_force_continuous(self, exp_params):
        """Test force_continuous parameter."""
        data = {
            "calcium": np.random.rand(3, 500),
            "discrete_vals": np.array([0, 1] * 250),  # Binary
            "multi_vals": np.array([0, 1, 2, 3, 4] * 100),  # Multiple values
            "continuous_vals": np.random.rand(500),
        }

        with patch(
            "driada.experiment.exp_build.construct_session_name", return_value="test"
        ):
            # Force multi_vals to be continuous
            exp = load_exp_from_aligned_data(
                "source",
                exp_params,
                data,
                force_continuous=["multi_vals", "continuous_vals"],
                verbose=False,
            )

        # Check discreteness
        assert exp.discrete_vals.discrete
        assert not exp.multi_vals.discrete  # Forced continuous
        assert not exp.continuous_vals.discrete

    def test_force_continuous_override(self, exp_params):
        """Test that force_continuous can override auto-detection."""
        data = {
            "calcium": np.random.rand(3, 500),
            "feature_a": np.array(
                [0, 1, 2, 3, 4] * 100
            ),  # Would be auto-detected as discrete
            "feature_b": np.array([0, 1] * 250),  # Binary, would be discrete
        }

        with patch(
            "driada.experiment.exp_build.construct_session_name", return_value="test"
        ):
            exp = load_exp_from_aligned_data(
                "source",
                exp_params,
                data,
                force_continuous=["feature_a"],  # Force feature_a to be continuous
                verbose=True,
            )

        # feature_a should be forced continuous despite having few unique values
        assert not exp.feature_a.discrete
        assert len(np.unique(exp.feature_a.data)) == 5  # Preserves all values

        # feature_b should remain discrete (auto-detected)
        assert exp.feature_b.discrete
        assert len(np.unique(exp.feature_b.data)) == 2

    def test_bad_frames_mask(self, basic_data, exp_params):
        """Test bad frames masking."""
        bad_frames = [10, 20, 30, 40, 50]  # Remove these frames

        with patch(
            "driada.experiment.exp_build.construct_session_name", return_value="test"
        ):
            exp = load_exp_from_aligned_data(
                "source", exp_params, basic_data, bad_frames=bad_frames, verbose=False
            )

        # Should have 5 fewer frames
        assert exp.n_frames == 495  # 500 - 5 bad frames

    def test_static_features_defaults(self, basic_data, exp_params):
        """Test default static features are set."""
        with patch(
            "driada.experiment.exp_build.construct_session_name", return_value="test"
        ):
            exp = load_exp_from_aligned_data(
                "source", exp_params, basic_data, verbose=False
            )

        # Check defaults are set as attributes
        assert hasattr(exp, "t_rise_sec")
        assert hasattr(exp, "t_off_sec")
        assert hasattr(exp, "fps")

    def test_static_features_override(self, basic_data, exp_params):
        """Test overriding static features."""
        custom_static = {
            "t_rise_sec": 1.0,
            "t_off_sec": 2.0,
            "fps": 20.0,
            "custom_param": "test",
        }

        with patch(
            "driada.experiment.exp_build.construct_session_name", return_value="test"
        ):
            exp = load_exp_from_aligned_data(
                "source",
                exp_params,
                basic_data,
                static_features=custom_static,
                verbose=False,
            )

        assert exp.t_rise_sec == 1.0
        assert exp.t_off_sec == 2.0
        assert exp.fps == 20.0
        assert exp.custom_param == "test"

    def test_reconstruct_spikes_parameter(self, exp_params):
        """Test reconstruct_spikes parameter is passed correctly."""
        data = {"calcium": np.random.rand(3, 500)}

        with patch(
            "driada.experiment.exp_build.construct_session_name", return_value="test"
        ):
            with patch(
                "driada.experiment.exp_base.Experiment.__init__", return_value=None
            ) as mock_init:
                load_exp_from_aligned_data(
                    "source",
                    exp_params,
                    data,
                    reconstruct_spikes="oasis",
                    verbose=False,
                )

                # Check reconstruct_spikes was passed
                _, kwargs = mock_init.call_args
                assert kwargs["reconstruct_spikes"] == "oasis"

    def test_verbose_output(self, basic_data, exp_params, capsys):
        """Test verbose output."""
        with patch(
            "driada.experiment.exp_build.construct_session_name", return_value="test"
        ):
            load_exp_from_aligned_data("source", exp_params, basic_data, verbose=True)

        captured = capsys.readouterr()
        assert "Building experiment test..." in captured.out
        assert "behaviour variables:" in captured.out
        assert "'position' continuous" in captured.out
        assert "'speed' continuous" in captured.out
        assert "'direction' discrete" in captured.out

    def test_empty_features_handling(self, exp_params):
        """Test handling of empty feature arrays."""
        data = {
            "calcium": np.random.rand(3, 500),
            "empty_feature": np.array([]),  # Empty array
            "good_feature": np.linspace(0, 1, 500),  # Clearly continuous
        }

        with patch(
            "driada.experiment.exp_build.construct_session_name", return_value="test"
        ):
            exp = load_exp_from_aligned_data("IABS", exp_params, data, verbose=False)

        # Empty feature should be removed
        assert not hasattr(exp, "empty_feature")
        assert hasattr(exp, "good_feature")


class TestLoadExperiment:
    """Test load_experiment function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_experiment(self):
        """Create a simple mock experiment that can be pickled."""
        return MockExperiment()

    def test_load_from_existing_pickle(self, temp_dir, mock_experiment):
        """Test loading from existing pickle file."""
        # Create pickle file
        exp_path = os.path.join(temp_dir, "test_exp", "Exp test_exp.pickle")
        os.makedirs(os.path.dirname(exp_path), exist_ok=True)

        with open(exp_path, "wb") as f:
            pickle.dump(mock_experiment, f)

        # Load experiment
        with patch(
            "driada.experiment.exp_build.construct_session_name",
            return_value="test_exp",
        ):
            exp, log = load_experiment(
                "IABS", {"test": "params"}, root=temp_dir, verbose=False
            )

        assert exp.signature == "Test Experiment"
        assert log is None  # No download log when loading from pickle

    def test_force_rebuild(self, temp_dir, mock_experiment):
        """Test force_rebuild parameter."""
        # Create existing pickle
        exp_path = os.path.join(temp_dir, "test_exp", "Exp test_exp.pickle")
        os.makedirs(os.path.dirname(exp_path), exist_ok=True)

        with open(exp_path, "wb") as f:
            pickle.dump(mock_experiment, f)

        # Create aligned data
        data_path = os.path.join(
            temp_dir, "test_exp", "Aligned data", "test_exp syn data.npz"
        )
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        np.savez(data_path, calcium=np.random.rand(5, 500))

        # Load with force_rebuild
        with patch(
            "driada.experiment.exp_build.construct_session_name",
            return_value="test_exp",
        ):
            with patch(
                "driada.experiment.exp_build.load_exp_from_aligned_data"
            ) as mock_load:
                mock_load.return_value = mock_experiment

                exp, log = load_experiment(
                    "IABS",
                    {"test": "params"},
                    root=temp_dir,
                    force_rebuild=True,
                    verbose=False,
                )

                # Should call load_exp_from_aligned_data, not load from pickle
                mock_load.assert_called_once()

    def test_download_from_gdrive(self, temp_dir):
        """Test downloading data from Google Drive."""
        with patch(
            "driada.experiment.exp_build.construct_session_name",
            return_value="test_exp",
        ):
            with patch(
                "driada.experiment.exp_build.initialize_iabs_router"
            ) as mock_router:
                with patch(
                    "driada.experiment.exp_build.download_gdrive_data"
                ) as mock_download:
                    mock_router.return_value = ("router", "pieces")
                    mock_download.return_value = (True, ["Download successful"])

                    # Create aligned data after "download"
                    data_path = os.path.join(
                        temp_dir, "test_exp", "Aligned data", "test_exp syn data.npz"
                    )
                    os.makedirs(os.path.dirname(data_path), exist_ok=True)
                    np.savez(data_path, calcium=np.random.rand(5, 500))

                    exp, log = load_experiment(
                        "IABS",
                        {"test": "params"},
                        root=temp_dir,
                        force_reload=True,
                        verbose=False,
                    )

                    # Check download was called
                    mock_download.assert_called_once()
                    assert log == ["Download successful"]

    def test_download_failure(self, temp_dir):
        """Test handling of download failure."""
        with patch(
            "driada.experiment.exp_build.construct_session_name",
            return_value="test_exp",
        ):
            with patch(
                "driada.experiment.exp_build.initialize_iabs_router"
            ) as mock_router:
                with patch(
                    "driada.experiment.exp_build.download_gdrive_data"
                ) as mock_download:
                    mock_router.return_value = ("router", "pieces")
                    mock_download.return_value = (False, ["Download failed"])

                    with pytest.raises(
                        FileNotFoundError, match="Cannot download test_exp"
                    ):
                        load_experiment(
                            "IABS",
                            {"test": "params"},
                            root=temp_dir,
                            force_reload=True,
                            verbose=False,
                        )

    def test_non_iabs_source_requires_data_path(self, temp_dir):
        """Test error when non-IABS source is used without data_path."""
        with pytest.raises(
            ValueError, match="For data source 'MyLab', you must provide the 'data_path' parameter"
        ):
            load_experiment("MyLab", {'name': 'test'}, root=temp_dir)

    def test_save_to_pickle_option(self, temp_dir):
        """Test save_to_pickle parameter."""
        # Create aligned data
        with patch(
            "driada.experiment.exp_build.construct_session_name",
            return_value="test_exp",
        ):
            data_path = os.path.join(
                temp_dir, "test_exp", "Aligned data", "test_exp syn data.npz"
            )
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            np.savez(data_path, calcium=np.random.rand(5, 500))

            exp, log = load_experiment(
                "IABS",
                {"test": "params"},
                root=temp_dir,
                save_to_pickle=True,
                verbose=False,
            )

            # Check pickle was created
            pickle_path = os.path.join(temp_dir, "test_exp", "Exp test_exp.pickle")
            assert os.path.exists(pickle_path)

    def test_custom_paths(self, temp_dir):
        """Test custom exp_path and data_path."""
        custom_exp_path = os.path.join(temp_dir, "custom_exp.pickle")
        custom_data_path = os.path.join(temp_dir, "custom_data.npz")

        # Create data file
        np.savez(custom_data_path, calcium=np.random.rand(5, 500))

        exp, log = load_experiment(
            "IABS",
            {"track": "HT", "animal_id": "test", "session": "1"},
            root=temp_dir,
            exp_path=custom_exp_path,
            data_path=custom_data_path,
            save_to_pickle=True,
            verbose=False,
        )

        # Check custom pickle path was used
        assert os.path.exists(custom_exp_path)

    def test_root_directory_creation(self, temp_dir):
        """Test root directory is created if it doesn't exist."""
        new_root = os.path.join(temp_dir, "new_root")
        assert not os.path.exists(new_root)

        with patch(
            "driada.experiment.exp_build.construct_session_name", return_value="test"
        ):
            with patch("driada.experiment.exp_build.load_exp_from_aligned_data"):
                # This should create the directory
                try:
                    load_experiment("IABS", {}, root=new_root, verbose=False)
                except:
                    pass  # We expect it to fail later, but directory should be created

        assert os.path.exists(new_root)

    def test_root_not_directory(self, temp_dir):
        """Test error when root is not a directory."""
        # Create a file instead of directory
        root_file = os.path.join(temp_dir, "not_a_dir")
        with open(root_file, "w") as f:
            f.write("test")

        with pytest.raises(ValueError, match="Root must be a folder"):
            load_experiment("IABS", {}, root=root_file)
    
    def test_generic_lab_loading(self, temp_dir):
        """Test loading experiment from generic (non-IABS) lab data."""
        # Create NPZ file with required data
        data_path = os.path.join(temp_dir, "my_data.npz")
        np.savez(
            data_path,
            calcium=np.random.rand(10, 1000),
            position=np.random.rand(1000),
            speed=np.random.rand(1000),
            trial_type=np.tile([0, 1, 2, 3], 250)  # Numeric discrete data
        )
        
        exp, log = load_experiment(
            "MyLab",
            {"name": "test_experiment"},
            data_path=data_path,
            root=temp_dir,
            verbose=False
        )
        
        assert exp.n_cells == 10
        assert exp.n_frames == 1000
        assert hasattr(exp, "position")
        assert hasattr(exp, "speed")
        assert hasattr(exp, "trial_type")
        assert log is None  # No download log for local files
    
    def test_generic_lab_multidimensional_features(self, temp_dir):
        """Test handling of 2D features in generic lab data."""
        # Create NPZ with 2D position data
        data_path = os.path.join(temp_dir, "2d_data.npz")
        np.savez(
            data_path,
            calcium=np.random.rand(10, 1000),  # More neurons and frames to avoid shuffle issues
            position=np.random.rand(2, 1000),  # 2D trajectory
            x_pos=np.random.rand(1000),
            y_pos=np.random.rand(1000)
        )
        
        exp, _ = load_experiment(
            "NeuroLab",
            {"subject": "rat1", "session": "day1"},
            data_path=data_path,
            root=temp_dir,
            static_features={"fps": 30.0},
            verbose=False
        )
        
        assert hasattr(exp, "position")
        assert exp.position.n_dim == 2  # Should be MultiTimeSeries with 2 components
        assert hasattr(exp, "x_pos")
        assert hasattr(exp, "y_pos")
    
    def test_generic_lab_scalar_warning(self, temp_dir, capsys):
        """Test warning for scalar values in NPZ file."""
        # Create NPZ with scalar value that should be ignored
        data_path = os.path.join(temp_dir, "scalar_data.npz")
        np.savez(
            data_path,
            calcium=np.random.rand(10, 1000),  # More neurons/frames to avoid issues
            position=np.random.rand(1000),
            fps=30.0,  # Scalar - should trigger warning
            description="test"  # Another scalar
        )
        
        exp, _ = load_experiment(
            "MyLab",
            {"name": "test"},
            data_path=data_path,
            root=temp_dir,
            static_features={"fps": 30.0},
            verbose=True
        )
        
        captured = capsys.readouterr()
        assert "Ignoring scalar value 'fps'" in captured.out
        assert "Ignoring scalar value 'description'" in captured.out
        # fps should exist as static feature (default or provided), just not as dynamic
        assert hasattr(exp, "fps")  # Should exist as static feature
        assert exp.fps == 30.0  # Should use the provided static value
        # description should not exist at all (not static or dynamic)
        assert not hasattr(exp, "description")
    
    def test_generic_lab_non_numeric_warning(self, temp_dir, capsys):
        """Test warning for non-numeric features in NPZ file."""
        # Create NPZ with string data that should be ignored
        data_path = os.path.join(temp_dir, "string_data.npz")
        
        # We need to use object dtype to store strings in numpy arrays
        trial_labels = np.array(['A', 'B', 'C', 'D'] * 125, dtype=object)
        
        np.savez(
            data_path,
            calcium=np.random.rand(10, 1000),  # More neurons/frames to avoid issues
            position=np.random.rand(1000),
            trial_labels=trial_labels  # String data - should trigger warning
        )
        
        exp, _ = load_experiment(
            "MyLab",
            {"name": "test"},
            data_path=data_path,
            root=temp_dir,
            verbose=True
        )
        
        captured = capsys.readouterr()
        assert "Ignoring non-numeric feature 'trial_labels'" in captured.out
        assert not hasattr(exp, "trial_labels")  # Should not be added
    
    def test_generic_lab_save_pickle(self, temp_dir):
        """Test saving generic lab experiment to pickle."""
        # Create NPZ file
        data_path = os.path.join(temp_dir, "data.npz")
        np.savez(
            data_path,
            calcium=np.random.rand(10, 1000),  # More neurons/frames
            position=np.random.rand(1000)
        )
        
        exp, _ = load_experiment(
            "MyLab",
            {"experiment": "navigation", "animal_id": "m1"},
            data_path=data_path,
            root=temp_dir,
            save_to_pickle=True,
            verbose=False
        )
        
        # Check pickle was created in expected location
        # Note: load_experiment creates path as root/expname/Exp expname.pickle
        expected_path = os.path.join(
            temp_dir, "navigation_m1", "Exp navigation_m1.pickle"
        )
        assert os.path.exists(expected_path)
        
        # Verify we can load from pickle
        exp2, _ = load_experiment(
            "MyLab",
            {"experiment": "navigation", "animal_id": "m1"},
            root=temp_dir,
            verbose=False
        )
        assert exp2.n_cells == exp.n_cells
        assert exp2.n_frames == exp.n_frames
    
    def test_generic_lab_missing_calcium(self, temp_dir):
        """Test error when calcium data is missing from NPZ."""
        data_path = os.path.join(temp_dir, "no_calcium.npz")
        np.savez(
            data_path,
            position=np.random.rand(500),
            speed=np.random.rand(500)
        )
        
        with pytest.raises(ValueError, match="NPZ file must contain 'calcium' key"):
            load_experiment(
                "MyLab",
                {"name": "test"},
                data_path=data_path,
                root=temp_dir
            )
    
    def test_generic_lab_invalid_npz(self, temp_dir):
        """Test error handling for invalid NPZ file."""
        # Create invalid file
        bad_path = os.path.join(temp_dir, "bad.npz")
        with open(bad_path, "w") as f:
            f.write("not a valid npz file")
        
        with pytest.raises(ValueError, match="Failed to load NPZ file"):
            load_experiment(
                "MyLab", 
                {"name": "test"},
                data_path=bad_path,
                root=temp_dir
            )
    
    def test_generic_lab_file_not_found(self, temp_dir):
        """Test error when data file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            load_experiment(
                "MyLab",
                {"name": "test"}, 
                data_path="/non/existent/file.npz",
                root=temp_dir
            )


class TestPickleFunctions:
    """Test save_exp_to_pickle and load_exp_from_pickle."""

    @pytest.fixture
    def temp_file(self):
        """Create a temporary file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pickle") as f:
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_save_and_load_pickle(self, temp_file):
        """Test saving and loading experiment to/from pickle."""
        # Create mock experiment
        exp = MockExperiment()
        exp.data = np.array([1, 2, 3])

        # Save
        save_exp_to_pickle(exp, temp_file, verbose=False)
        assert os.path.exists(temp_file)

        # Load
        loaded_exp = load_exp_from_pickle(temp_file, verbose=False)
        assert loaded_exp.signature == "Test Experiment"
        np.testing.assert_array_equal(loaded_exp.data, np.array([1, 2, 3]))

    def test_save_pickle_verbose(self, temp_file, capsys):
        """Test verbose output when saving."""
        exp = MockExperiment()
        exp.signature = "Test Exp"

        save_exp_to_pickle(exp, temp_file, verbose=True)

        captured = capsys.readouterr()
        assert f"Experiment Test Exp saved to {temp_file}" in captured.out

    def test_load_pickle_verbose(self, temp_file, capsys):
        """Test verbose output when loading."""
        exp = MockExperiment()
        exp.signature = "Test Exp"

        with open(temp_file, "wb") as f:
            pickle.dump(exp, f)

        load_exp_from_pickle(temp_file, verbose=True)

        captured = capsys.readouterr()
        assert f"Experiment Test Exp loaded from {temp_file}" in captured.out

    def test_load_nonexistent_pickle(self):
        """Test loading non-existent pickle file."""
        with pytest.raises(FileNotFoundError):
            load_exp_from_pickle("/non/existent/file.pickle")

    @pytest.mark.skip(reason="Path validation behavior is platform-dependent")
    def test_save_to_invalid_path(self):
        """Test saving to invalid path."""
        exp = MockExperiment()

        with pytest.raises(Exception):  # Could be OSError, PermissionError, etc.
            save_exp_to_pickle(exp, "/invalid/path/file.pickle")


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_is_garbage_edge_cases(self):
        """Test the improved is_garbage function with edge cases."""
        from driada.experiment.exp_build import load_exp_from_aligned_data

        # Access the is_garbage function (it's defined inside load_exp_from_aligned_data)
        # We'll test it indirectly through the main function

        data = {
            "calcium": np.random.rand(3, 500),
            "empty": np.array([]),  # Empty array
            "all_nan": np.full(500, np.nan),  # All NaN
            "single_value": np.ones(500),  # All same
            "nan_and_const": np.array([1] * 499 + [np.nan]),  # Constant with NaN
            "mixed": np.random.rand(500),  # Mixed values with good variation
        }

        with patch(
            "driada.experiment.exp_build.construct_session_name", return_value="test"
        ):
            exp = load_exp_from_aligned_data("source", {}, data, verbose=False)

        # Check what was kept/removed
        assert not hasattr(exp, "empty")
        assert not hasattr(exp, "all_nan")
        assert not hasattr(exp, "single_value")
        assert not hasattr(exp, "nan_and_const")
        assert hasattr(exp, "mixed")  # Should be kept

    def test_deepcopy_behavior(self):
        """Test that input data is not modified."""
        original_data = {
            "calcium": np.random.rand(3, 500),
            "feature": np.random.rand(500),
        }

        # Make a copy to compare later
        data_copy = {k: v.copy() for k, v in original_data.items()}

        with patch(
            "driada.experiment.exp_build.construct_session_name", return_value="test"
        ):
            load_exp_from_aligned_data("source", {}, original_data, verbose=False)

        # Original data should be unchanged
        for key in original_data:
            np.testing.assert_array_equal(original_data[key], data_copy[key])

    def test_large_bad_frames_list(self):
        """Test with many bad frames."""
        data = {"calcium": np.random.rand(5, 1000)}
        bad_frames = list(range(0, 1000, 2))  # Every other frame is bad

        with patch(
            "driada.experiment.exp_build.construct_session_name", return_value="test"
        ):
            exp = load_exp_from_aligned_data(
                "source", {}, data, bad_frames=bad_frames, verbose=False
            )

        assert exp.n_frames == 500  # Half the frames removed
