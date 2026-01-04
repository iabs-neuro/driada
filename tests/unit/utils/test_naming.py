"""Tests for naming utilities."""

import re
from driada.utils.naming import construct_session_name


def test_construct_session_name_old_track():
    """Test old track with non-standard naming."""
    params = {"track": "HT", "animal_id": "A123", "session": "5"}
    assert construct_session_name("IABS", params) == "A123_HT5"


def test_construct_session_name_standard_track():
    """Test standard track naming pattern."""
    params = {"track": "STFP", "animal_id": "B456", "session": "3"}
    assert construct_session_name("IABS", params) == "STFP_B456_3"


# Tests for generic (non-IABS) naming
def test_generic_lab_explicit_name():
    """Test generic lab with explicit 'name' parameter."""
    params = {"name": "my_experiment_001"}
    assert construct_session_name("MyLab", params) == "my_experiment_001"


def test_generic_lab_multiple_parameters():
    """Test generic lab naming with multiple common parameters."""
    params = {
        "experiment": "spatial_navigation",
        "subject": "rat42", 
        "session": "day3"
    }
    # Should combine available parameters
    assert construct_session_name("NeuroLab", params) == "spatial_navigation_rat42_day3"


def test_generic_lab_animal_id_session():
    """Test generic lab with animal_id and session."""
    params = {"animal_id": "mouse5", "session": "2"}
    assert construct_session_name("BrainLab", params) == "mouse5_2"


def test_generic_lab_date_parameter():
    """Test generic lab with date parameter."""
    params = {
        "subject": "monkey1",
        "date": "2024-01-15",
        "experiment": "reaching_task"
    }
    assert construct_session_name("PrimateResearch", params) == "reaching_task_monkey1_2024-01-15"


def test_generic_lab_partial_parameters():
    """Test with only some of the common parameters."""
    # Only experiment
    assert construct_session_name("Lab1", {"experiment": "test"}) == "test"
    
    # Only subject  
    assert construct_session_name("Lab2", {"subject": "sub1"}) == "sub1"
    
    # Only session
    assert construct_session_name("Lab3", {"session": "s1"}) == "s1"


def test_generic_lab_empty_params():
    """Test generic lab with empty parameters - should use timestamp."""
    params = {}
    result = construct_session_name("GenericLab", params)
    
    # Should be in format: GenericLab_YYYYMMDD_HHMMSS
    assert result.startswith("GenericLab_")
    # Check timestamp format (YYYYMMDD_HHMMSS)
    timestamp_pattern = r"GenericLab_\d{8}_\d{6}$"
    assert re.match(timestamp_pattern, result)


def test_generic_lab_timestamp_with_params():
    """Test that timestamp is added when no standard parameters exist."""
    params = {"custom_field": "value", "other_field": 123}
    result = construct_session_name("CustomLab", params)
    
    # Should use timestamp since no standard fields
    assert result.startswith("CustomLab_")
    timestamp_pattern = r"CustomLab_\d{8}_\d{6}$"
    assert re.match(timestamp_pattern, result)


def test_generic_lab_priority_order():
    """Test that 'name' parameter takes priority over others."""
    params = {
        "name": "explicit_name",
        "experiment": "should_be_ignored",
        "subject": "also_ignored",
        "session": "ignored_too"
    }
    assert construct_session_name("TestLab", params) == "explicit_name"


def test_generic_lab_string_conversion():
    """Test that non-string values are converted to strings."""
    params = {
        "experiment": "task",
        "animal_id": 42,  # Integer
        "session": 3.5    # Float
    }
    assert construct_session_name("NumLab", params) == "task_42_3.5"


def test_different_data_sources():
    """Test that different data sources work correctly."""
    params = {"subject": "s1", "session": "d1"}
    
    # Different labs should produce same format for same params
    assert construct_session_name("Lab_A", params) == "s1_d1"
    assert construct_session_name("Lab_B", params) == "s1_d1"
    assert construct_session_name("ResearchCenter", params) == "s1_d1"
