"""Tests for naming utilities."""

from driada.utils.naming import construct_session_name


def test_construct_session_name_old_track():
    """Test old track with non-standard naming."""
    params = {'track': 'HT', 'animal_id': 'A123', 'session': '5'}
    assert construct_session_name('IABS', params) == 'A123_HT5'


def test_construct_session_name_standard_track():
    """Test standard track naming pattern."""
    params = {'track': 'STFP', 'animal_id': 'B456', 'session': '3'}
    assert construct_session_name('IABS', params) == 'STFP_B456_3'