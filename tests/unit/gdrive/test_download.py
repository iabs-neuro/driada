import os
import pytest
from driada.gdrive.download import *
import shutil
import requests
import urllib3
import http.client


TEST_LINK = "https://drive.google.com/drive/folders/1rqV0A3Y-miX8QccmkiGEI5r-5K4RdjCj?usp=sharing"
TEST_DIR = os.path.join(os.getcwd(), "test dir")


def skip_on_network_error(func):
    """Decorator to skip tests on network-related errors."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (
            requests.exceptions.SSLError,
            requests.exceptions.ConnectionError,
            urllib3.exceptions.SSLError,
            urllib3.exceptions.MaxRetryError,
            http.client.RemoteDisconnected,
            OSError,
        ) as e:
            # Check if it's a network-related error
            error_msg = str(e).lower()
            if any(
                keyword in error_msg
                for keyword in ["ssl", "connection", "network", "remote end closed"]
            ):
                pytest.skip(f"Skipping test due to network error: {type(e).__name__}")
            else:
                raise

    return wrapper


@pytest.fixture(autouse=True, scope="function")
def cleanup_test_dir():
    """Fixture to clean up test directory before and after each test."""
    # Clean up before test
    shutil.rmtree(TEST_DIR, ignore_errors=True)

    yield  # Run the test

    # Clean up after test
    shutil.rmtree(TEST_DIR, ignore_errors=True)


@skip_on_network_error
def test_download():
    print(TEST_DIR)

    return_code, filenames, load_log = download_part_of_folder(
        TEST_DIR,  # path for downloaded data
        TEST_LINK,  # share link to google drive folder
        key="",  # part of filename to search for
        antikey=None,  # part of name to suppress
        whitelist=[],  # list of filenames to be downloaded regardless of their names
        extensions=[],  # allowed file extensions
    )

    assert len(os.listdir(TEST_DIR)) == 5


@skip_on_network_error
def test_download_redflag():
    print(TEST_DIR)

    return_code, filenames, load_log = download_part_of_folder(
        TEST_DIR,  # path for downloaded data
        TEST_LINK,  # share link to google drive folder
        key="",  # part of filename to search for
        antikey="redflag",  # part of name to suppress
        whitelist=[],  # list of filenames to be downloaded regardless of their names
        extensions=[],  # allowed file extensions
    )

    assert len(os.listdir(TEST_DIR)) == 4


@skip_on_network_error
def test_download_extension():
    print(TEST_DIR)

    return_code, filenames, load_log = download_part_of_folder(
        TEST_DIR,  # path for downloaded data
        TEST_LINK,  # share link to google drive folder
        key="",  # part of filename to search for
        antikey=None,  # part of name to suppress
        whitelist=[],  # list of filenames to be downloaded regardless of their names
        extensions=[".txt"],  # allowed file extensions
    )

    assert sorted(os.listdir(TEST_DIR)) == sorted(["test.txt", "test2.txt"])


@skip_on_network_error
def test_download_whitelist():
    print(TEST_DIR)

    return_code, filenames, load_log = download_part_of_folder(
        TEST_DIR,  # path for downloaded data
        TEST_LINK,  # share link to google drive folder
        key="",  # part of filename to search for
        antikey=None,  # part of name to suppress
        whitelist=[
            "white.xlsx"
        ],  # list of filenames to be downloaded regardless of their names
        extensions=[".txt"],  # allowed file extensions
    )

    assert sorted(os.listdir(TEST_DIR)) == sorted(
        ["test.txt", "test2.txt", "white.xlsx"]
    )


class TestGDriveModuleImports:
    """Test imports for Google Drive module components."""

    def test_import_auth(self):
        """Test importing GDrive authentication."""
        from driada.gdrive import auth

        assert hasattr(auth, "__file__")

    def test_import_upload(self):
        """Test importing GDrive upload utilities."""
        from driada.gdrive import upload

        assert hasattr(upload, "__file__")

    def test_import_gdrive_utils(self):
        """Test importing GDrive utilities."""
        from driada.gdrive import gdrive_utils

        assert hasattr(gdrive_utils, "__file__")

    def test_download_functions(self):
        """Test main download functions are accessible."""
        from driada.gdrive.download import download_part_of_folder, download_gdrive_data

        assert callable(download_part_of_folder)
        assert callable(download_gdrive_data)
