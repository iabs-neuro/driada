"""
Google Drive integration for DRIADA.

This module provides utilities for uploading and downloading data to/from Google Drive,
particularly useful for sharing experimental data and results.
"""

from .auth import desktop_auth, google_colab_auth
from .download import (
    download_gdrive_data,
    download_part_of_folder,
    initialize_iabs_router,
    retrieve_relevant_ids,
)
from .gdrive_utils import (
    GoogleDriveFile,
    download_and_parse_google_drive_link,
    id_from_link,
    parse_google_drive_file,
)
from .upload import save_file_to_gdrive

__all__ = [
    # Authentication
    "desktop_auth",
    "google_colab_auth",
    # Download functions
    "download_gdrive_data",
    "download_part_of_folder",
    "retrieve_relevant_ids",
    "initialize_iabs_router",
    # Upload functions
    "save_file_to_gdrive",
    # Utilities
    "GoogleDriveFile",
    "parse_google_drive_file",
    "download_and_parse_google_drive_link",
    "id_from_link",
]
