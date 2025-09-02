Google Drive Integration Module
===============================

.. automodule:: driada.gdrive
   :no-members:
   :noindex:

Utilities for uploading and downloading data to/from Google Drive,
particularly useful for sharing experimental data and results.

Module Components
-----------------

.. toctree::
   :maxdepth: 1

   gdrive/auth
   gdrive/download
   gdrive/upload
   gdrive/utils

Quick Links
-----------

**Authentication**
   * :func:`~driada.gdrive.desktop_auth` - Authenticate on desktop
   * :func:`~driada.gdrive.google_colab_auth` - Authenticate in Colab
   * :doc:`gdrive/auth` - Authentication utilities

**Download Functions**
   * :func:`~driada.gdrive.download_gdrive_data` - Download files/folders
   * :func:`~driada.gdrive.download_part_of_folder` - Selective download
   * :func:`~driada.gdrive.initialize_iabs_router` - Setup IABS router
   * :doc:`gdrive/download` - All download utilities

**Upload Functions**
   * :func:`~driada.gdrive.save_file_to_gdrive` - Upload files
   * :doc:`gdrive/upload` - Upload utilities

**Utilities**
   * :class:`~driada.gdrive.GoogleDriveFile` - File wrapper class
   * :func:`~driada.gdrive.parse_google_drive_file` - Parse Drive files
   * :func:`~driada.gdrive.id_from_link` - Extract ID from URL
   * :doc:`gdrive/utils` - General utilities

Usage Example
-------------

.. code-block:: python

   from driada.gdrive import (
       desktop_auth, 
       download_gdrive_data,
       save_file_to_gdrive
   )
   
   # Authenticate
   auth = desktop_auth()
   
   # Download data
   download_gdrive_data(
       auth,
       'https://drive.google.com/file/d/...',
       'local_path/'
   )
   
   # Upload results
   save_file_to_gdrive(
       auth,
       'results.h5',
       folder_id='...'
   )