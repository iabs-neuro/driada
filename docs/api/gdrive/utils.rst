Utility Functions
=================

.. currentmodule:: driada.gdrive

This module provides utility functions for Google Drive operations.

Classes
-------

.. autoclass:: driada.gdrive.gdrive_utils.GoogleDriveFile
   :members:
   :undoc-members:

Functions
---------

.. autofunction:: driada.gdrive.parse_google_drive_file
.. autofunction:: driada.gdrive.id_from_link
.. autofunction:: driada.gdrive.download_and_parse_google_drive_link

Usage Examples
--------------

File ID Extraction
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.gdrive import id_from_link
   
   # Extract ID from various URL formats
   urls = [
       'https://drive.google.com/file/d/1abc123.../view',
       'https://drive.google.com/open?id=1abc123...',
       'https://drive.google.com/drive/folders/1xyz789...',
       'https://docs.google.com/document/d/1doc456.../edit'
   ]
   
   for url in urls:
       file_id = id_from_link(url)
       print(f"ID: {file_id}")


Download and Parse Links
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.gdrive import download_and_parse_google_drive_link
   
   # Download and parse a Google Drive folder page
   folder_url = 'https://drive.google.com/drive/folders/1xyz789...'
   
   # Get folder info and its contents
   success, folder_file = download_and_parse_google_drive_link(
       folder_url,
       quiet=True,  # Suppress output
       name_part="exp"  # Filter by name
   )
   
   if success and folder_file:
       print(f"Folder: {folder_file['name']}")
       print(f"Contains {len(folder_file.get('children', []))} matching files")
   
   # Process children
   for child in folder_file.get('children', []):
       print(f"- {child['name']} ({child['id']})")
       print(f"  Type: {child['type']}")
       

GoogleDriveFile Class
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.gdrive import GoogleDriveFile
   
   # GoogleDriveFile is a simple data class
   gdfile = GoogleDriveFile(
       id='1abc123...',
       name='experiment_data.mat',
       type='file'
   )
   
   # Access attributes
   print(f"File: {gdfile.name}")
   print(f"ID: {gdfile.id}")
   print(f"Type: {gdfile.type}")
   
   # Use with download functions
   from driada.gdrive import download_gdrive_data
   download_gdrive_data(auth, gdfile, 'local_path/')


Parse Google Drive HTML
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.gdrive import parse_google_drive_file
   import requests
   
   # This is a low-level function used internally
   # parse_google_drive_file requires HTML content
   folder_url = 'https://drive.google.com/drive/folders/1xyz789...'
   
   # Get the raw HTML content first
   response = requests.get(folder_url)
   
   # Parse the content
   folder_file, children = parse_google_drive_file(
       folder_url,
       response.text,
       use_cookies=True
   )


Integration with Other Functions
--------------------------------

These utilities work seamlessly with download functions:

.. code-block:: python

   from driada.gdrive import (
       id_from_link, 
       download_gdrive_data,
       desktop_auth
   )
   
   # Authenticate
   auth = desktop_auth('path/to/client_secrets.json')
   
   # Extract ID from URL
   url = 'https://drive.google.com/file/d/1abc123.../view?usp=sharing'
   file_id = id_from_link(url)
   
   # Download using extracted ID
   download_gdrive_data(auth, file_id, 'local_data.mat')