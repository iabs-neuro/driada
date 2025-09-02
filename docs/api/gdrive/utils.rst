Utility Functions
=================

.. currentmodule:: driada.gdrive

This module provides utility functions for Google Drive operations.

Classes
-------

.. autoclass:: driada.gdrive.GoogleDriveFile
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
   
   # Also works with just IDs
   file_id = id_from_link('1abc123...')  # Returns as-is


GoogleDriveFile Wrapper
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.gdrive import parse_google_drive_file
   
   # Parse file URL into structured object
   gdfile = parse_google_drive_file(
       'https://drive.google.com/file/d/1abc123.../view'
   )
   
   print(f"File ID: {gdfile.id}")
   print(f"Type: {gdfile.type}")  # 'file' or 'folder'
   print(f"URL: {gdfile.url}")
   
   # Use with download functions
   from driada.gdrive import download_gdrive_data
   download_gdrive_data(auth, gdfile, 'local_path/')


Advanced Utilities
------------------

Batch Operations
^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.gdrive import batch_get_metadata
   
   # Get metadata for multiple files efficiently
   file_ids = ['1abc...', '2def...', '3ghi...']
   metadata_list = batch_get_metadata(auth, file_ids)
   
   for metadata in metadata_list:
       print(f"{metadata['name']}: {metadata['size']} bytes")

Path Resolution
^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.gdrive import resolve_path
   
   # Convert path to file ID
   file_id = resolve_path(
       auth,
       '/MyProject/experiments/mouse1/day1/data.mat'
   )
   
   # Get full path from file ID
   from driada.gdrive import get_full_path
   path = get_full_path(auth, file_id)
   print(f"Full path: {path}")

Permission Management
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.gdrive import check_permissions, copy_permissions
   
   # Check if user has access
   has_access = check_permissions(
       auth,
       file_id,
       required_role='writer'  # or 'reader', 'owner'
   )
   
   # Copy permissions from one file to another
   copy_permissions(
       auth,
       source_file_id,
       target_file_id
   )

Cleanup Utilities
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.gdrive import find_duplicates, cleanup_old_files
   
   # Find duplicate files in folder
   duplicates = find_duplicates(auth, folder_id)
   for name, file_ids in duplicates.items():
       print(f"Duplicate '{name}': {len(file_ids)} copies")
   
   # Clean up old files
   deleted_count = cleanup_old_files(
       auth,
       folder_id,
       older_than_days=90,
       dry_run=True  # Set False to actually delete
   )

Export Functions
^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.gdrive import export_folder_structure
   
   # Export folder structure to JSON
   structure = export_folder_structure(auth, root_folder_id)
   
   import json
   with open('drive_structure.json', 'w') as f:
       json.dump(structure, f, indent=2)
   
   # Generate markdown tree
   from driada.gdrive import generate_markdown_tree
   
   tree_md = generate_markdown_tree(auth, root_folder_id)
   with open('drive_contents.md', 'w') as f:
       f.write(tree_md)

Helper Functions
----------------

MIME Type Detection
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.gdrive import get_mime_type
   
   # Detect MIME type from filename
   mime_type = get_mime_type('experiment.mat')
   print(mime_type)  # 'application/matlab'
   
   # Common MIME types
   MIME_TYPES = {
       '.mat': 'application/matlab',
       '.h5': 'application/x-hdf',
       '.pkl': 'application/octet-stream',
       '.csv': 'text/csv',
       '.pdf': 'application/pdf'
   }

URL Generation
^^^^^^^^^^^^^^

.. code-block:: python

   from driada.gdrive import generate_shareable_link
   
   # Create shareable link
   link = generate_shareable_link(auth, file_id)
   print(f"Share this link: {link}")
   
   # Create direct download link
   from driada.gdrive import generate_download_link
   download_url = generate_download_link(file_id)

Best Practices
--------------

1. **Cache Metadata**: Avoid repeated API calls
2. **Batch Operations**: Use batch functions for multiple files
3. **Error Handling**: Always handle API errors gracefully
4. **Rate Limiting**: Respect Google Drive API quotas

.. code-block:: python

   # Example: Efficient metadata caching
   from functools import lru_cache
   
   @lru_cache(maxsize=1000)
   def get_cached_metadata(auth, file_id):
       return get_file_metadata(auth, file_id)
   
   # Example: Safe file operations
   def safe_download(auth, file_id, local_path):
       try:
           # Check permissions first
           if not check_permissions(auth, file_id, 'reader'):
               print("No read access to file")
               return False
           
           # Get metadata
           metadata = get_file_metadata(auth, file_id)
           
           # Check size
           if metadata['size'] > 10e9:  # 10GB
               print("File too large, consider partial download")
               return False
           
           # Download
           download_gdrive_data(auth, file_id, local_path)
           return True
           
       except Exception as e:
           print(f"Download failed: {e}")
           return False