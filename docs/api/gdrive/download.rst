Download Functions
==================

.. currentmodule:: driada.gdrive

This module provides functions for downloading files and folders from Google Drive.

Functions
---------

.. autofunction:: driada.gdrive.download_gdrive_data
.. autofunction:: driada.gdrive.download_part_of_folder
.. autofunction:: driada.gdrive.initialize_iabs_router

Usage Examples
--------------

Basic File Download
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.gdrive import desktop_auth, download_gdrive_data
   
   # Authenticate
   auth = desktop_auth()
   
   # Download a single file
   file_url = 'https://drive.google.com/file/d/1abc123.../view'
   download_gdrive_data(auth, file_url, 'local_data/experiment.mat')
   
   # Download from file ID
   file_id = '1abc123...'
   download_gdrive_data(auth, file_id, 'local_data/experiment.mat')

Folder Download
^^^^^^^^^^^^^^^

.. code-block:: python

   # Download entire folder
   folder_url = 'https://drive.google.com/drive/folders/1xyz789...'
   download_gdrive_data(
       auth, 
       folder_url, 
       'local_data/experiment_folder/',
       recursive=True  # Include subfolders
   )

Selective Download
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.gdrive import download_part_of_folder
   
   # Download only specific files from folder
   download_part_of_folder(
       auth,
       folder_id='1xyz789...',
       local_path='local_data/',
       file_pattern='*.mat',  # Only MATLAB files
       max_files=10          # Limit number of files
   )
   
   # Download with filters
   download_part_of_folder(
       auth,
       folder_id='1xyz789...',
       local_path='local_data/',
       file_filter=lambda f: f['size'] < 1e9,  # Files < 1GB
       recursive=True
   )


IABS Router Setup
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.gdrive import initialize_iabs_router
   
   # Setup router for IABS data
   router = initialize_iabs_router(
       auth,
       data_root='IABS_data/',
       cache_metadata=True
   )
   
   # Use router to access data
   experiment = router.get_experiment('mouse1/day1')
   calcium_data = router.get_calcium_data('mouse1/day1/calcium.h5')

Advanced Features
-----------------

Resumable Downloads
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Resume interrupted downloads
   from driada.gdrive import ResumableDownload
   
   downloader = ResumableDownload(
       auth,
       file_id='1abc123...',
       local_path='large_file.h5'
   )
   
   # Start/resume download
   downloader.download()
   
   # Check if complete
   if downloader.is_complete():
       print("Download finished!")

Checksum Verification
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Verify file integrity
   download_gdrive_data(
       auth,
       file_id,
       local_path,
       verify_checksum=True  # Verify MD5 after download
   )


Error Handling
--------------

Retry Logic
^^^^^^^^^^^

.. code-block:: python

   from driada.gdrive import download_with_retry
   
   # Automatic retry on failure
   download_with_retry(
       auth,
       file_id,
       local_path,
       max_retries=5,
       backoff_factor=2.0,  # Exponential backoff
       retry_on=[500, 502, 503, 504]  # HTTP errors to retry
   )

Handling Quota Errors
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.gdrive import handle_quota_error
   
   @handle_quota_error(wait_time=60)  # Wait 60s on quota error
   def download_many_files(auth, file_list):
       for file_id, path in file_list:
           download_gdrive_data(auth, file_id, path)

Performance Tips
----------------

1. **Batch Operations**: Use batch_download for multiple files
2. **Parallel Downloads**: Enable parallel mode for large datasets
3. **Chunk Size**: Adjust chunk_size based on connection speed
4. **Caching**: Cache metadata to avoid repeated API calls

.. code-block:: python

   # Optimized for large datasets
   from driada.gdrive import OptimizedDownloader
   
   downloader = OptimizedDownloader(
       auth,
       cache_dir='.gdrive_cache',
       parallel_downloads=4,
       chunk_size=10*1024*1024  # 10MB chunks
   )
   
   # Download entire dataset
   downloader.download_dataset(
       dataset_folder_id,
       local_root='./data',
       skip_existing=True
   )

Common Use Cases
----------------

**Downloading Experimental Data**:

.. code-block:: python

   # Standard workflow for DRIADA data
   auth = desktop_auth()
   
   # Download experiment folder
   exp_folder = 'https://drive.google.com/drive/folders/...'
   download_gdrive_data(
       auth,
       exp_folder,
       'experiments/mouse1_day1/',
       recursive=True
   )
   
   # Load the experiment
   from driada.experiment import load_experiment
   exp = load_experiment('experiments/mouse1_day1/data.mat')