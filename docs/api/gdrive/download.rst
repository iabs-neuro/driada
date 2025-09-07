Download Functions
==================

.. currentmodule:: driada.gdrive

This module provides functions for downloading files and folders from Google Drive.

Functions
---------

.. autofunction:: driada.gdrive.download.download_gdrive_data
.. autofunction:: driada.gdrive.download_part_of_folder
.. autofunction:: driada.gdrive.initialize_iabs_router

Usage Examples
--------------

Basic File Download
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.gdrive import desktop_auth, download_gdrive_data
   
   # Authenticate
   auth = desktop_auth('path/to/client_secrets.json')
   
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
   from driada.gdrive import desktop_auth, download_gdrive_data
   
   auth = desktop_auth('path/to/client_secrets.json')
   folder_url = 'https://drive.google.com/drive/folders/1xyz789...'
   success, log = download_gdrive_data(
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
       output='local_data/',
       folder='https://drive.google.com/drive/folders/1xyz789...',
       key='experiment',  # Files containing 'experiment'
       extensions=['.mat', '.npz'],  # Only these file types
       via_pydrive=True,
       gauth=auth,
       maxfiles=10
   )
   


IABS Router Setup
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.gdrive import initialize_iabs_router
   
   # IMPORTANT: Requires config.py setup first:
   # 1. Copy src/driada/gdrive/config_template.py to config.py
   # 2. Set IABS_ROUTER_URL to your Google Sheets export URL
   # 3. Add config.py to .gitignore
   
   # Download and initialize IABS data router
   router_df, data_pieces = initialize_iabs_router(root='/content')
   
   # router_df contains experiment metadata with Drive links
   # data_pieces lists downloadable data columns



Error Handling
--------------

Basic error handling example:

.. code-block:: python

   import time
   
   def download_with_retry(auth, file_id, local_path, max_retries=3):
       for attempt in range(max_retries):
           try:
               download_gdrive_data(auth, file_id, local_path)
               return True
           except Exception as e:
               print(f"Attempt {attempt + 1} failed: {e}")
               if attempt < max_retries - 1:
                   time.sleep(2 ** attempt)  # Exponential backoff
       return False

Common Use Cases
----------------

**Downloading Experimental Data**:

.. code-block:: python

   # Standard workflow for DRIADA data
   from driada.gdrive import desktop_auth, download_gdrive_data
   
   auth = desktop_auth('path/to/client_secrets.json')
   
   # Download experiment folder
   exp_folder = 'https://drive.google.com/drive/folders/...'
   success, log = download_gdrive_data(
       auth,
       exp_folder,
       'experiments/mouse1_day1/',
       recursive=True
   )
   
   # Load the experiment (requires exp_params)
   from driada.experiment import load_experiment
   exp_params = {'mouse': 'mouse1', 'date': 'day1', 'session': 1}
   exp = load_experiment('local', exp_params, data_path='experiments/mouse1_day1/data.mat')