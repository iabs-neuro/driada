Upload Functions
================

.. currentmodule:: driada.gdrive

This module provides functions for uploading files and data to Google Drive.

Functions
---------

.. autofunction:: driada.gdrive.upload.save_file_to_gdrive

Usage Examples
--------------

Basic File Upload
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.gdrive import desktop_auth, save_file_to_gdrive
   
   # Authenticate
   auth = desktop_auth('path/to/client_secrets.json')
   
   # Upload a single file
   file_id = save_file_to_gdrive(
       auth,
       local_path='results/analysis.h5',
       folder_id='1xyz789...',  # Parent folder ID
       file_name='mouse1_day1_analysis.h5'  # Optional rename
   )
   
   print(f"Uploaded file ID: {file_id}")


Upload with Version Control
---------------------------

.. code-block:: python

   # Upload with version in filename
   from driada.gdrive import desktop_auth, save_file_to_gdrive
   from datetime import datetime
   
   auth = desktop_auth('path/to/client_secrets.json')
   timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
   file_id = save_file_to_gdrive(
       auth,
       local_path='results.h5',
       folder_id='1xyz789...',
       file_name=f'results_{timestamp}.h5'
   )

Integration Examples
--------------------

Upload Analysis Results
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import save_exp_to_pickle, load_demo_experiment
   from driada.gdrive import save_file_to_gdrive
   
   # Load demo experiment
   exp = load_demo_experiment()
   # Run some analysis (example)
   results = {'neurons': exp.n_cells, 'frames': exp.n_frames}
   
   # Save locally
   save_exp_to_pickle(exp, 'processed_exp.pkl')
   
   # Use existing results folder on Drive
   auth = desktop_auth('path/to/client_secrets.json')
   results_folder = '1xyz789...'  # Your Drive folder ID
   
   # Upload results
   save_file_to_gdrive(auth, 'processed_exp.pkl', results_folder)
   save_file_to_gdrive(auth, 'analysis_plots.pdf', results_folder)

Error Handling
--------------

Simple Retry Pattern
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import time
   
   # Simple retry with exponential backoff
   def upload_with_retry(auth, local_path, folder_id, max_retries=3):
       for attempt in range(max_retries):
           try:
               file_id = save_file_to_gdrive(auth, local_path, folder_id)
               return file_id
           except Exception as e:
               print(f"Attempt {attempt + 1} failed: {e}")
               if attempt < max_retries - 1:
                   time.sleep(2 ** attempt)  # Exponential backoff
       raise Exception(f"Failed to upload after {max_retries} attempts")

Best Practices
--------------

1. **Organize with Folders**: Create logical folder structure
2. **Use Descriptive Names**: Include metadata in filenames
3. **Version Control**: Keep old versions or use timestamps
4. **Compression**: Compress large files/folders before upload
5. **Error Handling**: Always implement retry logic
6. **Batch Operations**: Group uploads for efficiency

.. code-block:: python

   # Example: Well-organized upload
   import os
   from datetime import datetime
   from driada.gdrive import desktop_auth, save_file_to_gdrive
   
   # Authenticate
   auth = desktop_auth('path/to/client_secrets.json')
   
   # Create timestamped filename
   timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
   experiment_id = 'EXP001'
   filename = f'{experiment_id}_results_{timestamp}.h5'
   
   # Upload with descriptive name
   file_id = save_file_to_gdrive(
       auth,
       local_path='results.h5', 
       folder_id='1xyz789...',
       file_name=filename
   )
   
   # Use existing organized structure
   project_folder = '1abc123...'  # Your project folder ID
   
   # Use existing date folder
   date_str = datetime.now().strftime('%Y-%m-%d')
   daily_folder = '1def456...'  # Your daily folder ID
   
   # Upload with metadata
   for exp_file in os.listdir('experiments/'):
       if exp_file.endswith('.mat'):
           # Extract metadata from filename
           parts = exp_file.split('_')
           mouse_id = parts[0]
           session = parts[1]
           
           # Upload with descriptive name
           file_id = save_file_to_gdrive(
               auth,
               f'experiments/{exp_file}',
               daily_folder,
               file_name=f'{mouse_id}_{session}_{date_str}.mat',
               description=f'Recording: {mouse_id}, Session: {session}'
           )