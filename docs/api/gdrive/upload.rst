Upload Functions
================

.. currentmodule:: driada.gdrive

This module provides functions for uploading files and data to Google Drive.

Functions
---------

.. autofunction:: driada.gdrive.save_file_to_gdrive

Usage Examples
--------------

Basic File Upload
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.gdrive import desktop_auth, save_file_to_gdrive
   
   # Authenticate
   auth = desktop_auth()
   
   # Upload a single file
   file_id = save_file_to_gdrive(
       auth,
       local_path='results/analysis.h5',
       folder_id='1xyz789...',  # Parent folder ID
       file_name='mouse1_day1_analysis.h5'  # Optional rename
   )
   
   print(f"Uploaded file ID: {file_id}")


Advanced Features
-----------------

Resumable Uploads
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.gdrive import ResumableUpload
   
   # For large files that might fail
   uploader = ResumableUpload(
       auth,
       local_path='huge_dataset.h5',
       folder_id='1xyz789...'
   )
   
   # Start/resume upload
   file_id = uploader.upload()
   
   # Check status
   print(f"Uploaded {uploader.bytes_uploaded}/{uploader.total_bytes} bytes")

Version Control
^^^^^^^^^^^^^^^

.. code-block:: python

   # Keep file versions
   file_id = save_file_to_gdrive(
       auth,
       local_path='analysis.py',
       folder_id='1xyz789...',
       keep_revision=True  # Don't overwrite, create new version
   )
   
   # Upload with version in filename
   from datetime import datetime
   timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
   save_file_to_gdrive(
       auth,
       local_path='results.h5',
       folder_id='1xyz789...',
       file_name=f'results_{timestamp}.h5'
   )

Metadata and Permissions
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Upload with custom metadata
   file_id = save_file_to_gdrive(
       auth,
       local_path='experiment.mat',
       folder_id='1xyz789...',
       description='Mouse M1, Day 5, CA1 recording',
       properties={
           'experiment_id': 'EXP001',
           'mouse_id': 'M1',
           'brain_region': 'CA1'
       }
   )
   
   # Set permissions
   from driada.gdrive import set_permissions
   
   # Make file readable by anyone with link
   set_permissions(
       auth,
       file_id,
       permission_type='anyone',
       role='reader',
       allow_discovery=False
   )
   
   # Share with specific email
   set_permissions(
       auth,
       file_id,
       permission_type='user',
       role='writer',
       email='collaborator@example.com'
   )

Compression
^^^^^^^^^^^

.. code-block:: python

   from driada.gdrive import upload_compressed
   
   # Automatically compress before upload
   file_id = upload_compressed(
       auth,
       local_path='large_folder/',
       folder_id='1xyz789...',
       compression='zip',  # or 'tar.gz'
       compression_level=6
   )

Integration Examples
--------------------

Upload Analysis Results
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import save_exp_to_pickle
   from driada.gdrive import save_file_to_gdrive, create_folder
   
   # Run analysis
   exp = load_experiment('data.mat')
   results = analyze_experiment(exp)
   
   # Save locally
   save_exp_to_pickle(exp, 'processed_exp.pkl')
   
   # Use existing results folder on Drive
   auth = desktop_auth()
   results_folder = '1xyz789...'  # Your Drive folder ID
   
   # Upload results
   save_file_to_gdrive(auth, 'processed_exp.pkl', results_folder)
   save_file_to_gdrive(auth, 'analysis_plots.pdf', results_folder)

Automated Pipeline
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.gdrive import DriveUploadPipeline
   
   # Setup automated upload pipeline
   pipeline = DriveUploadPipeline(
       auth,
       watch_directory='./results',
       drive_folder_id='1xyz789...',
       file_patterns=['*.h5', '*.pkl', '*.pdf'],
       upload_interval=300  # Check every 5 minutes
   )
   
   # Start monitoring
   pipeline.start()
   
   # Your analysis code runs here...
   
   # Stop when done
   pipeline.stop()

Error Handling
--------------

Retry on Failure
^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.gdrive import upload_with_retry
   
   # Automatic retry with exponential backoff
   file_id = upload_with_retry(
       auth,
       local_path='important_results.h5',
       folder_id='1xyz789...',
       max_retries=5,
       initial_delay=1.0
   )

Handling Quota Limits
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.gdrive import RateLimitedUploader
   
   # Respect API quotas
   uploader = RateLimitedUploader(
       auth,
       max_requests_per_minute=60,
       max_bytes_per_minute=100*1024*1024  # 100MB/min
   )
   
   # Upload many files
   for file_path in file_list:
       uploader.upload(file_path, folder_id)

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