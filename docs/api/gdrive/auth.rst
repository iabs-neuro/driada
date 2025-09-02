Authentication
==============

.. currentmodule:: driada.gdrive

This module handles authentication for Google Drive API access.

Functions
---------

.. autofunction:: driada.gdrive.desktop_auth
.. autofunction:: driada.gdrive.google_colab_auth

Usage Examples
--------------

Desktop Authentication
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.gdrive import desktop_auth
   
   # Authenticate on desktop/laptop
   auth = desktop_auth()
   
   # This will:
   # 1. Open a browser window for Google login
   # 2. Request Drive API permissions
   # 3. Save credentials locally for reuse
   
   # Use auth object for subsequent operations
   from driada.gdrive import download_gdrive_data
   download_gdrive_data(auth, file_url, local_path)

Google Colab Authentication
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.gdrive import google_colab_auth
   
   # In Google Colab environment
   auth = google_colab_auth()
   
   # This will:
   # 1. Use Colab's built-in auth
   # 2. Mount Google Drive if needed
   # 3. Return authenticated service


Configuration
-------------

First-time Setup
^^^^^^^^^^^^^^^^

1. **Enable Google Drive API**:
   
   - Go to `Google Cloud Console <https://console.cloud.google.com/>`_
   - Create a new project or select existing
   - Enable Google Drive API
   - Create credentials (OAuth2 or Service Account)

2. **Download Credentials**:
   
   - For OAuth2: Download client configuration
   - For Service Account: Download JSON key file

3. **Configure DRIADA**:

   .. code-block:: python
   
      # Create config file (first time only)
      from driada.gdrive import create_config
      
      create_config(
          client_id='your-client-id',
          client_secret='your-client-secret',
          redirect_uri='http://localhost:8080'
      )

Environment Variables
^^^^^^^^^^^^^^^^^^^^^

Set these for automatic configuration:

.. code-block:: bash

   # OAuth2 credentials
   export GDRIVE_CLIENT_ID="your-client-id"
   export GDRIVE_CLIENT_SECRET="your-client-secret"
   
   # Service account
   export GDRIVE_SERVICE_ACCOUNT_FILE="/path/to/key.json"
   
   # Token storage
   export GDRIVE_TOKEN_DIR="~/.driada/tokens"

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

**"Access blocked" error**:

.. code-block:: python

   # Use desktop auth with specific port
   auth = desktop_auth(port=8090)  # Try different port

**"Quota exceeded" error**:

.. code-block:: python

   # Add retry logic
   from driada.gdrive import auth_with_retry
   
   auth = auth_with_retry(
       max_retries=3,
       backoff_factor=2.0
   )

**Token expired**:

.. code-block:: python

   # Auto-refresh on operations
   from driada.gdrive import auto_refresh_decorator
   
   @auto_refresh_decorator
   def my_operation(auth):
       # Your code here
       pass

Security Best Practices
-----------------------

1. **Never commit credentials** to version control
2. **Use service accounts** for production
3. **Limit API scope** to minimum required:

   .. code-block:: python
   
      auth = desktop_auth(
          scopes=['https://www.googleapis.com/auth/drive.readonly']
      )

4. **Rotate credentials** regularly
5. **Monitor API usage** in Cloud Console

API Scopes
----------

Available scopes for different access levels:

- **Read-only**: ``drive.readonly``
- **File access**: ``drive.file``
- **Full access**: ``drive``
- **Metadata only**: ``drive.metadata.readonly``

Example with limited scope:

.. code-block:: python

   # Only request what you need
   auth = desktop_auth(
       scopes=[
           'https://www.googleapis.com/auth/drive.readonly',
           'https://www.googleapis.com/auth/drive.metadata'
       ]
   )