Authentication
==============

.. currentmodule:: driada.gdrive

This module handles authentication for Google Drive API access.

Functions
---------

.. autofunction:: driada.gdrive.auth.desktop_auth
.. autofunction:: driada.gdrive.auth.google_colab_auth

Usage Examples
--------------

Desktop Authentication
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.gdrive import desktop_auth
   
   # Authenticate on desktop/laptop  
   auth = desktop_auth('path/to/client_secrets.json')
   
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
   
      # OAuth credentials should be set via environment variables:
      # export GDRIVE_CLIENT_ID="your-client-id"
      # export GDRIVE_CLIENT_SECRET="your-client-secret"

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

- Ensure your OAuth2 credentials are properly configured in Google Cloud Console
- Check that redirect URIs include http://localhost:8080/
- Try using a different browser or clearing browser cache

**Note on Desktop Authentication**:

Desktop authentication (``desktop_auth``) is primarily tested within IABS lab environment 
and may throw unexpected errors in other setups. For most users, we recommend using 
Google Colab authentication or setting up service accounts.
   

Security Best Practices
-----------------------

1. **Never commit credentials** to version control
2. **Use service accounts** for production
3. **Limit API scope** to minimum required:

   .. code-block:: python
   
      # Scopes are configured in your OAuth2 client setup in Google Cloud Console
      from driada.gdrive import desktop_auth
      auth = desktop_auth('path/to/client_secrets.json')

4. **Rotate credentials** regularly
5. **Monitor API usage** in Cloud Console

API Scopes
----------

Available scopes for different access levels:

- **Read-only**: ``drive.readonly``
- **File access**: ``drive.file``
- **Full access**: ``drive``
- **Metadata only**: ``drive.metadata.readonly``

To limit scope, configure it when creating OAuth2 credentials in Google Cloud Console.