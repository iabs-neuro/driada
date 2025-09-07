from pydrive2.auth import GoogleAuth


def desktop_auth(secret_path):
    """Authenticate Google Drive access for desktop applications.

    Creates a Google Drive authentication object using OAuth2 credentials
    from a client secrets file. Opens a local web server for the OAuth2
    authentication flow.

    Parameters
    ----------
    secret_path : str
        Path to the client secrets JSON file downloaded from Google Cloud Console.
        This file contains OAuth2 credentials for your application. Must be a valid
        path to an existing JSON file.

    Returns
    -------
    gauth : GoogleAuth
        Authenticated GoogleAuth object that can be used with PyDrive2.

    Raises
    ------
    FileNotFoundError
        If secret_path does not exist.
    json.JSONDecodeError
        If secret_path contains invalid JSON.
    Exception
        If OAuth2 authentication fails (e.g., user cancels, network error).

    Notes
    -----
    This function is intended for desktop applications where a web browser
    can be opened for authentication. It will:
    1. Start a local web server (usually on port 8080)
    2. Open the default browser for Google authentication  
    3. Handle the OAuth2 callback automatically
    
    WARNING: This function modifies GoogleAuth.DEFAULT_SETTINGS globally,
    which affects all GoogleAuth instances in the process.

    The client secrets file can be obtained from:
    https://console.cloud.google.com/apis/credentials
    
    Requires a web browser and ability to bind to local ports.

    Examples
    --------
    >>> # Requires actual client_secrets.json file
    >>> gauth = desktop_auth('./client_secrets.json')  # doctest: +SKIP
    >>> from pydrive2.drive import GoogleDrive  # doctest: +SKIP
    >>> drive = GoogleDrive(gauth)  # doctest: +SKIP    """
    import os
    import json
    
    # Validate secret_path
    if not os.path.exists(secret_path):
        raise FileNotFoundError(f"Client secrets file not found: {secret_path}")
    
    # Validate it's valid JSON
    try:
        with open(secret_path, 'r') as f:
            json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in client secrets file: {e.msg}", e.doc, e.pos)
    
    gauth = GoogleAuth()
    GoogleAuth.DEFAULT_SETTINGS["client_config_file"] = secret_path
    # Create local webserver and auto handles authentication.
    gauth.LocalWebserverAuth()
    return gauth


def google_colab_auth():
    """Authenticate Google Drive access in Google Colab environment.

    Uses Google Colab's built-in authentication mechanism to obtain
    credentials for accessing Google Drive. This function only works
    when running in a Google Colab notebook.

    Returns
    -------
    gauth : GoogleAuth
        Authenticated GoogleAuth object that can be used with PyDrive2.

    Raises
    ------
    ImportError
        If not running in Google Colab (google.colab module not available).
    Exception
        If authentication fails (e.g., user denies permission, network error).

    Notes
    -----
    This function will prompt the user to:
    1. Click a link to authenticate with Google
    2. Copy and paste an authorization code
    
    The authentication persists for the duration of the Colab session.
    Credentials are tied to the Google account used in Colab.

    Examples
    --------
    >>> # In a Google Colab notebook:
    >>> gauth = google_colab_auth()  # doctest: +SKIP
    >>> from pydrive2.drive import GoogleDrive  # doctest: +SKIP
    >>> drive = GoogleDrive(gauth)  # doctest: +SKIP    """
    from google.colab import auth
    from oauth2client.client import GoogleCredentials

    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    return gauth
