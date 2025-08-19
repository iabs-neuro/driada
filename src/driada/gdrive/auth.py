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
        This file contains OAuth2 credentials for your application.

    Returns
    -------
    gauth : GoogleAuth
        Authenticated GoogleAuth object that can be used with PyDrive2.

    Notes
    -----
    This function is intended for desktop applications where a web browser
    can be opened for authentication. It will:
    1. Start a local web server
    2. Open the default browser for Google authentication
    3. Handle the OAuth2 callback automatically

    The client secrets file can be obtained from:
    https://console.cloud.google.com/apis/credentials

    Examples
    --------
    >>> gauth = desktop_auth('./client_secrets.json')
    >>> drive = GoogleDrive(gauth)
    """
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

    Notes
    -----
    This function will prompt the user to:
    1. Click a link to authenticate with Google
    2. Copy and paste an authorization code
    
    The authentication persists for the duration of the Colab session.

    Examples
    --------
    >>> # In a Google Colab notebook:
    >>> gauth = google_colab_auth()
    >>> drive = GoogleDrive(gauth)
    >>> # Now you can use drive to upload/download files
    """
    from google.colab import auth
    from oauth2client.client import GoogleCredentials

    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    return gauth
