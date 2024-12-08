from pydrive2.auth import GoogleAuth


def desktop_auth(secret_path):
    gauth = GoogleAuth()
    GoogleAuth.DEFAULT_SETTINGS['client_config_file'] = secret_path
    # Create local webserver and auto handles authentication.
    gauth.LocalWebserverAuth()
    return gauth


def google_colab_auth():
    from google.colab import auth
    from oauth2client.client import GoogleCredentials
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    return gauth