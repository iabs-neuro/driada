import os.path
from datetime import datetime
import pytz
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

from .gdrive_utils import *
from .download import retrieve_relevant_ids

def get_datetime():
    tz = pytz.timezone('Europe/Moscow')
    now = datetime.now(tz)

    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    return dt_string


def save_file_to_gdrive(data_router,
                        expname,
                        path_to_file,
                        link=None,
                        destination=None,
                        force_rewriting=False,
                        gauth=None):

    drive = GoogleDrive(gauth)

    if link is None:
        row = data_router[data_router['Эксперимент'] == expname]
        links = dict(zip(row.columns, row.values[0]))
        if destination not in links:
            raise ValueError(f'Wrong folder name: {destination}')
        link = links[destination]

    fid = id_from_link(link)

    dataname = os.path.basename(path_to_file)
    if force_rewriting:
        return_code, rel = retrieve_relevant_ids(link,
                                                 dataname,
                                                 whitelist=[],
                                                 extensions=['.npz', '.csv', 'xlsx'])
        if len(rel) != 0:
            if len(rel) > 1:
                print('More than one relevant file found, which is suspicious. Consider manual check')

            existing_file_data = rel[0]
            dat_id, dat_name = existing_file_data
            f = drive.CreateFile({'id': dat_id})

        else:
            print(f'Data for {dataname} not found in folder, nothing to rewrite')
            f = drive.CreateFile({'title': dataname, "parents": [{"kind": "drive#fileLink", "id": fid}]})

    else:
        date_time = get_datetime()
        ext = os.path.splitext(path_to_file)[1]
        dataname = dataname[:-len(ext)] + date_time + ext
        f = drive.CreateFile({'title': dataname, "parents": [{"kind": "drive#fileLink", "id": fid}]})

    f.SetContentFile(path_to_file)
    f.Upload()


def desktop_auth(secret_path):
    gauth = GoogleAuth()
    GoogleAuth.DEFAULT_SETTINGS['client_config_file'] = secret_path
    # Create local webserver and auto handles authentication.
    gauth.LocalWebserverAuth()
    return gauth


# TODO: test colab usage
def google_colab_auth():
    from google.colab import auth
    from oauth2client.client import GoogleCredentials
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
