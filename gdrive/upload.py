from datetime import datetime, date
import pytz
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

from ._gd_utils import *


def get_datetime():
    tz = pytz.timezone('Europe/Moscow')
    now = datetime.now(tz)

    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    return dt_string

def save_file_to_gdrive(data_router, expname, path_to_file = None, link = None,
                        destination = None, postfix = None, force_rewriting = False):

    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)

    if link is None:
        row = data_router[data_router['Эксперимент'] == expname]
        links = dict(zip(row.columns, row.values[0]))
        if destination not in links:
            raise ValueError(f'Wrong folder name: {destination}')
        link = links[destination]

    fid = id_from_link(link)

    if postfix is None:
        if destination == 'Aligned data':
            postfix = ' syn data'

    if path_to_file == None:
        path_to_file = join(expname, destination, expname + postfix + '.npz')

    if force_rewriting:
        dataname = expname + postfix + '.npz'
    else:
        date_time = get_datetime()
        dataname = expname + '_' + date_time + postfix + '.npz'

    f = drive.CreateFile({'title': dataname, "parents": [{"kind": "drive#fileLink", "id": fid}]})
    f.SetContentFile(path_to_file)
    f.Upload()