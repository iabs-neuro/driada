from os.path import join

from pydrive2.drive import GoogleDrive
import warnings
import wget
import gdown
import pandas as pd
import shutil
from pathlib import Path

from .gdrive_utils import *
from ..utils.output import *


def retrieve_relevant_ids(folder,
                          name_part,
                          prohibited_name_part='',
                          whitelist=[],
                          extensions=['.csv', '.xlsx']):

    return_code = True
    folder_page = client.get(folder)

    if folder_page.status_code != 200:
        return False, None

    gdrive_file, id_name_type_iter = parse_google_drive_file(
        folder,
        folder_page.text,
    )

    relevant = []
    if len(list(id_name_type_iter)) > MAX_NUMBER_FILES:
        raise MemoryError(
            f'The folder {folder} has {len(list(id_name_type_iter))} elements while max allowed number of files is {MAX_NUMBER_FILES}')

    for child_id, child_name, child_type in id_name_type_iter:
        if child_type != folder_type:
            if child_name in whitelist:
                relevant.append((child_id, child_name))
            elif name_part in child_name:
                if len(extensions) != 0 and Path(child_name).suffix in extensions or len(extensions) == 0:
                    if (prohibited_name_part is not None) and (prohibited_name_part not in child_name) or prohibited_name_part is None:
                        relevant.append((child_id, child_name))
            else:
                pass

        else:
            return_code, rel_sublist = retrieve_relevant_ids(folders_url + child_id,
                                                             name_part,
                                                             prohibited_name_part=prohibited_name_part,
                                                             whitelist=whitelist,
                                                             extensions=extensions)
            if not return_code:
                print(f'recursive search broke on folder {child_id}')
                break
            relevant.extend(rel_sublist)

    return return_code, relevant


def download_part_of_folder(
        output,  # path for downloaded data
        folder,  # share link to google drive folder
        key='',  # part of filename to search for
        antikey=None, # part of name to suppress
        whitelist=[],  # list of filenames to be downloaded regardless of their names
        extensions=['.csv', '.xlsx', '.npz'],  # allowed file extensions
        via_pydrive=False,  # pydrive requires authorization, but can download a big number of files,
        gauth=None,
        maxfiles=None):

    os.makedirs(output, exist_ok=True)

    with Capturing() as load_log:
        if via_pydrive:
            if gauth is None:
                raise ValueError('To use pydrive, you need to authenticate using one of the functions'
                                 ' in driada.gdrive.auth')
            drive = GoogleDrive(gauth)

            rel = []
            fid = id_from_link(folder)
            file_list = drive.ListFile({'q': f"'{fid}' in parents and trashed=false"}).GetList()
            if maxfiles is not None:
                file_list = file_list[:maxfiles]

            for f in file_list:
                if key in f['title']:
                    # print('title: %s, id: %s' % (f['title'],f['id']))
                    f.GetContentFile(join(output, f['title']))
                    rel.append((f['id'], f['title']))

            return_code = True

        else:
            return_code, rel = retrieve_relevant_ids(folder,
                                                     key,
                                                     prohibited_name_part=antikey,
                                                     whitelist=whitelist,
                                                     extensions=extensions)

            if return_code:
                for i, pair in enumerate(rel):
                    idx, name = rel[i]
                    gdown.download(id=idx, output=os.path.join(output, name))

            else:
                raise FileNotFoundError('Error in downloading procedure!')

        return return_code, rel, load_log


def download_gdrive_data(data_router,
                         expname,
                         whitelist=['Timing.xlsx'],
                         via_pydrive=False,
                         data_pieces=None,
                         tdir='DRIADA data',
                         gauth=None):

    with Capturing() as load_log:
        print('-------------------------------------------------------------')
        print(f'Extracting data for {expname} from Google Drive')
        print('-------------------------------------------------------------')

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)

            success = False
            available_exp = data_router['Эксперимент'].values
            if expname not in available_exp:
                print(f'{expname} not found in available experiments: {available_exp}')
                return success, load_log

            row = data_router[data_router['Эксперимент'] == expname]
            links = dict(zip(row.columns, row.values[0]))

            os.makedirs(join(tdir, expname), exist_ok=True)
            if data_pieces is None:
                data_pieces = [d for d in list(data_router.columns.values) if d not in ['Эксперимент', 'Краткое описание', 'Video', 'Aligned data', 'Computation results']]

            for key in data_pieces:
                if 'http' in links[key]:
                    print(f'Loading data: {key}...')
                    ddir = join(tdir, expname, key)
                    os.makedirs(ddir, exist_ok=True)
                    # gdown.download_folder(url = links[key], output = dir, quiet=False)
                    return_code, rel, folder_log = download_part_of_folder(ddir,
                                                                           links[key],
                                                                           key=expname,
                                                                           whitelist=whitelist,
                                                                           via_pydrive=via_pydrive,
                                                                           gauth=gauth)

                    load_log.extend(folder_log)

                    if len(rel) == 0:
                        os.rmdir(ddir)
                        print('No relevant data found at: ', links[key])

                    else:
                        loaded_names = [r[1] for r in rel]
                        for n in loaded_names:
                            print(n)
                        success = True

                    print('--------------------------')

            return success, load_log


def initialize_iabs_router(root='\\content'):
    router_name = 'IABS data router.xlsx'
    router_path = join(root, router_name)
    os.makedirs(root, exist_ok=True)
    if router_name in os.listdir(root):
        os.remove(router_path)

    global_data_table_url = 'https://docs.google.com/spreadsheets/d/130DDFAoAbmm0jcKLBF6xsWsQLDr2Zsj4cPuOYivXoM8/export?format=xlsx'
    wget.download(global_data_table_url, out=router_path)

    data_router = pd.read_excel(router_path)
    #data_router.fillna(method='ffill', inplace=True)
    data_router = data_router.replace("", None).ffill()

    data_pieces = [d for d in list(data_router.columns.values) if d not in ['Эксперимент',
                                                                            'Краткое описание',
                                                                            'Video',
                                                                            'Aligned data',
                                                                            'Computation results']
                   ]
    return data_router, data_pieces
