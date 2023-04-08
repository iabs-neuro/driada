from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import warnings
import wget
import pandas as pd

from ._gd_utils import *
from ..utils.output import *

def retrieve_relevant_ids(folder, name_part, whitelist=[], extentions=['.csv', '.xlsx']):
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
            if (name_part in child_name or child_name in whitelist) and os.path.splitext(child_name)[1] in extentions:
                relevant.append((child_id, child_name))

        else:
            return_code, rel_sublist = retrieve_relevant_ids(folders_url + child_id, name_part)
            if not return_code:
                print('recursive search broke on folder {child_id}')
                break
            relevant.extend(rel_sublist)

    return True, relevant


def download_part_of_folder(
        output,  # path for downloaded data
        folder,  # share link to google drive folder
        key='',  # part of filename to search for
        whitelist=[],  # list of filenames to be downloaded regardless of their names
        extentions=['.csv', '.xlsx', '.npz'],  # allowed file extentions
        via_pydrive=False  # pydrive requires authorization, but can download a big number of files
):
    with Capturing() as load_log:
        if via_pydrive:
            auth.authenticate_user()
            gauth = GoogleAuth()
            gauth.credentials = GoogleCredentials.get_application_default()
            drive = GoogleDrive(gauth)

            rel = []
            fid = id_from_link(folder)
            file_list = drive.ListFile({'q': f"'{fid}' in parents and trashed=false"}).GetList()
            for f in file_list:
                if key in f['title']:
                    # print('title: %s, id: %s' % (f['title'],f['id']))
                    f.GetContentFile(f['title'])
                    rel.append((f['id'], f['title']))

            relfiles = [f for f in os.listdir('/content') if os.path.isfile(join('/content', f)) and key in f]
            for f in relfiles:
                os.rename(join('/content', f), os.path.join('/content', output, f))
            return_code = True

        else:
            return_code, rel = retrieve_relevant_ids(folder,
                                                     key,
                                                     whitelist=whitelist,
                                                     extentions=extentions)

            if return_code:
                for i, pair in enumerate(rel):
                    id, name = rel[i]
                    gdown.download(id=id, output=os.path.join(output, name))

            else:
                raise FileNotFoundError('Error in downloading procedure!')

        return return_code, rel, load_log


def download_gdrive_data(data_router, expname, aligned_only=False, whitelist=['Timing.xlsx'], via_pydrive=False):
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

            os.makedirs(expname, exist_ok=True)

            if aligned_only:
                key = 'Aligned data'
                dir = os.path.join(expname, key)
                os.makedirs(dir, exist_ok=True)
                return_code, rel, folder_log = download_part_of_folder(dir,
                                                                       links[key],
                                                                       key=expname,
                                                                       via_pydrive=via_pydrive
                                                                       )

                load_log.extend(folder_log)

                if len(rel) == 0:
                    print(
                        f'No relevant aligned data found for session {expname}, consider using "aligned_only = False"')
                else:
                    success = True
                return success, load_log

            data_pieces = [d for d in list(data_router.columns.values) if d not in ['Эксперимент', 'Краткое описание', 'Video', 'Aligned data', 'Computation results']]
            for key in data_pieces:
                if 'http' in links[key]:
                    print(f'Loading data: {key}...')
                    dir = join(expname, key)
                    os.makedirs(dir, exist_ok=True)
                    # gdown.download_folder(url = links[key], output = dir, quiet=False)
                    return_code, rel, folder_log = download_part_of_folder(dir,
                                                                           links[key],
                                                                           key=expname,
                                                                           whitelist=whitelist,
                                                                           via_pydrive=via_pydrive
                                                                           )

                    load_log.extend(folder_log)

                    if len(rel) == 0:
                        os.rmdir(dir)
                        print('No relevant data found')
                    else:
                        success = True

                    print('--------------------------')

            return success, load_log


def initialize_router():
    if 'IABSexperimentsdata.xlsx' in os.listdir('/content'):
        os.remove('/content/IABSexperimentsdata.xlsx')

    global_data_table_url = 'https://docs.google.com/spreadsheets/d/130DDFAoAbmm0jcKLBF6xsWsQLDr2Zsj4cPuOYivXoM8/export?format=xlsx'
    wget.download(global_data_table_url)

    data_router = pd.read_excel('IABSexperimentsdata.xlsx')
    data_router.fillna(method='ffill', inplace=True)

    data_pieces = list(data_router.columns.values)
    data_pieces = [d for d in list(data_router.columns.values) if d not in ['Эксперимент', 'Краткое описание', 'Video',
                                                                            'Aligned data', 'Computation results']]
    return data_router, data_pieces