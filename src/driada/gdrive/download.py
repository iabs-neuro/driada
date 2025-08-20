import os
from os.path import join

from pydrive2.drive import GoogleDrive
import warnings
import wget
import gdown
import pandas as pd
from pathlib import Path

from .gdrive_utils import (
    parse_google_drive_file,
    id_from_link,
    client,
    folder_type,
    folders_url,
    MAX_NUMBER_FILES,
)
from ..utils.output import Capturing


def retrieve_relevant_ids(
    folder,
    name_part,
    prohibited_name_part="",
    whitelist=[],
    extensions=[".csv", ".xlsx"],
):
    """Retrieve file IDs and names from a Google Drive folder that match specified criteria.

    Recursively searches through a Google Drive folder and its subfolders to find files
    matching the given criteria. Files are selected based on name patterns, file extensions,
    and whitelist/blacklist rules.

    Parameters
    ----------
    folder : str
        URL of the Google Drive folder to search.
    name_part : str
        Substring that must be present in the file name for it to be included.
    prohibited_name_part : str, optional
        Substring that, if present in the file name, will exclude the file.
        Default is empty string (no exclusions).
    whitelist : list of str, optional
        List of exact file names that will be included regardless of other criteria.
        Default is empty list.
    extensions : list of str, optional
        List of allowed file extensions (e.g., ['.csv', '.xlsx']).
        If empty, all extensions are allowed. Default is ['.csv', '.xlsx'].

    Returns
    -------
    return_code : bool
        True if the operation completed successfully, False otherwise.
    relevant : list of tuple
        List of (file_id, file_name) tuples for files matching the criteria.

    Raises
    ------
    MemoryError
        If the folder contains more files than MAX_NUMBER_FILES (50).

    Notes
    -----
    The function recursively searches through subfolders and applies the same
    filtering criteria to all levels of the folder hierarchy.
    """

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
            f"The folder {folder} has {len(list(id_name_type_iter))} elements while max allowed number of files is {MAX_NUMBER_FILES}"
        )

    for child_id, child_name, child_type in id_name_type_iter:
        if child_type != folder_type:
            if child_name in whitelist:
                relevant.append((child_id, child_name))
            elif name_part in child_name:
                if (
                    len(extensions) != 0
                    and Path(child_name).suffix in extensions
                    or len(extensions) == 0
                ):
                    if (
                        (prohibited_name_part is not None)
                        and (prohibited_name_part not in child_name)
                        or prohibited_name_part is None
                    ):
                        relevant.append((child_id, child_name))
            else:
                pass

        else:
            return_code, rel_sublist = retrieve_relevant_ids(
                folders_url + child_id,
                name_part,
                prohibited_name_part=prohibited_name_part,
                whitelist=whitelist,
                extensions=extensions,
            )
            if not return_code:
                print(f"recursive search broke on folder {child_id}")
                break
            relevant.extend(rel_sublist)

    return return_code, relevant


def download_part_of_folder(
    output,  # path for downloaded data
    folder,  # share link to google drive folder
    key="",  # part of filename to search for
    antikey=None,  # part of name to suppress
    whitelist=[],  # list of filenames to be downloaded regardless of their names
    extensions=[".csv", ".xlsx", ".npz"],  # allowed file extensions
    via_pydrive=False,  # pydrive requires authorization, but can download a big number of files,
    gauth=None,
    maxfiles=None,
):
    """Download specific files from a Google Drive folder based on filtering criteria.

    Downloads files from a Google Drive folder that match specific name patterns and
    file extensions. Supports both gdown (no authentication) and PyDrive2 (requires
    authentication) methods.

    Parameters
    ----------
    output : str
        Local directory path where files will be downloaded.
    folder : str
        Google Drive folder share link.
    key : str, optional
        Substring that must be present in file names to be downloaded.
        Default is empty string (matches all).
    antikey : str or None, optional
        Substring that, if present in file names, will exclude them from download.
        Default is None.
    whitelist : list of str, optional
        List of exact file names to download regardless of other criteria.
        Default is empty list.
    extensions : list of str, optional
        List of allowed file extensions. Default is ['.csv', '.xlsx', '.npz'].
    via_pydrive : bool, optional
        If True, use PyDrive2 (requires authentication but supports more files).
        If False, use gdown (no auth but limited). Default is False.
    gauth : GoogleAuth object or None, optional
        PyDrive2 authentication object. Required if via_pydrive=True.
        Default is None.
    maxfiles : int or None, optional
        Maximum number of files to download (only used with PyDrive2).
        Default is None (no limit).

    Returns
    -------
    return_code : bool
        True if download completed successfully, False otherwise.
    rel : list of tuple
        List of (file_id, file_name) tuples for downloaded files.
    load_log : list
        Captured output log from the download process.

    Raises
    ------
    ValueError
        If via_pydrive=True but gauth is None.
    FileNotFoundError
        If download fails when not using PyDrive2.

    Examples
    --------
    >>> # Download CSV files containing 'experiment' in name
    >>> success, files, log = download_part_of_folder(
    ...     output='./data',
    ...     folder='https://drive.google.com/drive/folders/...',
    ...     key='experiment',
    ...     extensions=['.csv']
    ... )
    """

    os.makedirs(output, exist_ok=True)

    with Capturing() as load_log:
        if via_pydrive:
            if gauth is None:
                raise ValueError(
                    "To use pydrive, you need to authenticate using one of the functions"
                    " in driada.gdrive.auth"
                )
            drive = GoogleDrive(gauth)

            rel = []
            fid = id_from_link(folder)
            file_list = drive.ListFile(
                {"q": f"'{fid}' in parents and trashed=false"}
            ).GetList()
            if maxfiles is not None:
                file_list = file_list[:maxfiles]

            for f in file_list:
                if key in f["title"]:
                    # print('title: %s, id: %s' % (f['title'],f['id']))
                    f.GetContentFile(join(output, f["title"]))
                    rel.append((f["id"], f["title"]))

            return_code = True

        else:
            return_code, rel = retrieve_relevant_ids(
                folder,
                key,
                prohibited_name_part=antikey,
                whitelist=whitelist,
                extensions=extensions,
            )

            if return_code:
                for i, pair in enumerate(rel):
                    idx, name = rel[i]
                    gdown.download(id=idx, output=os.path.join(output, name))

            else:
                raise FileNotFoundError("Error in downloading procedure!")

        return return_code, rel, load_log


def download_gdrive_data(
    data_router,
    expname,
    whitelist=["Timing.xlsx"],
    via_pydrive=False,
    data_pieces=None,
    tdir="DRIADA data",
    gauth=None,
):
    """Download experimental data from Google Drive based on a data router table.

    Uses a data router DataFrame to locate and download experimental data files
    from Google Drive folders specified for each experiment.

    Parameters
    ----------
    data_router : pandas.DataFrame
        DataFrame containing experiment names and corresponding Google Drive links
        for different data types. Must have an 'Эксперимент' column.
    expname : str
        Name of the experiment to download data for. Must match an entry in
        the 'Эксперимент' column of data_router.
    whitelist : list of str, optional
        List of file names to always download regardless of naming patterns.
        Default is ['Timing.xlsx'].
    via_pydrive : bool, optional
        If True, use PyDrive2 for downloading (requires authentication).
        If False, use gdown. Default is False.
    data_pieces : list of str or None, optional
        List of data types (column names) to download. If None, downloads all
        available data types except certain excluded ones. Default is None.
    tdir : str, optional
        Target directory name for downloaded data. Default is 'DRIADA data'.
    gauth : GoogleAuth object or None, optional
        PyDrive2 authentication object. Required if via_pydrive=True.
        Default is None.

    Returns
    -------
    success : bool
        True if at least one file was successfully downloaded, False otherwise.
    load_log : list
        Captured output log from the download process.

    Notes
    -----
    The function creates a directory structure: tdir/expname/data_type/
    for organizing downloaded files. Data types excluded by default are:
    'Эксперимент', 'Краткое описание', 'Video', 'Aligned data', 'Computation results'.
    """

    with Capturing() as load_log:
        print("-------------------------------------------------------------")
        print(f"Extracting data for {expname} from Google Drive")
        print("-------------------------------------------------------------")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)

            success = False
            available_exp = data_router["Эксперимент"].values
            if expname not in available_exp:
                print(f"{expname} not found in available experiments: {available_exp}")
                return success, load_log

            row = data_router[data_router["Эксперимент"] == expname]
            links = dict(zip(row.columns, row.values[0]))

            os.makedirs(join(tdir, expname), exist_ok=True)
            if data_pieces is None:
                data_pieces = [
                    d
                    for d in list(data_router.columns.values)
                    if d
                    not in [
                        "Эксперимент",
                        "Краткое описание",
                        "Video",
                        "Aligned data",
                        "Computation results",
                    ]
                ]

            for key in data_pieces:
                if "http" in links[key]:
                    print(f"Loading data: {key}...")
                    ddir = join(tdir, expname, key)
                    os.makedirs(ddir, exist_ok=True)
                    # gdown.download_folder(url = links[key], output = dir, quiet=False)
                    return_code, rel, folder_log = download_part_of_folder(
                        ddir,
                        links[key],
                        key=expname,
                        whitelist=whitelist,
                        via_pydrive=via_pydrive,
                        gauth=gauth,
                    )

                    load_log.extend(folder_log)

                    if len(rel) == 0:
                        os.rmdir(ddir)
                        print("No relevant data found at: ", links[key])

                    else:
                        loaded_names = [r[1] for r in rel]
                        for n in loaded_names:
                            print(n)
                        success = True

                    print("--------------------------")

            return success, load_log


def initialize_iabs_router(root="\\content"):
    """Download and initialize the IABS data router from Google Sheets.

    Downloads the IABS (Institute for Advanced Brain Studies) data router
    spreadsheet from Google Drive and prepares it for use in data downloading.

    Parameters
    ----------
    root : str, optional
        Root directory where the router file will be saved.
        Default is '\\content' (typically for Google Colab).

    Returns
    -------
    data_router : pandas.DataFrame
        DataFrame containing experiment information and Google Drive links.
        Columns include experiment names and various data type links.
    data_pieces : list of str
        List of data type column names that can be downloaded, excluding
        metadata columns.

    Notes
    -----
    The function downloads the router from a hardcoded Google Sheets URL.
    It removes any existing router file before downloading the latest version.
    Empty cells in the DataFrame are forward-filled to handle merged cells.

    The following columns are excluded from data_pieces as they contain
    metadata rather than downloadable data:
    - 'Эксперимент' (Experiment name)
    - 'Краткое описание' (Brief description)
    - 'Video'
    - 'Aligned data'
    - 'Computation results'
    """
    router_name = "IABS data router.xlsx"
    router_path = join(root, router_name)
    os.makedirs(root, exist_ok=True)
    if router_name in os.listdir(root):
        os.remove(router_path)

    global_data_table_url = "https://docs.google.com/spreadsheets/d/130DDFAoAbmm0jcKLBF6xsWsQLDr2Zsj4cPuOYivXoM8/export?format=xlsx"
    wget.download(global_data_table_url, out=router_path)

    data_router = pd.read_excel(router_path)
    # data_router.fillna(method='ffill', inplace=True)
    data_router = data_router.replace("", None).ffill()

    data_pieces = [
        d
        for d in list(data_router.columns.values)
        if d
        not in [
            "Эксперимент",
            "Краткое описание",
            "Video",
            "Aligned data",
            "Computation results",
        ]
    ]
    return data_router, data_pieces
