import os.path
from datetime import datetime
import pytz
from pydrive2.drive import GoogleDrive

from .gdrive_utils import id_from_link
from .download import retrieve_relevant_ids


def get_datetime():
    """Get current datetime string in Moscow timezone.

    Returns
    -------
    str
        Formatted datetime string in format 'DD-MM-YYYY HH:MM:SS'.

    Examples
    --------
    >>> import re
    >>> datetime_str = get_datetime()
    >>> bool(re.match(r'\\d{2}-\\d{2}-\\d{4} \\d{2}:\\d{2}:\\d{2}', datetime_str))
    True
    """
    tz = pytz.timezone("Europe/Moscow")
    now = datetime.now(tz)

    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    return dt_string


def save_file_to_gdrive(
    data_router,
    expname,
    path_to_file,
    link=None,
    destination=None,
    force_rewriting=False,
    gauth=None,
):
    """Upload a file to Google Drive folder associated with an experiment.

    Uploads a local file to a Google Drive folder specified either directly
    via a link or through a data router table. Supports both creating new
    files and overwriting existing ones.

    Parameters
    ----------
    data_router : pandas.DataFrame
        DataFrame containing experiment names and Google Drive folder links.
        Must have an 'Эксперимент' column matching expname.
    expname : str
        Name of the experiment, used to look up folder links in data_router.
    path_to_file : str
        Local file path of the file to upload.
    link : str or None, optional
        Direct Google Drive folder link. If provided, overrides data_router lookup.
        Default is None.
    destination : str or None, optional
        Column name in data_router specifying which folder to use.
        Required if link is None. Default is None.
    force_rewriting : bool, optional
        If True, overwrites existing file with same name.
        If False, appends timestamp to filename. Default is False.
    gauth : GoogleAuth object
        PyDrive2 authentication object. Required for upload.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If destination is not found in data_router columns.

    Notes
    -----
    When force_rewriting=False, the uploaded file will have a timestamp
    appended to its name in format 'filename_DD-MM-YYYY HH:MM:SS.ext'.
    
    When force_rewriting=True and multiple files with the same name exist,
    a warning is printed and the first matching file is overwritten.

    Examples
    --------
    >>> # Upload with timestamp
    >>> save_file_to_gdrive(  # doctest: +SKIP
    ...     data_router, 'exp001', './results.csv',
    ...     destination='Results', gauth=auth
    ... )
    
    >>> # Overwrite existing file
    >>> save_file_to_gdrive(  # doctest: +SKIP
    ...     data_router, 'exp001', './results.csv',
    ...     destination='Results', force_rewriting=True, gauth=auth
    ... )
    """

    drive = GoogleDrive(gauth)

    if link is None:
        row = data_router[data_router["Эксперимент"] == expname]
        links = dict(zip(row.columns, row.values[0]))
        if destination not in links:
            raise ValueError(f"Wrong folder name: {destination}")
        link = links[destination]

    fid = id_from_link(link)

    dataname = os.path.basename(path_to_file)
    if force_rewriting:
        return_code, rel = retrieve_relevant_ids(
            link, dataname, whitelist=[], extensions=[".npz", ".csv", "xlsx"]
        )
        if len(rel) != 0:
            if len(rel) > 1:
                print(
                    "More than one relevant file found, which is suspicious. Consider manual check"
                )

            existing_file_data = rel[0]
            dat_id, dat_name = existing_file_data
            f = drive.CreateFile({"id": dat_id})

        else:
            print(f"Data for {dataname} not found in folder, nothing to rewrite")
            f = drive.CreateFile(
                {"title": dataname, "parents": [{"kind": "drive#fileLink", "id": fid}]}
            )

    else:
        date_time = get_datetime()
        ext = os.path.splitext(path_to_file)[1]
        dataname = dataname[: -len(ext)] + date_time + ext
        f = drive.CreateFile(
            {"title": dataname, "parents": [{"kind": "drive#fileLink", "id": fid}]}
        )

    f.SetContentFile(path_to_file)
    f.Upload()
