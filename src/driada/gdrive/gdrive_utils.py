import requests
import regex
from bs4 import BeautifulSoup
from itertools import islice
import json


class GoogleDriveFile(object):
    """Represent Google Drive file objects structure.

    Attributes
    ----------
    id: str
        Unique id, used to build the download URL.
    name: str
        Actual name, used as file name.
    type: str
        MIME type, or application/vnd.google-apps.folder if it is a folder
    children: List[GoogleDriveFile]
        If it is a directory, it contains the folder files/directories    """

    def __init__(self, id, name, type, children=None):
        """Initialize GoogleDriveFile instance.
        
        Parameters
        ----------
        id : str
            Unique file or folder ID from Google Drive.
        name : str
            Display name of the file or folder.
        type : str
            MIME type of the file, or 'application/vnd.google-apps.folder' for folders.
        children : List[GoogleDriveFile], optional
            Child items if this is a folder. Default is empty list.        """
        self.id = id
        self.name = name
        self.type = type
        self.children = children if children is not None else []

    def is_folder(self):
        """Check if the GoogleDriveFile is a folder.

        Returns
        -------
        bool
            True if the file is a folder, False otherwise.
            
        Notes
        -----
        Uses the global folder_type constant for comparison.        """
        return self.type == folder_type

    def __repr__(self):
        """Return string representation of GoogleDriveFile.
        
        Returns
        -------
        str
            Formatted string showing all attributes including children.
            
        Notes
        -----
        May produce long output if there are many children.        """
        template = "(id={id}, name={name}, type={type}, children={children})"
        return "GoogleDriveFile" + template.format(
            id=self.id,
            name=self.name,
            type=self.type,
            children=self.children,
        )


def parse_google_drive_file(folder, content, use_cookies=True):
    """Extract information about the current page file and its children.

    Parameters
    ----------
    folder : str
        URL of the Google Drive folder.
        Must be of the format 'https://drive.google.com/drive/folders/{id}'.
    content : str
        Google Drive's raw HTML content.
    use_cookies : bool, optional
        Whether to clear cookies. Default is True.

    Returns
    -------
    gdrive_file : GoogleDriveFile
        Current GoogleDriveFile object with empty children list.
    id_name_type_iter : list
        List of tuples (id, name, type) for each child item.
        
    Raises
    ------
    RuntimeError
        If folder information cannot be extracted from HTML.
        
    Notes
    -----
    Parses JavaScript data embedded in Google Drive HTML.
    Expects specific HTML structure and may break with Google Drive updates.    """
    folder_soup = BeautifulSoup(content, features="html.parser")

    if not use_cookies:
        client.cookies.clear()

    # finds the script tag with window['_DRIVE_ivd']
    encoded_data = None
    for script in folder_soup.select("script"):
        inner_html = script.decode_contents()

        if "_DRIVE_ivd" in inner_html:
            # first js string is _DRIVE_ivd, the second one is the encoded arr
            regex_iter = string_regex.finditer(inner_html)
            # get the second elem in the iter
            try:
                encoded_data = next(islice(regex_iter, 1, None)).group(1)
            except StopIteration:
                raise RuntimeError("Couldn't find the folder encoded JS string")
            break

    if encoded_data is None:
        raise RuntimeError(
            "Cannot retrieve the folder information from the link. "
            "You may need to change the permission to "
            "'Anyone with the link'."
        )

    # decodes the array and evaluates it as a python array
    decoded = encoded_data.encode("utf-8").decode("unicode_escape")
    folder_arr = json.loads(decoded)

    folder_contents = [] if folder_arr[0] is None else folder_arr[0]

    gdrive_file = GoogleDriveFile(
        id=folder.split("/")[-1],
        name=" - ".join(folder_soup.title.contents[0].split(" - ")[:-1]),
        type=folder_type,
    )

    id_name_type_iter = [
        (e[0], e[2].encode("raw_unicode_escape").decode("utf-8"), e[3])
        for e in folder_contents
    ]

    return gdrive_file, id_name_type_iter


def download_and_parse_google_drive_link(
    folder, quiet=False, use_cookies=True, remaining_ok=False, name_part=""
):
    """Get folder structure of Google Drive folder URL.

    Parameters
    ----------
    folder : str
        URL of the Google Drive folder.
        Must be of the format 'https://drive.google.com/drive/folders/{id}'.
    quiet : bool, optional
        Suppress terminal output. Default is False.
    use_cookies : bool, optional
        Flag to use cookies. Default is True.
    remaining_ok : bool, optional
        Allow processing if folder has ≥50 files (API limit).
        Default is False.
    name_part : str, optional
        Filter items by name substring. Default is empty string (no filter).

    Returns
    -------
    return_code : bool
        True if successful, False if failed (network error, permissions, etc.).
    gdrive_file : GoogleDriveFile or None
        Folder structure with nested children, or None if failed.
        
    Raises
    ------
    RuntimeError
        If folder has ≥50 files and remaining_ok is False.
        
    Notes
    -----
    Recursively processes subfolders. Limited to 50 items per folder
    due to Google Drive API restrictions.    """
    return_code = True

    folder_page = client.get(folder)

    if folder_page.status_code != 200:
        return False, None

    gdrive_file, id_name_type_iter = parse_google_drive_file(
        folder,
        folder_page.text,
    )

    for child_id, child_name, child_type in id_name_type_iter:
        if name_part in child_name:
            if child_type != folder_type:
                if not quiet:
                    print(
                        "Processing file",
                        child_id,
                        child_name,
                    )
                gdrive_file.children.append(
                    GoogleDriveFile(
                        id=child_id,
                        name=child_name,
                        type=child_type,
                    )
                )
                if not return_code:
                    return return_code, None
                continue

            if not quiet:
                print(
                    "Retrieving folder",
                    child_id,
                    child_name,
                )
            return_code, child = download_and_parse_google_drive_link(
                folders_url + child_id,
                use_cookies=use_cookies,
                quiet=quiet,
            )
            if not return_code:
                return return_code, None
            gdrive_file.children.append(child)

    has_at_least_max_files = len(gdrive_file.children) == MAX_NUMBER_FILES
    if not remaining_ok and has_at_least_max_files:
        err_msg = " ".join(
            [
                "The gdrive folder with url: {url}".format(url=folder),
                "has at least {max} files,".format(max=MAX_NUMBER_FILES),
                "gdrive can't download more than this limit,",
                "if you are ok with this,",
                "please run again with --remaining-ok flag.",
            ]
        )
        raise RuntimeError(err_msg)
    return return_code, gdrive_file


def id_from_link(link):
    """Extract the file or folder ID from a Google Drive URL.

    Parameters
    ----------
    link : str
        Google Drive URL containing the file or folder ID.
        Can be in format:
        - https://drive.google.com/drive/folders/{id}
        - https://drive.google.com/file/d/{id}/view
        - https://drive.google.com/open?id={id}

    Returns
    -------
    str
        The extracted file or folder ID.

    Raises
    ------
    ValueError
        If the link doesn't contain 'http'.

    Examples
    --------
    >>> id_from_link('https://drive.google.com/drive/folders/1a2b3c4d5e')
    '1a2b3c4d5e'
    >>> id_from_link('https://drive.google.com/open?id=xyz123')
    'xyz123'
    
    Notes
    -----
    Does not validate the extracted ID format. May return empty string
    or invalid IDs for malformed URLs.    """
    if "http" not in link:
        raise ValueError("Wrong link format")

    if "id=" in link:
        return link.split("id=")[-1].split("&")[0]
    else:
        return link.split("folders/")[-1].split("?")[0]


folders_url = "https://drive.google.com/drive/folders/"
files_url = "https://drive.google.com/uc?id="
folder_type = "application/vnd.google-apps.folder"

string_regex = regex.compile(r"'((?:[^'\\]|\\.)*)'")
MAX_NUMBER_FILES = 50

client = requests.session()
