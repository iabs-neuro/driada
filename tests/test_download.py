import os

from src.driada.gdrive.download import *
import shutil
import os


TEST_LINK = "https://drive.google.com/drive/folders/1rqV0A3Y-miX8QccmkiGEI5r-5K4RdjCj?usp=sharing"
TEST_DIR = os.path.join(os.getcwd(), 'test dir')


def test_download():
    print(TEST_DIR)
    shutil.rmtree(TEST_DIR, ignore_errors=True)

    return_code, filenames, load_log = download_part_of_folder(
                                                TEST_DIR,  # path for downloaded data
                                                TEST_LINK,  # share link to google drive folder
                                                key='',  # part of filename to search for
                                                antikey=None,  # part of name to suppress
                                                whitelist=[],  # list of filenames to be downloaded regardless of their names
                                                extensions=[],  # allowed file extensions
                                            )

    assert len(os.listdir(TEST_DIR)) == 5


def test_download_redflag():
    print(TEST_DIR)
    shutil.rmtree(TEST_DIR, ignore_errors=True)

    return_code, filenames, load_log = download_part_of_folder(
                                                TEST_DIR,  # path for downloaded data
                                                TEST_LINK,  # share link to google drive folder
                                                key='',  # part of filename to search for
                                                antikey='redflag',  # part of name to suppress
                                                whitelist=[],  # list of filenames to be downloaded regardless of their names
                                                extensions=[],  # allowed file extensions
                                            )

    assert len(os.listdir(TEST_DIR)) == 4


def test_download_extension():
    print(TEST_DIR)
    shutil.rmtree(TEST_DIR, ignore_errors=True)

    return_code, filenames, load_log = download_part_of_folder(
                                                TEST_DIR,  # path for downloaded data
                                                TEST_LINK,  # share link to google drive folder
                                                key='',  # part of filename to search for
                                                antikey=None,  # part of name to suppress
                                                whitelist=[],  # list of filenames to be downloaded regardless of their names
                                                extensions=['.txt'],  # allowed file extensions
                                            )

    assert os.listdir(TEST_DIR) == ['test.txt', 'test2.txt']


def test_download_whitelist():
    print(TEST_DIR)
    shutil.rmtree(TEST_DIR, ignore_errors=True)

    return_code, filenames, load_log = download_part_of_folder(
                                                TEST_DIR,  # path for downloaded data
                                                TEST_LINK,  # share link to google drive folder
                                                key='',  # part of filename to search for
                                                antikey=None,  # part of name to suppress
                                                whitelist=['white.xlsx'],  # list of filenames to be downloaded regardless of their names
                                                extensions=['.txt'],  # allowed file extensions
                                            )

    assert os.listdir(TEST_DIR) == ['test.txt', 'test2.txt', 'white.xlsx']


def test_erase():
    # just clears the test folder
    shutil.rmtree(TEST_DIR, ignore_errors=True)
