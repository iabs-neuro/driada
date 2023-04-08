
from .gdrive import download, upload
from .information import gcmi, ksg, info_main
from .signals import sig_main
from .utils import data, naming, output, plot


# TODO: probably add automatic scanning for every .py file (dangerous and not pythonic, but convenient)
'''
import glob
from os.path import dirname, basename, isfile, join
modules = glob.glob(join(dirname(__file__), "*", "*.py"), recursive=True)
__all__ = [basename(f)[:-3] for f in modules if isfile(f) and not basename(f).startswith('_')]
'''