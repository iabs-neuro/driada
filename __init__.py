
#from .gdrive.download import *
#from .gdrive.upload import *

from .information.info_main import *

from .signals.sig_main import *

from .networks.net_main import *

from .utils.data import *
from .utils.naming import *
from .utils.output import *
from .utils.plot import *


# TODO: probably add automatic scanning for every .py file (dangerous and not pythonic, but convenient)
'''
import glob
from os.path import dirname, basename, isfile, join
modules = glob.glob(join(dirname(__file__), "*", "*.py"), recursive=True)
__all__ = [basename(f)[:-3] for f in modules if isfile(f) and not basename(f).startswith('_')]
'''