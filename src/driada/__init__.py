
#from .gdrive.download import *
#from .gdrive.upload import *

# TODO: include environment.yml
# TODO: unit tests for calculations (https://goodresearch.dev/testing)
# TODO: document everything in Google Style (https://goodresearch.dev/docs)
# TODO: publish docs on Readthedocs (https://docs.readthedocs.io/en/stable/tutorial/index.html#importing-the-project-to-read-the-docs_
# TODO: update ReadMe (https://goodresearch.dev/pipelines)

# TODO: probably add automatic scanning for every .py file (dangerous and not pythonic, but convenient)
'''
import glob
from os.path import dirname, basename, isfile, join
modules = glob.glob(join(dirname(__file__), "*", "*.py"), recursive=True)
__all__ = [basename(f)[:-3] for f in modules if isfile(f) and not basename(f).startswith('_')]
'''

# TODO: make Readme structured as a numbered list, like in mango