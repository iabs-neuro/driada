# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

# Mock torch for documentation build
try:
    import torch
except ImportError:
    from unittest.mock import MagicMock
    import sys
    
    # Create mock for torch and its submodules
    torch_mock = MagicMock()
    torch_mock.nn = MagicMock()
    torch_mock.nn.Module = MagicMock
    torch_mock.nn.functional = MagicMock()
    torch_mock.utils = MagicMock()
    torch_mock.utils.data = MagicMock()
    torch_mock.utils.data.Dataset = MagicMock
    torch_mock.optim = MagicMock()
    torch_mock.Tensor = MagicMock
    torch_mock.device = MagicMock
    torch_mock.cuda = MagicMock()
    torch_mock.cuda.is_available = MagicMock(return_value=False)
    
    # Install the mock
    sys.modules['torch'] = torch_mock
    sys.modules['torch.nn'] = torch_mock.nn
    sys.modules['torch.nn.functional'] = torch_mock.nn.functional
    sys.modules['torch.utils'] = torch_mock.utils
    sys.modules['torch.utils.data'] = torch_mock.utils.data
    sys.modules['torch.optim'] = torch_mock.optim

# -- Project information -----------------------------------------------------
project = 'DRIADA'
copyright = '2025, DRIADA Contributors'
author = 'DRIADA Contributors'
release = '0.6.4'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_autodoc_typehints',
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
autodoc_typehints = 'description'
autodoc_mock_imports = [
    'torch', 
    'torch.nn', 
    'torch.nn.functional',
    'torch.utils',
    'torch.utils.data',
    'torch.optim',
    'tensorflow', 
    'sklearn', 
    'umap', 
    'numba', 
    'cvxpy'
]

# Doctest configuration
doctest_test_doctest_blocks = 'yes'
doctest_global_setup = '''
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import DRIADA modules
from driada import *
from driada.intense import *
from driada.dim_reduction import *
from driada.integration import *
from driada.experiment import *
'''

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False
}

# -- Extension configuration -------------------------------------------------
# Add any Sphinx extension module names here, as strings.