#!/usr/bin/env python3
"""
Run doctests on all documentation files to ensure code examples work.

This script:
1. Finds all .rst files with code examples
2. Runs doctests on them
3. Reports any failures
4. Can be used in CI/CD pipelines
"""

import os
import sys
import doctest
import importlib
from pathlib import Path
from typing import List, Dict, Tuple
import tempfile
import subprocess
import re

# Add src to path to import driada
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def find_rst_files(docs_dir: Path) -> List[Path]:
    """Find all .rst files in the documentation directory."""
    return list(docs_dir.rglob("*.rst"))


def find_python_files_with_doctests(src_dir: Path) -> List[Path]:
    """Find all Python files containing docstring examples (>>> patterns)."""
    python_files = []
    
    for py_file in src_dir.rglob("*.py"):
        # Skip __pycache__ and other non-source directories
        if '__pycache__' in str(py_file) or '.egg-info' in str(py_file):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Look for doctest patterns
                if '>>>' in content:
                    python_files.append(py_file)
        except (UnicodeDecodeError, IOError):
            # Skip files we can't read
            continue
    
    return sorted(python_files)


def extract_code_blocks(file_path: Path) -> List[Tuple[int, str]]:
    """Extract code blocks from RST file that should be tested.
    
    Returns:
        List of (line_number, code) tuples
    """
    code_blocks = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Look for code-block:: python
        if line.strip().startswith('.. code-block:: python'):
            i += 1
            # Skip empty lines
            while i < len(lines) and not lines[i].strip():
                i += 1
            
            # Extract code block
            if i < len(lines):
                # Get indentation level
                indent_line = lines[i]
                if indent_line.strip():
                    indent_level = len(indent_line) - len(indent_line.lstrip())
                    
                    code_lines = []
                    start_line = i + 1
                    
                    # Collect all lines with the same or greater indentation
                    while i < len(lines):
                        current_line = lines[i]
                        if current_line.strip():  # Non-empty line
                            current_indent = len(current_line) - len(current_line.lstrip())
                            if current_indent < indent_level:
                                break
                            # Remove the base indentation
                            code_lines.append(current_line[indent_level:])
                        else:
                            code_lines.append(current_line)
                        i += 1
                    
                    code = ''.join(code_lines).rstrip()
                    if code:
                        code_blocks.append((start_line, code))
        else:
            i += 1
    
    return code_blocks


def prepare_test_code(code: str, file_path: Path) -> str:
    """Prepare code for testing by adding necessary imports and setup."""
    # Common imports that examples might need
    setup_code = """
import numpy as np
import matplotlib.pyplot as plt
from driada import *
from driada.experiment import Experiment
from driada.information import TimeSeries, MultiTimeSeries
from driada.network import Network
from driada.intense import compute_cell_feat_significance
from driada.dim_reduction import MVData
from driada.utils import rescale, make_beautiful, compute_rate_map, brownian

# Optional imports
try:
    import torch
except ImportError:
    torch = None

try:
    import umap
except ImportError:
    # Mock UMAP for testing
    class MockUMAP:
        def __init__(self, *args, **kwargs):
            self.n_components = kwargs.get('n_components', 2)
        def fit_transform(self, X):
            # Return random embedding of correct shape
            return np.random.randn(X.shape[0], self.n_components)
    umap = type('umap', (), {'UMAP': MockUMAP})()

# Create a minimal real Experiment directly
np.random.seed(42)  # For reproducibility

# Don't create experiment here - let each example handle it
# This avoids conflicts and path issues
exp = None  # Will be loaded by examples as needed

# But define some basic parameters that examples might use
n_neurons = 10
n_frames = 1000
fps = 20.0

# Additional data that examples might reference
data = np.random.randn(100, 10)
spike_times = np.sort(np.random.uniform(0, 100, 50))
spike_counts = np.random.choice([0, 1, 2, 3, 4], size=300, p=[0.2, 0.3, 0.3, 0.15, 0.05])
positions = np.random.randn(100, 2)

# Load sample recording data if it exists
try:
    sample_data_path = os.path.join(os.path.dirname(__file__), '..', 'examples', 'example_data', 'sample_recording.npz')
    if os.path.exists(sample_data_path):
        sample_data = dict(np.load(sample_data_path))
        # Make it available as 'data' for examples that expect it
        if 'data' not in locals():
            data = sample_data
except:
    pass

# Also make paths to example files available
import os
# Get the actual script location, not the temp file location
actual_script_path = os.path.abspath(__file__)
if 'tmp' in actual_script_path:
    # We're in a temp file, use the known project path
    project_root = '/Users/nikita/PycharmProjects/driada2'
else:
    script_dir = os.path.dirname(actual_script_path)
    project_root = os.path.abspath(os.path.join(script_dir, '..'))

example_data_dir = os.path.join(project_root, 'examples', 'example_data')

# Make absolute paths available
sample_npz_path = os.path.join(example_data_dir, 'sample_recording.npz')
sample_pkl_path = os.path.join(example_data_dir, 'sample_experiment.pkl')

# Also make relative paths from project root
sample_npz_relpath = 'examples/example_data/sample_recording.npz'
sample_pkl_relpath = 'examples/example_data/sample_experiment.pkl'
# Create some example data that might be referenced
position = np.sin(np.linspace(0, 10 * np.pi, n_frames)) + 0.1 * np.random.randn(n_frames)
speed = np.abs(np.gradient(position) * fps) + 0.01

# Handle different neural_data shapes for different contexts
neural_data = np.random.randn(10, 100)  # Default 2D for most cases
neural_data_3d = np.random.randn(10, 5, 100)  # For RSA examples
trial_types = ['A', 'B', 'C', 'D', 'E']
auth = None  # Mock auth object for gdrive examples

# Prevent matplotlib from showing plots
plt.ioff()

# Add example file paths
sample_npz_path = '/Users/nikita/PycharmProjects/driada2/examples/example_data/sample_recording.npz'
sample_pkl_path = '/Users/nikita/PycharmProjects/driada2/examples/example_data/sample_experiment.pkl'
"""
    
    # Check if this is an example that needs special handling
    if 'gdrive' in str(file_path):
        # Skip actual Google Drive operations
        code = code.replace('desktop_auth()', 'None')
        code = code.replace('download_gdrive_data(', '# download_gdrive_data(')
        code = code.replace('save_file_to_gdrive(', '# save_file_to_gdrive(')
    
    # Use appropriate neural_data shape based on context
    if 'rsa' in str(file_path) and 'compute_rdm' in code:
        # RSA examples need 3D data
        setup_code = setup_code.replace(
            'neural_data = np.random.randn(10, 100)',
            'neural_data = np.random.randn(10, 5, 100)  # (n_neurons, n_conditions, n_timepoints)'
        )
    elif 'dimensionality/effective' in str(file_path):
        # Effective dimensionality expects (n_timepoints, n_neurons)
        setup_code = setup_code.replace(
            'neural_data = np.random.randn(10, 100)',
            'neural_data = np.random.randn(1000, 100)  # (n_timepoints, n_neurons)'
        )
    
    return setup_code + "\n" + code


def run_code_block(code: str, file_path: Path, line_num: int, timeout: int = 30) -> Tuple[bool, str]:
    """Run a code block and return success status and any error message."""
    try:
        # Prepare the code
        test_code = prepare_test_code(code, file_path)
        
        # Create a temporary file to run the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            temp_file = f.name
        
        try:
            # Run the code in a subprocess to isolate it
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout  # Use configurable timeout
            )
            
            if result.returncode != 0:
                return False, f"Error: {result.stderr}"
            
            return True, ""
            
        finally:
            # Clean up
            os.unlink(temp_file)
            
    except subprocess.TimeoutExpired:
        return False, "Timeout: Code took too long to execute"
    except Exception as e:
        return False, f"Exception: {str(e)}"


def test_rst_file(file_path: Path, timeout: int = 30) -> Dict:
    """Test all code blocks in an RST file using doctest.
    
    Returns:
        Dict with test results
    """
    import doctest
    import tempfile
    import subprocess
    
    results = {
        'file': file_path,
        'total_blocks': 0,
        'passed': 0,
        'failed': 0,
        'failures': []
    }
    
    # Create a test script that uses doctest.testfile()
    test_script = f'''#!/usr/bin/env python3
import sys
import os
import doctest
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Setup environment
os.environ["MPLBACKEND"] = "Agg"  # Non-interactive matplotlib

# Import and setup common modules
import numpy as np
np.random.seed(42)

# Import driada modules
try:
    from driada import *
    from driada.experiment import Experiment
    from driada.information import TimeSeries, MultiTimeSeries
    from driada.network import Network
    from driada.intense import compute_cell_feat_significance
    from driada.dim_reduction import MVData
    from driada.utils import rescale, make_beautiful, compute_rate_map, brownian
except ImportError as e:
    print(f"Warning: Could not import driada modules: {{e}}")

# Create globals dictionary
globs = {{
    'np': np,
    'os': os,
    'sys': sys,
    'plt': None,  # Will be imported on demand
}}

# Add driada imports to globals
try:
    globs.update(globals())
except:
    pass

# Optional imports
try:
    import torch
    globs['torch'] = torch
except ImportError:
    globs['torch'] = None

# Mock objects
class MockAuth:
    pass
globs['auth'] = MockAuth()

# Add sample data
globs['data'] = np.random.randn(100, 10)
globs['spike_times'] = np.sort(np.random.uniform(0, 100, 50))
globs['positions'] = np.random.randn(100, 2)
globs['n_neurons'] = 10
globs['n_frames'] = 1000
globs['fps'] = 20.0

# Run doctests on the RST file
try:
    failures, tests = doctest.testfile(
        "{file_path}",
        module_relative=False,
        globs=globs,
        optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS,
        verbose=False
    )
    print(f"TESTS={{tests}}")
    print(f"FAILURES={{failures}}")
    sys.exit(0 if failures == 0 else 1)
except Exception as e:
    print(f"ERROR: {{str(e)}}")
    import traceback
    traceback.print_exc()
    sys.exit(2)
'''
    
    # Write and run test script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        temp_file = f.name
    
    try:
        # Run the test script with timeout
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Parse output
        output = result.stdout + result.stderr
        
        # Extract test counts
        import re
        tests_match = re.search(r'TESTS=(\d+)', output)
        failures_match = re.search(r'FAILURES=(\d+)', output)
        
        if tests_match and failures_match:
            total_tests = int(tests_match.group(1))
            total_failures = int(failures_match.group(1))
            
            results['total_blocks'] = total_tests
            results['failed'] = total_failures
            results['passed'] = total_tests - total_failures
            
            if total_failures > 0:
                # Extract failure details if available
                results['failures'].append({
                    'test': str(file_path.name),
                    'error': 'Doctest failures (run with --verbose for details)',
                    'examples': []
                })
        else:
            # Could not parse output - use old method as fallback
            code_blocks = extract_code_blocks(file_path)
            results['total_blocks'] = len(code_blocks)
            
            for line_num, code in code_blocks:
                success, error = run_code_block(code, file_path, line_num, timeout)
                
                if success:
                    results['passed'] += 1
                else:
                    results['failed'] += 1
                    results['failures'].append({
                        'line': line_num,
                        'code': code,
                        'error': error
                    })
            
    except subprocess.TimeoutExpired:
        results['total_blocks'] = 1
        results['failed'] = 1
        results['failures'].append({
            'test': str(file_path.name),
            'error': f'Timeout: Tests took longer than {timeout} seconds',
            'examples': []
        })
        
    except Exception as e:
        results['total_blocks'] = 1
        results['failed'] = 1
        results['failures'].append({
            'test': str(file_path.name),
            'error': f"Error running tests: {str(e)}",
            'examples': []
        })
        
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_file)
        except:
            pass
    
    return results


def prepare_python_doctest_globals():
    """Prepare global namespace for Python docstring tests."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    # Common imports
    globs = {
        'np': np,
        'plt': plt,
    }
    
    # Import all driada modules
    try:
        import driada
        globs['driada'] = driada
        
        from driada.experiment import (
            Experiment, load_experiment, load_exp_from_aligned_data,
            save_exp_to_pickle, load_demo_experiment
        )
        from driada.information import TimeSeries, MultiTimeSeries
        from driada.network import Network
        from driada.intense import compute_cell_feat_significance
        from driada.dim_reduction import MVData
        from driada.utils import rescale, make_beautiful, compute_rate_map, brownian
        from driada.utils.data import create_correlated_gaussian_data
        from driada.utils.naming import construct_session_name
        
        # Add imports to globals
        globs['Experiment'] = Experiment
        globs['load_experiment'] = load_experiment
        globs['load_exp_from_aligned_data'] = load_exp_from_aligned_data
        globs['construct_session_name'] = construct_session_name
        globs['save_exp_to_pickle'] = save_exp_to_pickle
        globs['load_demo_experiment'] = load_demo_experiment
        globs['TimeSeries'] = TimeSeries
        globs['MultiTimeSeries'] = MultiTimeSeries
        globs['Network'] = Network
        globs['compute_cell_feat_significance'] = compute_cell_feat_significance
        globs['MVData'] = MVData
        globs['rescale'] = rescale
        globs['make_beautiful'] = make_beautiful
        globs['compute_rate_map'] = compute_rate_map
        globs['brownian'] = brownian
        globs['create_correlated_gaussian_data'] = create_correlated_gaussian_data
        
        # Add some test data
        globs['data'] = np.random.randn(100, 10)
        globs['positions'] = np.random.randn(100, 2)
        globs['spike_times'] = np.sort(np.random.uniform(0, 100, 50))
        
    except ImportError as e:
        print(f"Warning: Could not import driada modules: {e}")
    
    # Optional imports
    try:
        import torch
        globs['torch'] = torch
    except ImportError:
        globs['torch'] = None
    
    # Mock objects for external dependencies
    class MockAuth:
        """Mock authentication for gdrive examples."""
        pass
    
    globs['auth'] = MockAuth()
    
    # Add datetime for timestamp examples
    import datetime
    globs['datetime'] = datetime
    
    return globs


def run_python_doctests(file_path: Path, timeout: int = 30) -> Dict:
    """Run docstring tests for a Python file - simplified version.
    
    Returns:
        Dict with test results
    """
    import doctest
    import tempfile
    import subprocess
    
    results = {
        'file': file_path,
        'total_blocks': 0,
        'passed': 0, 
        'failed': 0,
        'failures': []
    }
    
    # Convert file path to module name
    # e.g., src/driada/utils/data.py -> driada.utils.data
    # Find the src directory in the path
    path_parts = file_path.parts
    if 'src' in path_parts:
        src_index = path_parts.index('src')
        module_parts = path_parts[src_index + 1:]
        module_name = '.'.join(module_parts).removesuffix('.py')
    else:
        # Fallback: assume file is under src/driada
        module_name = 'driada.' + file_path.stem
    
    # Create a simple test script
    test_script = f'''#!/usr/bin/env python3
import sys
import doctest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Mock datetime for consistent timestamps
import datetime as real_datetime
class MockDateTime:
    # Preserve other datetime attributes
    def __getattr__(self, name):
        return getattr(real_datetime, name)
    
    class datetime(real_datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            # Return a fixed datetime, optionally with timezone
            dt = real_datetime.datetime(2024, 1, 15, 14, 30, 52)
            if tz:
                dt = tz.localize(dt) if hasattr(tz, 'localize') else dt.replace(tzinfo=tz)
            return dt

# Run doctests
try:
    # Import the module using its full module path to preserve package context
    import importlib
    module = importlib.import_module("{module_name}")
    
    # Mock datetime if the module uses it for consistent output
    # But only do this for modules that we know need timestamp consistency
    # For gdrive.upload, we should NOT mock datetime as it breaks the tests
    if 'datetime' in module.__dict__ and "{module_name}" in ["driada.utils.naming"]:
        module.datetime = MockDateTime()
    
    # The datetime mocking is already handled above
    
    # Run doctests with ELLIPSIS flag for flexible matching
    failures, tests = doctest.testmod(
        module, 
        verbose=False,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    )
    
    print(f"TESTS={{tests}}")
    print(f"FAILURES={{failures}}")
    sys.exit(0 if failures == 0 else 1)
except Exception as e:
    print(f"ERROR: {{str(e)}}")
    import traceback
    traceback.print_exc()
    sys.exit(2)
'''
    
    # Write and run test script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        temp_file = f.name
    
    try:
        # Run the test script with timeout
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Parse output
        output = result.stdout + result.stderr
        
        # Extract test counts
        import re
        tests_match = re.search(r'TESTS=(\d+)', output)
        failures_match = re.search(r'FAILURES=(\d+)', output)
        
        if tests_match and failures_match:
            total_tests = int(tests_match.group(1))
            total_failures = int(failures_match.group(1))
            
            results['total_blocks'] = total_tests
            results['failed'] = total_failures
            results['passed'] = total_tests - total_failures
            
            if total_failures > 0 and 'Failed example' in output:
                # Extract failure details
                results['failures'].append({
                    'test': str(file_path.name),
                    'error': 'Doctest failures (run with --verbose for details)',
                    'examples': []
                })
        else:
            # Could not parse output, treat as error
            results['total_blocks'] = 1
            results['failed'] = 1
            results['failures'].append({
                'test': str(file_path.name), 
                'error': f"Could not parse test output:\n{output}",
                'examples': []
            })
            
    except subprocess.TimeoutExpired:
        results['total_blocks'] = 1
        results['failed'] = 1
        results['failures'].append({
            'test': str(file_path.name),
            'error': f'Timeout: Tests took longer than {timeout} seconds',
            'examples': []
        })
        
    except Exception as e:
        results['total_blocks'] = 1
        results['failed'] = 1
        results['failures'].append({
            'test': str(file_path.name),
            'error': f"Error running tests: {str(e)}",
            'examples': []
        })
        
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_file)
        except:
            pass
    
    return results


def run_python_doctests_simple(file_path: Path, timeout: int = 30) -> Dict:
    """Run docstring tests for a Python file.
    
    Returns:
        Dict with test results
    """
    import doctest
    import tempfile
    import subprocess
    
    results = {
        'file': file_path,
        'total_blocks': 0,
        'passed': 0, 
        'failed': 0,
        'failures': []
    }
    
    # Create a test script that will run the doctests
    test_script = f'''
import sys
import os
import doctest
import warnings

# Suppress warnings during doctest
warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, "{Path(__file__).parent.parent / "src"}")

# Setup environment
os.environ["MPLBACKEND"] = "Agg"  # Non-interactive matplotlib

# Import common modules
import numpy as np
np.random.seed(42)  # For reproducible examples

# Prepare globals
globs = {{
    'np': np,
    'os': os,
    'sys': sys,
}}

# Import driada modules
try:
    import driada
    from driada.experiment import (
        Experiment, load_experiment, load_exp_from_aligned_data,
        save_exp_to_pickle, load_demo_experiment
    )
    from driada.information import TimeSeries, MultiTimeSeries
    from driada.utils.naming import construct_session_name
    from driada.utils.data import create_correlated_gaussian_data
    
    # Add to globals
    globs['construct_session_name'] = construct_session_name
    globs['load_demo_experiment'] = load_demo_experiment
    globs['create_correlated_gaussian_data'] = create_correlated_gaussian_data
    globs['Experiment'] = Experiment
    globs['TimeSeries'] = TimeSeries
    globs['MultiTimeSeries'] = MultiTimeSeries
    
    # Mock datetime for consistent timestamp examples
    import datetime as real_datetime
    class MockDateTime:
        datetime = real_datetime.datetime
        @staticmethod
        def now():
            return real_datetime.datetime(2024, 1, 15, 14, 30, 52)
    
    # Use mock datetime for timestamp consistency
    datetime = MockDateTime()
    globs['datetime'] = datetime
    
except Exception as e:
    print(f"Import error: {{e}}")
    sys.exit(2)

# Custom output checker for lenient matching
class LenientOutputChecker(doctest.OutputChecker):
    def check_output(self, want, got, optionflags):
        # Always use these flags
        optionflags |= doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
        
        # Handle timestamp outputs
        if '_20' in want and '_20' in got:  # Likely a timestamp
            # Just check the format is similar
            import re
            # For timestamps, check if format matches (ignore actual numbers)
            # Replace all digits with \d+ pattern
            want_pattern = re.sub(r'\d+', r'\\d+', want.strip())
            # Remove any trailing comment
            if '#' in want_pattern:
                want_pattern = want_pattern.split('#')[0].strip()
            # Create regex pattern
            pattern = '^' + want_pattern + '$'
            if re.match(pattern, got.strip()):
                return True
        
        return super().check_output(want, got, optionflags)

# Run doctests on the file
try:
    # Create custom checker
    checker = LenientOutputChecker()
    
    # Run doctests with custom checker
    failures, tests = doctest.testfile(
        "{file_path}",
        module_relative=False,
        globs=globs,
        optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS,
        verbose=False,
        report=True,
        extraglobs=globs
    )
    
    # For modules, we need to use testmod instead
    if str("{file_path}").endswith('.py'):
        # Import the module directly
        import importlib.util
        spec = importlib.util.spec_from_file_location("test_module", "{file_path}")
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Update globals with module content
            globs.update(module.__dict__)
            
            # Run module doctests
            # Create a custom test runner with our checker
            runner = doctest.DocTestRunner(checker=checker, verbose=False, optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
            finder = doctest.DocTestFinder()
            
            # Find and run tests
            tests_found = finder.find(module)
            for test in tests_found:
                runner.run(test, globs)
            
            failures = runner.failures
            tests = runner.tries
    
    # Report results
    print(f"TESTS={{tests}}")
    print(f"FAILURES={{failures}}")
    
    # Exit with appropriate code
    sys.exit(0 if failures == 0 else 1)
    
except Exception as e:
    print(f"ERROR: {{str(e)}}")
    import traceback
    traceback.print_exc()
    sys.exit(2)
'''
    
    # Write test script to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        temp_file = f.name
    
    try:
        # Run the test script with timeout
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Parse output
        output = result.stdout + result.stderr
        
        # Extract test counts
        import re
        tests_match = re.search(r'TESTS=(\d+)', output)
        failures_match = re.search(r'FAILURES=(\d+)', output)
        
        if tests_match and failures_match:
            total_tests = int(tests_match.group(1))
            total_failures = int(failures_match.group(1))
            
            results['total_blocks'] = total_tests
            results['failed'] = total_failures
            results['passed'] = total_tests - total_failures
            
            if total_failures > 0:
                # Extract error details from output
                error_lines = []
                capture = False
                for line in output.split('\n'):
                    if '***Test Failed***' in line:
                        capture = True
                    if capture:
                        error_lines.append(line)
                
                results['failures'].append({
                    'test': str(file_path.name),
                    'error': '\n'.join(error_lines) if error_lines else output,
                    'examples': []
                })
        else:
            # Could not parse output, treat as error
            results['total_blocks'] = 1
            results['failed'] = 1
            results['failures'].append({
                'test': str(file_path.name), 
                'error': f"Could not parse test output:\n{output}",
                'examples': []
            })
            
    except subprocess.TimeoutExpired:
        results['total_blocks'] = 1
        results['failed'] = 1
        results['failures'].append({
            'test': str(file_path.name),
            'error': f'Timeout: Tests took longer than {timeout} seconds',
            'examples': []
        })
        
    except Exception as e:
        results['total_blocks'] = 1
        results['failed'] = 1
        results['failures'].append({
            'test': str(file_path.name),
            'error': f"Error running tests: {str(e)}",
            'examples': []
        })
        
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_file)
        except:
            pass
    
    return results


def main():
    """Main function to run all doctests."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(
        description='Run doctests on documentation files and Python modules',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test only RST documentation files (default)
  python run_doctests.py
  
  # Test only Python module docstrings  
  python run_doctests.py --python-modules
  
  # Test both RST files and Python modules
  python run_doctests.py --all
  
  # Test specific file or directory
  python run_doctests.py path/to/file.rst
  python run_doctests.py src/driada/utils --python-modules
        """
    )
    parser.add_argument('path', nargs='?', default=None, 
                       help='Path to specific file or directory to test')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Timeout for each code block in seconds (default: 30)')
    parser.add_argument('--python-modules', action='store_true',
                       help='Test Python module docstrings instead of RST files')
    parser.add_argument('--all', action='store_true',
                       help='Test both RST files and Python module docstrings')
    args = parser.parse_args()
    
    # Find directories
    script_dir = Path(__file__).parent
    docs_dir = script_dir.parent / "docs"
    src_dir = script_dir.parent / "src"
    
    # Determine test mode
    test_rst = not args.python_modules or args.all
    test_python = args.python_modules or args.all
    
    # Initialize file lists
    rst_files = []
    python_files = []
    
    # Handle specific path if provided
    if args.path:
        test_path = Path(args.path)
        if not test_path.is_absolute():
            # Try to resolve relative path
            if test_path.exists():
                test_path = test_path.absolute()
            elif (docs_dir / test_path).exists():
                test_path = docs_dir / test_path
            elif (src_dir / test_path).exists():
                test_path = src_dir / test_path
            else:
                print(f"Error: Path not found: {test_path}")
                sys.exit(1)
        
        if not test_path.exists():
            print(f"Error: Path not found: {test_path}")
            sys.exit(1)
            
        # Determine what to test based on path
        if test_path.is_file():
            if test_path.suffix == '.rst' and test_rst:
                rst_files = [test_path]
            elif test_path.suffix == '.py' and test_python:
                python_files = [test_path]
        else:
            if test_rst:
                rst_files = list(test_path.rglob("*.rst"))
            if test_python:
                python_files = find_python_files_with_doctests(test_path)
    else:
        # No path specified, use defaults
        if test_rst and docs_dir.exists():
            rst_files = find_rst_files(docs_dir)
        if test_python and src_dir.exists():
            python_files = find_python_files_with_doctests(src_dir)
    
    # Print summary of what we're testing
    print(f"Running doctests")
    print("=" * 80)
    if test_rst:
        print(f"RST files: {len(rst_files)} files found")
    if test_python:
        print(f"Python modules: {len(python_files)} files with docstring examples")
    
    # Track overall results
    all_results = []
    total_blocks = 0
    total_passed = 0
    total_failed = 0
    
    # Test RST files
    if rst_files:
        print("\n" + "="*80)
        print("Testing RST documentation files:")
        print("="*80)
        
        for rst_file in sorted(rst_files):
            relative_path = rst_file.relative_to(docs_dir) if rst_file.is_relative_to(docs_dir) else rst_file
            
            # Skip certain files that don't contain testable code
            skip_patterns = ['changelog', 'contributing', 'license', 'index.rst']
            if any(pattern in str(relative_path).lower() for pattern in skip_patterns):
                continue
            
            print(f"\nTesting {relative_path}...", end='', flush=True)
            
            results = test_rst_file(rst_file, args.timeout)
            results['type'] = 'rst'
            
            if results['total_blocks'] == 0:
                print(" (no code blocks)")
                continue
            
            all_results.append(results)
            total_blocks += results['total_blocks']
            total_passed += results['passed']
            total_failed += results['failed']
            
            if results['failed'] == 0:
                print(f" ✅ {results['passed']}/{results['total_blocks']} passed")
            else:
                print(f" ❌ {results['passed']}/{results['total_blocks']} passed, {results['failed']} failed")
    
    # Test Python modules
    if python_files:
        print("\n" + "="*80)
        print("Testing Python module docstrings:")
        print("="*80)
        
        for py_file in sorted(python_files):
            relative_path = py_file.relative_to(src_dir) if py_file.is_relative_to(src_dir) else py_file
            
            print(f"\nTesting {relative_path}...", end='', flush=True)
            
            results = run_python_doctests(py_file, args.timeout)
            results['type'] = 'python'
            
            if results['total_blocks'] == 0:
                print(" (no doctest examples)")
                continue
                
            all_results.append(results)
            total_blocks += results['total_blocks']
            total_passed += results['passed']
            total_failed += results['failed']
            
            if results['failed'] == 0:
                print(f" ✅ {results['passed']}/{results['total_blocks']} passed")
            else:
                print(f" ❌ {results['passed']}/{results['total_blocks']} passed, {results['failed']} failed")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"Total code blocks tested: {total_blocks}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    
    # Print detailed failures
    if total_failed > 0:
        print("\n" + "=" * 80)
        print("FAILURES:")
        print("=" * 80)
        
        # Group by type for cleaner output
        rst_failures = [r for r in all_results if r.get('type') == 'rst' and r['failed'] > 0]
        py_failures = [r for r in all_results if r.get('type') == 'python' and r['failed'] > 0]
        
        if rst_failures:
            print("\nRST Documentation Failures:")
            print("-" * 26)
            for result in rst_failures:
                relative_path = result['file'].relative_to(docs_dir) if result['file'].is_relative_to(docs_dir) else result['file']
                print(f"\n{relative_path}:")
                
                for failure in result['failures']:
                    if 'line' in failure:
                        print(f"\nLine {failure['line']}:")
                        print("Code:")
                        for line in failure['code'].split('\n')[:5]:
                            print(f"  {line}")
                        if len(failure['code'].split('\n')) > 5:
                            print("  ...")
                    print(f"Error: {failure['error']}")
        
        if py_failures:
            print("\nPython Module Docstring Failures:")
            print("-" * 33)
            for result in py_failures:
                relative_path = result['file'].relative_to(src_dir) if result['file'].is_relative_to(src_dir) else result['file']
                print(f"\n{relative_path}:")
                
                for failure in result['failures']:
                    print(f"\nTest: {failure.get('test', 'unknown')}")
                    if failure.get('examples'):
                        print("Examples that failed:")
                        for example in failure['examples'][:3]:
                            print(f"  {example.strip()}")
                    print(f"Error: {failure['error']}")
    
    # Generate report
    report_path = script_dir / "doctest_report.txt"
    with open(report_path, 'w') as f:
        f.write("Documentation and Docstring Test Report\n")
        f.write("======================================\n\n")
        
        # Summary by type
        rst_results = [r for r in all_results if r.get('type') == 'rst']
        py_results = [r for r in all_results if r.get('type') == 'python']
        
        if rst_results:
            rst_total = sum(r['total_blocks'] for r in rst_results)
            rst_passed = sum(r['passed'] for r in rst_results)
            rst_failed = sum(r['failed'] for r in rst_results)
            f.write(f"RST Documentation:\n")
            f.write(f"  Files tested: {len(rst_results)}\n")
            f.write(f"  Total blocks: {rst_total}\n")
            f.write(f"  Passed: {rst_passed}\n")
            f.write(f"  Failed: {rst_failed}\n\n")
        
        if py_results:
            py_total = sum(r['total_blocks'] for r in py_results)
            py_passed = sum(r['passed'] for r in py_results)
            py_failed = sum(r['failed'] for r in py_results)
            f.write(f"Python Docstrings:\n")
            f.write(f"  Files tested: {len(py_results)}\n")
            f.write(f"  Total examples: {py_total}\n")
            f.write(f"  Passed: {py_passed}\n")
            f.write(f"  Failed: {py_failed}\n\n")
        
        f.write(f"Overall Summary:\n")
        f.write(f"  Total tests: {total_blocks}\n")
        f.write(f"  Passed: {total_passed}\n")
        f.write(f"  Failed: {total_failed}\n\n")
        
        if total_failed > 0:
            f.write("Detailed Failures:\n")
            f.write("==================\n\n")
            
            # RST failures
            rst_failures = [r for r in all_results if r.get('type') == 'rst' and r['failed'] > 0]
            if rst_failures:
                f.write("RST Documentation:\n")
                for result in rst_failures:
                    relative_path = result['file'].relative_to(docs_dir) if result['file'].is_relative_to(docs_dir) else result['file']
                    f.write(f"{relative_path}:\n")
                    
                    for failure in result['failures']:
                        if 'line' in failure:
                            f.write(f"  Line {failure['line']}: {failure['error']}\n")
                        else:
                            f.write(f"  Error: {failure['error']}\n")
                    f.write("\n")
            
            # Python failures
            py_failures = [r for r in all_results if r.get('type') == 'python' and r['failed'] > 0]
            if py_failures:
                f.write("\nPython Module Docstrings:\n")
                for result in py_failures:
                    relative_path = result['file'].relative_to(src_dir) if result['file'].is_relative_to(src_dir) else result['file']
                    f.write(f"{relative_path}:\n")
                    
                    for failure in result['failures']:
                        f.write(f"  Test: {failure.get('test', 'unknown')}\n")
                        f.write(f"  Error: {failure['error']}\n")
                    f.write("\n")
    
    print(f"\nDetailed report saved to: {report_path}")
    
    # Exit with appropriate code
    if total_failed > 0:
        print("\n⚠️  Some documentation examples failed!")
        sys.exit(1)
    else:
        print("\n✅ All documentation examples passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()