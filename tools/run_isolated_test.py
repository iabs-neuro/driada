#!/usr/bin/env python
"""Script to run a single test file with proper torch handling."""

import sys
import os


def main():
    if len(sys.argv) < 3:
        print("Usage: run_isolated_test.py <test_file> <module>")
        sys.exit(1)

    test_file = sys.argv[1]
    module = sys.argv[2]

    # Set environment variable to disable numba caching to avoid conflicts
    os.environ["NUMBA_DISABLE_JIT"] = "1"
    os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache_test"

    # Pre-import problematic libraries to handle initialization issues

    # 1. Import numba first to avoid duplicate registration
    try:
        import numba
    except ImportError:
        pass
    except KeyError as e:
        if "duplicate registration" not in str(e):
            raise

    # 2. Import pynndescent which uses numba
    try:
        import pynndescent
    except ImportError:
        pass
    except KeyError:
        pass

    # 3. Pre-import torch to initialize it properly
    try:
        import torch

        # Force initialization
        _ = torch.tensor([1.0])
    except ImportError:
        pass  # Torch not installed
    except RuntimeError as e:
        if "_has_torch_function" not in str(e):
            raise

    # 4. Import ssqueezepy if available to handle its torch import
    try:
        import ssqueezepy
    except ImportError:
        pass

    # Now run pytest directly in this process
    import pytest

    # Ensure we're in the project root so pytest finds conftest files
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # Run with coverage
    # Important: Don't use -c /dev/null as it prevents conftest discovery
    exit_code = pytest.main(
        [
            test_file,
            f"--cov={module}",
            "--cov-report=json",
            "--cov-report=term",
            "-q",
            "--tb=short",
        ]
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
