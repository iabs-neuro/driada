#!/usr/bin/env python
"""
Wrapper script that runs INTENSE analysis with automatic timeout recovery.

If the analysis freezes, kills the process and restarts with --skip-computed
to resume from the last saved experiment.

Usage
-----
    # Run with 30 minute timeout per attempt
    python tools/run_intense_with_timeout.py --dir "DRIADA data" --output-dir INTENSE --timeout 1800

    # All arguments after --timeout are passed to run_intense_analysis.py
    python tools/run_intense_with_timeout.py --timeout 3600 --dir "DRIADA data" --output-dir INTENSE --parallel-backend threading
"""

import subprocess
import sys
import time
from pathlib import Path


def run_with_timeout(args, timeout):
    """Run analysis script with timeout.

    Parameters
    ----------
    args : list
        Arguments to pass to run_intense_analysis.py
    timeout : int
        Timeout in seconds

    Returns
    -------
    str
        'success' - completed normally
        'timeout' - killed due to timeout
        'error' - failed with error
    """
    cmd = [sys.executable, str(Path(__file__).parent / "run_intense_analysis.py")] + args

    print(f"\n{'='*60}")
    print(f"Starting analysis (timeout: {timeout}s)")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)

    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            # Don't capture output - let it print to console
        )
        if result.returncode == 0:
            return 'success'
        else:
            return 'error'
    except subprocess.TimeoutExpired:
        print(f"\n{'!'*60}")
        print("TIMEOUT - process killed, will restart with --skip-computed")
        print('!'*60)
        return 'timeout'
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 'interrupted'


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Run INTENSE analysis with automatic timeout recovery',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--timeout', type=int, default=1800,
                        help='Timeout in seconds per attempt (default: 1800 = 30 min)')
    parser.add_argument('--max-retries', type=int, default=100,
                        help='Maximum number of restart attempts (default: 100)')

    # Parse only known args, pass rest to run_intense_analysis.py
    args, remaining = parser.parse_known_args()

    # Ensure --skip-computed is in the args for restarts
    if '--skip-computed' not in remaining:
        remaining.append('--skip-computed')

    attempt = 0
    start_time = time.time()

    while attempt < args.max_retries:
        attempt += 1
        print(f"\n{'#'*60}")
        print(f"ATTEMPT {attempt}/{args.max_retries}")
        print('#'*60)

        result = run_with_timeout(remaining, args.timeout)

        if result == 'success':
            elapsed = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"COMPLETED SUCCESSFULLY after {attempt} attempt(s)")
            print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
            print('='*60)
            return 0

        elif result == 'interrupted':
            print("\nExiting due to user interrupt")
            return 1

        elif result == 'error':
            print(f"\nScript exited with error, retrying...")
            # Continue to retry

        elif result == 'timeout':
            print(f"\nTimeout on attempt {attempt}, restarting...")
            # Continue to retry

        # Small delay before retry
        time.sleep(2)

    print(f"\nMax retries ({args.max_retries}) exceeded!")
    return 1


if __name__ == '__main__':
    sys.exit(main())
