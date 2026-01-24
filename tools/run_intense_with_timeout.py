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

import os
import subprocess
import sys
import time
import platform
import tempfile
from pathlib import Path

# Limit BLAS threads to prevent conflicts with joblib parallelism
# Inherited by subprocesses
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

IS_WINDOWS = platform.system() == 'Windows'


def kill_process_tree(pid):
    """Kill a process and all its children.

    On Windows, uses taskkill /T /F for reliable tree killing.
    On Unix, uses psutil if available, otherwise just kills the process.
    """
    print(f"Killing process tree (PID: {pid})...")

    if IS_WINDOWS:
        # taskkill /T kills entire tree, /F forces termination
        try:
            subprocess.run(
                ['taskkill', '/T', '/F', '/PID', str(pid)],
                capture_output=True,
                timeout=30
            )
        except Exception as e:
            print(f"taskkill failed: {e}")
    else:
        # Unix: try psutil for tree kill, fallback to regular kill
        try:
            import psutil
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            parent.kill()
        except ImportError:
            # No psutil, just kill the main process
            import signal
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        except Exception as e:
            print(f"kill failed: {e}")


def run_with_timeout(args, timeout):
    """Run analysis script with timeout using Popen and polling.

    Uses explicit polling loop instead of subprocess.run(timeout=...) to avoid
    freezing when the child process hangs. Kills entire process tree on timeout.

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

    # Create temp log file to prevent pipe deadlock
    log_file = tempfile.NamedTemporaryFile(
        mode='w+',
        delete=False,
        suffix='.log',
        prefix='intense_timeout_'
    )
    log_path = log_file.name

    print(f"\n{'='*60}")
    print(f"Starting analysis (timeout: {timeout}s)")
    print(f"Command: {' '.join(cmd)}")
    print(f"Log file: {log_path}")
    print('='*60)

    try:
        # Use Popen for manual timeout control
        # Redirect output to file to prevent pipe deadlock
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT
        )
        start_time = time.time()
        poll_interval = 1.0  # Check every second
        heartbeat_interval = 60  # Print heartbeat every minute
        last_heartbeat = start_time

        while True:
            # Check if process completed
            retcode = proc.poll()
            if retcode is not None:
                if retcode == 0:
                    return 'success'
                else:
                    return 'error'

            # Check timeout
            now = time.time()
            elapsed = now - start_time
            if elapsed > timeout:
                print(f"\n{'!'*60}")
                print(f"TIMEOUT after {elapsed:.0f}s - killing process tree")
                print('!'*60)
                kill_process_tree(proc.pid)
                # Wait a moment for cleanup
                time.sleep(2)
                return 'timeout'

            # Periodic heartbeat to show wrapper is alive
            if now - last_heartbeat >= heartbeat_interval:
                remaining = timeout - elapsed
                print(f"[watchdog] {elapsed:.0f}s elapsed, {remaining:.0f}s until timeout")
                last_heartbeat = now

            # Brief sleep before next poll
            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\nInterrupted by user - killing process...")
        if proc.poll() is None:
            kill_process_tree(proc.pid)
        return 'interrupted'

    finally:
        # Close log file
        if 'log_file' in locals():
            log_file.close()


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

    # Force threading backend on Windows for stability
    if IS_WINDOWS and '--parallel-backend' not in ' '.join(remaining):
        remaining.extend(['--parallel-backend', 'threading'])
        print("Note: Auto-enabled threading backend for Windows stability")

    attempt = 0
    start_time = time.time()

    while attempt < args.max_retries:
        attempt += 1

        # Auto-enable --skip-computed on retries to resume from last checkpoint
        if attempt > 1 and '--skip-computed' not in ' '.join(remaining):
            remaining.append('--skip-computed')
            print("Note: Auto-enabled --skip-computed for retry attempt")

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
