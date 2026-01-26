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

# Force unbuffered output so subprocess output reaches log file immediately
os.environ["PYTHONUNBUFFERED"] = "1"

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


def stream_log_file(log_file, last_position):
    """Read and print new content from log file.

    Returns
    -------
    tuple
        (new_position, had_output) - new file position and whether new content was found
    """
    try:
        log_file.flush()
        os.fsync(log_file.fileno())
    except (OSError, ValueError):
        pass  # File may be closed or invalid

    log_file.seek(last_position)
    new_content = log_file.read()

    if new_content:
        print(new_content, end='', flush=True)
        return log_file.tell(), True

    return last_position, False


def get_output_dir_from_args(args):
    """Extract --output-dir value from argument list.

    Returns
    -------
    Path or None
        Output directory path if found, None otherwise.
    """
    for i, arg in enumerate(args):
        if arg == '--output-dir' and i + 1 < len(args):
            return Path(args[i + 1])
        if arg.startswith('--output-dir='):
            return Path(arg.split('=', 1)[1])
    return None


def run_with_timeout(args, timeout, stall_timeout=120):
    """Run analysis script with timeout using Popen and polling.

    Uses explicit polling loop instead of subprocess.run(timeout=...) to avoid
    freezing when the child process hangs. Kills entire process tree on timeout
    or stall (no output for too long).

    Parameters
    ----------
    args : list
        Arguments to pass to run_intense_analysis.py
    timeout : int
        Wall-clock timeout in seconds
    stall_timeout : int, optional
        Kill process if no output for this many seconds (default: 120)

    Returns
    -------
    str
        'success' - completed normally
        'timeout' - killed due to wall-clock timeout
        'stall' - killed due to no output (process frozen)
        'error' - failed with error
    """
    cmd = [sys.executable, str(Path(__file__).parent / "run_intense_analysis.py")] + args

    # Determine log directory: use output_dir/logs/ if available, else system temp
    output_dir = get_output_dir_from_args(args)
    if output_dir:
        log_dir = output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
    else:
        log_dir = Path(tempfile.gettempdir())

    # Create log file with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = log_dir / f'intense_timeout_{timestamp}.log'
    log_file = open(log_path, 'w+', encoding='utf-8')

    print(f"\n{'='*60}")
    print(f"Starting analysis (timeout: {timeout}s, stall_timeout: {stall_timeout}s)")
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

        log_position = 0
        last_output_time = start_time  # Track when we last saw output

        while True:
            # Stream any new log content to console
            new_position, had_output = stream_log_file(log_file, log_position)
            if had_output:
                log_position = new_position
                last_output_time = time.time()  # Reset stall timer

            # Check if process completed
            retcode = proc.poll()
            if retcode is not None:
                # Final flush of remaining output
                stream_log_file(log_file, log_position)
                if retcode == 0:
                    return 'success'
                else:
                    return 'error'

            now = time.time()
            elapsed = now - start_time

            # Check wall-clock timeout
            if elapsed > timeout:
                stream_log_file(log_file, log_position)
                print(f"\n{'!'*60}")
                print(f"TIMEOUT after {elapsed:.0f}s - killing process tree")
                print('!'*60)
                kill_process_tree(proc.pid)
                time.sleep(2)
                return 'timeout'

            # Check stall timeout (no output for too long)
            stall_duration = now - last_output_time
            if stall_duration > stall_timeout:
                stream_log_file(log_file, log_position)
                print(f"\n{'!'*60}")
                print(f"STALL DETECTED: No output for {stall_duration:.0f}s - killing process tree")
                print('!'*60)
                kill_process_tree(proc.pid)
                time.sleep(2)
                return 'stall'

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        if 'log_position' in locals():
            stream_log_file(log_file, log_position)
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
    parser.add_argument('--stall-timeout', type=int, default=120,
                        help='Kill process if no output for this many seconds (default: 120 = 2 min)')
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

        result = run_with_timeout(remaining, args.timeout, args.stall_timeout)

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

        elif result == 'stall':
            print(f"\nProcess stalled on attempt {attempt}, restarting...")
            # Continue to retry

        # Small delay before retry
        time.sleep(2)

    print(f"\nMax retries ({args.max_retries}) exceeded!")
    return 1


if __name__ == '__main__':
    sys.exit(main())
