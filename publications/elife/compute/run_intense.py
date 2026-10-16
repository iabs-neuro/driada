"""INTENSE for the FOF sessions: the selectivity results all figures of real data read.

Settings: tools/run_intense_analysis.py of the library with its defaults
(type-based feature representations, place_field rule for neurons selective to
both place and a zone), seed 42, frame rate 30. Sessions that already have
results are skipped, so an interrupted run can be restarted.

The synchronized FOF files carry no frame rate, so it is supplied here; Windows
power throttling is lifted for the whole process tree.

Run:  python publications/elife/compute/run_intense.py [session ...]
      without arguments every file in the synchronized folder is processed;
      a session is named as in the file, e.g. FOF_F28_1D
"""
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import elife_paths

elife_paths.add_tools_to_path()

DATA = elife_paths.SYNC_DIR
OUT = elife_paths.INTENSE_ROOT
QOS = os.path.join(HERE, "set_high_qos.py")
FPS = 30.0


def _lift_throttling():
    import psutil
    seen = set()
    me = psutil.Process()
    while True:
        pids = [me.pid] + [c.pid for c in me.children(recursive=True)]
        new = [str(p) for p in pids if p not in seen]
        if new:
            subprocess.run([sys.executable, QOS, *new], capture_output=True)
            seen.update(int(p) for p in new)
        time.sleep(15)


def main():
    import driada
    print(f"[INFO] driada {driada.__version__} from {os.path.dirname(driada.__file__)}", flush=True)
    if os.name == "nt":
        threading.Thread(target=_lift_throttling, daemon=True).start()

    import selectivity_dynamics.loader as loader
    from selectivity_dynamics.filters import DEFAULT_ZONE_RULE
    print(f"[INFO] tools from {os.path.dirname(loader.__file__)}; zone rule {DEFAULT_ZONE_RULE}", flush=True)
    orig_meta = loader.get_npz_metadata

    def with_fps(path):
        meta = dict(orig_meta(path))
        if meta.get('fps') is None:
            meta['fps'] = FPS
        return meta

    loader.get_npz_metadata = with_fps

    import run_intense_analysis
    sessions = sys.argv[1:]
    source = ([os.path.join(DATA, f"{s}_aligned.npz") for s in sessions] if sessions
              else ['--dir', DATA])
    sys.argv = ['run_intense_analysis.py', *source, '--output-dir', OUT,
                '--representation', 'by_type', '--seed', '42', '--skip-computed']
    run_intense_analysis.main()


if __name__ == '__main__':
    main()
