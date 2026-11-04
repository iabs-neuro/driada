# -*- coding: utf-8 -*-
"""Finish the MEC decode after an interruption, check the session files, run the whole analysis.

Safe to start again after another interruption: only sessions still without a file are decoded, and
run_xyfix.py skips the sessions it has finished. The output of the decode goes to a file, so that it
survives a restart of the machine.

usage: python -X utf8 -B mec/finish_decode_then_recompute.py --section DIR --stream-from BYTE
                                                             [--decode-python EXE] [--out DIR]
       --section: folder with traces_section.sql (the _cell__traces section of the dump) and
                  traces_section_offsets.txt; --stream-from: mec/find_resume_offset.py;
       --decode-python: the interpreter with datajoint 0.12.7, default this one;
       --out default: <[mec] recompute_dir>/parts
"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_recompute_primary as RC  # noqa: E402

P = RC.P
NEW = P.get("mec", "sessions_dir")


def say(msg, fh):
    line = f"{datetime.now():%H:%M:%S} {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def unreadable():
    bad = []
    for f in sorted(NEW.glob("*.npz")):
        try:
            d = np.load(f)
            T = d["calcium"].shape[1]
            if any(d[k].shape != (T,) for k in ("x", "y", "speed", "head")):
                bad.append(f.stem)
        except Exception:
            bad.append(f.stem)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", type=Path, required=True, help="folder with traces_section.sql and its offsets")
    ap.add_argument("--stream-from", type=int, required=True)
    ap.add_argument("--decode-python", default=sys.executable, help="interpreter with datajoint 0.12.7")
    ap.add_argument("--out", type=Path, default=P.get("mec", "recompute_dir") / "parts")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fh = open(a.out / "finish_decode.log", "a", encoding="utf-8")
    missing = sorted(RC.session_files() - {f.stem for f in NEW.glob("*.npz")})
    say(f"sessions without a file: {len(missing)}", fh)
    if missing:
        cmd = [a.decode_python, "-B", str(HERE / "decode_cohort_primary_resume.py"),
               "--dump", str(a.section / "traces_section.sql"),
               "--offsets", str(a.section / "traces_section_offsets.txt"),
               "--stream-from", str(a.stream_from), "--sessions", *missing]
        rc = subprocess.run(cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT).returncode
        say(f"decode finished with exit code {rc}", fh)
        if rc != 0:
            sys.exit(rc)
    bad = unreadable()
    if bad:
        say(f"[FAILED] unreadable or inconsistent session files: {bad}; the analysis is not started", fh)
        sys.exit(2)
    say(f"[OK] {len(list(NEW.glob('*.npz')))} session files read; starting the analysis", fh)
    cmd = [sys.executable, "-X", "utf8", "-B", str(HERE / "run_recompute_primary.py"), "--out", str(a.out)]
    sys.exit(subprocess.run(cmd, cwd=HERE).returncode)


if __name__ == "__main__":
    main()
