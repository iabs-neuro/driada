# -*- coding: utf-8 -*-
"""Run multisession_fig4_controls.py on three mouse groups in parallel, then merge.

Opts every worker out of Windows power throttling.

Each group writes its own results_<g>/sessions.jsonl under <output>/multisession,
so a restart resumes where it stopped. The merged table and summary go to
results_all/; Figure 4 reads results_all/sessions.csv once it is copied to the
cache directory as fig4_multisession_sessions.csv.
"""
import os
import subprocess
import sys
import ctypes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import elife_paths
OUT = os.path.join(elife_paths.OUTPUT_DIR, "multisession")
os.makedirs(OUT, exist_ok=True)


def high_qos(pid):
    if os.name != "nt":
        return
    class PPT(ctypes.Structure):
        _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]
    k = ctypes.WinDLL("kernel32")
    h = k.OpenProcess(0x0200, False, pid)
    st = PPT(1, 1, 0)
    k.SetProcessInformation(h, 4, ctypes.byref(st), ctypes.sizeof(st))
    k.CloseHandle(h)

GROUPS = {
    "A": ["F28", "F29", "F37", "F48", "F53", "F54"],
    "B": ["F34", "F35", "F38", "F40"],
    "C": ["F30", "F36", "F41"],
}
SETTINGS = {"SYMMETRIZATION": "union", "N_RANDOM": "10", "N_SHUF": "3", "N_SHUF2": "10000", "MIN_FILT": "10"}

procs = {}
for g, mice in GROUPS.items():
    env = dict(os.environ, OUT_DIR=os.path.join(OUT, f"results_{g}"), **SETTINGS)
    log = open(os.path.join(OUT, f"log_{g}.txt"), "a", encoding="utf-8")
    procs[g] = subprocess.Popen([sys.executable, "-W", "ignore", "multisession_fig4_controls.py", *mice],
                                cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
    high_qos(procs[g].pid)
    print(f"[INFO] group {g} started: {mice}", flush=True)
codes = {g: p.wait() for g, p in procs.items()}
print(f"[INFO] exit codes: {codes}", flush=True)

env = dict(os.environ, OUT_DIR=os.path.join(OUT, "results_all"),
           MERGE_DIRS=";".join(os.path.join(OUT, f"results_{g}") for g in GROUPS))
subprocess.run([sys.executable, "-W", "ignore", "multisession_fig4_controls.py"], cwd=HERE, env=env, check=True)
sys.exit(0 if all(c == 0 for c in codes.values()) else 1)
