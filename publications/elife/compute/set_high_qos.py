# -*- coding: utf-8 -*-
"""Opt running processes out of Windows power throttling (EcoQoS) by PID.

Usage: python set_high_qos.py <pid> [<pid> ...] [--restore]
--restore hands the decision back to Windows (ControlMask=0), which is the default state.
"""
import ctypes, sys
from ctypes import wintypes

class PPT(ctypes.Structure):
    _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

k = ctypes.WinDLL("kernel32", use_last_error=True)
k.OpenProcess.restype = wintypes.HANDLE
k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
k.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
k.CloseHandle.argtypes = [wintypes.HANDLE]
PROCESS_SET_INFORMATION = 0x0200
restore = "--restore" in sys.argv
st = PPT(1, 0, 0) if restore else PPT(1, 1, 0)
for pid in [int(a) for a in sys.argv[1:] if a != "--restore"]:
    h = k.OpenProcess(PROCESS_SET_INFORMATION, False, pid)
    if not h:
        print(f"[FAILED] {pid}: OpenProcess error {ctypes.get_last_error()}")
        continue
    ok = k.SetProcessInformation(h, 4, ctypes.byref(st), ctypes.sizeof(st))
    print(f"[{'OK' if ok else 'FAILED'}] {pid} {'restored' if restore else 'high QoS'}"
          + ("" if ok else f" error {ctypes.get_last_error()}"))
    k.CloseHandle(h)
