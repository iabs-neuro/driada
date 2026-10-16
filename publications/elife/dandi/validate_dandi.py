"""Check the packed NWB files the way the DANDI archive will, without contacting it.

Three checks, none of which needs an account or a key:

1. NWB schema validation (``pynwb.validate``) and NWB Inspector with the
   ``dandi`` configuration on every file. The gate: no error, no CRITICAL and
   no BEST_PRACTICE_VIOLATION message. Suggestions are listed, not gated.
2. ``dandi validate`` on every file. Outside a Dandiset it always reports
   ``DANDI.NO_DANDISET_FOUND``; that one is counted apart, the rest is gated.
3. Dry run of ``dandi organize`` over all files: the names the archive client
   would give must be the names the files already have.

Run:  python publications/elife/dandi/validate_dandi.py [--out <folder>] [--report <folder>]
"""
import argparse
import contextlib
import csv
import glob
import io
import logging
import os
import sys
import tempfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import elife_paths

GATED = {"ERROR", "PYNWB_VALIDATION", "CRITICAL", "BEST_PRACTICE_VIOLATION"}
NOT_IN_DANDISET = "DANDI.NO_DANDISET_FOUND"


def inspect_files(paths, root):
    """Run schema validation and the inspector; return one row per message."""
    from nwbinspector import inspect_nwbfile, load_config
    from pynwb import validate as validate_schema

    config = load_config("dandi")
    rows = []
    for path in paths:
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        result = validate_schema(paths=[path])
        errors = result[0] if isinstance(result, tuple) else result
        for err in errors:
            rows.append({"file": rel, "tool": "pynwb.validate", "level": "PYNWB_VALIDATION",
                         "check": type(err).__name__, "location": "", "message": str(err)})
        for msg in inspect_nwbfile(nwbfile_path=path, config=config):
            rows.append({"file": rel, "tool": "nwbinspector", "level": msg.importance.name,
                         "check": msg.check_function_name, "location": msg.location or "",
                         "message": msg.message})
    return rows


def dandi_validate_files(paths, root):
    """Run the validator of the archive client; return one row per result."""
    from dandi.validate import validate

    rows = []
    for path in paths:
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        for res in validate(path):
            rows.append({"file": rel, "tool": "dandi.validate",
                         "level": getattr(res.severity, "name", str(res.severity)),
                         "check": res.id, "location": "", "message": str(res.message)})
    return rows


def organize_dry_run(paths, root):
    """Ask ``dandi organize`` how it would name the files.

    Returns
    -------
    dict
        Current relative path -> proposed relative path. A file the organizer
        leaves where it is maps to its own path.
    str
        Everything the organizer printed.
    """
    from dandi.consts import dandiset_metadata_file
    from dandi.organize import FileOperationMode, organize

    # The organizer needs a Dandiset folder to plan against; a stub in a
    # temporary place keeps the data folder free of files that are not data.
    with tempfile.TemporaryDirectory() as stub:
        with open(os.path.join(stub, dandiset_metadata_file), "w", encoding="utf-8") as fh:
            fh.write("identifier: '000000'\nname: local dry run\n")
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        logger = logging.getLogger("dandi")
        logger.addHandler(handler)
        old_level = logger.level
        logger.setLevel(logging.INFO)
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                organize([os.path.abspath(p) for p in paths], dandiset_path=stub,
                         files_mode=FileOperationMode.DRY)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)
    text = buf.getvalue()

    proposed = {}
    for line in text.splitlines():
        if line.startswith("DRY:") and " -> " in line:
            src, dst = line[len("DRY:"):].split(" -> ", 1)
            rel = os.path.relpath(src.strip(), root).replace(os.sep, "/")
            proposed[rel] = dst.strip().replace(os.sep, "/")
    return proposed, text


def main():
    default_out = os.environ.get("DRIADA_ELIFE_DANDI",
                                 os.path.join(elife_paths.DATA_ROOT, "dandi_nwb"))
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=default_out, help="folder with the packed NWB files")
    parser.add_argument("--report", default=elife_paths.OUTPUT_DIR,
                        help="folder for validation_messages.csv and organize_dry_run.log")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.out, "sub-*", "*.nwb")))
    if not paths:
        print(f"[FAILED] no NWB files under {args.out}")
        return 1
    os.makedirs(args.report, exist_ok=True)

    rows = inspect_files(paths, args.out) + dandi_validate_files(paths, args.out)
    with open(os.path.join(args.report, "validation_messages.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "tool", "level", "check", "location", "message"])
        writer.writeheader()
        writer.writerows(rows)

    ok = True
    print(f"files checked: {len(paths)}")
    for tool in ("pynwb.validate", "nwbinspector", "dandi.validate"):
        counts = Counter((r["level"], r["check"]) for r in rows if r["tool"] == tool)
        print(f"{tool}: {sum(counts.values())} messages")
        for (level, check), n in sorted(counts.items()):
            files = len({r["file"] for r in rows if r["tool"] == tool and r["check"] == check})
            print(f"    {level:<26} {check:<48} {n} messages in {files} files")

    blocking = [r for r in rows if r["tool"] != "dandi.validate" and r["level"] in GATED]
    print(f"{'[OK]' if not blocking else '[FAILED]'} inspector gate: "
          f"{len(blocking)} errors, critical messages and best-practice violations")
    ok &= not blocking

    # The client reports inspector violations as warnings; only its errors
    # stop an upload, and one of them is the missing Dandiset itself.
    dandi_errors = [r for r in rows if r["tool"] == "dandi.validate"
                    and r["level"] == "ERROR" and r["check"] != NOT_IN_DANDISET]
    outside = sum(r["check"] == NOT_IN_DANDISET for r in rows)
    print(f"{'[OK]' if not dandi_errors else '[FAILED]'} dandi validate: {len(dandi_errors)} errors "
          f"besides {outside} x {NOT_IN_DANDISET} (expected: the folder is not a Dandiset yet)")
    ok &= not dandi_errors

    proposed, text = organize_dry_run(paths, args.out)
    with open(os.path.join(args.report, "organize_dry_run.log"), "w", encoding="utf-8") as fh:
        fh.write(text)
    current = [os.path.relpath(p, args.out).replace(os.sep, "/") for p in paths]
    renamed = {rel: proposed[rel] for rel in current if proposed.get(rel, rel) != rel}
    for rel, new in sorted(renamed.items()):
        print(f"    organize would rename {rel} -> {new}")
    print(f"{'[OK]' if not renamed else '[FAILED]'} dandi organize dry run: "
          f"{len(current) - len(renamed)} of {len(current)} names kept, "
          f"{len(proposed)} files mentioned by the organizer")
    # A file the organizer did not mention was not planned at all, which is
    # not the same as planned under its present name.
    ok &= not renamed and len(proposed) == len(current)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
