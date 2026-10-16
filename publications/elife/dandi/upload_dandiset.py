"""Upload the NWB files to the draft of the Dandiset and check what arrived.

The API key is read from --key and handed to `dandi upload` through the
environment, never printed. After the upload every local file must be present
on the server with the same size, and the server must hold nothing else.

    python upload_dandiset.py --key <file> --nwb-dir <folder with dandiset.yaml>
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from dandi.dandiapi import DandiAPIClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--nwb-dir", required=True)
    ap.add_argument("--instance", default="dandi")
    args = ap.parse_args()

    root = Path(args.nwb_dir)
    ident = None
    for line in (root / "dandiset.yaml").read_text(encoding="utf-8").splitlines():
        if line.startswith("identifier:"):
            ident = line.split(":", 1)[1].strip().strip("'\"")
    token = Path(args.key).read_text().strip()

    env = dict(os.environ, DANDI_API_KEY=token, PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, "-c", "from dandi.cli.command import main; main()", "upload", "-i", args.instance,
           "--existing", "refresh"]
    print("[RUN]", " ".join(cmd), "in", root, flush=True)
    rc = subprocess.call(cmd, cwd=root, env=env)
    print("[INFO] dandi upload exit code", rc, flush=True)

    local = {p.relative_to(root).as_posix(): p.stat().st_size
             for p in root.rglob("*.nwb")}
    client = DandiAPIClient.for_dandi_instance(args.instance, token=token)
    ds = client.get_dandiset(ident, "draft")
    remote = {a.path: a.size for a in ds.get_assets()}

    missing = sorted(set(local) - set(remote))
    extra = sorted(set(remote) - set(local))
    wrong = sorted(p for p in set(local) & set(remote) if local[p] != remote[p])
    print(f"[INFO] local {len(local)} files, server {len(remote)} assets")
    for name, lst in (("missing on server", missing), ("extra on server", extra),
                      ("size differs", wrong)):
        for p in lst:
            print(f"[FAILED] {name}: {p}")
    ok = rc == 0 and not (missing or extra or wrong)
    print("[OK] upload verified" if ok else "[FAILED] upload not complete")
    print("URL:", f"{client.dandi_instance.gui}/dandiset/{ident}/draft")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
