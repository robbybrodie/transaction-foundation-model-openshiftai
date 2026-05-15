#!/usr/bin/env python3
"""
zippy.py — zip only git-tracked files.

Uses `git ls-files` so the archive contains exactly what git knows about:
no git history, no untracked files, no gitignored paths.

Usage:
    python zippy.py                   # -> <reponame>-YYYYMMDD-HHMMSS.zip
    python zippy.py archive.zip       # -> archive.zip
    python zippy.py /tmp/archive.zip  # -> /tmp/archive.zip
"""

import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path


def git_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True,
    )
    return Path(result.stdout.strip())


def tracked_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True, text=True, check=True,
        cwd=root,
    )
    return [root / line for line in result.stdout.splitlines() if line]


def main() -> None:
    try:
        root = git_root()
    except subprocess.CalledProcessError:
        sys.exit("error: not inside a git repository")

    if len(sys.argv) > 1:
        out = Path(sys.argv[1])
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = Path(f"{root.name}-{timestamp}.zip")

    files = tracked_files(root)
    if not files:
        sys.exit("error: git ls-files returned no files")

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            if not f.exists():
                print(f"  skip (missing): {f.relative_to(root)}")
                continue
            arcname = f.relative_to(root)
            zf.write(f, arcname)
            print(f"  added: {arcname}")

    size_mb = out.stat().st_size / 1_048_576
    print(f"\n{len(zf.namelist())} files → {out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
