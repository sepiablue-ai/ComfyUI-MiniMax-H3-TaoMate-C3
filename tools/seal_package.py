# SPDX-License-Identifier: GPL-3.0-only
"""Maintainer: record the reviewed distribution files after editing; never install anything."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP = {".git", "__pycache__", ".venv", ".pytest_cache", ".setup", "tmp"}


def main():
    files = {}
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if any(part in SKIP for part in relative.parts) or not path.is_file() or relative.as_posix() == "package-files.json":
            continue
        if path.suffix.lower() in {".pyc", ".mp4", ".webm", ".wav", ".safetensors", ".pth", ".pt", ".log"}:
            raise RuntimeError(f"Unexpected artifact in distribution: {relative}")
        files[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT / "package-files.json").write_text(json.dumps({"files": files}, indent=2) + "\n", encoding="utf-8")
    print(f"Sealed {len(files)} distribution files")


if __name__ == "__main__":
    main()
