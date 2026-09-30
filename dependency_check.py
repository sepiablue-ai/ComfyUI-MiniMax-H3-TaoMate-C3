# SPDX-License-Identifier: GPL-3.0-only
"""Pin the upstream internals used by the C3 wrapper, without load-order coupling."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_dependency(root, entry):
    root = Path(root).resolve()
    errors = []
    for relative, expected in entry["files"].items():
        path = root / relative
        if not path.is_file():
            errors.append(f"missing {relative}")
        elif digest(path) != expected:
            # Git for Windows can check out the same upstream text as LF or CRLF.
            # Binary fingerprints, including the native DLL, remain byte-exact.
            if path.suffix == ".py" or path.name == "BUILD_ID":
                data = path.read_bytes().replace(b"\r\n", b"\n")
                variants = (data, data.replace(b"\n", b"\r\n"))
                if any(hashlib.sha256(value).hexdigest() == expected for value in variants):
                    continue
            errors.append(f"changed {relative}")
    # This optional file takes precedence over the pinned DLL in upstream code.
    if (root / "native/library_path.txt").exists():
        errors.append("native/library_path.txt overrides the pinned native binary")
    return errors


@lru_cache(maxsize=1)
def require_h3_runtime():
    try:
        import h3_optimizations
    except ImportError as exc:
        raise RuntimeError("H3-Optimizations is not loaded. Run this package's setup.bat and restart ComfyUI.") from exc
    root = Path(h3_optimizations.__file__).resolve().parent.parent
    entry = json.loads((HERE / "dependency-lock.json").read_text(encoding="utf-8"))["dependencies"][0]
    problems = verify_dependency(root, entry)
    if problems:
        raise RuntimeError(f"C3 requires H3-Optimizations {entry['commit']}. "
                           f"Loaded from {root}: {'; '.join(problems[:8])}. "
                           "Run setup.bat -CheckOnly; do not mix a different H3 package or local DLL.")
    return root
