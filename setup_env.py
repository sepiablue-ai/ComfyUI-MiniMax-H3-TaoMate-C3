# SPDX-License-Identifier: GPL-3.0-only
"""Install C3 and locked custom nodes into an existing ComfyUI. No pip/global edits."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from dependency_check import digest, verify_dependency
from cache import cache_file, manifest
from cache_contract import require_c3_cache

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
NAME = "ComfyUI-MiniMax-H3-TaoMate-C3"
CACHE = Path("models/h3_preconverted/taomate_c3_fc1")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def isolated_python():
    executable = Path(sys.executable).resolve()
    embedded = any(executable.parent.glob("python*._pth"))
    if sys.prefix == sys.base_prefix and not embedded:
        raise RuntimeError("Global Python is refused. Use the target ComfyUI venv or portable embedded Python.")
    return executable


def run(arguments, cwd=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    subprocess.run([str(item) for item in arguments], cwd=cwd, env=env, check=True)


def files_to_install():
    rows = read(HERE / "package-files.json")["files"]
    for relative, expected in rows.items():
        path = (HERE / relative).resolve()
        if not path.is_relative_to(HERE) or not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"Package file is missing or changed: {relative}. Refresh package-files.json only after reviewing the source changes.")
    return [*rows, "package-files.json"]


def conflicts(node_roots, target):
    found = []
    for node_root in node_roots:
        if not node_root.is_dir():
            continue
        for entry in node_root.iterdir():
            if entry.name.endswith(".disabled") or entry.name.startswith(".") or entry.resolve() == target.resolve():
                continue
            if entry.is_dir() and (entry / "h3_optimizations").is_dir() and entry.name != "H3-Optimizations":
                found.append(str(entry) + " contains another h3_optimizations package")
            paths = [entry] if entry.suffix == ".py" else [entry / "__init__.py", entry / "nodes.py"]
            for path in paths:
                if not path.is_file():
                    continue
                text = path.read_text(encoding="utf-8-sig", errors="replace")
                if any(name in text for name in ("H3C2HeldFFN8", "H3C3HeldOut", "H3TaoMateW4A4Loader")):
                    found.append(str(entry) + " may register duplicate C3 node IDs")
                    break
    return sorted(set(found))


def dependency_locations(entry, node_roots, default_root):
    matches = list(dict.fromkeys((root / entry["name"]).resolve() for root in node_roots
                                if (root / entry["name"]).exists()))
    if len(matches) > 1:
        raise RuntimeError(f"Multiple {entry['name']} directories: {matches}. Disable duplicates before setup.")
    return matches[0] if matches else default_root / entry["name"]


def install_dependency(entry, destination):
    if destination.exists():
        problems = verify_dependency(destination, entry)
        if problems:
            raise RuntimeError(f"Existing {destination} differs from locked {entry['commit']}: {problems[:8]}. "
                               "Stop ComfyUI, back it up outside active custom_nodes (or rename with .disabled), then rerun. Setup never resets existing repos.")
        print(f"Dependency verified: {entry['name']} {entry['commit']}", flush=True)
        return
    # The staging folder ends in .disabled, and publication is one same-volume rename.
    with tempfile.TemporaryDirectory(prefix="c3-install-", suffix=".disabled", dir=destination.parent) as temporary:
        stage = Path(temporary) / entry["name"]
        run(["git", "-c", "core.longpaths=true", "init", stage])
        run(["git", "-C", stage, "config", "core.longpaths", "true"])
        run(["git", "-C", stage, "remote", "add", "origin", entry["url"]])
        run(["git", "-C", stage, "fetch", "--depth", "1", "origin", entry["commit"]])
        run(["git", "-C", stage, "checkout", "--detach", "FETCH_HEAD"])
        problems = verify_dependency(stage, entry)
        if problems:
            raise RuntimeError(f"Downloaded dependency failed integrity verification: {entry['name']} {problems[:8]}")
        if destination.exists():
            raise RuntimeError(f"Destination appeared during setup: {destination}")
        stage.rename(destination)
    print(f"Installed dependency: {entry['name']} {entry['commit']}", flush=True)


def copy_package(target, files):
    if target.resolve() == HERE:
        return
    if target.exists():
        differences = [relative for relative in files
                       if not (target / relative).is_file() or digest(target / relative) != digest(HERE / relative)]
        if differences:
            raise RuntimeError(f"Existing package differs: {target}: {differences[:8]}. Back it up and remove it from active custom_nodes before reinstalling.")
        return
    with tempfile.TemporaryDirectory(prefix="c3-package-", suffix=".disabled", dir=target.parent) as temporary:
        stage = Path(temporary) / NAME
        for relative in files:
            out = stage / relative
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(HERE / relative, out)
        stage.rename(target)


def check_cache(directory, source, full=False):
    data = manifest(directory, source=source, require_gate=False)
    require_c3_cache(data)
    if full:
        for row in data["fc1"]:
            if digest(cache_file(directory, row)) != row["sha256"]:
                raise RuntimeError(f"FC1 shard hash mismatch: {row['file']}")
    return data


def reuse_cache(source, target, model):
    data = check_cache(source, model, full=True)
    if target.exists():
        check_cache(target, model, full=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="c3-cache-", dir=target.parent) as temporary:
        stage = Path(temporary) / target.name
        stage.mkdir()
        for row in data["fc1"]:
            origin, destination = cache_file(source, row), stage / row["file"]
            try:
                os.link(origin, destination)
            except OSError:
                shutil.copyfile(origin, destination)
        data = dict(data, mode="fc1_only")
        data.pop("gate", None)
        data.pop("gate_source", None)
        (stage / "manifest.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        check_cache(stage, model, full=True)
        stage.rename(target)


def find_models(folder_paths, verify=False):
    results, problems = {}, []
    for row in read(HERE / "models.json")["models"]:
        names = [name for name in folder_paths.get_filename_list(row["category"])
                 if Path(name).name == row["name"]]
        if len(names) != 1:
            problems.append(f"{row['name']}: expected one model entry, found {len(names)}; see docs/MODELS.md")
            continue
        path = Path(folder_paths.get_full_path_or_raise(row["category"], names[0]))
        if path.stat().st_size != row["bytes"]:
            problems.append(f"{row['name']}: wrong file size ({path.stat().st_size}, expected {row['bytes']}). Same-named VAE files can differ; see docs/MODELS.md")
            continue
        if verify and digest(path) != row["sha256"]:
            problems.append(f"{row['name']}: SHA-256 mismatch")
            continue
        results[row["name"]] = {"path": path, "workflow_name": names[0], "category": row["category"]}
    return results, problems


def check_environment(root):
    # Read-only metadata inspection; importing Comfy's execution engine is an explicit separate check.
    expected = read(HERE / "compatibility.json")["python_packages"]
    for package, version in expected.items():
        try:
            current = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"{package} is missing from this ComfyUI Python. See compatibility.json; no pip command has been run.") from exc
        if current != version:
            print(f"NOTE: {package}={current}; validated version={version}. Run tools/check_runtime.py --native before generating.", flush=True)
    for relative in ("comfy/quant_ops.py", "comfy/ldm/minimax/model.py", "comfy_extras/nodes_minimax_h3.py"):
        if not (root / relative).is_file():
            raise RuntimeError(f"ComfyUI lacks {relative}. Use a MiniMax H3-capable ComfyUI (validated 0.37.0).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--cache-source", type=Path)
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--verify-models", action="store_true")
    args = parser.parse_args()
    if args.check_only and (args.build_cache or args.cache_source):
        parser.error("--check-only is read-only; run cache installation separately")
    if args.build_cache and args.cache_source:
        parser.error("Choose --build-cache or --cache-source, not both")
    executable = isolated_python()
    root = args.comfy_root.resolve()
    if not (root / "main.py").is_file() or not (root / "comfy/sd.py").is_file():
        parser.error("--comfy-root must contain main.py and comfy/sd.py")
    # folder_paths creates input/ at import time. Do not import it during a
    # read-only check if that would create a directory in a fresh checkout.
    if args.check_only and not (root / "input").is_dir():
        raise RuntimeError("CHECK NOT READY: ComfyUI input directory is absent. Run setup without -CheckOnly first. No files written.")
    files = files_to_install()
    check_environment(root)
    sys.path.insert(0, str(root))
    import folder_paths
    from utils.extra_config import load_extra_path_config
    config = root / "extra_model_paths.yaml"
    if config.is_file():
        load_extra_path_config(str(config))
    node_root = root / "custom_nodes"
    node_roots = list(dict.fromkeys([node_root, *map(Path, folder_paths.get_folder_paths("custom_nodes"))]))
    target = node_root / NAME
    duplicate = conflicts(node_roots, target)
    if duplicate:
        raise RuntimeError("Duplicate/legacy nodes detected:\n  " + "\n  ".join(duplicate) +
                           "\nStop ComfyUI, back up those entries, rename them with .disabled, then rerun. No entries were removed.")
    dependencies = [(row, dependency_locations(row, node_roots, node_root))
                    for row in read(HERE / "dependency-lock.json")["dependencies"]]
    missing_dependencies = []
    for entry, destination in dependencies:
        if destination.exists():
            problems = verify_dependency(destination, entry)
            if problems:
                raise RuntimeError(f"Incompatible dependency: {destination}: {problems[:8]}. Expected {entry['commit']}. Preserve/disable it before installing the pinned version.")
        else:
            missing_dependencies.append(entry["name"])
    if missing_dependencies and not args.check_only and shutil.which("git") is None:
        raise RuntimeError("Git for Windows is needed to fetch the locked custom-node revisions. Install Git or place the verified dependency directories manually.")
    if target.exists() and target.resolve() != HERE:
        differences = [relative for relative in files
                       if not (target / relative).is_file() or digest(target / relative) != digest(HERE / relative)]
        if differences:
            raise RuntimeError(f"Existing C3 package differs: {differences[:8]}. Preserve/disable the old package first.")
    models, model_problems = find_models(folder_paths, args.verify_models)
    diffusion_name = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    model = models.get(diffusion_name, {}).get("path")
    cache = root / CACHE
    # Match the W4A4-VSA installer: prepared models enable the first offline conversion.
    build_cache = args.build_cache or (not args.check_only and not args.cache_source
                                      and not model_problems and not cache.exists())
    if cache.exists():
        check_cache(cache, model, full=True)
    if args.cache_source or args.build_cache:
        if model is None:
            raise RuntimeError("The exact C3 diffusion model is required for cache setup. " + "; ".join(model_problems))
        if args.cache_source:
            check_cache(args.cache_source.resolve(), model, full=True)
    image = root / "input/taomate_c3/first_frame.png"
    if image.exists() and digest(image) != digest(HERE / "examples/first_frame.png"):
        raise RuntimeError(f"Sample input already differs: {image}; setup will not overwrite it.")
    print(f"Python: {executable}\nComfyUI: {root}\nPackage: {target}", flush=True)
    ready_problems = [*model_problems]
    if not cache.exists():
        ready_problems.append("FC1 cache missing; use -CacheSource <existing fl2va cache> or -BuildCache")
    if args.check_only:
        if not target.is_dir():
            ready_problems.append("C3 package is not installed")
        ready_problems += [f"Missing custom node: {name}" for name in missing_dependencies]
        if not image.is_file():
            ready_problems.append("Bundled workflow input is not installed")
        if ready_problems:
            raise RuntimeError("CHECK NOT READY:\n  " + "\n  ".join(ready_problems))
        print("CHECK PASS: pinned nodes, models (size; SHA if -VerifyModels), all FC1 shard hashes, sample input. No files written; no GPU execution.", flush=True)
        return
    node_root.mkdir(parents=True, exist_ok=True)
    for entry, destination in dependencies:
        install_dependency(entry, destination)
    copy_package(target, files)
    image.parent.mkdir(parents=True, exist_ok=True)
    if not image.exists():
        shutil.copyfile(HERE / "examples/first_frame.png", image)
    if args.cache_source and not cache.exists():
        reuse_cache(args.cache_source.resolve(), cache, model)
    elif build_cache and not cache.exists():
        print("Preparing the first FC1 cache on the selected GPU (50 layers; no video generation).", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        # Convert in a new directory and publish only its completed manifest.
        with tempfile.TemporaryDirectory(prefix="c3-convert-", dir=cache.parent) as temporary:
            stage = Path(temporary) / cache.name
            run([executable, "-B", "-X", "utf8", HERE / "convert.py", "--comfy-root", root,
                 "--fc1-only", "--gpu", args.gpu, "--model", model, "--output", stage])
            check_cache(stage, model, full=True)
            stage.rename(cache)
    for item in models.values():
        if item["workflow_name"] != item["path"].name:
            print(f"Select model in workflow: {item['category']} = {item['workflow_name']}", flush=True)
    print(f"NODE INSTALL COMPLETE: {target}\nRestart ComfyUI and open workflows/TaoMate_C3.json.", flush=True)
    ready_problems = [*model_problems]
    if not cache.exists():
        ready_problems.append("FC1 cache missing: run setup.bat -BuildCache, or -CacheSource <existing fl2va cache>")
    if ready_problems:
        print("GENERATION NOT READY:\n  " + "\n  ".join(ready_problems), flush=True)
    else:
        print("Assets ready. Run tools/check_runtime.py --comfy-root <ComfyUI> --native to check node registration and the small native INT4 path. No video was generated by setup.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"Setup stopped: {exc}", file=sys.stderr)
        raise SystemExit(1)
