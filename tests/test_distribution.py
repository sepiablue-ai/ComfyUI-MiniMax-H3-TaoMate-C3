# SPDX-License-Identifier: GPL-3.0-only
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import shutil
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup_env
from dependency_check import verify_dependency
from migrate_workflow import migrate
from cache_contract import C3_MODEL_SHA256


class InstallationTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(os.environ.get("C3_TEST_TMP", ROOT / "tmp"))
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="c3-test-", dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_reject_global_python(self):
        with patch.object(sys, "prefix", sys.base_prefix), patch.object(Path, "glob", return_value=iter([])):
            with self.assertRaisesRegex(RuntimeError, "Global Python"):
                setup_env.isolated_python()

    def test_dependency_modification_and_native_override_detected(self):
        file = self.root / "__init__.py"
        file.write_bytes(b"original")
        entry = {"files": {file.name: hashlib.sha256(b"original").hexdigest()}}
        self.assertEqual(verify_dependency(self.root, entry), [])
        file.write_bytes(b"modified")
        self.assertIn("changed __init__.py", verify_dependency(self.root, entry))
        (self.root / "native").mkdir()
        (self.root / "native/library_path.txt").write_text("other.dll")
        self.assertEqual(len(verify_dependency(self.root, entry)), 2)

    def test_conflicting_dependency_never_fetches_or_overwrites(self):
        file = self.root / "__init__.py"
        file.write_bytes(b"user edit")
        entry = {"name": "dependency", "commit": "test", "files": {file.name: "wrong"}}
        with patch.object(setup_env, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "differs"):
                setup_env.install_dependency(entry, self.root)
        run.assert_not_called()
        self.assertEqual(file.read_bytes(), b"user edit")

    def test_dependency_accepts_git_line_endings_but_not_code_or_binary_changes(self):
        file = self.root / "__init__.py"
        original = b"value = 1\r\n"
        entry = {"files": {file.name: hashlib.sha256(original).hexdigest()}}
        for data in (original, original.replace(b"\r\n", b"\n")):
            file.write_bytes(data)
            self.assertEqual(verify_dependency(self.root, entry), [])
        file.write_bytes(b"value = 2\n")
        self.assertEqual(verify_dependency(self.root, entry), ["changed __init__.py"])
        binary = self.root / "native.dll"
        binary.write_bytes(original.replace(b"\r\n", b"\n"))
        entry = {"files": {binary.name: hashlib.sha256(original).hexdigest()}}
        self.assertEqual(verify_dependency(self.root, entry), ["changed native.dll"])

    @unittest.skipUnless(shutil.which("git"), "Git is required for checkout verification")
    def test_git_checkout_preserves_package_fingerprints(self):
        repository = self.root / "repo"
        repository.mkdir()
        files = setup_env.files_to_install()
        for relative in files:
            target = repository / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        def git(*arguments):
            subprocess.run(["git", "-c", "core.autocrlf=true", "-c", "core.excludesFile=",
                            "-C", str(repository), *arguments], check=True, capture_output=True)
        git("init", "--quiet")
        git("add", ".")
        checkout = self.root / "checkout"
        checkout.mkdir()
        git("checkout-index", "--all", "--prefix=" + checkout.as_posix() + "/")
        with patch.object(setup_env, "HERE", checkout):
            self.assertEqual(set(setup_env.files_to_install()), set(files))
        for relative in files:
            self.assertEqual((ROOT / relative).read_bytes(), (checkout / relative).read_bytes(), relative)

    def test_failed_fetch_is_cleaned_and_never_published(self):
        entry = {"name": "dependency", "url": "https://invalid.example/repo.git", "commit": "test"}
        with patch.object(setup_env, "run", side_effect=RuntimeError("network interrupted")):
            with self.assertRaisesRegex(RuntimeError, "network interrupted"):
                setup_env.install_dependency(entry, self.root / "dependency")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_duplicate_legacy_pack_detected_but_disabled_pack_ignored(self):
        target = self.root / setup_env.NAME
        legacy = self.root / "ComfyUI-Sayaka-H3-C2C3"
        legacy.mkdir()
        (legacy / "__init__.py").write_text('NODE_CLASS_MAPPINGS={"H3C3HeldOut": object}')
        (legacy / "h3_optimizations").mkdir()
        self.assertTrue(setup_env.conflicts([self.root], target))
        legacy.rename(legacy.with_suffix(".disabled"))
        self.assertEqual(setup_env.conflicts([self.root], target), [])

    def test_package_reinstall_preserves_user_edits(self):
        target = self.root / "installed"
        target.mkdir()
        (target / "README.md").write_text("user edit")
        with self.assertRaisesRegex(RuntimeError, "differs"):
            setup_env.copy_package(target, ["README.md"])
        self.assertEqual((target / "README.md").read_text(), "user edit")

    def make_cache(self):
        source = self.root / "元 cache"
        source.mkdir()
        rows = []
        for i in range(50):
            file = source / f"block_{i:02d}.safetensors"
            file.write_bytes(f"test shard {i}".encode())
            rows.append({"block": i, "file": file.name, "file_size": file.stat().st_size,
                         "sha256": hashlib.sha256(file.read_bytes()).hexdigest()})
        data = {"format": "h3-streaming-v2-cache-1", "complete": True, "fc1": rows,
                "model": {"sha256": C3_MODEL_SHA256},
                "gate": {"file": "unused_gate"}, "gate_source": {"note": "must not be copied"}}
        (source / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
        return source

    def test_fc1_only_reuse_no_gate_and_source_manifest_unchanged(self):
        source = self.make_cache()
        before = (source / "manifest.json").read_bytes()
        target = self.root / "新しい cache"
        setup_env.reuse_cache(source, target, None)
        data = json.loads((target / "manifest.json").read_text())
        self.assertEqual(data["mode"], "fc1_only")
        self.assertNotIn("gate", data)
        self.assertNotIn("gate_source", data)
        self.assertEqual(len(list(target.iterdir())), 51)
        self.assertEqual((source / "manifest.json").read_bytes(), before)
        setup_env.check_cache(target, None, full=True)

    def test_corrupt_cache_never_published(self):
        source = self.make_cache()
        file = source / "block_03.safetensors"
        file.write_bytes(b"x" * file.stat().st_size)
        target = self.root / "output"
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            setup_env.reuse_cache(source, target, None)
        self.assertFalse(target.exists())

    def test_cache_path_traversal_rejected(self):
        source = self.make_cache()
        data = json.loads((source / "manifest.json").read_text())
        data["fc1"][0]["file"] = "../outside.safetensors"
        (source / "manifest.json").write_text(json.dumps(data))
        with self.assertRaisesRegex(RuntimeError, "directly inside"):
            setup_env.check_cache(source, None)

    def test_premerged_lora_cache_rejected_despite_same_base_model(self):
        source = self.make_cache()
        data = json.loads((source / "manifest.json").read_text())
        data["premerged_lora"] = {"source_sha256": "same TaoMate used in graph"}
        (source / "manifest.json").write_text(json.dumps(data))
        with self.assertRaisesRegex(RuntimeError, "LoRA twice"):
            setup_env.reuse_cache(source, self.root / "output", None)
        self.assertFalse((self.root / "output").exists())

    def test_ref2va_cache_rejected(self):
        source = self.make_cache()
        data = json.loads((source / "manifest.json").read_text())
        data["model"]["sha256"] = "different base model"
        (source / "manifest.json").write_text(json.dumps(data))
        with self.assertRaisesRegex(RuntimeError, "unmerged FL2VA"):
            setup_env.check_cache(source, None)

    def test_check_only_does_not_create_missing_comfy_input(self):
        (self.root / "comfy").mkdir()
        (self.root / "comfy/sd.py").touch()
        (self.root / "main.py").touch()
        before = sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*"))
        with patch.object(sys, "argv", ["setup_env.py", "--comfy-root", str(self.root), "--check-only"]):
            with patch.object(setup_env, "isolated_python", return_value=Path(sys.executable)):
                with self.assertRaisesRegex(RuntimeError, "input directory is absent"):
                    setup_env.main()
        self.assertEqual(before, sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*")))


class DistributionTests(unittest.TestCase):
    def test_adopted_code_is_byte_identical(self):
        provenance = json.loads((ROOT / "docs/source-provenance.json").read_text())
        for name, expected in provenance["unchanged_sources"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected, name)

    def test_registration_without_upstream_or_torch(self):
        name = "c3_distribution_test"
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.get_filename_list = lambda _: []
        with patch.dict(sys.modules, {"folder_paths": folder_paths}):
            spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            self.assertEqual(set(module.NODE_CLASS_MAPPINGS), {"H3TaoMateW4A4Loader", "H3C2HeldFFN8", "H3C3HeldOut"})
            self.assertNotIn("h3_optimizations", sys.modules)
            self.assertNotIn("torch", sys.modules)
        for key in list(sys.modules):
            if key == name or key.startswith(name + "."):
                del sys.modules[key]

    def test_legacy_migration_preserves_numerical_graph(self):
        graph = json.loads((ROOT / "workflows/TaoMate_C3.api.json").read_text())
        legacy = json.loads(json.dumps(graph))
        legacy["127"]["class_type"] = "H3V2PreconvertedLoader"
        legacy["127"]["inputs"]["cache_directory"] = "C:/old/machine/cache"
        legacy["901"]["inputs"]["image"] = "baseline016_case01_frame000.png"
        self.assertEqual(migrate(legacy), graph)
        self.assertEqual(legacy["127"]["class_type"], "H3V2PreconvertedLoader")
        del legacy["960"]
        with self.assertRaisesRegex(ValueError, "not the adopted C3"):
            migrate(legacy)

    def test_workflow_fixed_contract(self):
        graph = json.loads((ROOT / "workflows/TaoMate_C3.api.json").read_text())
        self.assertEqual(graph["129"]["inputs"]["noise_seed"], 43)
        self.assertEqual(graph["140"]["inputs"]["chunks"], 8)
        self.assertEqual(graph["912"]["inputs"]["video_budget"], 0.1)
        self.assertFalse(graph["912"]["inputs"]["denser_early_late_steps"])
        self.assertEqual(graph["124"]["inputs"]["sigmas"], "1.0, 0.9611650705337524, 0.8533332943916321, 0.0")
        self.assertEqual(graph["960"]["inputs"]["model"], ["140", 0])
        self.assertEqual(graph["142"]["inputs"]["model"], ["960", 0])
        self.assertEqual([graph["131"]["inputs"][k] for k in ("width", "height", "length")], [1024, 1792, 124])

    def test_gui_links_and_widget_values_match_api(self):
        graph = json.loads((ROOT / "workflows/TaoMate_C3.api.json").read_text(encoding="utf-8"))
        ui = json.loads((ROOT / "workflows/TaoMate_C3.json").read_text(encoding="utf-8"))
        schemas = json.loads((ROOT / "docs/workflow-schema.json").read_text(encoding="utf-8"))
        links = {link[0]: link for link in ui["links"]}
        ui_nodes = {node["id"]: node for node in ui["nodes"]}
        for item in ui["nodes"]:
            if item["type"] == "MarkdownNote":
                continue
            reference = graph[str(item["id"])]
            self.assertEqual(item["type"], reference["class_type"])
            if item["type"] == "SaveVideo":
                # ComfyUI 0.37's frontend uses dotted dynamic-combo widget names.
                named = item["widgets_values_named"]
                self.assertEqual(named["format"], reference["inputs"]["format"])
                self.assertEqual(named["format.codec"], reference["inputs"]["codec"])
                self.assertEqual(named["codec"], reference["inputs"]["codec"])
                self.assertEqual(named["format.codec.encoding"], "auto")
                self.assertEqual(named["codec.encoding"], "auto")
                self.assertEqual(named["filename_prefix"], reference["inputs"]["filename_prefix"])
                self.assertEqual(item["widgets_values"], list(named.values()))
                link = links[item["inputs"][0]["link"]]
                self.assertEqual([str(link[1]), link[2]], reference["inputs"]["video"])
                self.assertEqual(link[3:5], [item["id"], 0])
                continue
            actual = {}
            widgets = iter(item["widgets_values"])
            for group in ("required", "optional"):
                for name, spec in schemas[item["type"]]["input"].get(group, {}).items():
                    kind, options = spec[0], spec[1] if len(spec) > 1 else {}
                    if isinstance(kind, list) or kind in {"STRING", "INT", "FLOAT", "BOOLEAN", "COMBO"}:
                        value = next(widgets)
                        if name in reference["inputs"]:
                            actual[name] = value
                        else:
                            self.assertEqual(value, options.get("default"))
                        if options.get("control_after_generate"):
                            self.assertEqual(next(widgets), "fixed")
            self.assertEqual(list(widgets), [])
            for slot, input_ in enumerate(item["inputs"]):
                if input_["link"] is None:
                    continue
                link = links[input_["link"]]
                self.assertEqual(link[3:5], [item["id"], slot])
                output = ui_nodes[link[1]]["outputs"][link[2]]
                self.assertIn(link[0], output["links"])
                self.assertEqual(output["type"], link[5])
                actual[input_["name"]] = [str(link[1]), link[2]]
            self.assertEqual(actual, reference["inputs"])


if __name__ == "__main__":
    unittest.main()
