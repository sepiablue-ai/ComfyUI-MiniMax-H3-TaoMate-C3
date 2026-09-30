# SPDX-License-Identifier: GPL-3.0-only
"""Load real Comfy node classes and validate the C3 graph; no server or video generation."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dependency_check import verify_dependency
from setup_env import isolated_python, conflicts


async def check(args):
    executable = isolated_python()
    comfy = args.comfy_root.resolve()
    if not (comfy / "comfy/sd.py").is_file():
        raise RuntimeError("--comfy-root must contain comfy/sd.py")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(comfy))
    import comfy.options
    comfy.options.enable_args_parsing()
    sys.argv = [sys.argv[0], "--disable-pinned-memory", "--disable-comfy-compiler"]
    if not args.native:
        sys.argv.append("--cpu")
    import folder_paths
    from utils.extra_config import load_extra_path_config
    config = args.comfy_root / "extra_model_paths.yaml"
    if config.is_file():
        load_extra_path_config(str(config))
    import nodes
    node_root = args.comfy_root / "custom_nodes"
    package = node_root / "ComfyUI-MiniMax-H3-TaoMate-C3"
    problems = conflicts(list(map(Path, folder_paths.get_folder_paths("custom_nodes"))), package)
    if problems:
        raise RuntimeError("Duplicate nodes: " + "; ".join(problems))
    dependencies = json.loads((ROOT / "dependency-lock.json").read_text(encoding="utf-8"))["dependencies"]
    packs = []
    for entry in dependencies:
        candidates = list(dict.fromkeys((Path(p) / entry["name"]).resolve()
                                       for p in folder_paths.get_folder_paths("custom_nodes")
                                       if (Path(p) / entry["name"]).is_dir()))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one {entry['name']}, found {candidates}")
        issues = verify_dependency(candidates[0], entry)
        if issues:
            raise RuntimeError(f"Invalid {entry['name']}: {issues[:8]}")
        packs.append(candidates[0])
    # Core files needed by this workflow only; do not load other user node packs.
    for module in ("nodes_custom_sampler.py", "nodes_minimax_h3.py", "nodes_audio.py", "nodes_video.py", "nodes_model_advanced.py"):
        if not await nodes.load_custom_node(str(args.comfy_root / "comfy_extras" / module), module_parent="comfy_extras"):
            raise RuntimeError(f"Could not load core node file: {module}")
    ordered = [package, *packs] if args.package_first else [*packs, package]
    for path in ordered:
        if not await nodes.load_custom_node(str(path)):
            raise RuntimeError(f"Could not register {path}")
    graph = json.loads((ROOT / "workflows/TaoMate_C3.api.json").read_text(encoding="utf-8"))
    # Comfy supports nested model directories; resolve the bundled basename selections.
    categories = {"H3TaoMateW4A4Loader": ("unet_name", "diffusion_models"),
                  "VAELoader": ("vae_name", "vae"), "CLIPLoader": ("clip_name", "text_encoders"),
                  "LoraLoaderModelOnly": ("lora_name", "loras")}
    for node in graph.values():
        if node["class_type"] in categories:
            field, category = categories[node["class_type"]]
            choices = [name for name in folder_paths.get_filename_list(category)
                       if Path(name).name == Path(node["inputs"][field]).name]
            if len(choices) == 1:
                node["inputs"][field] = choices[0]
    missing = sorted({row["class_type"] for row in graph.values()} - nodes.NODE_CLASS_MAPPINGS.keys())
    if missing:
        raise RuntimeError(f"Missing workflow nodes: {missing}")
    import execution
    validation = await execution.validate_prompt("c3-install-check", graph, None)
    if not validation[0]:
        raise RuntimeError("ComfyUI prompt validation failed: " + json.dumps(validation, ensure_ascii=True, default=str))
    # Validate the modern GUI's dotted dynamic-combo settings too; legacy API uses flat format/codec.
    gui_path = ROOT / "workflows/TaoMate_C3.json"
    if gui_path.is_file():
        gui = json.loads(gui_path.read_text(encoding="utf-8"))
        save = next(n for n in gui["nodes"] if n["type"] == "SaveVideo")
        modern = json.loads(json.dumps(graph))
        modern[str(save["id"])]["inputs"] = {
            "video": graph[str(save["id"])]["inputs"]["video"], **save["widgets_values_named"]}
        gui_validation = await execution.validate_prompt("c3-gui-install-check", modern, None)
        if not gui_validation[0]:
            raise RuntimeError("GUI save-node validation failed: " + json.dumps(gui_validation, ensure_ascii=True, default=str))
    # Exercise the lazy import after all custom nodes have registered, in either order.
    public = nodes.NODE_CLASS_MAPPINGS["H3C3HeldOut"]
    module = sys.modules[public.__module__]
    h3_root = module.require_h3_runtime()
    from importlib import import_module
    implementation = import_module(public.__module__.rsplit(".", 1)[0] + ".runtime.h3_c3_held_out")
    assert implementation.H3C3HeldOut is not None
    if args.schema_output:
        schema = {}
        for name in sorted({row["class_type"] for row in graph.values()}):
            cls = nodes.NODE_CLASS_MAPPINGS[name]
            if hasattr(cls, "GET_NODE_INFO_V1"):
                schema[name] = cls.GET_NODE_INFO_V1()
            else:
                schema[name] = {"input": cls.INPUT_TYPES(), "output": cls.RETURN_TYPES,
                                "output_name": getattr(cls, "RETURN_NAMES", cls.RETURN_TYPES),
                                "output_node": getattr(cls, "OUTPUT_NODE", False)}
        args.schema_output.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    native = None
    if args.native:
        subprocess.run([str(executable), "-B", "-X", "utf8", str(ROOT / "convert.py"),
                        "--comfy-root", str(args.comfy_root), "--gpu", args.gpu, "--fc1-only", "--check"], check=True)
        from h3_optimizations.native import loader
        native = {"abi": loader.load().h3_int8_abi_version()}
        if native["abi"] != 4:
            raise RuntimeError(f"Unexpected H3 native ABI: {native}")
    result = {"ok": True, "nodes_in_workflow": len(graph), "node_types": sorted({n["class_type"] for n in graph.values()}),
              "package_first": args.package_first, "graph_validation": "PASS", "lazy_import": "PASS",
              "h3_root": str(h3_root), "native_check": native, "videos_generated": 0, "server_started": False}
    if args.report:
        args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comfy-root", required=True, type=Path)
    ap.add_argument("--package-first", action="store_true", help="Verify custom-node load-order independence")
    ap.add_argument("--native", action="store_true", help="Small GPU INT4 offload/reload check and DLL ABI check")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--schema-output", type=Path)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    asyncio.run(check(args))


if __name__ == "__main__":
    main()
