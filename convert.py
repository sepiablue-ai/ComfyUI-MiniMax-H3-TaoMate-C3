# SPDX-License-Identifier: GPL-3.0-only
"""Offline conversion / compatibility check. Uses the user's existing ComfyUI Python.

No downloads, installs, server startup, overwrites, or file deletion.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", required=True, type=Path)
    parser.add_argument("--model", type=Path, help="Existing INT8 ConvRot diffusion safetensors")
    parser.add_argument("--gate", type=Path, help="Existing fasth3_vsa_gate.safetensors")
    parser.add_argument("--output", type=Path, help="NEW cache directory (never overwrite)")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES for this process only")
    parser.add_argument("--check", action="store_true", help="API and small native INT4 smoke check only")
    parser.add_argument("--verify", type=Path, help="Check existing cache SHA-256; optionally verify --model / --gate")
    parser.add_argument("--fc1-only", action="store_true", help="Create/check FC1-only cache without Gate or Sol-Attn")
    args = parser.parse_args()
    if not (args.comfy_root / "comfy/sd.py").is_file():
        parser.error("--comfy-root must point to ComfyUI containing comfy/sd.py")
    if args.fc1_only and args.gate:
        parser.error("--gate cannot be combined with --fc1-only")
    if not args.check and not args.verify and not all((args.model, args.output)):
        parser.error("Conversion requires --model and --output")
    if not args.check and not args.verify and not args.fc1_only and not args.gate:
        parser.error("Gate conversion also requires --gate; use --fc1-only for FC1 alone")
    if args.output and args.output.exists():
        parser.error("Output already exists; choose a new directory. Existing files are preserved.")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.comfy_root.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import comfy.options
    comfy.options.enable_args_parsing()
    sys.argv = [sys.argv[0], "--disable-pinned-memory", "--disable-comfy-compiler"]
    import torch
    import comfy.ops
    from safetensors import safe_open
    from safetensors.torch import save_file, load_file
    from comfy_kitchen.tensor.convrot_w4a4 import TensorCoreConvRotW4A4Layout as Layout, QuantizedTensor
    import comfy_kitchen.backends.cuda._C as native
    from cache import (FORMAT, LAYOUT, check_environment, check_fc1_environment, require, identity,
                       sha256, manifest, cache_file, validate_fc1_state)

    if args.verify:
        data = manifest(args.verify, args.model, require_gate=not args.fc1_only)
        for row in data["fc1"] + ([] if args.fc1_only else [data["gate"]]):
            require(sha256(cache_file(args.verify, row)) == row["sha256"], f"SHA-256 mismatch: {row['file']}")
        sources = [(args.model, data["model"])]
        if not args.fc1_only:
            sources.append((args.gate, data["gate_source"]))
        for path, info in sources:
            if path:
                require(sha256(path) == info["sha256"], "Source content SHA-256 mismatch")
        print("Cache SHA-256 verification PASS (50 FC1 shards" + ("" if args.fc1_only else " + Gate") + ")", flush=True)
        return

    print(json.dumps(check_fc1_environment() if args.fc1_only else check_environment()), flush=True)
    calls = [0]
    native_linear = native.cutlass_int4_dequant

    def counted(*a, **kw):
        result = native_linear(*a, **kw)
        require(bool(result), "Native CUTLASS INT4 declined execution")
        calls[0] += 1
        return result

    native.cutlass_int4_dequant = counted

    def make_linear(state):
        layer = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16).Linear(5376, 28672, bias=False, device="cpu")
        layer.load_state_dict(dict(state), strict=True)
        return layer

    try:
        with torch.inference_mode():
            torch.manual_seed(43)
            if args.check:
                w = torch.randn(1024, 5376, device="cuda", dtype=torch.bfloat16)
                packed, params = Layout.quantize(w)
                q = QuantizedTensor(packed, LAYOUT, params)
                x = torch.randn(1024, 5376, device="cuda", dtype=torch.bfloat16)
                a = torch.nn.functional.linear(x, q)
                b = torch.nn.functional.linear(x, q.to("cpu").to("cuda"))
                require(torch.equal(a, b) and bool(torch.isfinite(a).all()) and calls[0] == 2,
                        "Native INT4 GPU/CPU/GPU smoke check failed")
                print("Native INT4 + CPU offload/reload PASS. No files written.", flush=True)
                return

            # Validate source headers before creating any output directory.
            model_id = identity(args.model)
            report = {"format": FORMAT, "complete": False,
                      "model": {"identity": model_id, "sha256": sha256(args.model)},
                      "fc1": []}
            if args.fc1_only:
                report["mode"] = "fc1_only"
            else:
                report["gate_source"] = {"identity": identity(args.gate), "sha256": sha256(args.gate)}
            args.output.mkdir(parents=True, exist_ok=False)
            started = time.perf_counter()
            x = torch.randn(1024, 5376, device="cuda", dtype=torch.bfloat16)
            with safe_open(str(args.model), framework="pt", device="cpu") as source:
                keys = set(source.keys())
                for i in range(50):
                    prefix = f"blocks.{i}.mlp.fc1."
                    state = {k[len(prefix):]: source.get_tensor(k) for k in keys if k.startswith(prefix)}
                    require(set(state) == {"weight", "weight_scale", "comfy_quant"}, "Unsupported source FC1 keys")
                    old = make_linear(state)
                    require(getattr(old.weight, "_layout_cls", None) == "TensorWiseINT8Layout"
                            and old.weight._params.convrot, "Source FC1 must be native INT8 ConvRot")
                    logical = old.weight.dequantize()
                    require(logical.dtype == torch.bfloat16 and bool(torch.isfinite(logical).all()), "Non-finite logical FC1")
                    packed, params = Layout.quantize(logical.to("cuda"), convrot_groupsize=256, stochastic_rounding=0)
                    runtime = QuantizedTensor(packed, LAYOUT, params)
                    cpu = runtime.to("cpu")
                    old.weight = torch.nn.Parameter(cpu, requires_grad=False)
                    old.quant_format, old.layout_type = "convrot_w4a4", LAYOUT
                    native_state = {prefix + k: v.detach().contiguous() for k, v in old.state_dict().items()}
                    validate_fc1_state(native_state, i)
                    path = args.output / f"block_{i:02d}.safetensors"
                    save_file(native_state, str(path), metadata={"format": FORMAT, "block": str(i)})
                    saved = load_file(str(path), device="cpu")
                    require(all(torch.equal(v, saved[k]) for k, v in native_state.items()), "FC1 save/reload mismatch")
                    restored = make_linear({k[len(prefix):]: v for k, v in saved.items()})
                    require(all(getattr(restored.weight._params, k) == getattr(cpu._params, k)
                                for k in ("orig_dtype", "orig_shape", "convrot_groupsize", "quant_group_size", "linear_dtype", "transposed")),
                            "Restored FC1 layout metadata mismatch")
                    before = calls[0]
                    reference = torch.nn.functional.linear(x, runtime)
                    restored.to("cuda")
                    actual = restored(x)
                    restored.to("cpu")
                    require(restored.weight.device.type == "cpu" and restored.weight._params.scale.device.type == "cpu", "CPU offload failed")
                    restored.to("cuda")
                    again = restored(x)
                    require(calls[0] - before == 3 and bool(torch.isfinite(actual).all())
                            and torch.equal(reference, actual) and torch.equal(actual, again), "Native FC1 roundtrip mismatch")
                    report["fc1"].append({"block": i, "file": path.name, "file_size": path.stat().st_size,
                                          "sha256": sha256(path), "native_roundtrip_exact": True})
                    print(f"FC1 {i + 1}/50: saved, native INT4, CPU offload/reload EXACT", flush=True)
                    del state, old, logical, packed, params, runtime, cpu, native_state, saved, restored, reference, actual, again

            if not args.fc1_only:
                output = {}
                previous_threads = torch.get_num_threads()
                try:
                    torch.set_num_threads(8)
                    with safe_open(str(args.gate), framework="pt", device="cpu") as source:
                        for i in range(50):
                            w = source.get_tensor(f"blocks.{i}.attn.to_gate_compress.weight").contiguous().to(torch.bfloat16)
                            require(tuple(w.shape) == (7168, 5376) and bool(torch.isfinite(w).all()), "Invalid source Gate")
                            scale = (w.abs().max() / 127.0).float()
                            require(scale.item() > 0, "Zero Gate scale")
                            output[f"blocks.{i}.weight"] = (w / scale).round_().clamp_(-128, 127).to(torch.int8)
                            output[f"blocks.{i}.scale"] = scale
                    path = args.output / "gate_int8.safetensors"
                    save_file(output, str(path), metadata={"format": FORMAT})
                    with safe_open(str(path), framework="pt", device="cpu") as saved:
                        require(set(saved.keys()) == set(output) and all(torch.equal(v, saved.get_tensor(k)) for k, v in output.items()), "Gate save/reload mismatch")
                finally:
                    torch.set_num_threads(previous_threads)
                report["gate"] = {"file": path.name, "file_size": path.stat().st_size,
                                  "sha256": sha256(path), "blocks": 50}
            report.update(complete=True, conversion_sec=time.perf_counter() - started)
            # Completion manifest is written last. Partial directories cannot load.
            with (args.output / "manifest.json").open("x", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            contents = "FC1 50/50" + ("" if args.fc1_only else " + Gate 50/50")
            print(f"COMPLETE: {contents}; {args.output}", flush=True)
    finally:
        native.cutlass_int4_dequant = native_linear


if __name__ == "__main__":
    main()
