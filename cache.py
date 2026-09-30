# SPDX-License-Identifier: GPL-3.0-only
"""Portable native safetensors overlays. Runtime loading never quantizes weights."""
import hashlib
import json
import time
from pathlib import Path

FORMAT = "h3-streaming-v2-cache-1"
LAYOUT = "TensorCoreConvRotW4A4Layout"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path):
    path = Path(path)
    with path.open("rb") as f:
        n = int.from_bytes(f.read(8), "little")
        require(0 < n <= min(path.stat().st_size - 8, 64 * 1024 * 1024), "Invalid safetensors header")
        header = f.read(n)
    # Portable across copies/renames: do not bind to a local path or mtime.
    return {"size": path.stat().st_size, "header_sha256": hashlib.sha256(header).hexdigest()}


def cache_file(directory, row):
    directory = Path(directory).resolve()
    path = (directory / row["file"]).resolve()
    require(path.parent == directory, "Cache file must be directly inside its directory")
    require(path.is_file() and path.stat().st_size == row["file_size"], f"Cache size/missing file: {path.name}")
    return path


def manifest(directory, source=None, require_gate=True):
    directory = Path(directory)
    data = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    require(data.get("format") == FORMAT and data.get("complete") is True, "Incomplete or incompatible cache; run convert.py")
    require([r["block"] for r in data["fc1"]] == list(range(50)), "FC1 cache must cover blocks 0..49 exactly")
    if source is not None:
        require(data["model"]["identity"] == identity(source), "Source model differs from the cache; convert the matching model")
    if require_gate:
        require(data.get("mode") != "fc1_only" and isinstance(data.get("gate"), dict),
                "Gate cache is required for Streaming VSA")
    for row in data["fc1"] + ([data["gate"]] if require_gate else []):
        cache_file(directory, row)
    return data


def check_fc1_environment():
    import inspect
    import importlib.metadata
    import torch
    import comfy.sd
    import comfy.ops
    import comfy.quant_ops
    from comfy_kitchen.tensor.convrot_w4a4 import TensorCoreConvRotW4A4Layout
    import comfy_kitchen.backends.cuda._C as native
    import comfy_kitchen.backends.cuda as cuda_backend
    require("convrot_w4a4" in comfy.quant_ops.QUANT_ALGOS, "ComfyUI lacks native convrot_w4a4 serialization")
    require("disable_dynamic" in inspect.signature(comfy.sd.load_diffusion_model_state_dict).parameters,
            "ComfyUI lacks the required Dynamic VRAM loader API")
    require(hasattr(comfy.ops, "mixed_precision_ops") and hasattr(native, "cutlass_int4_dequant"), "Native INT4 backend missing")
    require(torch.cuda.is_available(), "CUDA is required; CPU fallback is not supported")
    dev = torch.cuda.current_device()
    native_supported = getattr(cuda_backend, "_cuda_device_supports_native_int4_mma", None)
    require(native_supported is not None and native_supported(torch.empty(0, device=torch.device("cuda", dev))),
            "This GPU/configuration selects an INT8 fallback instead of native ConvRot INT4; see README")
    return {"torch": torch.__version__, "comfy-kitchen": importlib.metadata.version("comfy-kitchen"),
            "gpu": torch.cuda.get_device_name(dev), "layout": TensorCoreConvRotW4A4Layout.__name__}


def check_vsa_environment():
    import torch
    import comfy_kitchen as ck
    from comfy_extras import nodes_sparse_attention as sparse

    info = check_fc1_environment()
    require(sparse.PRODUCER_CHUNK == 4096 and sparse.HEAD_DIM == 128, "Unsupported H3 sparse API constants")
    require(ck.sol_attn_is_available(torch.device("cuda", torch.cuda.current_device())),
            "Compiled Sol-Attn kernel unavailable on this GPU")
    return info


def check_environment():
    """Keep the original full FC1 + Streaming VSA compatibility check."""
    return check_vsa_environment()


def validate_fc1_state(state, index):
    import torch
    prefix = f"blocks.{index}.mlp.fc1."
    require(set(state) == {prefix + k for k in ("weight", "weight_scale", "comfy_quant")}, "Invalid FC1 shard keys")
    w, s, q = (state[prefix + k] for k in ("weight", "weight_scale", "comfy_quant"))
    require(w.dtype == torch.int8 and tuple(w.shape) == (28672, 2688), "Invalid FC1 packed tensor")
    require(s.dtype == torch.float32 and tuple(s.shape) == (28672,), "Invalid FC1 scales")
    require(q.dtype == torch.uint8 and q.numel() < 4096, "Invalid FC1 quantization metadata")
    conf = json.loads(q.numpy().tobytes())
    require(conf.get("format") == "convrot_w4a4" and conf.get("convrot_groupsize") == 256
            and conf.get("linear_dtype", "int4") == "int4", "Unsupported FC1 quantization configuration")


def validate_model(model):
    blocks = model.get_model_object("diffusion_model").blocks
    require(len(blocks) == 50, "Expected 50 MiniMax H3 blocks")
    for block in blocks:
        w = block.mlp.fc1.weight
        require(getattr(w, "_layout_cls", None) == LAYOUT, "FC1 layout was not restored")
        require(w.device.type == "cpu" and w._params.scale.device.type == "cpu", "FC1 backing data/scales must stay on CPU")
        p = w._params
        require(p.convrot_groupsize == 256 and p.quant_group_size == 64 and p.linear_dtype == "int4"
                and not p.transposed, "Unsupported restored FC1 parameters")


def load_model(source, directory, model_options=None, disable_dynamic=False):
    import comfy.sd
    import comfy.utils
    started = time.perf_counter()
    data = manifest(directory, source, require_gate=False)
    state, metadata = comfy.utils.load_torch_file(str(source), return_metadata=True)
    begin = time.perf_counter()
    for row in data["fc1"]:
        shard = comfy.utils.load_torch_file(str(cache_file(directory, row)))
        validate_fc1_state(shard, row["block"])
        state.update(shard)
    weight_seconds = time.perf_counter() - begin
    model = comfy.sd.load_diffusion_model_state_dict(state, model_options=model_options or {},
                                                    metadata=metadata, disable_dynamic=disable_dynamic)
    require(model is not None, "Unsupported diffusion model")
    validate_model(model)
    model.cached_patcher_init = (load_model, (str(source), str(directory), model_options))
    model.h3_v2_load_info = {"fc1_blocks": 50, "w4a4_load_sec": weight_seconds,
                            "preparation_sec": time.perf_counter() - started,
                            "runtime_weight_conversion_sec": 0.0, "dynamic": model.is_dynamic()}
    return model


def load_gates(directory):
    import torch
    import comfy.utils
    data = manifest(directory)
    state = comfy.utils.load_torch_file(str(cache_file(directory, data["gate"])))
    expected = {f"blocks.{i}.{field}" for i in range(50) for field in ("weight", "scale")}
    require(set(state) == expected, "Gate cache must have 50 weight/scale pairs")
    gates = {}
    for i in range(50):
        w, scale = state[f"blocks.{i}.weight"], state[f"blocks.{i}.scale"]
        require(w.dtype == torch.int8 and tuple(w.shape) == (7168, 5376), "Invalid gate tensor")
        require(scale.dtype == torch.float32 and scale.numel() == 1
                and bool(torch.isfinite(scale).all()) and scale.item() > 0, "Invalid gate scale")
        try:
            w, scale = w.pin_memory(), scale.pin_memory()
        except RuntimeError:
            pass
        gates[i] = (w, scale)
    return gates
