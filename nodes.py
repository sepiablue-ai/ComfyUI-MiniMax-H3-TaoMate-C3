# SPDX-License-Identifier: GPL-3.0-only
"""Public nodes. Register without importing other custom-node packages first."""
import logging
from pathlib import Path

import folder_paths

from .cache import check_fc1_environment, load_model, manifest
from .cache_contract import require_c3_cache
from .dependency_check import require_h3_runtime


class H3TaoMateW4A4Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
            "cache_directory": ("STRING", {"default": "h3_preconverted/taomate_c3_fc1"}),
        }}

    RETURN_TYPES = ("MODEL", "H3_V2_CACHE")
    RETURN_NAMES = ("model", "cache")
    FUNCTION = "load"
    CATEGORY = "MiniMax H3/TaoMate C3"
    DESCRIPTION = "C3 FC1 W4A4 loader. Run setup.bat to prepare the offline cache. No VSA Gate is needed."

    def load(self, unet_name, cache_directory):
        source = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        directory = Path(cache_directory).expanduser()
        if not directory.is_absolute():
            directory = Path(folder_paths.models_dir) / directory
        directory = directory.resolve()
        if not (directory / "manifest.json").is_file():
            raise RuntimeError(f"C3 FC1 cache is missing: {directory}. Run setup.bat -BuildCache or -CacheSource <cache>.")
        require_c3_cache(manifest(directory, source, require_gate=False))
        check_fc1_environment()
        model = load_model(source, directory)
        logging.info("[TaoMate C3] %s", model.h3_v2_load_info)
        return model, str(directory)


class H3C2HeldFFN8:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",),
                             "chunks": ("INT", {"default": 8, "min": 8, "max": 8}),
                             "seq_threshold": ("INT", {"default": 4096, "min": 4096, "max": 4096})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "MiniMax H3/TaoMate C3"
    DESCRIPTION = "Hold W4A4 FC1 / native INT8 FC2 per block. Fixed FFN8, threshold 4096."

    def apply(self, model, chunks, seq_threshold):
        require_h3_runtime()
        from .runtime.h3_c2_held_ffn8 import H3C2HeldFFN8 as Impl
        return Impl().apply(model, chunks, seq_threshold)


class H3C3HeldOut:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "MiniMax H3/TaoMate C3"
    DESCRIPTION = "Hold an independent INT8 attention-out copy per call, after the H3 final plan."

    def apply(self, model):
        require_h3_runtime()
        from .runtime.h3_c3_held_out import H3C3HeldOut as Impl
        return Impl().apply(model)


NODE_CLASS_MAPPINGS = {
    "H3TaoMateW4A4Loader": H3TaoMateW4A4Loader,
    "H3C2HeldFFN8": H3C2HeldFFN8,
    "H3C3HeldOut": H3C3HeldOut,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3TaoMateW4A4Loader": "TaoMate C3 · FC1 W4A4 Loader",
    "H3C2HeldFFN8": "TaoMate C3 · Held FFN8 (C2)",
    "H3C3HeldOut": "TaoMate C3 · Held Attention Out",
}
