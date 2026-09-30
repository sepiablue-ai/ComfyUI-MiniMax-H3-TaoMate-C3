"""Experimental C2 FFN8 replacement holding native W4A4/INT8 MLP weights per block."""

from __future__ import annotations

import logging
import types

import torch
import comfy.ops

from h3_optimizations.memory.linear import HeldMLP


def _held_forward(self, x, *, block, chunks, seq_threshold, announced):
    # Keep KJNodes' exact chunk boundaries and output assignment semantics.
    parts = torch.chunk(x, chunks, dim=0) if x.shape[0] > seq_threshold and chunks > 1 else (x,)
    output = torch.empty_like(x)
    offset = 0
    with HeldMLP(self, parts[0]) as held:
        for part in parts:
            expanded = held.fc1(part)
            comfy.ops.run_every_op()
            value, mode = held.fc2_swiglu(expanded, native=True)
            if mode != "held_tensorwise_int8_native":
                raise RuntimeError(f"C2 held MLP must retain native FC2 SwiGLU, got {mode}")
            if block == 0 and not announced[0]:
                logging.info("[C2 HeldMLP] executed block 0, chunks=%d, native FC2 SwiGLU", len(parts))
                announced[0] = True
            output[offset:offset + part.shape[0]] = value
            offset += part.shape[0]
            del expanded, value
    return output


class H3C2HeldFFN8:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",),
                             "chunks": ("INT", {"default": 8, "min": 8, "max": 8}),
                             "seq_threshold": ("INT", {"default": 4096, "min": 4096, "max": 4096})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "H3/C2 experimental"

    def apply(self, model, chunks, seq_threshold):
        if int(chunks) != 8 or int(seq_threshold) != 4096:
            raise ValueError("C2 candidate is fixed to FFN8 and threshold 4096")
        patched = model.clone()
        diffusion = patched.get_model_object("diffusion_model")
        blocks = getattr(diffusion, "blocks", None)
        if blocks is None or len(blocks) != 50:
            raise RuntimeError("C2 candidate expects 50 H3 blocks")
        if not hasattr(patched, "object_patches"):
            raise RuntimeError("ModelPatcher object patch inventory is unavailable")
        for index, block in enumerate(blocks):
            key = f"diffusion_model.blocks.{index}.mlp.forward"
            if key in patched.object_patches:
                raise RuntimeError(f"MLP forward already patched: {key}; refusing double FFN chunking")
            if not hasattr(block.mlp, "fc1") or not hasattr(block.mlp, "fc2"):
                raise RuntimeError(f"H3 block {index} has no FC1/FC2")
            announced = [False]

            def forward(self_module, x, *, _index=index, _announced=announced):
                return _held_forward(self_module, x, block=_index, chunks=8,
                                     seq_threshold=4096, announced=_announced)

            patched.add_object_patch(key, types.MethodType(forward, block.mlp))
        logging.info("[C2 HeldMLP] selected for 50 MLPs; FFN8, threshold=4096")
        return (patched,)


NODE_CLASS_MAPPINGS = {"H3C2HeldFFN8": H3C2HeldFFN8}
NODE_DISPLAY_NAME_MAPPINGS = {"H3C2HeldFFN8": "C2 Held W4A4 INT8 FFN8"}
