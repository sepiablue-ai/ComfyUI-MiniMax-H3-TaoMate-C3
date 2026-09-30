"""C3 candidate: hold one independent native INT8 out weight per attention call."""

from __future__ import annotations

from dataclasses import replace
import logging
import math
import torch
import torch.nn.functional as F

import comfy.ops
import comfy.patcher_extension
from comfy.quant_ops import QuantizedTensor

from h3_optimizations.attention.sparse.kitchen_streamed_q import (
    PreparedStreamedSparseKitchen,
    StreamedSparseKitchenBackend,
)
from h3_optimizations.patch import (
    OWNER_MARKER, SIGNATURE_MARKER, configure_backend, installation_signature,
    key_for,
)
from h3_optimizations.qkv.int8 import HeldConvRotINT8Linear


def _freevars(function):
    cells = function.__closure__ or ()
    return {name: cell.cell_contents for name, cell in zip(function.__code__.co_freevars, cells)}


class _HeldOut:
    def __init__(self, module, sample):
        self.module = module
        self.sample = sample
        self.weight = None
        self.calls = 0
        self.original_bytes = 0

    def __enter__(self):
        binding = HeldConvRotINT8Linear(self.module, self.sample)
        binding.__enter__()
        try:
            source = binding.weight
            if not isinstance(source, QuantizedTensor):
                raise RuntimeError("C3 out must use a native quantized weight")
            qdata = source._qdata
            scale = source._params.scale
            if qdata.dtype != torch.int8 or qdata.ndim != 2 or not torch.is_tensor(scale):
                raise RuntimeError("C3 out INT8 representation changed")
            # Comfy may reuse an async cast buffer when QKV is acquired next.
            # Retain only a private quantized copy, then release the cast handle.
            own_qdata = qdata.clone(memory_format=torch.contiguous_format)
            own_scale = scale.clone(memory_format=torch.contiguous_format)
            if own_qdata.untyped_storage().data_ptr() == qdata.untyped_storage().data_ptr():
                raise RuntimeError("C3 out qdata copy aliases the cast buffer")
            if own_scale.untyped_storage().data_ptr() == scale.untyped_storage().data_ptr():
                raise RuntimeError("C3 out scale copy aliases the cast buffer")
            self.weight = QuantizedTensor(
                own_qdata,
                source._layout_cls,
                replace(source._params, scale=own_scale),
            )
            self.original_bytes = own_qdata.numel() * own_qdata.element_size() + own_scale.numel() * own_scale.element_size()
            return self
        finally:
            binding.__exit__(None, None, None)

    def __call__(self, x):
        if self.weight is None:
            raise RuntimeError("C3 out weight was released")
        comfy.ops.run_every_op()
        self.calls += 1
        return F.linear(x, self.weight, None)

    def __exit__(self, _type, _value, _traceback):
        self.weight = None
        self.sample = None
        return False


class _OutProxy:
    def __init__(self, module, held):
        self._module = module
        self.out_proj = held

    def __getattr__(self, name):
        return getattr(self._module, name)


class _HeldOutBackend:
    def __init__(self, base, blocks):
        self.base = base
        self.block_ids = {id(block.attn): index for index, block in enumerate(blocks)}
        self.announced = set()

    def __getattr__(self, name):
        return getattr(self.base, name)

    def installation_signature(self):
        return ("C3-held-out-int8", installation_signature(self.base))

    def execute_projected(self, module, prepared):
        if not isinstance(prepared, PreparedStreamedSparseKitchen):
            raise RuntimeError("C3 candidate requires streamed sparse Kitchen")
        original = getattr(module, "_module", module)
        if prepared.projected.module is not original:
            raise RuntimeError("C3 attention module identity changed")
        x = prepared.projected.x
        if x is None or x.ndim != 2 or x.dtype not in (torch.bfloat16, torch.float16):
            raise RuntimeError("C3 requires live BF16/FP16 attention rows")
        output_width = int(original.heads) * int(original.head_dim)
        if output_width != int(original.out_proj.weight.shape[1]):
            raise RuntimeError("C3 out input width differs from attention heads")
        block = self.block_ids.get(id(original))
        if block is None:
            raise RuntimeError("C3 attention block is not registered")
        expected = math.ceil(int(x.shape[0]) / int(self.base.query_chunk_rows))
        sample = torch.empty((1, output_width), dtype=x.dtype, device=x.device)
        with _HeldOut(original.out_proj, sample) as held:
            result = self.base.execute_projected(_OutProxy(original, held), prepared)
            if held.calls != expected:
                raise RuntimeError(f"C3 block {block} projected {held.calls} chunks, expected {expected}")
            if block in (0, 49) and block not in self.announced:
                logging.info("[C3 HeldOut] executed block %d, chunks=%d, private_int8_bytes=%d",
                             block, held.calls, held.original_bytes)
                self.announced.add(block)
            return result


def _configure_after_plan(patched):
        diffusion = patched.get_model_object("diffusion_model")
        blocks = getattr(diffusion, "blocks", ())
        if len(blocks) != 50:
            raise RuntimeError("C3 candidate expects exactly 50 H3 blocks")
        current = []
        for index in range(50):
            forward = patched.object_patches.get(key_for(index))
            if forward is None or not getattr(forward, OWNER_MARKER, False):
                raise RuntimeError(f"C3 requires a package-owned attention patch at block {index}")
            signature = getattr(forward, SIGNATURE_MARKER, None)
            if not isinstance(signature, tuple) or len(signature) != 5 or signature[4]:
                raise RuntimeError("C3 requires native out INT8 without runtime float conversion")
            inner = _freevars(forward).get("_forward")
            if inner is None:
                raise RuntimeError("C3 cannot inspect the installed attention forward")
            values = _freevars(inner)
            backend, projector = values.get("backend"), values.get("projector")
            if not isinstance(backend, StreamedSparseKitchenBackend) or projector is None:
                raise RuntimeError("C3 requires streamed sparse Kitchen and its projector")
            current.append((backend, projector, signature))
        base, projector, signature = current[0]
        if any(item[0] is not base or item[1] is not projector or item[2] != signature for item in current):
            raise RuntimeError("C3 attention patch configuration differs across blocks")
        backend = _HeldOutBackend(base, blocks)
        _, changed = configure_backend(
            patched, backend, projector=projector,
            projector_fallback_to_original=signature[2],
            backend_fallback_to_dense=signature[3],
            force_out_proj_int8=False, force_rebuild=True,
        )
        if changed != 50:
            raise RuntimeError(f"C3 installed {changed}/50 attention patches")
        _place_c3_wrapper_last(patched)
        logging.info("[C3 HeldOut] selected 50 attention blocks; native out INT8, private per-call copy")


_C3_KEY = "c3_held_out_after_h3_plan"


def _place_c3_wrapper_last(patched):
    wrapper_type = comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING
    rows = patched.wrappers.get(wrapper_type, {})
    if _C3_KEY not in rows:
        raise RuntimeError("C3 prepare wrapper disappeared")
    saved = rows.pop(_C3_KEY)
    rows[_C3_KEY] = saved


def _on_clone(_parent, child):
    _place_c3_wrapper_last(child)
    rows = child.callbacks.get(comfy.patcher_extension.CallbacksMP.ON_CLONE, {})
    if _C3_KEY not in rows:
        raise RuntimeError("C3 clone callback disappeared")
    saved = rows.pop(_C3_KEY)
    rows[_C3_KEY] = saved


def _after_h3_plan(executor, model, *args, **kwargs):
    _configure_after_plan(model)
    return executor(model, *args, **kwargs)


def _prepare_wrapper(executor, model, *args, **kwargs):
    # The upstream wrapper appends its final plan reconciliation. Append our
    # replacement after that, before Comfy patches the model onto the GPU.
    remaining = executor.wrappers[executor.idx + 1:]
    if not any(getattr(item, "__name__", "") == "_finalize_prepare_sampling" for item in remaining):
        raise RuntimeError("C3 wrapper ran before the upstream final-plan wrapper")
    executor.wrappers.append(_after_h3_plan)
    return executor(model, *args, **kwargs)


class H3C3HeldOut:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "H3/C3 experimental"

    def apply(self, model):
        patched = model.clone()
        wrapper_type = comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING
        keys = list(patched.wrappers.get(wrapper_type, {}))
        if "h3_optimizations_finalize_plan" not in keys:
            raise RuntimeError("C3 requires upstream final-plan wrapper")
        if _C3_KEY in keys:
            raise RuntimeError("C3 out wrapper is already installed")
        patched.add_wrapper_with_key(wrapper_type, _C3_KEY, _prepare_wrapper)
        patched.add_callback_with_key(comfy.patcher_extension.CallbacksMP.ON_CLONE,
                                      _C3_KEY, _on_clone)
        logging.info("[C3 HeldOut] armed after the H3 final-plan wrapper")
        return (patched,)


NODE_CLASS_MAPPINGS = {"H3C3HeldOut": H3C3HeldOut}
NODE_DISPLAY_NAME_MAPPINGS = {"H3C3HeldOut": "C3 Held Attention Out INT8"}
