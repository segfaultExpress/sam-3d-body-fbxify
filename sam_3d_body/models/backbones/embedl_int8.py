"""Drop-in replacement for Dinov3Backbone using embedl's INT8 encoder.

embedl/sam-3d-body (Hugging Face, gated) ships the SAM 3D Body DINOv3 ViT-H+/16
encoder quantized to INT8 (sensitive layers kept FP16) in two formats:

  * ``embedl_sam3dbody_int8.pt2``            torch.export ExportedProgram
  * ``embedl_sam3dbody_int8.onnx`` (+.data)   ONNX with Q/DQ nodes, for TensorRT

Both take a ``(1, 3, 512, 512)`` ImageNet-normalized crop and return a
``(1, 1280, 32, 32)`` feature map, which is exactly what ``Dinov3Backbone.forward``
returns for the dinov3 checkpoint. The decoder, prompt encoder, hand decoder and MHR
head stay in PyTorch, so Fast inference (skip_keypoint_prompt), keypoint prompts and
hand refinement keep working unchanged.

Modes
  pt2  eager PyTorch execution of the exported graph. Numerically the quantized
       model, but Q/DQ ops run as fake-quant in fp32: expect fidelity parity with
       TensorRT INT8 but no speedup (often a slowdown). Useful for quality checks.
  trt  TensorRT engine built once from the INT8 ONNX (FP16 + INT8 builder flags,
       explicit-quantization). Cached next to the ONNX. This is the fast path.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import torch
from torch import nn

ENCODER_PT2 = "embedl_sam3dbody_int8.pt2"
ENCODER_ONNX = "embedl_sam3dbody_int8.onnx"
ENCODER_ONNX_DATA = "embedl_sam3dbody_int8.onnx.data"
QUANTIZED_MODES = ("pt2", "trt")

DINOV3_HPLUS_EMBED_DIM = 1280
DINOV3_PATCH_SIZE = 16


def _engine_path_for(onnx_path: str, device_name: str) -> str:
    """TensorRT engines are tied to GPU + TRT version; key the cache file by both."""
    import tensorrt as trt

    tag = device_name.lower().replace(" ", "_").replace("/", "_")
    return f"{os.path.splitext(onnx_path)[0]}.{tag}.trt{trt.__version__}.engine"


class _TrtEncoder:
    """Minimal TensorRT runner that exchanges torch CUDA tensors (no pycuda)."""

    def __init__(self, onnx_path: str, device: torch.device, workspace_gb: float = 6.0,
                 fp16: bool = True, int8: bool = True, rebuild: bool = False):
        import tensorrt as trt

        self.trt = trt
        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)
        engine_path = _engine_path_for(onnx_path, torch.cuda.get_device_name(device))
        if rebuild or not os.path.exists(engine_path):
            self._build(onnx_path, engine_path, workspace_gb, fp16, int8)
        self.engine_path = engine_path
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine {engine_path}")
        self.context = self.engine.create_execution_context()

        self.input_name = None
        self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name
        if self.input_name is None or self.output_name is None:
            raise RuntimeError("TensorRT engine must have one input and one output tensor")
        self.input_dtype = self._torch_dtype(self.engine.get_tensor_dtype(self.input_name))
        self.output_dtype = self._torch_dtype(self.engine.get_tensor_dtype(self.output_name))
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        self._out = torch.empty(self.output_shape, dtype=self.output_dtype, device=device)

    def _torch_dtype(self, trt_dtype):
        trt = self.trt
        return {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.BF16: torch.bfloat16,
            trt.DataType.INT8: torch.int8,
            trt.DataType.INT32: torch.int32,
        }[trt_dtype]

    def _build(self, onnx_path: str, engine_path: str, workspace_gb: float, fp16: bool, int8: bool):
        trt = self.trt
        print(f"[embedl-int8] Building TensorRT engine from {onnx_path} (this takes a few minutes)...",
              flush=True)
        t0 = time.time()
        builder = trt.Builder(self.logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, self.logger)
        if not parser.parse_from_file(onnx_path):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"ONNX parse failed:\n{errors}")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
        # Explicit quantization: the Q/DQ nodes carry calibrated scales, so no calibrator.
        if int8 and hasattr(trt.BuilderFlag, "INT8"):
            config.set_flag(trt.BuilderFlag.INT8)
        if fp16 and hasattr(trt.BuilderFlag, "FP16"):
            config.set_flag(trt.BuilderFlag.FP16)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build failed")
        tmp = engine_path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(serialized)
        os.replace(tmp, engine_path)
        print(f"[embedl-int8] Engine saved to {engine_path} in {time.time() - t0:.0f}s", flush=True)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.device, dtype=self.input_dtype).contiguous()
        if tuple(x.shape) != self.input_shape:
            raise ValueError(f"TensorRT encoder expects input {self.input_shape}, got {tuple(x.shape)}")
        stream = torch.cuda.current_stream(self.device)
        self.context.set_tensor_address(self.input_name, x.data_ptr())
        self.context.set_tensor_address(self.output_name, self._out.data_ptr())
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        # Clone so the caller owns the memory; the next call overwrites _out.
        return self._out.clone()


class EmbedlInt8Backbone(nn.Module):
    """Quantized DINOv3 encoder with the Dinov3Backbone interface.

    The runner (ExportedProgram module or TensorRT context) is intentionally not
    registered as a submodule: ``model.half()`` / ``convert_to_fp16`` must not touch
    its quantization constants. Inputs are cast to the runner's dtype and outputs are
    cast back to the input dtype so the rest of SAM3DBody is unaware of the swap.
    """

    def __init__(
        self,
        quantized_dir: str,
        mode: str = "trt",
        device: Optional[torch.device] = None,
        embed_dim: int = DINOV3_HPLUS_EMBED_DIM,
        patch_size: int = DINOV3_PATCH_SIZE,
        trt_workspace_gb: float = 6.0,
        trt_rebuild: bool = False,
    ):
        super().__init__()
        if mode not in QUANTIZED_MODES:
            raise ValueError(f"quantized backbone mode must be one of {QUANTIZED_MODES}, got {mode!r}")
        self.mode = mode
        self.quantized_dir = quantized_dir
        self.patch_size = patch_size
        self.embed_dim = self.embed_dims = embed_dim
        self.device_ = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = f"embedl_int8_{mode}"
        # Keep nn.Module bookkeeping happy (device/dtype queries) without exposing weights.
        self.register_buffer("_anchor", torch.zeros(1), persistent=False)

        t0 = time.time()
        if mode == "pt2":
            path = os.path.join(quantized_dir, ENCODER_PT2)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Quantized encoder not found: {path}")
            ep = torch.export.load(path)
            runner = ep.module()
            runner.eval()
            runner.to(self.device_)
            self._runner_dtype = torch.float32
        else:
            if self.device_.type != "cuda":
                raise RuntimeError("TensorRT quantized backbone requires a CUDA device")
            path = os.path.join(quantized_dir, ENCODER_ONNX)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Quantized encoder not found: {path}")
            runner = _TrtEncoder(path, self.device_, workspace_gb=trt_workspace_gb, rebuild=trt_rebuild)
            self._runner_dtype = runner.input_dtype
        object.__setattr__(self, "_runner", runner)
        self.source_path = path
        print(f"[embedl-int8] Loaded quantized encoder ({mode}) from {path} in {time.time() - t0:.1f}s",
              flush=True)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, extra_embed=None) -> torch.Tensor:
        assert extra_embed is None, "Not Implemented Yet"
        out_dtype = x.dtype
        x = x.to(device=self.device_, dtype=self._runner_dtype)
        if x.shape[0] == 1:
            y = self._runner(x)
        else:
            # Exports are batch-1 static; run per person and stack.
            y = torch.cat([self._runner(x[i : i + 1]) for i in range(x.shape[0])], dim=0)
        return y.to(out_dtype)

    def get_layer_depth(self, param_name: str, prefix: str = "encoder."):
        return 0, 1


def quantized_files_present(quantized_dir: str, mode: str) -> bool:
    if mode == "pt2":
        return os.path.exists(os.path.join(quantized_dir, ENCODER_PT2))
    if mode == "trt":
        return os.path.exists(os.path.join(quantized_dir, ENCODER_ONNX)) and os.path.exists(
            os.path.join(quantized_dir, ENCODER_ONNX_DATA)
        )
    return False


def swap_in_quantized_backbone(model, quantized_dir: str, mode: str, **kwargs) -> EmbedlInt8Backbone:
    """Replace ``model.backbone`` with the embedl INT8 encoder and free the fp32 one.

    Only valid for the dinov3 checkpoint (ViT-H+/16, 512x512, 1280-d features).
    """
    backbone_type = str(model.cfg.MODEL.BACKBONE.TYPE)
    if not backbone_type.startswith("dinov3"):
        raise ValueError(
            f"embedl INT8 encoder is a DINOv3 ViT-H+ export; model backbone is {backbone_type!r}. "
            "Use --model dinov3."
        )
    old = model.backbone
    new = EmbedlInt8Backbone(quantized_dir, mode=mode, device=model.device, **kwargs)
    model.backbone = new
    model.backbone_dtype = torch.float32
    del old
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return new
