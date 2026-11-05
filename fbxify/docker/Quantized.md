# Quantized encoder (embedl INT8) — install & benchmark

This branch can swap the SAM 3D Body **DINOv3 ViT-H+/16 encoder** (840 M params, ~85 % of the
compute) for the INT8 export published by Embedl at
[huggingface.co/embedl/sam-3d-body](https://huggingface.co/embedl/sam-3d-body).
Everything after the encoder (prompt encoder, body/hand decoders, MHR head) stays in PyTorch, so
keypoint prompts, hand refinement and the **Fast inference** checkbox keep working unchanged.

| Mode (`FBXIFY_QUANTIZED_BACKBONE`) | What runs | Needs | Use for |
|---|---|---|---|
| `none` (default) | stock fp32 DINOv3 from `model.ckpt` | – | production today |
| `pt2` | `embedl_sam3dbody_int8.pt2` via `torch.export.load` | torch ≥ 2.8 | quality checks; Q/DQ runs as fake-quant in fp32, so **no speedup** |
| `trt` | TensorRT engine built once from `embedl_sam3dbody_int8.onnx` (INT8 + FP16) | `tensorrt-cu12 ≥ 10.16` | the fast path |

Only the **dinov3** checkpoint is supported (`FBXIFY_MODEL=dinov3` / `--model dinov3`); the ViT-H
checkpoint uses a different encoder.

In the [model catalogue](../../README.md#model-catalogue) these are the encoder entries
`embedl_int8_trt` and `embedl_int8_pt2` (shipped disabled, pinned to a Hugging Face revision with
sha256 for every file). Enable one in your `models.json` overlay and it is fetched at start-up and
offered in the **Encoder** dropdown; `--encoder embedl_int8_trt` / `FBXIFY_ENCODER` and the legacy
`--quantized_backbone trt` / `FBXIFY_QUANTIZED_BACKBONE=trt` all select the same entry.

## 1. Get access

Both repos are gated. Log into Hugging Face and accept the licences on:

1. https://huggingface.co/facebook/sam-3d-body-dinov3 (upstream checkpoint, required anyway)
2. https://huggingface.co/embedl/sam-3d-body (INT8 encoder; Embedl Models Community Licence — no redistribution as a hosted service)

Create a read token at https://huggingface.co/settings/tokens and export it as `HF_TOKEN`.

## 2. Download the files

Into `CHECKPOINTS_DIR/embedl-sam-3d-body` (override with `FBXIFY_QUANTIZED_DIR`). The app also
downloads them on first use when `HF_TOKEN` is set, but doing it once up front avoids a slow
first start (the ONNX weights are ~3.4 GB).

```bash
pip install -U huggingface_hub
# TensorRT path (recommended)
hf download embedl/sam-3d-body embedl_sam3dbody_int8.onnx embedl_sam3dbody_int8.onnx.data sam3dbody_faces.npy \
   --local-dir checkpoints/embedl-sam-3d-body
# torch.export path (optional, quality reference)
hf download embedl/sam-3d-body embedl_sam3dbody_int8.pt2 --local-dir checkpoints/embedl-sam-3d-body
```

Windows: `huggingface-cli download ...` works the same (`pip install huggingface_hub[cli]`).

### Check the DINOv3 checkpoint itself

The INT8 encoder is paired with the decoder/MHR head from `checkpoints/sam-3d-body-dinov3/model.ckpt`,
so that file must be the real DINOv3 checkpoint (**2,109,129,346 bytes**). The ViT-H checkpoint is
1,691,205,237 bytes; if it was copied into the dinov3 folder, none of its encoder weights match the
DINOv3 architecture and `load_state_dict(strict=False)` silently leaves the whole encoder randomly
initialised (different poses every process). The loader now raises
`RuntimeError: ... backbone tensors are missing ...` in that case. Fix:

```bash
hf download facebook/sam-3d-body-dinov3 model.ckpt --local-dir checkpoints/sam-3d-body-dinov3
```

## 3. Install

### Docker (recommended)

The worker image does not ship TensorRT. `Dockerfile.quantized` layers TensorRT + ONNX on top of
`mordommin94/fbxify-worker:latest` and copies this checkout:

```bash
docker build -f fbxify/docker/Dockerfile.quantized -t mordommin94/fbxify-worker:quantized .
```

Serve the worker API with the INT8 encoder:

```bash
docker run --rm --gpus all --shm-size=8g -p 8000:8000 \
  -e FBXIFY_MODEL=dinov3 -e FBXIFY_QUANTIZED_BACKBONE=trt -e HF_TOKEN=$HF_TOKEN \
  -v "$PWD/checkpoints":/fbxify/checkpoints -v "$PWD/cache":/fbxify/cache \
  mordommin94/fbxify-worker:quantized
```

With docker-compose: `FBXIFY_MODEL=dinov3 FBXIFY_QUANTIZED_BACKBONE=trt docker compose -f fbxify/docker/docker-compose.yml up`
(point the worker `image:` at the quantized tag).

The first `trt` start builds the engine (a few minutes) and caches it next to the ONNX as
`embedl_sam3dbody_int8.<gpu>.trt<version>.engine`. Engines are GPU- and TensorRT-version-specific;
keep the checkpoints volume writable so the cache persists.

### Bare metal (conda / venv)

On top of `fbxify/Install.md`:

```bash
pip install "tensorrt-cu12>=10.16,<11" onnx      # CUDA 12 driver/toolkit (matches torch cu128)
# pip install "tensorrt>=11" onnx                 # CUDA 13 stack instead
python -c "import tensorrt as trt; print(trt.__version__)"
```

Then run anything with `--quantized_backbone trt` (or `pt2`), e.g.

```bash
python fbxify/app.py --model dinov3 --quantized_backbone trt
python -m fbxify.cli --model dinov3 --quantized_backbone trt --fast_inference input/20f.mp4 ...
```

Notes from Embedl: TensorRT 11 (CUDA 13) removed the weakly-typed FP16/INT8 builder flags; the
builder in `sam_3d_body/models/backbones/embedl_int8.py` sets them when present, which is the
TensorRT 10 path we test. On TRT 11 you would cast the fp32 ONNX to fp16 first (their
`demo3d_trt.py` shows how).

## 4. Benchmark matrix (quantized × fast)

`fbxify/cli_bench_quantized.py` runs one configuration per process:
load → 20 frames of `input/20f.mp4` (bboxes from `input/20f_bboxes.txt`) → flush CUDA cache →
20 evenly spaced frames of `input/ai.mp4` (detector) → `results.json` + `poses_{a,b}.npz`.
`fbxify/cli_compare_quantized.py` then compares every configuration against `fp32-normal`
(per-vertex mm, arms/legs/hands/head keypoint mm and px, joint-rotation degrees) and renders
overlay + heatmap panels. Frames where a configuration reconstructed a *different person* than
the baseline (camera translation jump > 30 cm or mean vertex shift > 15 cm) are reported as
`target_switch_frames` and excluded from the `stable` pose-precision numbers, so a crowd-scene
identity flip does not masquerade as quantization noise.

```bash
HF_TOKEN=hf_xxx ./fbxify/docker/run_quant_bench.sh       # Linux/macOS/WSL
set HF_TOKEN=hf_xxx & fbxify\docker\run_quant_bench.bat  # Windows
```

Outputs land in `workspace/quant_bench/`:

```
fp32-normal/ fp32-fast/ trt-normal/ trt-fast/   results.json, poses_a.npz, poses_b.npz, frames_*/
compare/summary.json                            numbers used by the canvas
compare/<config>/pass_<a|b>_frame_NN_*.png       skeleton overlay | baseline mesh | candidate mesh | heatmap front | side
compare/grid_pass_<a|b>_frame_NN.png             all configurations for the same frame
```

Run a subset with `run_quant_bench.sh fp32-normal fp32-fast`, or `QUANT_MODE=pt2` for the
torch.export variant.

## Troubleshooting

- `embedl INT8 encoder files ... not found` — download step 2, or set `HF_TOKEN` and make the
  checkpoints mount writable.
- `cudaErrorInsufficientDriver` — you installed a cu13 TensorRT wheel on a CUDA 12 driver; use
  `tensorrt-cu12`.
- Engine build OOM — lower the workspace: `EmbedlInt8Backbone(..., trt_workspace_gb=3)`; 12 GB
  cards build fine with the default 6 GB.
- Different GPU than the cached engine — delete the `.engine` file or pass
  `--trt_rebuild` to the bench.
