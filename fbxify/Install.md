# FBXify: All-in-One Installation (No Docker)

This guide covers installing FBXify with the Gradio UI and GPU processing in one environment.

## Prerequisites

- **OS:** Linux (Ubuntu 22.04+ recommended) or Windows with WSL2
- **GPU:** NVIDIA GPU with CUDA 12.x support
- **Blender:** 5.0 (for FBX export)
- **Python:** 3.12

## 1. System Dependencies (Linux)

```bash
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-dev python3.12-venv python3-pip \
  git ffmpeg build-essential cmake wget \
  libgl1 libegl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
  libosmesa6 libosmesa6-dev
```

On Windows, install Python 3.12 and ensure `ffmpeg` is in PATH.

## 2. Blender 5.0

Download and install Blender 5.0 so `blender` is in PATH:

- Linux: https://download.blender.org/release/Blender5.0/blender-5.0.0-linux-x64.tar.xz
- Windows: https://download.blender.org/release/Blender5.0/

Then install mediapipe in Blender's Python:

```bash
<blender_install>/5.0/python/bin/python3.11 -m ensurepip
<blender_install>/5.0/python/bin/python3.11 -m pip install mediapipe
```

## 3. Clone and Create venv

```bash
git clone https://github.com/segfaultExpress/sam-3d-body-fbxify.git
cd sam-3d-body-fbxify
python3.12 -m venv .venv
source .venv/bin/activate   # Linux/Mac
# or: .venv\Scripts\activate   # Windows
```

## 4. Install PyTorch (CUDA)

Install PyTorch with CUDA 12.8 support first:

```bash
pip install "numpy<2"
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
```

## 5. Install Python Dependencies

From the repo root:

```bash
pip install -r fbxify/requirements-full.txt
```

## 6. Special Packages (install in order)

These require special flags and must be installed separately:

```bash
pip install --no-build-isolation xtcocotools
pip install "git+https://github.com/facebookresearch/detectron2.git@a1ce2f9" --no-build-isolation --no-deps
```

For flash-attn, try:

```bash
pip install flash-attn
```

If that fails (e.g. no pre-built wheel for your platform), build from source:

```bash
pip install ninja packaging
pip install flash-attn --no-build-isolation
```

## 7. Model Checkpoints

Request access at https://huggingface.co/facebook/sam-3d-body-dinov3 (or `sam-3d-body-vith`), then:

```bash
huggingface-cli login
huggingface-cli download facebook/sam-3d-body-dinov3 --local-dir checkpoints/sam-3d-body-dinov3
```

## 8. Environment Variables

```bash
export PYOPENGL_PLATFORM=osmesa   # or egl on headless Linux
export MOMENTUM_ENABLED=True
export MPLBACKEND=Agg
export WANDB_MODE=offline
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

On Windows (PowerShell):

```powershell
$env:PYOPENGL_PLATFORM="osmesa"
$env:MOMENTUM_ENABLED="True"
$env:MPLBACKEND="Agg"
$env:WANDB_MODE="offline"
$env:PYTHONPATH="$PWD;$env:PYTHONPATH"
```

## 9. Run

```bash
python fbxify/app.py
```

Open http://localhost:7444

## Optional: Detector Cache

To avoid re-downloading the ViTDet detector on first run, you can pre-download it and set `SAM3D_DETECTOR_PATH` to a folder containing `model_final_f05665.pkl`. Otherwise it downloads automatically on first use.

## Troubleshooting

- **OpenGL/EGL errors on headless:** Use `export PYOPENGL_PLATFORM=osmesa` or `egl`.
- **Blender not found:** Ensure `blender` is in PATH or the Blender executable path is correct.
- **Checkpoint not found:** Run the huggingface-cli download step and ensure `checkpoints/sam-3d-body-dinov3` exists with `model.ckpt` and `assets/mhr_model.pt`.
