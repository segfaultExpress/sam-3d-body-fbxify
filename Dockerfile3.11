# Build from sam-3d-body repo root:
#   docker build -t sam-3d-body .
#
# Example run:
#   docker run --gpus all --shm-size=8g -it \
#     -v $PWD:/workspace/sam-3d-body \
#     sam-3d-body \
#     python demo.py \
#       --image_folder /workspace/sam-3d-body/example_images \
#       --output_folder /workspace/sam-3d-body/output \
#       --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
#       --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ---- System deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-venv python3-pip \
    git \
    ffmpeg \
    build-essential \
    cmake \
    wget \
    # OpenGL libs (we'll use EGL, OSMesa is mostly harmless to keep)
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libosmesa6 \
    libosmesa6-dev \
    libegl1-mesa \
    && rm -rf /var/lib/apt/lists/*

# ---- Blender installation ----
RUN cd /opt && \
    wget https://download.blender.org/release/Blender5.0/blender-5.0.0-linux-x64.tar.xz && \
    tar -xvf blender-5.0.0-linux-x64.tar.xz && \
    mv blender-5.0.0-linux-x64 blender && \
    rm -f /usr/local/bin/blender && \
    ln -s /opt/blender/blender /usr/local/bin/blender && \
    rm -f blender-5.0.0-linux-x64.tar.xz

# ---- Blender Python setup ----
RUN /opt/blender/5.0/python/bin/python3.11 -m ensurepip && \
    /opt/blender/5.0/python/bin/python3.11 -m pip install --upgrade pip && \
    /opt/blender/5.0/python/bin/python3.11 -m pip install mediapipe

# ---- Python venv ----
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# ---- Core Python tooling ----
RUN pip install --upgrade pip setuptools wheel

# ---- NumPy + modern cu118 PyTorch (key part) ----
RUN pip install "numpy<2"
RUN pip install --index-url https://download.pytorch.org/whl/cu118 \
    torch torchvision torchaudio

# ---- Copy repo ----
WORKDIR /workspace
COPY . .

# ---- SAM-3D-Body Python deps ----
RUN pip install \
      pytorch-lightning \
      roma \
      pyrender \
      opencv-python \
      yacs \
      scikit-image \
      einops \
      timm \
      dill \
      pandas \
      rich \
      hydra-core \
      hydra-submitit-launcher \
      hydra-colorlog \
      pyrootutils \
      webdataset \
      chump \
      "networkx==3.2.1" \
      joblib \
      seaborn \
      wandb \
      appdirs \
      ffmpeg-python \
      cython \
      jsonlines \
      pytest \
      xtcocotools \
      loguru \
      optree \
      fvcore \
      black \
      pycocotools \
      tensorboard \
      huggingface_hub \
      pyopengl \
      pyopengl-accelerate \
      git+https://github.com/microsoft/MoGe.git

# ---- Detectron2 (commit used in their INSTALL.md) ----
RUN pip install \
      "git+https://github.com/facebookresearch/detectron2.git@a1ce2f9" \
      --no-build-isolation --no-deps

# ---- Headless rendering + paths ----
ENV PYOPENGL_PLATFORM=egl
ENV MPLBACKEND=Agg
ENV WANDB_MODE=offline
ENV PYTHONPATH=/workspace:${PYTHONPATH}

CMD ["python", "demo.py", "--help"]
