# Docker Build Variants

## Overview

| Variant | Image | Use case |
|---------|-------|----------|
| **simple** | — | Standalone. Runs the entire thing locally (UI + GPU processing in one container). Use this unless you know what you're doing. |
| **ui** | `fbxify-ui` | Gradio web interface only (no GPU). Connects to a remote worker. |
| **worker** | `fbxify-worker` | GPU processing backend. Runs FastAPI API for pose estimation and FBX generation. |

---

## Simple

- **Purpose:** Standalone deployment — UI and GPU processing in one container
- **Use when:** You want everything running locally without the UI/worker split. Again, use this unless you know what you're doing.

**Build:** `build_docker_simple.bat`

---

## UI (`Dockerfile.ui`)

- **Base:** `python:3.12-slim` (lightweight, no CUDA)
- **Contents:** Gradio app, no SAM 3D Body models, no Blender, no GPU deps
- **Exposes:** Port 7444
- **Requires:** `FBXIFY_REMOTE_WORKER_URL` pointing to a worker container (e.g. `http://worker:8000`)
- **Use when:** You want a separate lightweight UI that delegates all processing to a remote worker (e.g. in docker-compose with ui + worker)

**Build:** `build_docker_ui.bat` → `fbxify-ui`

---

## Worker (`Dockerfile.worker`)

- **Base:** `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`
- **Contents:** Full stack — SAM 3D Body, Blender, PyTorch, detectron2, flash-attn, FastAPI/uvicorn
- **Exposes:** Port 8000 (pose/FBX job API)
- **Requires:** GPU (`--gpus all`), `--shm-size=8g`
- **Use when:** You need the GPU processing backend, either standalone or paired with the UI container

**Build:** `build_docker_worker.bat` → `fbxify-worker`

---

## Typical setups

- **Simple:** Run everything in one container — Gradio UI with local GPU backend.
- **Split (ui + worker):** Run `fbxify-ui` and `fbxify-worker` together; UI calls the worker API.
- **Worker only:** Run `fbxify-worker` and call its API directly (e.g. from another app or the host).

---

## Run UI + worker locally

**1. Build both images** (if not already built):
```
build_docker_ui.bat
build_docker_worker.bat
```

**2. Ensure checkpoints exist** at `checkpoints/sam-3d-body-vith/` (or set `CHECKPOINTS_DIR` to your path).

**3. Start both:**
```
run_ui_worker.bat
```

Or with docker compose directly (from repo root):
```
docker compose -f fbxify/docker/docker-compose.yml up
```

**4. Open** http://localhost:7444

The worker starts first and loads models (~30s). The UI waits for the worker to be healthy before starting.
