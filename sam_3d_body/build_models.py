# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import torch

from .models.meta_arch import SAM3DBody
from .utils.config import get_config
from .utils.checkpoint import load_state_dict


def load_sam_3d_body(
    checkpoint_path: str = "",
    device: str = "cuda",
    mhr_path: str = "",
    require_backbone_weights: bool = True,
):
    """Build SAM3DBody from ``model_config.yaml`` next to ``checkpoint_path`` and load it.

    ``require_backbone_weights=False`` skips the encoder-coverage check; use it only when
    the backbone is replaced right after loading (e.g. the embedl INT8 encoder).
    """
    print("Loading SAM 3D Body model...")
    
    # Check the current directory, and if not present check the parent dir.
    model_cfg = os.path.join(os.path.dirname(checkpoint_path), "model_config.yaml")
    if not os.path.exists(model_cfg):
        # Looks at parent dir
        model_cfg = os.path.join(
            os.path.dirname(os.path.dirname(checkpoint_path)), "model_config.yaml"
        )

    model_cfg = get_config(model_cfg)

    # Disable face for inference
    model_cfg.defrost()
    model_cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = mhr_path
    model_cfg.freeze()

    # Initialze the model
    model = SAM3DBody(model_cfg)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    load_state_dict(model, state_dict, strict=False)
    if require_backbone_weights:
        _assert_backbone_weights_loaded(model, state_dict, checkpoint_path, model_cfg)

    model = model.to(device)
    model.eval()
    return model, model_cfg


def _assert_backbone_weights_loaded(model, state_dict, checkpoint_path, model_cfg):
    """Fail loudly if the checkpoint does not cover the image encoder.

    ``load_state_dict(strict=False)`` only logs a warning when keys are missing, so a
    checkpoint built for a different backbone (e.g. the ViT-H ``model.ckpt`` placed in
    the ``sam-3d-body-dinov3`` folder) silently leaves the whole DINOv3 encoder at its
    random initialisation. Every process then produces different, meaningless poses.
    """
    backbone_keys = [k for k in model.backbone.state_dict().keys()]
    if not backbone_keys:
        return
    missing = [k for k in backbone_keys if f"backbone.{k}" not in state_dict]
    if len(missing) * 2 > len(backbone_keys):
        raise RuntimeError(
            f"{len(missing)}/{len(backbone_keys)} backbone tensors are missing from "
            f"{checkpoint_path} (backbone type {model_cfg.MODEL.BACKBONE.TYPE!r}); the image "
            "encoder would run with random weights. The checkpoint does not belong to this "
            "model_config.yaml - re-download the matching model.ckpt "
            "(facebook/sam-3d-body-dinov3 is ~2.1 GB, facebook/sam-3d-body-vith ~1.7 GB)."
        )


def _hf_download(repo_id):
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(repo_id=repo_id)
    return os.path.join(local_dir, "model.ckpt"), os.path.join(local_dir, "assets", "mhr_model.pt")


def load_sam_3d_body_hf(repo_id, **kwargs):
    ckpt_path, mhr_path = _hf_download(repo_id)
    return load_sam_3d_body(checkpoint_path=ckpt_path, mhr_path=mhr_path)
