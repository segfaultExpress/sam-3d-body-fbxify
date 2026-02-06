# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch
from torch.utils.data import default_collate


class NoCollate:
    def __init__(self, data):
        self.data = data


def prepare_batch(
    img,
    transform,
    boxes,
    masks=None,
    masks_score=None,
    cam_int=None,
):
    """A helper function to prepare data batch for SAM 3D Body model inference."""
    height, width = img.shape[:2]

    # construct batch data samples
    data_list = []
    for idx in range(boxes.shape[0]):
        data_info = dict(img=img)
        data_info["bbox"] = boxes[idx]  # shape (4,)
        data_info["bbox_format"] = "xyxy"

        if masks is not None:
            data_info["mask"] = masks[idx].copy()
            if masks_score is not None:
                data_info["mask_score"] = masks_score[idx]
            else:
                data_info["mask_score"] = np.array(1.0, dtype=np.float32)
        else:
            data_info["mask"] = np.zeros((height, width, 1), dtype=np.uint8)
            data_info["mask_score"] = np.array(0.0, dtype=np.float32)

        data_list.append(transform(data_info))

    batch = default_collate(data_list)

    max_num_person = batch["img"].shape[0]
    for key in [
        "img",
        "img_size",
        "ori_img_size",
        "bbox_center",
        "bbox_scale",
        "bbox",
        "affine_trans",
        "mask",
        "mask_score",
    ]:
        if key in batch:
            batch[key] = batch[key].unsqueeze(0).float()
    if "mask" in batch:
        batch["mask"] = batch["mask"].unsqueeze(2)
    batch["person_valid"] = torch.ones((1, max_num_person))

    if cam_int is not None:
        batch["cam_int"] = cam_int.to(batch["img"])
    else:
        # Default camera intrinsics according image size
        batch["cam_int"] = torch.tensor(
            [
                [
                    [(height**2 + width**2) ** 0.5, 0, width / 2.0],
                    [0, (height**2 + width**2) ** 0.5, height / 2.0],
                    [0, 0, 1],
                ]
            ],
        ).to(batch["img"])

    batch["img_ori"] = [NoCollate(img)]
    return batch

""" BATCHING SUPPORT FOR FBXIFY """
def prepare_multiframe_batch(
    frame_list,
    transform,
    cam_int_list=None,
):
    """
    Build a single batch from multiple frames for batched pose inference.

    Each frame is (img, boxes) or (img, boxes, masks, masks_score).
    Produces batch with shape (K, N_max, ...) where K = len(frame_list) and
    N_max = max number of persons across frames. Frames with fewer persons
    are padded (repeat last person); use person_valid to mask invalid slots.

    Args:
        frame_list: List of per-frame inputs. Each element is either
            (img, boxes) or (img, boxes, masks, masks_score).
            boxes: (N_k, 4) in xyxy format.
        transform: Same transform as in prepare_batch (Compose with GetBBoxCenterScale, etc.).
        cam_int_list: Optional list of K camera intrinsics (tensors or None).
            If None for a frame, default intrinsics from image size are used.

    Returns:
        batch: Dict with batch["img"] shape (K, N_max, C, H, W), batch["person_valid"] (K, N_max),
            batch["cam_int"] (K, 3, 3), batch["img_ori"] = [NoCollate(img_0), ...].
    """
    K = len(frame_list)
    if K == 0:
        raise ValueError("frame_list must not be empty")

    # Normalize frame_list entries to (img, boxes, masks, masks_score)
    normalized = []
    for item in frame_list:
        if len(item) == 2:
            img, boxes = item[0], item[1]
            masks, masks_score = None, None
        elif len(item) >= 4:
            img, boxes, masks, masks_score = item[0], item[1], item[2], item[3]
        else:
            raise ValueError("Each frame must be (img, boxes) or (img, boxes, masks, masks_score)")
        normalized.append((img, boxes, masks, masks_score))

    # Build single-frame batch for each frame
    single_batches = []
    n_per_frame = []
    for i, (img, boxes, masks, masks_score) in enumerate(normalized):
        boxes = np.asarray(boxes)
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)
        if len(boxes) == 0:
            raise ValueError(f"Frame {i} has zero boxes; cannot include in multiframe batch.")
        cam_int = cam_int_list[i] if cam_int_list is not None and i < len(cam_int_list) else None
        single = prepare_batch(img, transform, boxes, masks, masks_score, cam_int)
        single_batches.append(single)
        n_per_frame.append(single["img"].shape[1])

    N_max = max(n_per_frame)
    device = single_batches[0]["img"].device
    dtype = single_batches[0]["img"].dtype

    # Pad each single batch to N_max along the person dimension and stack to (K, N_max, ...)
    batch_keys = [
        "img", "img_size", "ori_img_size", "bbox_center", "bbox_scale", "bbox",
        "affine_trans", "mask", "mask_score",
    ]
    stacked = {}
    for key in batch_keys:
        if key not in single_batches[0]:
            continue
        parts = []
        for k, single in enumerate(single_batches):
            t = single[key]  # (1, N_k, ...)
            n_k = t.shape[1]
            if n_k < N_max:
                # Repeat last person to pad
                last = t[:, -1:].expand(1, N_max - n_k, *t.shape[2:]).clone()
                t = torch.cat([t, last], dim=1)
            parts.append(t)
        stacked[key] = torch.cat(parts, dim=0)

    # person_valid: (K, N_max), 1 where valid, 0 for padding
    person_valid = torch.zeros((K, N_max), device=device, dtype=dtype)
    for k, n_k in enumerate(n_per_frame):
        person_valid[k, :n_k] = 1

    stacked["person_valid"] = person_valid

    # cam_int: stack to (K, 3, 3) so get_ray_condition indexing [:, None, None, None, [0,1], [2,2]] works
    cam_parts = []
    for single in single_batches:
        c = single["cam_int"]
        while c.dim() > 2 and c.shape[0] == 1:
            c = c.squeeze(0)
        if c.dim() != 2 or c.shape != (3, 3):
            c = c.reshape(3, 3)
        cam_parts.append(c)
    cam_stacked = torch.stack(cam_parts, dim=0)  # (K, 3, 3)
    stacked["cam_int"] = cam_stacked.to(device=device, dtype=dtype)

    # mask: currently (K, N_max, ..., 1) after unsqueeze(2) in prepare_batch; ensure same
    if "mask" in stacked and stacked["mask"].dim() == 5:
        pass
    elif "mask" in stacked and stacked["mask"].dim() == 4:
        stacked["mask"] = stacked["mask"].unsqueeze(2)

    stacked["img_ori"] = [NoCollate(normalized[k][0]) for k in range(K)]
    stacked["n_per_frame"] = n_per_frame  # list of ints for splitting outputs

    return stacked

""" BATCHING SUPPORT FOR FBXIFY """