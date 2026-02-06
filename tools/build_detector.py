# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from pathlib import Path
from typing import List, Union

import numpy as np
import torch


class HumanDetector:
    def __init__(self, name="vitdet", device="cuda", **kwargs):
        self.device = device

        if name == "vitdet":
            print("########### Using human detector: ViTDet...")
            self.detector = load_detectron2_vitdet(**kwargs)
            self.detector_func = run_detectron2_vitdet
            self.detector_func_batch = run_detectron2_vitdet_batch

            self.detector = self.detector.to(self.device)
            self.detector.eval()
        else:
            raise NotImplementedError

    def run_human_detection(self, img, **kwargs):
        return self.detector_func(self.detector, img, **kwargs)

    """ BATCHING SUPPORT FOR FBXIFY """
    def run_human_detection_batch(
        self, list_of_images: List[Union[np.ndarray, "torch.Tensor"]], **kwargs
    ) -> List[np.ndarray]:
        """
        Run human detection on a list of images in one forward pass.

        Args:
            list_of_images: List of images (each HxWx3 BGR numpy or tensor).

        Returns:
            List of box arrays, each shape (N_i, 4) in xyxy format.
        """
        if len(list_of_images) == 0:
            return []
        if len(list_of_images) == 1:
            boxes = self.run_human_detection(list_of_images[0], **kwargs)
            return [boxes if boxes is not None and len(boxes) > 0 else np.empty((0, 4), dtype=np.float32)]
        return self.detector_func_batch(self.detector, list_of_images, **kwargs)
    """ BATCHING SUPPORT FOR FBXIFY """

def load_detectron2_vitdet(path=""):
    """
    Load vitdet detector similar to 4D-Humans demo.py approach.
    Checkpoint is automatically downloaded from the hardcoded URL.
    """
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import instantiate, LazyConfig

    # Get config file from tools directory (same folder as this file)
    cfg_path = Path(__file__).parent / "cascade_mask_rcnn_vitdet_h_75ep.py"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {cfg_path}. "
            "Make sure cascade_mask_rcnn_vitdet_h_75ep.py exists in the tools directory."
        )

    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = (
        "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        if path == ""
        else os.path.join(path, "model_final_f05665.pkl")
    )
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = instantiate(detectron2_cfg.model)
    checkpointer = DetectionCheckpointer(detector)
    checkpointer.load(detectron2_cfg.train.init_checkpoint)

    detector.eval()
    return detector


def run_detectron2_vitdet(
    detector,
    img,
    det_cat_id: int = 0,
    bbox_thr: float = 0.5,
    nms_thr: float = 0.3,
    default_to_full_image: bool = True,
):
    import detectron2.data.transforms as T

    height, width = img.shape[:2]

    IMAGE_SIZE = 1024
    transforms = T.ResizeShortestEdge(short_edge_length=IMAGE_SIZE, max_size=IMAGE_SIZE)
    img_transformed = transforms(T.AugInput(img)).apply_image(img)
    img_transformed = torch.as_tensor(
        img_transformed.astype("float32").transpose(2, 0, 1)
    )
    inputs = {"image": img_transformed, "height": height, "width": width}

    with torch.no_grad():
        det_out = detector([inputs])

    det_instances = det_out[0]["instances"]
    valid_idx = (det_instances.pred_classes == det_cat_id) & (
        det_instances.scores > bbox_thr
    )
    if valid_idx.sum() == 0 and default_to_full_image:
        boxes = np.array([0, 0, width, height]).reshape(1, 4)
    else:
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

    # Sort boxes to keep a consistent output order
    sorted_indices = np.lexsort(
        (boxes[:, 3], boxes[:, 2], boxes[:, 1], boxes[:, 0])
    )  # shape: [len(boxes),]
    boxes = boxes[sorted_indices]
    return boxes


def run_detectron2_vitdet_batch(
    detector,
    list_of_imgs,
    det_cat_id: int = 0,
    bbox_thr: float = 0.5,
    nms_thr: float = 0.3,
    default_to_full_image: bool = False,
) -> List[np.ndarray]:
    """
    Run ViTDet on a list of images in one forward pass.

    Args:
        detector: Detectron2 model.
        list_of_imgs: List of images (each HxWx3 BGR numpy).

    Returns:
        List of box arrays, each shape (N_i, 4) xyxy.
    """
    import detectron2.data.transforms as T

    IMAGE_SIZE = 1024
    transforms = T.ResizeShortestEdge(short_edge_length=IMAGE_SIZE, max_size=IMAGE_SIZE)
    inputs_list = []
    for img in list_of_imgs:
        height, width = img.shape[:2]
        img_transformed = transforms(T.AugInput(img)).apply_image(img)
        img_tensor = torch.as_tensor(
            img_transformed.astype("float32").transpose(2, 0, 1)
        )
        inputs_list.append({
            "image": img_tensor,
            "height": height,
            "width": width,
        })

    with torch.no_grad():
        det_outputs = detector(inputs_list)

    result = []
    for i, det_out in enumerate(det_outputs):
        det_instances = det_out["instances"]
        valid_idx = (det_instances.pred_classes == det_cat_id) & (
            det_instances.scores > bbox_thr
        )
        height = inputs_list[i]["height"]
        width = inputs_list[i]["width"]
        if valid_idx.sum() == 0 and default_to_full_image:
            boxes = np.array([0, 0, width, height]).reshape(1, 4)
        elif valid_idx.sum() == 0:
            boxes = np.empty((0, 4), dtype=np.float32)
        else:
            boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        sorted_indices = np.lexsort(
            (boxes[:, 3], boxes[:, 2], boxes[:, 1], boxes[:, 0])
        )
        boxes = boxes[sorted_indices]
        result.append(boxes)
    return result
