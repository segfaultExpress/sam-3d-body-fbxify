# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from pathlib import Path
from typing import List, Union

import numpy as np
import torch

VITDET_WEIGHTS_FILENAME = "model_final_f05665.pkl"
VITDET_WEIGHTS_URL = (
    "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/"
    + VITDET_WEIGHTS_FILENAME
)

# Registry of detector types: name -> (loader, per-image runner, batch runner).
# Loaders take (device, **kwargs) and return a model; runners take (model, img(s), **run_kwargs)
# and return xyxy boxes sorted with np.lexsort like run_detectron2_vitdet. Register your own
# with ``register_detector_type`` and reference the type name from fbxify/models.json.
DETECTOR_TYPES = {}


def register_detector_type(name, loader, runner, batch_runner=None):
    DETECTOR_TYPES[name] = (loader, runner, batch_runner)


class HumanDetector:
    def __init__(self, name="vitdet", device="cuda", options=None, **kwargs):
        """
        name: detector type ("vitdet" / "detectron2_vitdet", "ultralytics", or a registered type).
        options: default keyword arguments merged into every run call (e.g. imgsz for YOLO).
        kwargs: forwarded to the loader (path=..., weights=...).
        """
        self.device = device
        self.name = name
        self.options = dict(options or {})

        if name in ("vitdet", "detectron2_vitdet"):
            print("########### Using human detector: ViTDet...")
            self.detector = load_detectron2_vitdet(**kwargs)
            self.detector_func = run_detectron2_vitdet
            self.detector_func_batch = run_detectron2_vitdet_batch

            self.detector = self.detector.to(self.device)
            self.detector.eval()
        elif name in ("ultralytics", "yolo"):
            print("########### Using human detector: Ultralytics YOLO...")
            self.detector = load_ultralytics(device=self.device, **kwargs)
            self.detector_func = run_ultralytics
            self.detector_func_batch = run_ultralytics_batch
        elif name in DETECTOR_TYPES:
            loader, runner, batch_runner = DETECTOR_TYPES[name]
            print(f"########### Using human detector: {name}...")
            self.detector = loader(device=self.device, **kwargs)
            self.detector_func = runner
            self.detector_func_batch = batch_runner or (
                lambda det, imgs, **kw: [runner(det, im, **kw) for im in imgs]
            )
        else:
            raise NotImplementedError(
                f"Unknown detector type {name!r}. Known: vitdet, ultralytics, {', '.join(DETECTOR_TYPES) or '-'}"
            )

    def run_human_detection(self, img, **kwargs):
        return self.detector_func(self.detector, img, **{**self.options, **kwargs})

    def release(self):
        """Drop the model so the caller can free GPU memory (gc + empty_cache)."""
        self.detector = None
        self.detector_func = None
        self.detector_func_batch = None

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
        return self.detector_func_batch(self.detector, list_of_images, **{**self.options, **kwargs})
    """ BATCHING SUPPORT FOR FBXIFY """


def _sort_boxes(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if len(boxes) == 0:
        return boxes
    order = np.lexsort((boxes[:, 3], boxes[:, 2], boxes[:, 1], boxes[:, 0]))
    return boxes[order]


# --------------------------------------------------------------------------- Ultralytics (opt in)
# Ultralytics is AGPL-3.0 and is not a dependency of this repo. Install it yourself
# (``pip install ultralytics``) and enable a detector of type "ultralytics" in fbxify/models.json.


def load_ultralytics(weights, device="cuda", **_unused):
    """Load a YOLO (Ultralytics) detector from a local .pt file listed in the model catalogue."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "The 'ultralytics' package is not installed. It is AGPL-3.0 licensed and opt-in: "
            "run `pip install ultralytics` in the worker image to use YOLO detectors."
        ) from exc
    if not weights or not os.path.isfile(weights):
        raise FileNotFoundError(f"Ultralytics weights not found: {weights!r}")
    model = YOLO(weights, task="detect")
    try:
        model.to(device)
    except Exception:
        pass
    return model


def _ultralytics_predict(model, imgs, det_cat_id, bbox_thr, nms_thr, imgsz, classes, half, max_det):
    with torch.no_grad():
        results = model.predict(
            imgs,
            imgsz=int(imgsz),
            conf=float(bbox_thr),
            iou=float(nms_thr),
            classes=list(classes) if classes is not None else [int(det_cat_id)],
            half=bool(half),
            max_det=int(max_det),
            verbose=False,
        )
    out = []
    for res in results:
        if res.boxes is None or len(res.boxes) == 0:
            out.append(np.empty((0, 4), dtype=np.float32))
            continue
        xyxy = res.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        out.append(_sort_boxes(xyxy))
    return out


def run_ultralytics(
    detector,
    img,
    det_cat_id: int = 0,
    bbox_thr: float = 0.5,
    nms_thr: float = 0.3,
    default_to_full_image: bool = True,
    imgsz: int = 640,
    classes=None,
    half: bool = False,
    max_det: int = 100,
):
    """Same contract as run_detectron2_vitdet: BGR image in, sorted xyxy boxes out."""
    height, width = img.shape[:2]
    boxes = _ultralytics_predict(detector, [img], det_cat_id, bbox_thr, nms_thr, imgsz, classes, half, max_det)[0]
    if len(boxes) == 0 and default_to_full_image:
        boxes = np.array([[0, 0, width, height]], dtype=np.float32)
    return boxes


def run_ultralytics_batch(
    detector,
    list_of_imgs,
    det_cat_id: int = 0,
    bbox_thr: float = 0.5,
    nms_thr: float = 0.3,
    default_to_full_image: bool = False,
    imgsz: int = 640,
    classes=None,
    half: bool = False,
    max_det: int = 100,
) -> List[np.ndarray]:
    if len(list_of_imgs) == 0:
        return []
    boxes_list = _ultralytics_predict(detector, list(list_of_imgs), det_cat_id, bbox_thr, nms_thr, imgsz, classes, half, max_det)
    if default_to_full_image:
        for i, img in enumerate(list_of_imgs):
            if len(boxes_list[i]) == 0:
                h, w = img.shape[:2]
                boxes_list[i] = np.array([[0, 0, w, h]], dtype=np.float32)
    return boxes_list

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
    local_pkl = os.path.join(path, VITDET_WEIGHTS_FILENAME) if path else ""
    if local_pkl and os.path.isfile(local_pkl):
        detectron2_cfg.train.init_checkpoint = local_pkl
    else:
        # Not fetched yet: let detectron2 download into its iopath cache (the worker entrypoint
        # links that cache to CACHE_DIR/videt_checkpoint, the same folder the catalogue uses).
        if local_pkl:
            print(f"ViTDet weights not found at {local_pkl}; downloading via detectron2.", flush=True)
        detectron2_cfg.train.init_checkpoint = VITDET_WEIGHTS_URL
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
