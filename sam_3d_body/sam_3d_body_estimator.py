# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import Any, Dict, List, Optional, Union

import cv2

import numpy as np
import torch

from sam_3d_body.data.transforms import (
    Compose,
    GetBBoxCenterScale,
    TopdownAffine,
    VisionTransformWrapper,
)

from sam_3d_body.data.utils.io import load_image
from sam_3d_body.data.utils.prepare_batch import prepare_batch, prepare_multiframe_batch
from sam_3d_body.utils import recursive_to
from torchvision.transforms import ToTensor


""" BATCHING FIX FOR FBXIFY - SAFE_LEN FIX """ 
def _safe_len(obj):
    """len() that handles 0-d numpy arrays gracefully."""
    if isinstance(obj, np.ndarray) and obj.ndim == 0:
        return 0
    return len(obj)
""" BATCHING FIX FOR FBXIFY - SAFE_LEN FIX """ 

class SAM3DBodyEstimator:
    def __init__(
        self,
        sam_3d_body_model,
        model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    ):
        self.device = sam_3d_body_model.device
        self.model, self.cfg = sam_3d_body_model, model_cfg
        self.detector = human_detector
        self.sam = human_segmentor
        self.fov_estimator = fov_estimator
        self.thresh_wrist_angle = 1.4

        # For mesh visualization
        self.faces = self.model.head_pose.faces.cpu().numpy()

        if self.detector is None:
            print("No human detector is used...")
        if self.sam is None:
            print("Mask-condition inference is not supported...")
        if self.fov_estimator is None:
            print("No FOV estimator... Using the default FOV!")

        self.transform = Compose(
            [
                GetBBoxCenterScale(),
                TopdownAffine(input_size=self.cfg.MODEL.IMAGE_SIZE, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )
        self.transform_hand = Compose(
            [
                GetBBoxCenterScale(padding=0.9),
                TopdownAffine(input_size=self.cfg.MODEL.IMAGE_SIZE, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )

    @torch.no_grad()
    def process_one_image(
        self,
        img: Union[str, np.ndarray],
        bboxes: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        cam_int: Optional[np.ndarray] = None,
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        use_mask: bool = False,
        inference_type: str = "full",
    ):
        """
        Perform model prediction in top-down format: assuming input is a full image.

        Args:
            img: Input image (path or numpy array)
            bboxes: Optional pre-computed bounding boxes
            masks: Optional pre-computed masks (numpy array). If provided, SAM2 will be skipped.
            det_cat_id: Detection category ID
            bbox_thr: Bounding box threshold
            nms_thr: NMS threshold
            inference_type:
                - full: full-body inference with both body and hand decoders
                - body: inference with body decoder only (still full-body output)
                - hand: inference with hand decoder only (only hand output)
        """

        # clear all cached results
        self.batch = None
        self.image_embeddings = None
        self.output = None
        self.prev_prompt = []
        torch.cuda.empty_cache()

        if type(img) == str:
            img = load_image(img, backend="cv2", image_format="bgr")
            image_format = "bgr"
        else:
            print("####### Please make sure the input image is in RGB format")
            image_format = "rgb"
        height, width = img.shape[:2]

        if bboxes is not None:
            boxes = bboxes.reshape(-1, 4)
            self.is_crop = True
        elif self.detector is not None:
            if image_format == "rgb":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                image_format = "bgr"
            boxes = self.detector.run_human_detection(
                img,
                det_cat_id=det_cat_id,
                bbox_thr=bbox_thr,
                nms_thr=nms_thr,
                default_to_full_image=False,
            )
            self.is_crop = True
        else:
            boxes = np.array([0, 0, width, height]).reshape(1, 4)
            self.is_crop = False

        # If there are no detected humans, don't run prediction
        """ BATCHING FIX FOR FBXIFY - SAFE_LEN FIX """ 
        if _safe_len(boxes) == 0:
            return []
        """ BATCHING FIX FOR FBXIFY - SAFE_LEN FIX """ 

        # The following models expect RGB images instead of BGR
        if image_format == "bgr":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Handle masks - either provided externally or generated via SAM2
        masks_score = None
        if masks is not None:
            # Use provided masks - ensure they match the number of detected boxes
            print(f"Using provided masks: {masks.shape}")
            assert (
                bboxes is not None
            ), "Mask-conditioned inference requires bboxes input!"
            masks = masks.reshape(-1, height, width, 1).astype(np.uint8)
            masks_score = np.ones(
                len(masks), dtype=np.float32
            )  # Set high confidence for provided masks
            use_mask = True
        elif use_mask and self.sam is not None:
            print("Running SAM to get mask from bbox...")
            # Generate masks using SAM2
            masks, masks_score = self.sam.run_sam(img, boxes)
        else:
            masks, masks_score = None, None

        #################### Construct batch data samples ####################
        batch = prepare_batch(img, self.transform, boxes, masks, masks_score)

        #################### Run model inference on an image ####################
        batch = recursive_to(batch, "cuda")
        self.model._initialize_batch(batch)

        # Handle camera intrinsics
        # - either provided externally or generated via default FOV estimator
        if cam_int is not None:
            cam_int = cam_int.to(batch["img"])
            batch["cam_int"] = cam_int.clone()
        elif self.fov_estimator is not None:
            input_image = batch["img_ori"][0].data
            cam_int = self.fov_estimator.get_cam_intrinsics(input_image).to(
                batch["img"]
            )
            batch["cam_int"] = cam_int.clone()
        else:
            cam_int = batch["cam_int"].clone()

        outputs = self.model.run_inference(
            img,
            batch,
            inference_type=inference_type,
            transform_hand=self.transform_hand,
            thresh_wrist_angle=self.thresh_wrist_angle,
        )
        if inference_type == "full":
            pose_output, batch_lhand, batch_rhand, _, _ = outputs
        else:
            pose_output = outputs

        out = pose_output["mhr"]
        out = recursive_to(out, "cpu")
        out = recursive_to(out, "numpy")
        all_out = []
        for idx in range(batch["img"].shape[1]):
            all_out.append(
                {
                    "bbox": batch["bbox"][0, idx].cpu().numpy(),
                    "focal_length": out["focal_length"][idx],
                    "pred_keypoints_3d": out["pred_keypoints_3d"][idx],
                    "pred_keypoints_2d": out["pred_keypoints_2d"][idx],
                    "pred_vertices": out["pred_vertices"][idx],
                    "pred_cam_t": out["pred_cam_t"][idx],
                    "pred_pose_raw": out["pred_pose_raw"][idx],
                    "global_rot": out["global_rot"][idx],
                    "body_pose_params": out["body_pose"][idx],
                    "hand_pose_params": out["hand"][idx],
                    "scale_params": out["scale"][idx],
                    "shape_params": out["shape"][idx],
                    "expr_params": out["face"][idx],
                    "mask": masks[idx] if masks is not None else None,
                    "pred_joint_coords": out["pred_joint_coords"][idx],
                    "pred_global_rots": out["joint_global_rots"][idx],
                    "mhr_model_params": out["mhr_model_params"][idx],
                }
            )

            if inference_type == "full":
                all_out[-1]["lhand_bbox"] = np.array(
                    [
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][0]
                            - batch_lhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][1]
                            - batch_lhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][0]
                            + batch_lhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][1]
                            + batch_lhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                    ]
                )
                all_out[-1]["rhand_bbox"] = np.array(
                    [
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][0]
                            - batch_rhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][1]
                            - batch_rhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][0]
                            + batch_rhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][1]
                            + batch_rhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                    ]
                )

        return all_out


    """ BATCHING SUPPORT FOR FBXIFY """
    @torch.no_grad()
    def process_batch_of_frames(
        self,
        frames: List[Dict[str, Any]],
        inference_type: str = "full",
    ) -> List[List[Dict[str, Any]]]:
        """
        Run pose estimation on multiple frames in one batched body forward, then
        hand refinement per frame. Returns the same structure as calling
        process_one_image for each frame: list of (list of per-person dicts per frame).

        Args:
            frames: List of dicts, each with "img" (path or ndarray), "bboxes" (xyxy, shape Nx4),
                optional "cam_int" (tensor or array). Optional "masks", "masks_score".
            inference_type: "full", "body", or "hand".

        Returns:
            [ all_out_frame_0, ..., all_out_frame_K-1 ] where each all_out_frame_i
            is a list of per-person dicts (same as process_one_image).
        """
        if not frames:
            return []
        if inference_type != "full":
            # Body-only or hand-only: no batching benefit; fall back to per-frame.
            return [
                self.process_one_image(
                    f["img"],
                    bboxes=f["bboxes"],
                    cam_int=f.get("cam_int"),
                    inference_type=inference_type,
                )
                for f in frames
            ]

        # Load images and normalize to RGB
        images = []
        cam_int_list = []
        for f in frames:
            img = f["img"]
            if isinstance(img, str):
                img = load_image(img, backend="cv2", image_format="bgr")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            cam_int_list.append(f.get("cam_int"))

        # Build frame_list for prepare_multiframe_batch: (img, boxes, masks, masks_score)
        frame_list = []
        for i, f in enumerate(frames):
            boxes = np.asarray(f["bboxes"])
            if boxes.ndim == 1:
                boxes = boxes.reshape(1, -1)
            """ BATCHING FIX FOR FBXIFY - SAFE_LEN FIX """ 
            if boxes.ndim == 0 or _safe_len(boxes) == 0:
                raise ValueError(f"Frame {i} has zero bboxes.")
            # if len(boxes) == 0:
            #     raise ValueError(f"Frame {i} has zero bboxes.")
            """ BATCHING FIX FOR FBXIFY - SAFE_LEN FIX """ 
            masks = f.get("masks")
            masks_score = f.get("masks_score")
            frame_list.append((images[i], boxes, masks, masks_score))

        batch = prepare_multiframe_batch(frame_list, self.transform, cam_int_list)
        n_per_frame = batch.pop("n_per_frame")
        K = len(n_per_frame)
        N_max = max(n_per_frame)

        batch = recursive_to(batch, "cuda")
        self.model._initialize_batch(batch)
        if self.fov_estimator is not None and all(c is None for c in cam_int_list):
            # Run FOV once on first frame and broadcast - avoids K sequential FOV forwards
            inp = batch["img_ori"][0].data
            c = self.fov_estimator.get_cam_intrinsics(inp)
            c = c.to(batch["img"]) if torch.is_tensor(c) else torch.tensor(c, device=batch["img"].device, dtype=batch["img"].dtype)
            while c.dim() > 2 and c.shape[0] == 1:
                c = c.squeeze(0)
            if c.dim() != 2 or c.shape != (3, 3):
                c = c.reshape(3, 3)
            batch["cam_int"] = c.unsqueeze(0).expand(K, 3, 3).clone()

        images = [batch["img_ori"][k].data for k in range(K)]
        pose_output, batch_lhand, batch_rhand = self.model.run_inference_batched(
            images=images,
            batch=batch,
            transform_hand=self.transform_hand,
            n_per_frame=n_per_frame,
            thresh_wrist_angle=self.thresh_wrist_angle,
        )

        out = pose_output["mhr"]
        out = recursive_to(out, "cpu")
        out = recursive_to(out, "numpy")

        # 4 syncs total (not 4*K) - move full batches once, then index
        bbox_cpu = batch["bbox"].cpu().numpy()
        bc_l = batch_lhand["bbox_center"].cpu().numpy()
        bs_l = batch_lhand["bbox_scale"].cpu().numpy()
        bc_r = batch_rhand["bbox_center"].cpu().numpy()
        bs_r = batch_rhand["bbox_scale"].cpu().numpy()
        lhand_bbox_cpu = []
        rhand_bbox_cpu = []
        for k in range(K):
            n_k = n_per_frame[k]
            for idx in range(n_k):
                lhand_bbox_cpu.append(np.array([
                    bc_l[k, idx, 0] - bs_l[k, idx, 0] / 2,
                    bc_l[k, idx, 1] - bs_l[k, idx, 1] / 2,
                    bc_l[k, idx, 0] + bs_l[k, idx, 0] / 2,
                    bc_l[k, idx, 1] + bs_l[k, idx, 1] / 2,
                ]))
                rhand_bbox_cpu.append(np.array([
                    bc_r[k, idx, 0] - bs_r[k, idx, 0] / 2,
                    bc_r[k, idx, 1] - bs_r[k, idx, 1] / 2,
                    bc_r[k, idx, 0] + bs_r[k, idx, 0] / 2,
                    bc_r[k, idx, 1] + bs_r[k, idx, 1] / 2,
                ]))

        person_idx = 0
        result = []
        for k in range(K):
            n_k = n_per_frame[k]
            all_out_k = []
            start = k * N_max
            masks_k = frame_list[k][2] if frame_list[k][2] is not None else None
            for idx in range(n_k):
                global_idx = start + idx
                all_out_k.append({
                    "bbox": bbox_cpu[k, idx],
                    "focal_length": out["focal_length"][global_idx],
                    "pred_keypoints_3d": out["pred_keypoints_3d"][global_idx],
                    "pred_keypoints_2d": out["pred_keypoints_2d"][global_idx],
                    "pred_vertices": out["pred_vertices"][global_idx],
                    "pred_cam_t": out["pred_cam_t"][global_idx],
                    "pred_pose_raw": out["pred_pose_raw"][global_idx],
                    "global_rot": out["global_rot"][global_idx],
                    "body_pose_params": out["body_pose"][global_idx],
                    "hand_pose_params": out["hand"][global_idx],
                    "scale_params": out["scale"][global_idx],
                    "shape_params": out["shape"][global_idx],
                    "expr_params": out["face"][global_idx],
                    "mask": masks_k[idx] if masks_k is not None and idx < len(masks_k) else None,
                    "pred_joint_coords": out["pred_joint_coords"][global_idx],
                    "pred_global_rots": out["joint_global_rots"][global_idx],
                    "mhr_model_params": out["mhr_model_params"][global_idx],
                })
                all_out_k[-1]["lhand_bbox"] = lhand_bbox_cpu[person_idx]
                all_out_k[-1]["rhand_bbox"] = rhand_bbox_cpu[person_idx]
                person_idx += 1
            result.append(all_out_k)
        return result

    """ BATCHING SUPPORT FOR FBXIFY """