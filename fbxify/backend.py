"""
Backend abstraction for pose estimation and FBX generation.

LocalBackend delegates to FbxifyManager (in-process, loads models).
RemoteBackend calls worker API via HTTP (no model loading on UI).
"""
from __future__ import annotations

import json
import os
import tempfile
import shutil
import uuid
import time
from typing import Any, Dict, List, Optional, Tuple, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from fbxify.fbxify_manager import FbxifyManager
    from fbxify.tracking.tracking_manager import TrackingManager


class Backend(Protocol):
    """Protocol for pose estimation and FBX generation backends."""

    def run_pose_estimation(
        self,
        input_file: Any,
        tracking_mode: Any,
        bbox_file: Any,
        num_people: Any,
        missing_bbox_behavior: Any,
        fov_method: Any,
        fov_file: Any,
        sample_number: Any,
        precision: Any,
        output_tracking_bbox: Any,
        tracking_config: Any,
        frame_batch_size: Any,
        detection_batch_size: Any,
        lang: str,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[Any, Any, Optional[str], Any, Any]:
        """Run pose estimation. Returns (pose_json_files, selected_json, pose_json_state, generate_fbx_btn, estimate_pose_btn)."""
        ...

    def run_fbx_generation(
        self,
        pose_json_file: Any,
        profile_name: Any,
        use_root_motion: Any,
        auto_floor: Any,
        include_mesh: Any,
        include_extrinsics: Any,
        extrinsics_sample_rate: Any,
        extrinsics_scale: Any,
        extrinsics_invert_quaternion: Any,
        extrinsics_invert_translation: Any,
        extrinsics_file: Any,
        create_camera: Any,
        camera_zoom: Any,
        camera_scene: Any,
        use_personalized_body: Any,
        lod: Any,
        outlier_removal_percent: Any,
        export_personalized_body_obj: Any,
        graph_refinement: Any,
        input_file: Any,
        refinement_config: Any,
        lang: str,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[Optional[List], Any, Any]:
        """Run FBX generation. Returns (output_files, estimate_pose_btn, generate_fbx_btn)."""
        ...

    def run_detection_only(
        self,
        input_file: Any,
        detection_batch_size: Any,
        progress_callback: Optional[callable] = None,
    ) -> Optional[str]:
        """Run detection only, return MOT file path."""
        ...

    def rerun_tracking(
        self,
        estimation_file: Any,
        step_through: Any,
        debug_start_frame: Any,
        tracking_config_params: Dict[str, Any],
    ) -> Tuple[Any, Any, Optional[str], Any]:
        """Re-run tracking on estimation JSON. Returns (pose_json, fbx_json, selected, generate_btn)."""
        ...

    def cancel_current_job(self) -> None:
        """Signal to cancel any running job."""
        ...


class CancelledError(Exception):
    """Raised when a running job is cancelled."""
    pass


class LocalBackend:
    """Backend that delegates to FbxifyManager (in-process)."""

    def __init__(self, manager: "FbxifyManager", tracking_manager: "TrackingManager"):
        self.manager = manager
        self.tracking_manager = tracking_manager

    def run_pose_estimation(
        self,
        input_file: Any,
        tracking_mode: Any,
        bbox_file: Any,
        num_people: Any,
        missing_bbox_behavior: Any,
        fov_method: Any,
        fov_file: Any,
        sample_number: Any,
        precision: Any,
        output_tracking_bbox: Any,
        tracking_config: Any,
        frame_batch_size: Any,
        detection_batch_size: Any,
        lang: str,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[Any, Any, Optional[str], Any, Any]:
        from fbxify.pose_estimation_manager import CancelledError as _CancelledError
        from fbxify.utils import render_tracking_bbox_overlay, export_mot_bboxes as write_mot_bboxes, _is_video_path
        from fbxify.i18n import Translator
        import gradio as gr

        translator = Translator(lang)
        temp_dir = None

        def _extract_first_json_path(files: List) -> Optional[str]:
            if files is None:
                return None
            for item in (files if isinstance(files, list) else [files]):
                path = item.name if hasattr(item, "name") else item
                if isinstance(path, str) and path.lower().endswith(".json"):
                    return path
            return None

        try:
            self.manager.estimation_manager.clear_cancel()
            if input_file is None:
                return (
                    gr.update(),
                    gr.update(),
                    None,
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                )

            from fbxify.gradio_ui.entry_section import (
                TRACKING_MODE_BBOX,
                TRACKING_MODE_COUNT,
                TRACKING_MODE_INFERENCE,
                TRACKING_MODE_INFERENCE_BBOX,
            )
            if isinstance(tracking_mode, list):
                tracking_mode = tracking_mode[0] if tracking_mode else TRACKING_MODE_COUNT
            use_bbox = tracking_mode == TRACKING_MODE_BBOX or tracking_mode == TRACKING_MODE_INFERENCE_BBOX

            if use_bbox and bbox_file is None:
                raise ValueError(translator.t("errors.bbox_file_required"))

            bbox_dict = None
            if use_bbox:
                bbox_dict = self.manager.prepare_bboxes(bbox_file.name)
                unique_person_ids = set()
                for bboxes in bbox_dict.values():
                    for bbox in bboxes:
                        if len(bbox) > 0:
                            unique_person_ids.add(bbox[0])
                num_people = len(unique_person_ids) if unique_person_ids else 0
            elif tracking_mode == TRACKING_MODE_COUNT:
                if num_people <= 0:
                    raise ValueError(translator.t("errors.num_people_required"))
            else:
                num_people = None

            file_path = input_file.name
            file_ext = os.path.splitext(file_path)[1].lower()
            is_video = file_ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]
            fps = 30.0
            if is_video:
                frame_paths, temp_dir, fps = self.manager.prepare_video(file_path)
            else:
                frame_paths = [file_path]
                temp_dir = None

            fov_file_path = fov_file.name if fov_file else None
            self.manager.set_camera_intrinsics(fov_method, fov_file_path, frame_paths, sample_number)

            precision_value = "fp32"
            if isinstance(precision, str):
                if precision.lower().startswith("bf16"):
                    precision_value = "bf16"
                elif precision.lower().startswith("fp16"):
                    precision_value = "fp16"
            self.manager.estimation_manager.set_inference_options(precision=precision_value)

            def prog(progress_value, description):
                if progress_callback:
                    progress_callback(progress_value * 0.95 if "Detecting" not in (description or "") else progress_value * 0.5, description)

            output_id = uuid.uuid4().hex
            output_dir = tempfile.gettempdir()
            estimation_json_path = os.path.join(output_dir, f"pose_outputs_{output_id}.json")

            estimation_results = self.manager.estimation_manager.estimate_all_frames(
                frame_paths,
                num_people=num_people,
                bbox_dict=bbox_dict,
                progress_callback=prog,
                missing_bbox_behavior=missing_bbox_behavior if use_bbox else "Run Detection",
                lang=lang,
                frame_batch_size=int(frame_batch_size) if frame_batch_size else None,
                detection_batch_size=int(detection_batch_size) if detection_batch_size else 1,
            )

            tracking_metadata = None
            if tracking_mode == TRACKING_MODE_INFERENCE or tracking_mode == TRACKING_MODE_INFERENCE_BBOX:
                mode_label = "Inference Tracking + BBOX File" if tracking_mode == TRACKING_MODE_INFERENCE_BBOX else "Inference Tracking"
                tracking_metadata = self.tracking_manager.run(estimation_results, tracking_config, mode=mode_label)

            source_name = os.path.basename(file_path)
            output_files = [estimation_json_path]

            if output_tracking_bbox:
                overlay_suffix = ".mp4" if is_video else ".png"
                overlay_path = os.path.join(output_dir, f"pose_outputs_{output_id}_overlay{overlay_suffix}")
                overlay_out = render_tracking_bbox_overlay(frame_paths, estimation_results, overlay_path, fps=fps)
                if overlay_out:
                    output_files.append(overlay_out)

            if (tracking_mode == TRACKING_MODE_INFERENCE or tracking_mode == TRACKING_MODE_INFERENCE_BBOX) and tracking_config and tracking_config.export_mot_bboxes:
                mot_path = os.path.join(output_dir, f"pose_outputs_{output_id}_mot.txt")
                mot_out = write_mot_bboxes(estimation_results, mot_path)
                if mot_out:
                    output_files.append(mot_out)

            self.manager.estimation_manager.save_estimation_results(
                estimation_results,
                estimation_json_path,
                source_name=source_name,
                num_people=num_people,
                tracking_metadata=tracking_metadata,
                output_files=output_files,
                fps=fps,
            )

        except _CancelledError:
            return (
                gr.update(value=None),
                gr.update(value=None),
                None,
                gr.update(interactive=False),
                gr.update(interactive=(input_file is not None)),
            )
        except Exception as e:
            raise
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

        selected_json = _extract_first_json_path(output_files)
        return (
            gr.update(value=output_files),
            gr.update(value=selected_json),
            selected_json,
            gr.update(interactive=(selected_json is not None)),
            gr.update(interactive=(input_file is not None)),
        )

    def run_fbx_generation(
        self,
        pose_json_file: Any,
        profile_name: Any,
        use_root_motion: Any,
        auto_floor: Any,
        include_mesh: Any,
        include_extrinsics: Any,
        extrinsics_sample_rate: Any,
        extrinsics_scale: Any,
        extrinsics_invert_quaternion: Any,
        extrinsics_invert_translation: Any,
        extrinsics_file: Any,
        create_camera: Any,
        camera_zoom: Any,
        camera_scene: Any,
        use_personalized_body: Any,
        lod: Any,
        outlier_removal_percent: Any,
        export_personalized_body_obj: Any,
        graph_refinement: Any,
        input_file: Any,
        refinement_config: Any,
        lang: str,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[Optional[List], Any, Any]:
        import gradio as gr
        from fbxify.i18n import Translator

        translator = Translator(lang)
        output_files: List[str] = []

        try:
            self.manager.estimation_manager.clear_cancel()
            if pose_json_file is None:
                raise ValueError(translator.t("errors.pose_json_file_required"))

            json_path = pose_json_file.name if hasattr(pose_json_file, "name") else pose_json_file
            lod_int = int(lod) if lod is not None else -1
            outlier_percent = float(outlier_removal_percent) if outlier_removal_percent else 10.0
            extrinsics_file_path = extrinsics_file.name if (include_extrinsics and extrinsics_file) else None
            camera_scene_path = camera_scene.name if (create_camera and camera_scene) else None

            def prog(progress_value, description):
                if progress_callback:
                    progress_callback(progress_value * 0.3, description)

            process_result = self.manager.process_from_estimation_json(
                json_path,
                profile_name,
                use_root_motion,
                fps=None,
                refinement_config=refinement_config,
                progress_callback=prog,
                lod=lod_int if include_mesh else -1,
                use_personalized_body=use_personalized_body if include_mesh else False,
                outlier_removal_percent=outlier_percent if (include_mesh and use_personalized_body) else 10.0,
                lang=lang,
                auto_floor=auto_floor,
                extrinsics_file=extrinsics_file_path,
                extrinsics_sample_rate=int(extrinsics_sample_rate) if extrinsics_sample_rate else 0,
                extrinsics_scale=float(extrinsics_scale) if extrinsics_scale else 0.0,
                extrinsics_invert_quaternion=bool(extrinsics_invert_quaternion),
                extrinsics_invert_translation=bool(extrinsics_invert_translation),
                collect_refinement_logs=bool(graph_refinement),
            )

            def export_prog(progress_value, description):
                if progress_callback:
                    progress_callback(0.3 + progress_value * 0.6, description)

            lod_fbx_path = None
            if include_mesh and lod_int >= 0:
                from fbxify.cli_common import resolve_lod_fbx_path
                lod_fbx_path = resolve_lod_fbx_path(process_result.profile_name, lod_int)

            fbx_paths = self.manager.export_fbx_files(
                process_result.profile_name,
                process_result.joint_to_bone_mappings,
                process_result.root_motions,
                process_result.frame_paths,
                process_result.fps,
                export_prog,
                lod=lod_int if include_mesh else -1,
                mesh_obj_paths=process_result.mesh_obj_paths,
                lod_fbx_path=lod_fbx_path,
                lang=lang,
                height_offset=process_result.height_offset,
                metadata_extras=process_result.metadata_extras,
                create_camera=bool(create_camera),
                camera_scene_path=camera_scene_path,
                camera_zoom=float(camera_zoom) if camera_zoom else 0.0,
                extrinsics_file=extrinsics_file_path,
            )
            output_files.extend(fbx_paths or [])
            if export_personalized_body_obj and process_result.mesh_obj_paths:
                for mesh_path in process_result.mesh_obj_paths.values():
                    if mesh_path and os.path.exists(mesh_path):
                        output_files.append(mesh_path)
            if graph_refinement and process_result.refinement_logs:
                try:
                    from fbxify.refinement.refinement_graphs import convert_refinement_logs
                    output_files.extend(convert_refinement_logs(process_result.refinement_logs))
                except Exception:
                    pass

        except Exception as e:
            raise

        return (
            output_files if output_files else None,
            gr.update(interactive=(input_file is not None)),
            gr.update(interactive=(pose_json_file is not None)),
        )

    def run_detection_only(
        self,
        input_file: Any,
        detection_batch_size: Any,
        progress_callback: Optional[callable] = None,
    ) -> Optional[str]:
        if input_file is None:
            return None
        file_path = input_file.name if hasattr(input_file, "name") else input_file
        file_ext = os.path.splitext(file_path)[1].lower()
        is_video = file_ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        temp_dir = None
        try:
            if is_video:
                frame_paths, temp_dir, _ = self.manager.prepare_video(file_path)
            else:
                frame_paths = [file_path]
            if not frame_paths:
                return None
            output_id = uuid.uuid4().hex
            output_dir = tempfile.gettempdir()
            mot_path = os.path.join(output_dir, f"detection_{output_id}.txt")
            batch_size = int(detection_batch_size) if detection_batch_size else 1
            self.manager.estimation_manager.run_detection_only(
                frame_paths, mot_path, progress_callback=progress_callback, detection_batch_size=batch_size
            )
            return mot_path
        finally:
            if temp_dir and os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def rerun_tracking(
        self,
        estimation_file: Any,
        step_through: Any,
        debug_start_frame: Any,
        tracking_config_params: Dict[str, Any],
    ) -> Tuple[Any, Any, Optional[str], Any]:
        import gradio as gr
        from fbxify.gradio_ui.tracking_section import build_tracking_config_from_gui
        from fbxify.tracking.tracking_manager import apply_frame_assignments

        if estimation_file is None:
            return (gr.update(), gr.update(), None, gr.update())
        path = estimation_file[0] if isinstance(estimation_file, list) else (estimation_file.name if hasattr(estimation_file, "name") else estimation_file)
        if not path or not os.path.isfile(path):
            return (gr.update(), gr.update(), None, gr.update())
        try:
            estimation_results, metadata = self.manager.estimation_manager.load_estimation_results(path)
            config = build_tracking_config_from_gui(**tracking_config_params)
            tracking_metadata = self.tracking_manager.run(
                estimation_results,
                config,
                mode="Re-run Tracking (debug)",
                debug_per_frame=True,
                step_through=bool(step_through),
                debug_start_frame=int(debug_start_frame or 0),
            )
            frame_assignments = tracking_metadata.get("frame_assignments") or {}
            remapped_results = apply_frame_assignments(estimation_results, frame_assignments)
            output_id = uuid.uuid4().hex
            output_dir = tempfile.gettempdir()
            new_path = os.path.join(output_dir, f"pose_outputs_rerun_{output_id}.json")
            save_tracking = dict(tracking_metadata)
            if "tracklets" in save_tracking:
                save_tracking["tracklets"] = [{k: v for k, v in t.items() if k != "detections"} for t in save_tracking["tracklets"]]
            self.manager.estimation_manager.save_estimation_results(
                remapped_results,
                new_path,
                source_name=metadata.get("source", "unknown"),
                num_people=metadata.get("num_people"),
                tracking_metadata=save_tracking,
                output_files=[new_path],
                fps=metadata.get("fps"),
            )
            return (
                gr.update(value=[new_path]),
                gr.update(value=new_path),
                new_path,
                gr.update(interactive=True),
            )
        except Exception as e:
            raise

    def cancel_current_job(self) -> None:
        self.manager.estimation_manager.cancel_current_job()


class RemoteBackend:
    """Backend that calls worker API via HTTP."""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = (api_key or os.environ.get("FBXIFY_SHARED_SECRET", "") or "").strip()

    def _headers(self) -> Dict[str, str]:
        """Headers for worker requests. Includes auth when FBXIFY_SHARED_SECRET is set."""
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def run_pose_estimation(
        self,
        input_file: Any,
        tracking_mode: Any,
        bbox_file: Any,
        num_people: Any,
        missing_bbox_behavior: Any,
        fov_method: Any,
        fov_file: Any,
        sample_number: Any,
        precision: Any,
        output_tracking_bbox: Any,
        tracking_config: Any,
        frame_batch_size: Any,
        detection_batch_size: Any,
        lang: str,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[Any, Any, Optional[str], Any, Any]:
        import requests
        import gradio as gr

        try:
            files = {}
            if input_file:
                files["input_file"] = (os.path.basename(getattr(input_file, "name", str(input_file)) or "input"), open(getattr(input_file, "name", input_file), "rb"))
            if bbox_file:
                files["bbox_file"] = (os.path.basename(getattr(bbox_file, "name", str(bbox_file)) or "bbox"), open(getattr(bbox_file, "name", bbox_file), "rb"))
            if fov_file:
                files["fov_file"] = (os.path.basename(getattr(fov_file, "name", str(fov_file)) or "fov"), open(getattr(fov_file, "name", fov_file), "rb"))

            if isinstance(tracking_mode, list):
                tracking_mode = tracking_mode[0] if tracking_mode else "count"
            payload = {
                "tracking_mode": tracking_mode,
                "num_people": int(num_people) if num_people else 1,
                "missing_bbox_behavior": missing_bbox_behavior or "Run Detection",
                "fov_method": fov_method or "Default",
                "sample_number": int(sample_number) if sample_number else 1,
                "precision": str(precision).lower()[:4] if precision else "fp32",
                "output_tracking_bbox": bool(output_tracking_bbox),
                "frame_batch_size": int(frame_batch_size) if frame_batch_size else 1,
                "detection_batch_size": int(detection_batch_size) if detection_batch_size else 1,
                "lang": lang,
            }
            if tracking_config is not None:
                tc = tracking_config.to_dict() if hasattr(tracking_config, "to_dict") else tracking_config
                payload["tracking_config"] = json.dumps(tc) if not isinstance(tc, str) else tc

            r = requests.post(f"{self.base_url}/jobs/pose", data=payload, files=files or None, headers=self._headers(), timeout=300)
            r.raise_for_status()
            job_id = r.json().get("job_id")
            if not job_id:
                raise RuntimeError("No job_id in response")

            for f in files.values():
                if hasattr(f[1], "close"):
                    f[1].close()

            output_files = []
            selected_json = None
            while True:
                time.sleep(1)
                status_r = requests.get(f"{self.base_url}/jobs/{job_id}", headers=self._headers(), timeout=30)
                status_r.raise_for_status()
                data = status_r.json()
                status = data.get("status", "")
                if status == "completed":
                    for filename in data.get("output_files", []):
                        fn = os.path.basename(filename) if isinstance(filename, str) else filename
                        url = f"{self.base_url}/jobs/{job_id}/files/{fn}"
                        dl = requests.get(url, headers=self._headers(), timeout=120)
                        dl.raise_for_status()
                        local_path = os.path.join(tempfile.gettempdir(), fn)
                        with open(local_path, "wb") as out:
                            out.write(dl.content)
                        output_files.append(local_path)
                    selected_json = next((f for f in output_files if f.endswith(".json")), output_files[0] if output_files else None)
                    break
                elif status == "failed":
                    raise RuntimeError(data.get("error", "Job failed"))
                if progress_callback:
                    progress_callback(data.get("progress", 0) / 100.0, data.get("message", "Processing..."))

            return (
                gr.update(value=output_files),
                gr.update(value=selected_json),
                selected_json,
                gr.update(interactive=bool(selected_json)),
                gr.update(interactive=(input_file is not None)),
            )
        except Exception as e:
            raise

    def run_fbx_generation(
        self,
        pose_json_file: Any,
        profile_name: Any,
        use_root_motion: Any,
        auto_floor: Any,
        include_mesh: Any,
        include_extrinsics: Any,
        extrinsics_sample_rate: Any,
        extrinsics_scale: Any,
        extrinsics_invert_quaternion: Any,
        extrinsics_invert_translation: Any,
        extrinsics_file: Any,
        create_camera: Any,
        camera_zoom: Any,
        camera_scene: Any,
        use_personalized_body: Any,
        lod: Any,
        outlier_removal_percent: Any,
        export_personalized_body_obj: Any,
        graph_refinement: Any,
        input_file: Any,
        refinement_config: Any,
        lang: str,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[Optional[List], Any, Any]:
        import requests
        import gradio as gr

        try:
            json_path = pose_json_file.name if hasattr(pose_json_file, "name") else pose_json_file
            files = {"pose_json_file": (os.path.basename(json_path), open(json_path, "rb"))}
            payload = {
                "profile_name": profile_name,
                "use_root_motion": use_root_motion,
                "auto_floor": auto_floor,
                "include_mesh": include_mesh,
                "include_extrinsics": include_extrinsics,
                "extrinsics_sample_rate": int(extrinsics_sample_rate or 0),
                "extrinsics_scale": float(extrinsics_scale or 0),
                "extrinsics_invert_quaternion": bool(extrinsics_invert_quaternion),
                "extrinsics_invert_translation": bool(extrinsics_invert_translation),
                "create_camera": bool(create_camera),
                "camera_zoom": float(camera_zoom or 0),
                "use_personalized_body": use_personalized_body,
                "lod": int(lod) if lod is not None else -1,
                "outlier_removal_percent": float(outlier_removal_percent or 10),
                "export_personalized_body_obj": bool(export_personalized_body_obj),
                "graph_refinement": bool(graph_refinement),
                "lang": lang,
            }
            if extrinsics_file:
                files["extrinsics_file"] = (os.path.basename(getattr(extrinsics_file, "name", extrinsics_file)), open(getattr(extrinsics_file, "name", extrinsics_file), "rb"))
            if camera_scene:
                files["camera_scene"] = (os.path.basename(getattr(camera_scene, "name", camera_scene)), open(getattr(camera_scene, "name", camera_scene), "rb"))
            if refinement_config is not None and hasattr(refinement_config, "profiles"):
                try:
                    from dataclasses import asdict
                    cfg_dict = {
                        "do_spike_fix": getattr(refinement_config, "do_spike_fix", True),
                        "do_rotation_smoothing": getattr(refinement_config, "do_rotation_smoothing", True),
                        "do_vector_smoothing": getattr(refinement_config, "do_vector_smoothing", True),
                        "do_root_motion_fix": getattr(refinement_config, "do_root_motion_fix", True),
                        "do_interpolate_missing_keyframes": getattr(refinement_config, "do_interpolate_missing_keyframes", False),
                        "do_foot_planting": getattr(refinement_config, "do_foot_planting", True),
                        "profiles": {},
                    }
                    for k, v in refinement_config.profiles.items():
                        if hasattr(v, "__dataclass_fields__"):
                            cfg_dict["profiles"][k] = asdict(v)
                        elif isinstance(v, dict):
                            cfg_dict["profiles"][k] = v
                    if hasattr(refinement_config, "foot_planting_config") and refinement_config.foot_planting_config:
                        fp = refinement_config.foot_planting_config
                        cfg_dict["foot_planting_config"] = asdict(fp) if hasattr(fp, "__dataclass_fields__") else {}
                    cfg_path = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
                    json.dump(cfg_dict, cfg_path)
                    cfg_path.close()
                    files["refinement_config"] = ("refinement_config.json", open(cfg_path.name, "rb"))
                except Exception:
                    pass

            r = requests.post(f"{self.base_url}/jobs/fbx", data=payload, files=files, headers=self._headers(), timeout=300)
            r.raise_for_status()
            job_id = r.json().get("job_id")
            if not job_id:
                raise RuntimeError("No job_id in response")

            for f in files.values():
                if hasattr(f, "close"):
                    f.close()
                elif isinstance(f, tuple) and hasattr(f[1], "close"):
                    f[1].close()

            output_files = []
            while True:
                time.sleep(1)
                status_r = requests.get(f"{self.base_url}/jobs/{job_id}", headers=self._headers(), timeout=30)
                status_r.raise_for_status()
                data = status_r.json()
                if data.get("status") == "completed":
                    for fn in data.get("output_files", []):
                        url = f"{self.base_url}/jobs/{job_id}/files/{fn}"
                        dl = requests.get(url, headers=self._headers(), timeout=120)
                        dl.raise_for_status()
                        local_path = os.path.join(tempfile.gettempdir(), fn)
                        with open(local_path, "wb") as out:
                            out.write(dl.content)
                        output_files.append(local_path)
                    break
                elif data.get("status") == "failed":
                    raise RuntimeError(data.get("error", "Job failed"))
                if progress_callback:
                    progress_callback(data.get("progress", 0) / 100.0, data.get("message", "Processing..."))

            return (
                output_files if output_files else None,
                gr.update(interactive=(input_file is not None)),
                gr.update(interactive=(pose_json_file is not None)),
            )
        except Exception as e:
            raise

    def run_detection_only(
        self,
        input_file: Any,
        detection_batch_size: Any,
        progress_callback: Optional[callable] = None,
    ) -> Optional[str]:
        import requests

        if not input_file:
            return None
        file_path = input_file.name if hasattr(input_file, "name") else input_file
        with open(file_path, "rb") as f:
            files = {"input_file": (os.path.basename(file_path), f)}
            r = requests.post(
                f"{self.base_url}/jobs/detection",
                data={"detection_batch_size": int(detection_batch_size or 1)},
                files=files,
                headers=self._headers(),
                timeout=300,
            )
        r.raise_for_status()
        job_id = r.json().get("job_id")
        while True:
            time.sleep(1)
            s = requests.get(f"{self.base_url}/jobs/{job_id}", headers=self._headers(), timeout=30)
            s.raise_for_status()
            d = s.json()
            if d.get("status") == "completed":
                mot_fn = d.get("mot_path")
                if not mot_fn:
                    return None
                url = f"{self.base_url}/jobs/{job_id}/files/{mot_fn}"
                dl = requests.get(url, headers=self._headers(), timeout=60)
                dl.raise_for_status()
                local_path = os.path.join(tempfile.gettempdir(), mot_fn)
                with open(local_path, "wb") as out:
                    out.write(dl.content)
                return local_path
            elif d.get("status") == "failed":
                raise RuntimeError(d.get("error", "Job failed"))

    def rerun_tracking(
        self,
        estimation_file: Any,
        step_through: Any,
        debug_start_frame: Any,
        tracking_config_params: Dict[str, Any],
    ) -> Tuple[Any, Any, Optional[str], Any]:
        import requests
        import gradio as gr

        if not estimation_file:
            return (gr.update(), gr.update(), None, gr.update())
        path = estimation_file[0] if isinstance(estimation_file, list) else (estimation_file.name if hasattr(estimation_file, "name") else estimation_file)
        if not path or not os.path.isfile(path):
            return (gr.update(), gr.update(), None, gr.update())
        try:
            with open(path, "rb") as f:
                files = {"estimation_file": (os.path.basename(path), f)}
                r = requests.post(
                    f"{self.base_url}/jobs/rerun_tracking",
                    data={
                        "step_through": bool(step_through),
                        "debug_start_frame": int(debug_start_frame or 0),
                        **tracking_config_params,
                    },
                    files=files,
                    headers=self._headers(),
                    timeout=300,
                )
            r.raise_for_status()
            job_id = r.json().get("job_id")
            while True:
                time.sleep(1)
                s = requests.get(f"{self.base_url}/jobs/{job_id}", headers=self._headers(), timeout=30)
                s.raise_for_status()
                d = s.json()
                if d.get("status") == "completed":
                    fns = d.get("output_files", [])
                    if not fns:
                        raise RuntimeError("No output from rerun tracking")
                    fn = fns[0]
                    url = f"{self.base_url}/jobs/{job_id}/files/{fn}"
                    dl = requests.get(url, headers=self._headers(), timeout=60)
                    dl.raise_for_status()
                    new_path = os.path.join(tempfile.gettempdir(), fn)
                    with open(new_path, "wb") as out:
                        out.write(dl.content)
                    return (
                        gr.update(value=[new_path]),
                        gr.update(value=new_path),
                        new_path,
                        gr.update(interactive=True),
                    )
                elif d.get("status") == "failed":
                    raise RuntimeError(d.get("error", "Job failed"))
        except Exception as e:
            raise

    def cancel_current_job(self) -> None:
        import requests
        try:
            requests.post(f"{self.base_url}/jobs/cancel", headers=self._headers(), timeout=5)
        except Exception:
            pass
