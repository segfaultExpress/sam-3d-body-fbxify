import os
import argparse
import cv2
import tempfile
import shutil
import numpy as np
from fbxify.pose_estimator import PoseEstimator
from fbxify.utils import export_to_fbx
from fbxify.metadata import PROFILES
from fbxify.i18n import Translator, SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE
import gradio as gr

VITH_CHECKPOINT_PATH = "/workspace/checkpoints/sam-3d-body-vith"
DINOV3_CHECKPOINT_PATH = "/workspace/checkpoints/sam-3d-body-dinov3"

def parse_args():
    parser = argparse.ArgumentParser()
    # can choose between vith and dinov3
    parser.add_argument("--model", type=str, default="vith", choices=["vith", "dinov3"])
    parser.add_argument(
        "--detector_name",
        default="vitdet",
        type=str,
        help="Human detection model for demo (Default `vitdet`, add your favorite detector if needed).",
    )
    parser.add_argument(
        "--detector_path",
        default="",
        type=str,
        help="Path to human detection model folder (or set SAM3D_DETECTOR_PATH)",
    )
    parser.add_argument(
        "--fov_name",
        default="moge2",
        type=str,
        help="FOV estimation model for demo (Default `moge2`, add your favorite fov estimator if needed).",
    )
    parser.add_argument(
        "--fov_path",
        default="",
        type=str,
        help="Path to fov estimation model folder (or set SAM3D_FOV_PATH)",
    )
    return parser.parse_args()

def create_app(estimator):
    """Create Gradio app"""
    
    # Initialize translator with default language
    translator = Translator(DEFAULT_LANGUAGE)
        
    def extract_frames_from_video(video_path, temp_dir):
        """Extract all frames from MP4 video and save to temp directory."""
        cap = cv2.VideoCapture(video_path)
        frame_paths = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_path = os.path.join(temp_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            frame_count += 1
        
        cap.release()
        return frame_paths
    
    def prepare_video(input_video):
        """Split MP4 video into frames"""
        if input_video is None:
            return None
        
        # Create temp directory for frames
        temp_dir = tempfile.mkdtemp(prefix="sam3d_frames_")
        frame_paths = extract_frames_from_video(input_video.name, temp_dir)
        return frame_paths

    def prepare_bboxes(bbox_file):
        if bbox_file is None:
            return None

        bbox_dict = {}
        with open(bbox_file.name, "r") as f:
            try:
                for line in f:
                    if line.strip() == "":
                        continue
                    frame_index, person_id, x1, y1, w, h, conf, x_world, y_world, z_world = line.strip().split(",")
                    # Convert frame_index to int to standardize keys (handles '1', '1.0', etc.)
                    frame_index_int = int(float(frame_index))
                    if frame_index_int not in bbox_dict:
                        bbox_dict[frame_index_int] = []
                    bbox_dict[frame_index_int].append((person_id, x1, y1, w, h, conf, x_world, y_world, z_world))
            except Exception as e:
                raise ValueError(f"Error preparing bboxes: {e}")
        return bbox_dict

    def process(input_file, profile_name, use_bbox, bbox_file, num_people, fov_method, fov_file, sample_number, use_root_motion, progress=gr.Progress()):
        """Process image or video file."""
        try:
            if input_file is None:
                return None

            # there must either be a valid number of people (>0) or a bbox file
            if use_bbox and bbox_file is None:
                raise ValueError(translator.t("errors.bbox_file_required"))

            bbox_dict = None
            if use_bbox:
                bbox_dict = prepare_bboxes(bbox_file)
                num_people = len(bbox_dict)
            else:
                if num_people <= 0:
                    raise ValueError(translator.t("errors.num_people_required"))

            file_path = input_file.name
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                frame_paths = prepare_video(input_file)
            else:
                frame_paths = [input_file.name]
            
            # Handle FOV estimation caching based on method
            if fov_method == "File":
                if fov_file is None:
                    raise ValueError(translator.t("errors.camera_intrinsics_required"))
                # Load camera intrinsics from file
                estimator.cache_cam_int_from_file(fov_file.name)
            elif fov_method == "Sample":
                if estimator.estimator.fov_estimator is None:
                    raise ValueError(translator.t("errors.fov_estimator_required"))
                # Sample and average camera intrinsics from images
                sample_num = int(sample_number) if sample_number is not None else 1
                estimator.cache_cam_int_from_images(frame_paths, average_of=sample_num)
            # else: "Default" - built-in FOV estimator runs every frame (default behavior)

            joint_to_bone_mappings = {}
            root_motions = {} if use_root_motion else None
            # If bbox_dict exists, get sorted keys to allow starting from 0, 1, or any index
            for keyframe_index, frame_path in enumerate(frame_paths):
                # proper MOT bbox formatting starts at 1, offset keyframe index by 1 (sorry if you like 0-indexing for bboxes)
                bboxes = bbox_dict[keyframe_index + 1] if bbox_dict is not None and keyframe_index + 1 in bbox_dict else None
                print(f"Processing frame {keyframe_index + 1} of {len(frame_paths)}")

                try:
                    results = estimator.process_single_frame(
                        profile_name,
                        frame_path,
                        keyframe_index,
                        joint_to_bone_mappings=joint_to_bone_mappings,
                        root_motions=root_motions,
                        num_people=num_people,
                        bboxes=bboxes
                    )
                    
                    # Only process results if any people were detected
                    if results and len(results) > 0:
                        for id in results.keys():
                            result = results[id]
                            joint_to_bone_mappings[id] = result["joint_to_bone_mapping"]
                    else:
                        print(f"  Warning: No people detected in frame {keyframe_index + 1}, skipping...")
                except ValueError as e:
                    # Handle "No people detected" or similar errors gracefully
                    if "No people detected" in str(e) or "person_index" in str(e):
                        print(f"  Warning: {str(e)} in frame {keyframe_index + 1}, skipping...")
                        continue
                    else:
                        # Re-raise other ValueError exceptions
                        raise

                progress((keyframe_index * 0.5) / len(frame_paths), desc=translator.t("progress.processing_keyframes"))

            # Collect all FBX paths for each person
            fbx_paths = []
            num_keyframes = len(frame_paths)
            for person_index, id in enumerate(joint_to_bone_mappings.keys()):
                root_motion = root_motions[id] if use_root_motion and root_motions is not None and id in root_motions else []
                fbx_path = output({
                    "metadata": estimator.create_metadata(profile_name, id, num_keyframes=num_keyframes),
                    "joint_to_bone_mapping": joint_to_bone_mappings[id],
                    "root_motion": root_motion,
                    "rest_pose": estimator.get_armature_rest_pose(profile_name)
                })
                if fbx_path is not None:
                    fbx_paths.append(fbx_path)

                progress((person_index * 0.5) / len(joint_to_bone_mappings.keys()), desc=translator.t("progress.processing_person"))
        except Exception as e:
            # Catch other exceptions and display them as well
            error_type = type(e).__name__
            error_msg = str(e)
            if error_msg:
                raise gr.Error(translator.t("errors.error_occurred", error_type=error_type, error_msg=error_msg))
            else:
                raise gr.Error(translator.t("errors.error_occurred_no_msg", error_type=error_type))
        
        # Return list of FBX paths (or single path if only one, for backward compatibility)
        if len(fbx_paths) == 0:
            return None
        elif len(fbx_paths) == 1:
            return fbx_paths[0]
        else:
            return fbx_paths

    def output(result):
        if result is None:
            return None

        fbx_path = export_to_fbx(
            result["metadata"],
            result["joint_to_bone_mapping"],
            result["root_motion"],
            result["rest_pose"],
            estimator.faces
        )
        return fbx_path
    
    def get_ui_texts(lang: str = DEFAULT_LANGUAGE):
        """Get all UI texts for a given language."""
        t = Translator(lang)
        features = t.get("app.features", [])
        usage = t.get("app.usage", [])
        
        features_text = "\n".join([f"- {f}" for f in features])
        usage_text = "\n".join([f"{i+1}. {u}" for i, u in enumerate(usage)])
        
        return {
            "title": t.t("app.title"),
            "heading": t.t("app.heading"),
            "features_title": t.t("app.features_title"),
            "features": features_text,
            "usage_title": t.t("app.usage_title"),
            "usage": usage_text,
            "profile": t.t("ui.profile"),
            "input_file": t.t("ui.input_file"),
            "use_bbox": t.t("ui.use_bbox"),
            "bbox_file": t.t("ui.bbox_file"),
            "num_people": t.t("ui.num_people"),
            "fov_method": t.t("ui.fov_method"),
            "fov_method_info": t.t("ui.fov_method_info"),
            "fov_file": t.t("ui.fov_file"),
            "sample_number": t.t("ui.sample_number"),
            "sample_number_info": t.t("ui.sample_number_info"),
            "use_root_motion": t.t("ui.use_root_motion"),
            "generate_btn": t.t("ui.generate_btn"),
            "output_file": t.t("ui.output_file"),
        }
    
    def update_ui_language(lang: str):
        """Update UI elements with new language."""
        texts = get_ui_texts(lang)
        # Update translator for process function
        nonlocal translator
        translator = Translator(lang)
        
        features = translator.get("app.features", [])
        usage = translator.get("app.usage", [])
        features_text = "\n".join([f"- {f}" for f in features])
        usage_text = "\n".join([f"{i+1}. {u}" for i, u in enumerate(usage)])
        description_text = f"### {texts['features_title']}\n{features_text}\n\n### {texts['usage_title']}\n{usage_text}"
        
        return (
            gr.update(value=f"## {texts['heading']}"),  # heading
            gr.update(value=description_text),  # description
            gr.update(label=texts["profile"]),  # profile_name
            gr.update(label=texts["input_file"]),  # input_file
            gr.update(label=texts["use_bbox"]),  # use_bbox
            gr.update(label=texts["bbox_file"]),  # bbox_file
            gr.update(label=texts["num_people"]),  # num_people
            gr.update(label=texts["fov_method"], info=texts["fov_method_info"]),  # fov_method
            gr.update(label=texts["fov_file"]),  # fov_file
            gr.update(label=texts["sample_number"], info=texts["sample_number_info"]),  # sample_number
            gr.update(label=texts["use_root_motion"]),  # use_root_motion
            gr.update(value=texts["generate_btn"]),  # generate_btn
            gr.update(label=texts["output_file"]),  # output_file
        )
    
    with gr.Blocks(title=translator.t("app.title")) as app:
        # Title and heading
        heading_md = gr.Markdown(f"## {translator.t('app.heading')}")
        
        # Description with features and usage
        features = translator.get("app.features", [])
        usage = translator.get("app.usage", [])
        features_text = "\n".join([f"- {f}" for f in features])
        usage_text = "\n".join([f"{i+1}. {u}" for i, u in enumerate(usage)])
        description_text = f"### {translator.t('app.features_title')}\n{features_text}\n\n### {translator.t('app.usage_title')}\n{usage_text}"
        description_md = gr.Markdown(description_text)
        
        # Language selector dropdown (visible for manual override)
        lang_selector = gr.Dropdown(
            label="🌐 Language / 言語",
            choices=[("English", "en"), ("日本語", "ja")],
            value=DEFAULT_LANGUAGE,
            interactive=True
        )
        
        # Feel free to add your own! But mixamo can also do remapping post-this via the Rokoko plugin, etc.
        profile_name = gr.Dropdown(
            label=translator.t("ui.profile"),
            choices=list(PROFILES.keys()),
            value=list(PROFILES.keys())[0]
        )
        with gr.Row():
            with gr.Column():
                input_file = gr.File(
                    label=translator.t("ui.input_file"),
                    file_types=["image", "video"]
                )
                use_bbox = gr.Checkbox(
                    label=translator.t("ui.use_bbox"),
                    value=True
                )
                with gr.Row():
                    bbox_file = gr.File(
                        label=translator.t("ui.bbox_file"),
                        file_types=[".txt"],
                        visible=True
                    )
                    num_people = gr.Number(
                        label=translator.t("ui.num_people"),
                        value=1,
                        precision=0,
                        visible=False
                    )
                
                # FOV Estimation Options
                fov_method = gr.Dropdown(
                    label=translator.t("ui.fov_method"),
                    choices=["Default", "File", "Sample"],
                    value="Default",
                    info=translator.t("ui.fov_method_info")
                )
                fov_file = gr.File(
                    label=translator.t("ui.fov_file"),
                    file_types=[".txt"],
                    visible=False
                )
                sample_number = gr.Number(
                    label=translator.t("ui.sample_number"),
                    value=1,
                    precision=0,
                    minimum=1,
                    visible=False,
                    info=translator.t("ui.sample_number_info")
                )
                use_root_motion = gr.Checkbox(
                    label=translator.t("ui.use_root_motion"),
                    value=True
                )
                
                generate_btn = gr.Button(
                    translator.t("ui.generate_btn"),
                    variant="primary"
                )
            with gr.Column():
                output_file = gr.File(
                    label=translator.t("ui.output_file"),
                    interactive=False
                )
        
        def toggle_bbox_inputs(use_bbox_value):
            """Toggle visibility of bbox_file and num_people based on checkbox."""
            return (
                gr.update(visible=use_bbox_value),
                gr.update(visible=not use_bbox_value)
            )
        
        def toggle_fov_inputs(fov_method_value):
            """Toggle visibility of fov_file and sample_number based on FOV method selection."""
            if fov_method_value == "File":
                return (
                    gr.update(visible=True),
                    gr.update(visible=False)
                )
            elif fov_method_value == "Sample":
                return (
                    gr.update(visible=False),
                    gr.update(visible=True)
                )
            else:  # "Default"
                return (
                    gr.update(visible=False),
                    gr.update(visible=False)
                )
        
        # Language change handler
        def on_lang_change(lang):
            return update_ui_language(lang)
        
        # Detect browser language and update UI on page load
        def detect_and_set_language():
            """Detect browser language and return it for initial setup."""
            # This will be called on page load
            return DEFAULT_LANGUAGE
        
        # Load language on page load - detect browser language via JavaScript
        app.load(
            fn=detect_and_set_language,
            inputs=[],
            outputs=[lang_selector],
            js="""
            function() {
                const lang = navigator.language || navigator.userLanguage;
                const langCode = lang.split('-')[0].toLowerCase();
                const supportedLangs = ['en', 'ja'];
                const detectedLang = supportedLangs.includes(langCode) ? langCode : 'en';
                return detectedLang;
            }
            """
        )
        
        # Also add a custom script that runs after page load to trigger language update
        app.load(
            fn=None,
            inputs=[],
            outputs=[],
            js="""
            function() {
                // Wait for Gradio to fully initialize
                setTimeout(() => {
                    const lang = navigator.language || navigator.userLanguage;
                    const langCode = lang.split('-')[0].toLowerCase();
                    const supportedLangs = ['en', 'ja'];
                    const detectedLang = supportedLangs.includes(langCode) ? langCode : 'en';
                    
                    // Find language selector by looking for dropdown with language options
                    const selects = document.querySelectorAll('select');
                    for (const select of selects) {
                        const options = Array.from(select.options);
                        const hasEn = options.some(opt => opt.value === 'en');
                        const hasJa = options.some(opt => opt.value === 'ja');
                        if (hasEn && hasJa && options.length === 2) {
                            select.value = detectedLang;
                            // Trigger input event which Gradio listens to
                            select.dispatchEvent(new Event('input', { bubbles: true }));
                            select.dispatchEvent(new Event('change', { bubbles: true }));
                            break;
                        }
                    }
                }, 500);
            }
            """
        )
        
        lang_selector.change(
            fn=on_lang_change,
            inputs=[lang_selector],
            outputs=[heading_md, description_md, profile_name, input_file, use_bbox, bbox_file, num_people, fov_method, fov_file, sample_number, use_root_motion, generate_btn, output_file]
        )
        
        use_bbox.change(fn=toggle_bbox_inputs, inputs=[use_bbox], outputs=[bbox_file, num_people])
        fov_method.change(fn=toggle_fov_inputs, inputs=[fov_method], outputs=[fov_file, sample_number])
        generate_btn.click(fn=process, inputs=[input_file, profile_name, use_bbox, bbox_file, num_people, fov_method, fov_file, sample_number, use_root_motion], outputs=output_file)
    
    return app

if __name__ == "__main__":
    args = parse_args()
    if args.model == "vith":
        checkpoint_base_path = VITH_CHECKPOINT_PATH
    elif args.model == "dinov3":
        checkpoint_base_path = DINOV3_CHECKPOINT_PATH
    else:
        raise ValueError(f"Invalid model: {args.model}")

    checkpoint_path = os.path.join(checkpoint_base_path, "model.ckpt")
    mhr_path = os.path.join(checkpoint_base_path, "assets", "mhr_model.pt")
    
    # Get detector path from args or environment variable
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    # Get FOV path from args or environment variable
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", None)
    
    estimator = PoseEstimator(
        checkpoint_path=checkpoint_path,
        mhr_path=mhr_path,
        detector_name=args.detector_name,
        detector_path=detector_path,
        fov_name=args.fov_name,
        fov_path=fov_path
    )

    app = create_app(estimator)
    app.launch(
        server_name="0.0.0.0",
        server_port=7444,
        share=True,
    )