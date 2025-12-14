import os
import argparse
import cv2
import tempfile
import shutil
from fbxify.pose_estimator import PoseEstimator
from fbxify.utils import export_to_fbx
from fbxify.metadata import PROFILES
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
    """Gradioアプリを作成"""
        
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
                    if frame_index not in bbox_dict:
                        bbox_dict[frame_index] = []
                    bbox_dict[frame_index].append((person_id, x1, y1, w, h, conf, x_world, y_world, z_world))
            except Exception as e:
                raise ValueError(f"Error preparing bboxes: {e}")
        return bbox_dict

    def process(input_file, profile_name, use_bbox, bbox_file, num_people, progress=gr.Progress()):
        """Process image or video file."""
        try:
            if input_file is None:
                return None

            # there must either be a valid number of people (>0) or a bbox file
            if use_bbox and bbox_file is None:
                raise ValueError("BBOX file must be provided when 'use bbox' is enabled")

            bbox_dict = None
            if use_bbox:
                bbox_dict = prepare_bboxes(bbox_file)
                num_people = len(bbox_dict)
            else:
                if num_people <= 0:
                    raise ValueError("Number of people must be greater than 0")

            file_path = input_file.name
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                frame_paths = prepare_video(input_file)
            else:
                frame_paths = [input_file.name]

            joint_to_bone_mappings = {}
            for keyframe_index, frame_path in enumerate(frame_paths):
                print(f"Processing frame {keyframe_index + 1} of {len(frame_paths)}")
                bboxes = bbox_dict[keyframe_index] if bbox_dict is not None else None

                try:
                    results = estimator.process_single_frame(profile_name, frame_path, keyframe_index, joint_to_bone_mappings=joint_to_bone_mappings, num_people=num_people, bboxes=bboxes)
                    
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

                progress((keyframe_index * 0.5) / len(frame_paths), desc="📦 キーフレームを処理中...")

            # Collect all FBX paths for each person
            fbx_paths = []
            num_keyframes = len(frame_paths)
            for person_index, id in enumerate(joint_to_bone_mappings.keys()):
                fbx_path = output({
                    "metadata": estimator.create_metadata(profile_name, id, num_keyframes=num_keyframes),
                    "joint_to_bone_mapping": joint_to_bone_mappings[id],
                    "rest_pose": estimator.get_armature_rest_pose(profile_name)
                })
                if fbx_path is not None:
                    fbx_paths.append(fbx_path)

                progress((person_index * 0.5) / len(joint_to_bone_mappings.keys()), desc="📦 人物を処理中...")
        except Exception as e:
            # Catch other exceptions and display them as well
            raise gr.Error(f"An error occurred: {str(e)}")
        
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
            result["rest_pose"],
            estimator.faces
        )
        return fbx_path
    
    with gr.Blocks(title="SAM 3D Body → Unity FBX") as app:
        gr.Markdown("## 🧍‍♂️ SAM 3D Body → Unity Humanoid FBX")
        gr.Markdown("""
        ### 機能
        - 1枚の画像からUnity Humanoid互換のFBXファイルを生成
        - MP4動画から複数フレームのアニメーションを生成
        - 3Dメッシュ + スケルトン + **アニメーション**
        
        ### 使い方
        1. 画像またはMP4動画をアップロード
        2. 「FBXを生成」ボタンをクリック
        3. 生成されたFBXをダウンロード → Unityにインポート
        """)
        
        # Feel free to add your own! But mixamo can also do remapping post-this via the Rokoko plugin, etc.
        profile_name = gr.Dropdown(label="📊 プロファイル", choices=list(PROFILES.keys()), value=list(PROFILES.keys())[0])
        with gr.Row():
            with gr.Column():
                input_file = gr.File(label="📁 画像/動画ファイル", file_types=["image", "video"])
                use_bbox = gr.Checkbox(label="👥 人物を指定", value=True)
                with gr.Row():
                    bbox_file = gr.File(label="📁 BBOXファイル", file_types=["txt"], visible=True)
                    num_people = gr.Number(label="👥 人数", value=1, precision=0, visible=False)
                generate_btn = gr.Button("🚀 FBXを生成", variant="primary")
            with gr.Column():
                output_file = gr.File(label="📦 生成されたFBX (複数可)", interactive=False)
        
        def toggle_bbox_inputs(use_bbox_value):
            """Toggle visibility of bbox_file and num_people based on checkbox."""
            return (
                gr.update(visible=use_bbox_value),
                gr.update(visible=not use_bbox_value)
            )
        
        use_bbox.change(fn=toggle_bbox_inputs, inputs=[use_bbox], outputs=[bbox_file, num_people])
        generate_btn.click(fn=process, inputs=[input_file, profile_name, use_bbox, bbox_file, num_people], outputs=output_file)
    
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