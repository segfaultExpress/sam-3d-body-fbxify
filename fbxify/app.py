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
    return parser.parse_args()

def create_app(estimator):
    """Gradioアプリを作成"""
    
    def process_single_frame(profile_name, image_path, keyframe_index, joint_to_bone_mapping=None):
        """Process a single frame and return the result dict without exporting to FBX."""
        return estimator.process_single_frame(profile_name, image_path, keyframe_index, joint_to_bone_mapping)
    
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
    
    def process_video(input_video, profile_name, progress=gr.Progress()):
        """Process MP4 video by extracting frames and processing each one."""
        if input_video is None:
            return None
        
        # Create temp directory for frames
        temp_dir = tempfile.mkdtemp(prefix="sam3d_frames_")
        
        try:
            progress(0, desc="🎬 動画からフレームを抽出中...")
            frame_paths = extract_frames_from_video(input_video.name, temp_dir)
            total_frames = len(frame_paths)
            
            if total_frames == 0:
                return None
            
            # Process first frame to get base structure
            progress(0.01, desc=f"🖼️ フレーム 1/{total_frames} を処理中...")
            first_result = process_single_frame(profile_name, frame_paths[0], 0, joint_to_bone_mapping=None)
            
            # Use first frame's structure as base
            joint_to_bone_mapping = first_result["joint_to_bone_mapping"]
            rest_pose = first_result["rest_pose"]
            vertices = first_result["vertices"]
            
            # Process remaining frames - append to existing joint_to_bone_mapping
            for frame_idx in range(1, total_frames):
                frame_progress = (frame_idx + 1) / total_frames * 0.5
                progress(frame_progress, desc=f"🖼️ フレーム {frame_idx + 1}/{total_frames} を処理中...")
                
                # Process this frame - it will append to the existing joint_to_bone_mapping
                process_single_frame(profile_name, frame_paths[frame_idx], frame_idx, joint_to_bone_mapping=joint_to_bone_mapping)
                # The joint_to_bone_mapping is modified in-place by populate_joint_mapping
            
            # Create metadata with correct num_keyframes
            metadata = estimator.create_metadata(profile_name, num_keyframes=total_frames)
            
            progress(0.5, desc="📦 FBXファイルを生成中...")
            fbx_path = export_to_fbx(
                metadata,
                joint_to_bone_mapping,
                rest_pose,
                vertices,
                estimator.faces
            )
            
            progress(1.0, desc="✅ 完了!")
            return fbx_path
        
        finally:
            # Clean up temp directory
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
    
    def process_image(input_file, profile_name, progress=gr.Progress()):
        """Process image or video file."""
        if input_file is None:
            return None
        
        # Detect file type by extension
        file_path = input_file.name
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            # Process as video
            return process_video(input_file, profile_name, progress)
        else:
            # Process as image
            progress(0, desc="🖼️ 画像を解析中...")
            result = estimator.process_image(profile_name, file_path)
            
            progress(0.5, desc="📦 FBXファイルを生成中...")
            fbx_path = export_to_fbx(
                result["metadata"],
                result["joint_to_bone_mapping"],
                result["rest_pose"],
                result["vertices"],
                estimator.faces
            )
            
            progress(1.0, desc="✅ 完了!")
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
                generate_btn = gr.Button("🚀 FBXを生成", variant="primary")
            with gr.Column():
                output_file = gr.File(label="📦 生成されたFBX", interactive=False)
        
        generate_btn.click(fn=process_image, inputs=[input_file, profile_name], outputs=output_file)
    
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
    
    estimator = PoseEstimator(checkpoint_path, mhr_path)

    app = create_app(estimator)
    app.launch(
        server_name="0.0.0.0",
        server_port=7444,
        share=True,
        open_browser=True,
    )