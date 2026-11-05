## FBXify: Export to FBX Format

![FBX Export Example](example_fbx.gif?raw=true)

This fork extends SAM 3D Body with **FBXify**, a tool that allows you to export estimated poses as FBX files compatible with Unity, Mixamo, and other 3D animation tools. Upload images or videos to generate armature-only FBX files that can be retargeted to any MHR, Mixamo, or Unity character.

## FBXify: Installation & Usage

### Installation

**Quick start:** Run the build script from the repo root. It will try to build Docker first; if Docker is not available, it will offer to set up a conda environment or install Python dependencies.

- Windows: `build.bat`
- Linux/Mac: `./build.sh`

**Side Note - Checkpoints:** Once you have gained access to the HuggingFace checkpoint files ([facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3) or [facebook/sam-3d-body-vith](https://huggingface.co/facebook/sam-3d-body-vith)), replace `your_hf_token_here` in `run.bat` / `run.sh` with your HuggingFace token to download the checkpoints automatically.

**Option A: All-in-one (no Docker)** - See [fbxify/Install.md](fbxify/Install.md) for step-by-step instructions and `fbxify/requirements-full.txt` for Python dependencies.

**Option B: Docker** - See [fbxify/docker/ReadMe.md](fbxify/docker/ReadMe.md) for UI, worker, and simple image variants.

**Experimental: quantized INT8 encoder (TensorRT)** - See [fbxify/docker/Quantized.md](fbxify/docker/Quantized.md) to swap the DINOv3 encoder for [embedl/sam-3d-body](https://huggingface.co/embedl/sam-3d-body) and run the quantized × fast benchmark matrix.

**Start the server** (Docker):
- Linux/Mac: `./run.sh`
- Windows: `./run.bat`

**Start the server** (Manual, no Docker):
- Linux/Mac: `./start_server.sh`
- Windows: `./start_server.bat`
- Or use python: `python ./fbxify/app.py`

**Access the web interface** - Open your browser to `http://localhost:7444`, then upload and process images or videos.

### Features

- **Export FBX files** from SAM 3D Body pose estimates. Export just an armature, or through the MHR armature selection an estimate of the person's body can also be exported
- **Support for multiple rig formats:** MHR, Mixamo, and Unity character remapping
- **Video processing:** Either automatically detects the closest person or allows bbox submitting to applies all keyframes to create an action for each person passed
- **Image processing:** Single frame pose estimation and export
- **Web-based interface:** Easy-to-use Gradio interface for uploading and processing files
- **Integration support for outputs from [BoxMOT](https://github.com/mikel-brostrom/boxmot), [gta-link](https://github.com/sjc042/gta-link) and [colmap](https://github.com/colmap/colmap):** Use outputs from these repos to save computation time by supplying your own bboxes or camera intrinsics
- **Mocap-style refinement:** Configurable refinement methods to try to smooth videos into better blender actions. See `fbxify/refinement/presets` for examples.

Note: The exported FBX files contain only the armature (skeleton), not the mesh. You can retarget or directly move the action to any MHR, Mixamo, or Unity character.

### Acknowledgments

Special thanks to [tori29umai0123](https://github.com/tori29umai0123) for their starter code that inspired this fork. See the original GitHub issue: [A script that outputs a human animation fbx file that can be loaded in Unity (incomplete) #66](https://github.com/facebookresearch/sam-3d-body/issues/66).

## Mapping

FBXify supports remapping poses to different rig formats. For detailed information on how to configure mappings for custom rigs, see the [Mapping Guide](fbxify/Mapping_Guide.md).

### Supported Rigs

- **MHR (Momentum Human Rig):** Native support with direct rotation mapping
- **Mixamo:** Standard rigs can use the Rokoko Retargeting plugin, or custom mappings can be configured
- **Unity:** Unity Humanoid-compatible rigs with custom bone mappings

### Mapping Methods

The system supports three mapping methods:

1. **`direct_rotation`**: Direct mapping for bones, primarily useful for MHR due to the perpendicular nature of MHR bone rotations
2. **`keypoint_with_global_rot_roll`**: Extends keypoint-based mapping by adding roll rotation from MHR rig bones, useful for rigs that need proper bone orientation
3. **Keypoint-based mapping**: Uses keypoints from the MHR pose estimation to drive bone rotations

## Model catalogue

Which bbox detector, image encoder and SAM 3D Body checkpoint fbxify uses is driven by one JSON
file, [`fbxify/models.json`](fbxify/models.json). Three sections, one entry per model:

| Section | Shipped entries | Purpose |
| --- | --- | --- |
| `detectors` | `vitdet` (default) | person bbox when no bbox file is given |
| `encoders` | `fp32` (default), `embedl_int8_trt`, `embedl_int8_pt2` (off) | image encoder variant inside SAM 3D Body |
| `body_models` | `vith` (default), `dinov3` | SAM 3D Body checkpoint + MHR head |

Rules:

- **Only `"enabled": true` entries are fetched** when the container starts (`python -m fbxify.model_catalog fetch`
  in both entrypoints, repeated by the worker's start-up) and only those appear in the dropdowns.
  `"default": true` is what loads first. Nothing disabled is downloaded or touched.
- **Sources** say where files come from: `huggingface` (`repo_id`, pinned `revision`, `files`),
  `url` (single file), `local` (already on one of your mounts - checked, never downloaded) or `none`.
  `${CHECKPOINTS_DIR}` and `${CACHE_DIR}` expand in paths.
- **Checksums are required for the shipped defaults and optional for your own.** Every shipped
  download carries `sha256` + `size` and is verified after fetch (`python -m fbxify.model_catalog verify --deep`
  re-hashes everything). Custom entries without a checksum are only checked for presence.
- Hugging Face repos are gated: accept the licence on the model page and pass `HF_TOKEN` to the container.
- The shipped catalogue only lists weights whose licences are compatible with this repo. Other
  detector families (for example anything loaded through the `ultralytics` package, which is AGPL-3.0
  and not installed by default) are yours to add from your own overlay, under your own licence terms.

### Switching at runtime

The three dropdowns on the pose tab (body model, encoder, detector) are filled from the catalogue.
Picking a different one on the next job unloads the current weights, flushes the CUDA cache and loads
the new ones (one resident set at a time, ComfyUI-style). INT8 encoders only pair with `dinov3`;
the UI/API refuse other combinations up front.

Worker API:

- `GET /models` (alias `GET /api/models`): catalogue with `status` (`ready`, `missing`,
  `size-mismatch`, `disabled`), `loaded` flags and `defaults`. `GET /health` also reports `models`.
- `POST /jobs/pose` and `POST /jobs/detection` accept optional form fields `body_model`, `encoder`, `detector`
  (catalogue **names only**, never paths). Unknown or disabled names return 400.
- `POST /models/select` swaps now (409 while a job holds the GPU); `POST /models/fetch` re-reads the
  catalogue and downloads enabled entries that are missing; `POST /reload` rebuilds everything.

CLI: `--model`, `--encoder`, `--detector` on `fbxify.cli` / `fbxify.cli_pose_estimation` / `fbxify.app`
take the same names (`--quantized_backbone trt` still works as an alias for `--encoder embedl_int8_trt`).
Environment: `FBXIFY_MODEL`, `FBXIFY_ENCODER` (or `FBXIFY_QUANTIZED_BACKBONE`), `FBXIFY_DETECTOR` set the
initial selection and force-enable those entries.

### Adding your own detector or pose variant

Put a `models.json` next to your checkpoints (`$CHECKPOINTS_DIR/models.json`) or point
`FBXIFY_MODELS_JSON` at one. It is overlaid on the shipped file: same name patches the shipped entry
(for example `{"embedl_int8_trt": {"enabled": true}}` under `encoders`), a new name adds an entry, `null` removes one, and
`"default": true` on your entry takes the default away from the shipped one. `sha256` / `size` on
your own files are optional; when present they are verified like the shipped ones.

```json
{
  "schema": "fbxify.models/1",
  "detectors": {
    "dancers_detector": {
      "display_name": "Person detector fine-tuned on our footage",
      "enabled": true, "default": true,
      "type": "ultralytics",
      "licence": "AGPL-3.0",
      "source": { "kind": "local", "path": "/workspace/models/dancers_detector.pt" },
      "options": { "imgsz": 960, "classes": [0] }
    },
    "rtdetr_from_hub": {
      "display_name": "My detector on the Hub",
      "enabled": false,
      "type": "ultralytics",
      "source": {
        "kind": "huggingface", "repo_id": "me/my-detector", "revision": "main",
        "files": [ { "path": "best.pt", "size": 12345678 } ]
      },
      "local_dir": "${CACHE_DIR}/my-detector"
    }
  },
  "body_models": {
    "mhr_finetune": {
      "display_name": "SAM 3D Body (our fine-tune)",
      "enabled": true,
      "type": "sam_3d_body",
      "source": { "kind": "local", "path": "/workspace/models/sam3d-finetune" }
    }
  },
  "encoders": {
    "my_int8": {
      "display_name": "INT8 export (TensorRT)",
      "enabled": true,
      "type": "embedl_int8",
      "options": { "mode": "trt" },
      "requires_body": ["dinov3"],
      "source": { "kind": "local", "path": "/workspace/models/my-int8-export" }
    }
  }
}
```

What each `type` expects on disk:

- `sam_3d_body`: `model.ckpt`, `model_config.yaml` and `assets/mhr_model.pt` in the directory
  (override file names with `options.checkpoint_file` / `options.mhr_file`).
- `detectron2_vitdet`: `model_final_f05665.pkl` in the directory (the config is the repo's).
- `ultralytics`: one `.pt` weights file (`source.path` may point straight at it). `options` go to
  `model.predict` (`imgsz`, `classes`, `half`, `max_det`); the job's confidence/NMS thresholds
  still come from the caller like for ViTDet.
- `embedl_int8`: `embedl_sam3dbody_int8.onnx` + `.onnx.data` for `mode: trt`, or
  `embedl_sam3dbody_int8.pt2` for `mode: pt2`. Set `requires_body` to the checkpoints it was exported from.
- `checkpoint` (encoders): no files; uses the encoder stored in the body-model checkpoint.

A detector architecture that is neither detectron2 nor Ultralytics needs a loader/runner pair
registered with `tools.build_detector.register_detector_type(name, loader, runner, batch_runner)`;
the runner must return xyxy boxes sorted like `run_detectron2_vitdet`. Accept a `return_scores=True`
keyword and return `(boxes, scores)` if you want the confidence modes below to see your detector's
scores; without it the boxes have no confidence and the modes fall back as described. Then reference
that `name` as the entry's `type`.

Check what the catalogue sees with `python -m fbxify.model_catalog list --all` (add `--json` for scripts).

## Choosing who gets posed

Posing is the expensive step, so the tracking mode decides which boxes in each frame are posed.
`tracking_mode` is the same value in the GUI, `--tracking_mode` on the CLIs and the form field on
`POST /jobs/pose`.

| Mode | Keeps per frame | Needs |
|---|---|---|
| `count` (Closest N People) | the N tallest boxes | `num_people` |
| `confidence_top` (Highest Confidence N People) | the N boxes with the highest confidence | `num_people` |
| `confidence_filter` (Filter Confidence N People) | every box at or above the threshold | `min_confidence` (0–1, default 0.5) |
| `bbox` | every box in the bbox file, with the file's ids | bbox file |
| `inference`, `inference_bbox` | every detected box, ids assigned by the tracker | optional bbox file |

Where the confidence comes from:

- **Your own bbox file.** MOTChallenge rows `frame,id,x,y,w,h,conf[,x,y,z]`; the seventh column is
  the confidence. `-1` (or a blank) means unknown. Give the three selector modes a bbox file and they
  select from your boxes instead of running the detector, so a box list from your own tracker or a
  manual annotation drives which people get posed.
- **The detector.** Without a bbox file the person detector runs and its per-box score is the
  confidence. Detection results exported as MOT carry that score in the conf column (`-1` when the
  detector gave none).

A box without a confidence never breaks a run: it sorts after every scored box in
`confidence_top`, and is dropped by `confidence_filter`. If no box in a frame has a confidence,
`confidence_top` behaves like `count` for that frame and `confidence_filter` keeps the frame as is.
Boxes keep their ids and world coordinates through selection.

Note what the number means. A stock person detector's score is "how sure am I this is a person",
not "how important is this person": in a crowd the ranking says nothing about who the dancers are.
Use `confidence_top` to trust a detector you have tuned, or hand in a bbox file whose conf column
encodes your own ranking (for example `1.0` for the performers and `0.1` for the audience).

API example, three people with your own confidence-ranked boxes:

```bash
curl -X POST http://localhost:8000/jobs/pose \
  -F "input_file=@clip.mp4" -F "bbox_file=@boxes.txt" \
  -F "tracking_mode=confidence_top" -F "num_people=3"
```

The same with the detector's scores and a threshold:

```bash
curl -X POST http://localhost:8000/jobs/pose \
  -F "input_file=@clip.mp4" -F "tracking_mode=confidence_filter" -F "min_confidence=0.7"
```

`count` now keeps the N tallest boxes when a detector is loaded; earlier releases posed every
detected person in that mode and only used N when no detector was available.

### Adding Custom Rigs

To add support for a custom rig:

1. Extract your skeleton structure and rest pose using the provided Blender scripts:
   - `extract_armature_bone_struct_for_mapping.py`
   - `extract_armature_skeleton_and_rest_pose.py`
2. Ensure your armature is selected and in rest pose
3. Copy the console output into a JSON file and add it to `metadata.py`'s `PROFILES`
4. Configure the mapping method and parameters for each bone

For complete instructions, see [fbxify/Mapping_Guide.md](fbxify/Mapping_Guide.md).

## TODO

Future improvements planned for FBXify:

1. Extend the mesh capabilities by providing lod\<x\>_mixamo.fbx, in addition to MHR
2. Facial animation integration (MHR specific maybe?)
3. Add option to scale the armature based on estimator returned values

# SAM 3D

SAM 3D Body is one part of SAM 3D, a pair of models for object and human mesh reconstruction. If you’re looking for SAM 3D Objects, [click here](https://github.com/facebookresearch/sam-3d-objects).

# SAM 3D Body: Robust Full-Body Human Mesh Recovery

<p align="left">
<a href="https://ai.meta.com/research/publications/sam-3d-body-robust-full-body-human-mesh-recovery/"><img src='https://img.shields.io/badge/Meta_AI-Paper-4A90E2?logo=meta&logoColor=white' alt='Paper'></a>
<a href="https://ai.meta.com/blog/sam-3d/"><img src='https://img.shields.io/badge/Project_Page-Blog-9B72F0?logo=googledocs&logoColor=white' alt='Blog'></a>
<a href="https://huggingface.co/datasets/facebook/sam-3d-body-dataset"><img src='https://img.shields.io/badge/🤗_Hugging_Face-Dataset-F59500?logoColor=white' alt='Dataset'></a>
<a href="https://www.aidemos.meta.com/segment-anything/editor/convert-body-to-3d"><img src='https://img.shields.io/badge/🤸_Playground-Live_Demo-E85D5D?logoColor=white' alt='Live Demo'></a>
</p>

[Xitong Yang](https://scholar.google.com/citations?user=k0qC-7AAAAAJ&hl=en)\*, [Devansh Kukreja](https://www.linkedin.com/in/devanshkukreja)\*, [Don Pinkus](https://www.linkedin.com/in/don-pinkus-9140702a)\*, [Anushka Sagar](https://www.linkedin.com/in/anushkasagar), [Taosha Fan](https://scholar.google.com/citations?user=3PJeg1wAAAAJ&hl=en), [Jinhyung Park](https://jindapark.github.io/)⚬, [Soyong Shin](https://yohanshin.github.io/)⚬, [Jinkun Cao](https://www.jinkuncao.com/), [Jiawei Liu](https://jia-wei-liu.github.io/), [Nicolas Ugrinovic](https://www.iri.upc.edu/people/nugrinovic/), [Matt Feiszli](https://scholar.google.com/citations?user=A-wA73gAAAAJ&hl=en&oi=ao)†, [Jitendra Malik](https://people.eecs.berkeley.edu/~malik/)†, [Piotr Dollar](https://pdollar.github.io/)†, [Kris Kitani](https://kriskitani.github.io/)†

***Meta Superintelligence Labs***

*Core Contributor,  ⚬Intern, †Project Lead

![SAM 3D Body Model Architecture](assets/model_diagram.png?raw=true)

**SAM 3D Body (3DB)** is a promptable model for single-image full-body 3D human mesh recovery (HMR). Our method demonstrates state-of-the-art performance, with strong generalization and consistent accuracy in diverse in-the-wild conditions. 3DB estimates the human pose of the body, feet, and hands based on the [Momentum Human Rig](https://github.com/facebookresearch/MHR) (MHR), a new parametric mesh representation that decouples skeletal structure and surface shape for improved accuracy and interpretability.

3DB employs an encoder-decoder architecture and supports auxiliary prompts, including 2D keypoints and masks, enabling user-guided inference similar to the SAM family of models. Our model is trained on high-quality annotations from a multi-stage annotation pipeline using differentiable optimization, multi-view geometry, dense keypoint detection, and a data engine to collect and annotated data covering both common and rare poses across a wide range of viewpoints.

## Qualitative Results

<table>
<thead>
<tr>
<th align="center">Input</th>
<th align="center"><strong>SAM 3D Body</strong></th>
<th align="center">CameraHMR</th>
<th align="center">NLF</th>
<th align="center">HMR2.0b</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center"><img src="assets/qualitative_comparisons/sample1/input_bbox.png" alt="Sample 1 Input" width="160"></td>
<td align="center"><img src="assets/qualitative_comparisons/sample1/SAM 3D Body.png" alt="Sample 1 - SAM 3D Body" width="160"></td>
<td align="center"><img src="assets/qualitative_comparisons/sample1/camerahmr.png" alt="Sample 1 - CameraHMR" width="160"></td>
<td align="center"><img src="assets/qualitative_comparisons/sample1/nlf.png" alt="Sample 1 - NLF" width="160"></td>
<td align="center"><img src="assets/qualitative_comparisons/sample1/4dhumans.png" alt="Sample 1 - 4DHumans (HMR2.0b)" width="160"></td>
</tr>
<tr>
<td align="center"><img src="assets/qualitative_comparisons/sample2/input_bbox.png" alt="Sample 2 Input" width="160"></td>
<td align="center"><img src="assets/qualitative_comparisons/sample2/SAM 3D Body.png" alt="Sample 2 - SAM 3D Body" width="160"></td>
<td align="center"><img src="assets/qualitative_comparisons/sample2/camerahmr.png" alt="Sample 2 - CameraHMR" width="160"></td>
<td align="center"><img src="assets/qualitative_comparisons/sample2/nlf.png" alt="Sample 2 - NLF" width="160"></td>
<td align="center"><img src="assets/qualitative_comparisons/sample2/4dhumans.png" alt="Sample 2 - 4DHumans (HMR2.0b)" width="160"></td>
</tr>
<tr>
<td align="center"><img src="assets/qualitative_comparisons/sample3/input_bbox.png" alt="Sample 3 Input" width="160"></td>
<td align="center"><img src="assets/qualitative_comparisons/sample3/SAM 3D Body.png" alt="Sample 3 - SAM 3D Body" width="160"></td>
<td align="center"><img src="assets/qualitative_comparisons/sample3/camerahmr.png" alt="Sample 3 - CameraHMR" width="160"></td>
<td align="center"><img src="assets/qualitative_comparisons/sample3/nlf.png" alt="Sample 3 - NLF" width="160"></td>
<td align="center"><img src="assets/qualitative_comparisons/sample3/4dhumans.png" alt="Sample 3 - 4DHumans (HMR2.0b)" width="160"></td>
</tr>
<tr>
<td align="center"><img src="assets/qualitative_comparisons/sample4/input_bbox.png" alt="Sample 4 Input" width="160"></td>
<td align="center"><img src="assets/qualitative_comparisons/sample4/SAM 3D Body.png" alt="Sample 4 - SAM 3D Body" width="160"></td>
<td align="center"><img src="assets/qualitative_comparisons/sample4/camerahmr.png" alt="Sample 4 - CameraHMR" width="160"></td>
<td align="center"><img src="assets/qualitative_comparisons/sample4/nlf.png" alt="Sample 4 - NLF" width="160"></td>
<td align="center"><img src="assets/qualitative_comparisons/sample4/4dhumans.png" alt="Sample 4 - 4DHumans (HMR2.0b)" width="160"></td>
</tr>
</tbody>
</table>

*Our SAM 3D Body demonstrates superior reconstruction quality with more accurate pose estimation, better shape recovery, and improved handling of occlusions and challenging viewpoints compared to existing approaches.*

## Latest updates

**11/19/2025** -- Checkpoints Launched, Dataset Released, Web Demo and Paper are out!

## Installation
See [INSTALL.md](INSTALL.md) for instructions for python environment setup and model checkpoint access.

## Getting Started

3DB can reconstruct 3D full-body human mesh from a single image, optionally with keypoint/mask prompts and/or hand refinement from the hand decoder. 

For a quick start, run our demo script for model inference and visualization with models from [Hugging Face](https://huggingface.co/facebook) (please make sure to follow [INSTALL.md](INSTALL.md) to request access to our checkpoints.).

```bash
# Download assets from HuggingFace
hf download facebook/sam-3d-body-dinov3 --local-dir checkpoints/sam-3d-body-dinov3

# Run demo script
python demo.py \
    --image_folder <path_to_images> \
    --output_folder <path_to_output> \
    --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
    --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
```

You can also try the following lines of code with models loaded directly from [Hugging Face](https://huggingface.co/facebook)

```python
import cv2
import numpy as np
from notebook.utils import setup_sam_3d_body
from tools.vis_utils import visualize_sample_together

# Set up the estimator
estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")

# Load and process image
img_bgr = cv2.imread("path/to/image.jpg")
outputs = estimator.process_one_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

# Visualize and save results
rend_img = visualize_sample_together(img_bgr, outputs, estimator.faces)
cv2.imwrite("output.jpg", rend_img.astype(np.uint8))
```

For a complete demo with visualization, see [notebook/demo_human.ipynb](notebook/demo_human.ipynb).


## Model Description

### SAM 3D Body checkpoints

The table below shows the performance of SAM 3D Body checkpoints released on 11/19/2025.

|      **Backbone (size)**       | **3DPW (MPJPE)** |    **EMDB (MPJPE)**     | **RICH (PVE)** | **COCO (PCK@.05)** |  **LSPET (PCK@.05)** | **Freihand (PA-MPJPE)** 
| :------------------: | :----------: | :--------------------: | :-----------------: | :----------------: | :----------------: | :----------------: |
|  DINOv3-H+ (840M) <br /> ([config](https://huggingface.co/facebook/sam-3d-body-dinov3/blob/main/model_config.yaml), [checkpoint](https://huggingface.co/facebook/sam-3d-body-dinov3/blob/main/model.ckpt))   |      54.8      |          61.7         |       60.3        |       86.5        | 68.0 | 5.5
|   ViT-H  (631M) <br /> ([config](https://huggingface.co/facebook/sam-3d-body-vith/blob/main/model_config.yaml), [checkpoint](https://huggingface.co/facebook/sam-3d-body-vith/blob/main/model.ckpt))    |     54.8   |         62.9         |       61.7        |        86.8       | 68.9 |  5.5


## SAM 3D Body Dataset
The SAM 3D Body data is released on [Hugging Face](https://huggingface.co/datasets/facebook/sam-3d-body-dataset). Please follow the [instructions](./data/README.md) to download and process the data.

## SAM 3D Objects

[SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) is a foundation model that reconstructs full 3D shape geometry, texture, and layout from a single image.

As a way to combine the strengths of both **SAM 3D Objects** and **SAM 3D Body**, we provide an example notebook that demonstrates how to combine the results of both models such that they are aligned in the same frame of reference. Check it out [here](https://github.com/facebookresearch/sam-3d-objects/blob/main/notebook/demo_3db_mesh_alignment.ipynb).

## License

The SAM 3D Body model checkpoints and code are licensed under [SAM License](./LICENSE).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Contributors

The SAM 3D Body project was made possible with the help of many contributors:
Vivian Lee, George Orlin, Nikhila Ravi, Andrew Westbury, Jyun-Ting Song, Zejia Weng, Xizi Zhang, Yuting Ye, Federica Bogo, Ronald Mallet, Ahmed Osman, Rawal Khirodkar, Javier Romero, Carsten Stoll, Jean-Charles Bazin, Sofien Bouaziz, Yuan Dong, Su Zhaoen, Fabian Prada, Alexander Richard, Michael Zollhoefer, Roman Rädle, Sasha Mitts, Michelle Chan, Yael Yungster, Azita Shokrpour, Helen Klein, Mallika Malhotra, Ida Cheng, Eva Galper.

## Citing SAM 3D Body

If you use SAM 3D Body or the SAM 3D Body dataset in your research, please use the following BibTeX entry.

```bibtex
@article{yang2025sam3dbody,
  title={SAM 3D Body: Robust Full-Body Human Mesh Recovery},
  author={Yang, Xitong and Kukreja, Devansh and Pinkus, Don and Sagar, Anushka and Fan, Taosha and Park, Jinhyung and Shin, Soyong and Cao, Jinkun and Liu, Jiawei and Ugrinovic, Nicolas and Feiszli, Matt and Malik, Jitendra and Dollar, Piotr and Kitani, Kris},
  journal={arXiv preprint; identifier to be added},
  year={2025}
}
```
