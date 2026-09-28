"""Plumbing tests for Fast SAM 3D Body skip_keypoint_prompt.

Avoids importing SAM 3D Body (torchvision/checkpoints). Asserts wiring in source.

Run: python -m pytest fbxify/tests/test_fast_inference_flag.py -v
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_estimator_defaults_skip_keypoint_prompt_off():
    src = _read("sam_3d_body/sam_3d_body_estimator.py")
    assert "self.skip_keypoint_prompt = False" in src
    assert "skip_keypoint_prompt=self.skip_keypoint_prompt" in src


def test_run_inference_accepts_and_honors_skip_flag():
    src = _read("sam_3d_body/models/meta_arch/sam3d_body.py")
    assert "skip_keypoint_prompt: bool = False" in src
    assert "do_keypoint_prompt = not skip_keypoint_prompt" in src
    assert "if do_keypoint_prompt and keypoint_prompt.numel() != 0:" in src
    assert "if (not skip_keypoint_prompt) and keypoint_prompt.numel() != 0:" in src


def test_manager_set_inference_options_wires_flag():
    src = _read("fbxify/pose_estimation_manager.py")
    assert "skip_keypoint_prompt: Optional[bool] = None" in src
    assert "self.estimator.skip_keypoint_prompt = new_val" in src


def test_cli_and_api_expose_fast_inference():
    assert "--fast_inference" in _read("fbxify/cli.py")
    assert "--fast_inference" in _read("fbxify/cli_pose_estimation.py")
    api = _read("fbxify/api.py")
    assert "fast_inference: bool = Form(False)" in api
    assert '"fast_inference": fast_inference' in api
    assert "skip_keypoint_prompt = bool(params.get(\"fast_inference\", False))" in api
    backend = _read("fbxify/backend.py")
    assert "skip_keypoint_prompt=bool(fast_inference)" in backend
    assert '"fast_inference": "true" if fast_inference else "false"' in backend
    app = _read("fbxify/app.py")
    assert "fast_inference=bool(fast_inference)" in app
    assert "entry_components['fast_inference']" in app
    assert 'cmd_parts.append("--fast_inference")' in app
    ui = _read("fbxify/gradio_ui/entry_section.py")
    assert "components['fast_inference'] = gr.Checkbox(" in ui
