"""Plumbing tests for the per-job body-only pose option and body-only frame batching.

Avoids importing SAM 3D Body (torchvision/checkpoints). Asserts wiring in source.

Run: python -m pytest fbxify/tests/test_body_only_option.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_estimator_batches_body_only_and_falls_back_for_hand_only():
    src = _read("sam_3d_body/sam_3d_body_estimator.py")
    # The old blanket fallback is gone: only hand-only goes frame by frame.
    assert "Body-only or hand-only: no batching benefit" not in src
    assert 'if inference_type == "hand":' in src
    # Body-only runs the same multi-frame body forward the full path starts with.
    assert 'if inference_type == "body":' in src
    assert 'pose_output = self.model.forward_step(batch, decoder_type="body")' in src
    # Hand boxes are only emitted when the hand decoders ran.
    assert "if batch_lhand is not None:" in src
    assert 'all_out_k[-1]["lhand_bbox"] = lhand_bbox_cpu[person_idx]' in src


def test_manager_exposes_body_only_and_passes_inference_type():
    src = _read("fbxify/pose_estimation_manager.py")
    assert 'self.inference_type = "full"' in src
    assert "body_only: Optional[bool] = None" in src
    assert 'new_type = "body" if body_only else "full"' in src
    # Both the batched and the per-frame paths honour the selection.
    assert "frames_list, inference_type=self.inference_type" in src
    assert "inference_type=self.inference_type,\n        )" in src
    assert 'inference_type="full"' not in src
    # The choice is written to the estimation file's metadata and read back.
    assert '"type": self.inference_type,' in src
    assert '"body_only": self.inference_type == "body",' in src
    assert '"inference": data.get("inference", None),' in src


def test_cli_api_and_gui_expose_body_only():
    for rel in ("fbxify/cli.py", "fbxify/cli_pose_estimation.py"):
        cli = _read(rel)
        assert '"--body_only"' in cli
        assert "body_only=bool(args.body_only)" in cli
    api = _read("fbxify/api.py")
    assert "body_only: bool = Form(False)" in api
    assert '"body_only": body_only' in api
    assert 'body_only = bool(params.get("body_only", False))' in api
    backend = _read("fbxify/backend.py")
    assert "body_only=bool(body_only)" in backend
    assert '"body_only": "true" if body_only else "false"' in backend
    app = _read("fbxify/app.py")
    assert "body_only=bool(body_only)" in app
    assert app.count("entry_components['body_only']") == 3
    assert 'cmd_parts.append("--body_only")' in app
    ui = _read("fbxify/gradio_ui/entry_section.py")
    assert "components['body_only'] = gr.Checkbox(" in ui
    assert 'gr.update(label=t.t("ui.body_only"), info=t.t("ui.body_only_info"))' in ui


def test_i18n_and_postman_carry_the_option():
    for lang in ("en", "es", "fr", "ja"):
        data = json.loads(_read(f"fbxify/i18n/{lang}.json"))
        assert data["ui"]["body_only"]
        assert data["ui"]["body_only_info"]
    collection = json.loads(_read("fbxify/postman/fbxify-worker-api.postman_collection.json"))
    text = json.dumps(collection)
    assert text.count('"body_only"') == 2
