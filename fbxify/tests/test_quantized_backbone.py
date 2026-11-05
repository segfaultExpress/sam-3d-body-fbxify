"""Plumbing tests for the embedl INT8 encoder swap (FBXIFY_QUANTIZED_BACKBONE).

Source-level wiring assertions (no checkpoints) plus a behavioural test of
EmbedlInt8Backbone with a fake runner (needs torch only).

Run: python -m pytest fbxify/tests/test_quantized_backbone.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _load_embedl_module():
    """Import embedl_int8.py by path so the sam_3d_body package __init__ (torchvision) is not needed."""
    import importlib.util

    path = ROOT / "sam_3d_body" / "models" / "backbones" / "embedl_int8.py"
    spec = importlib.util.spec_from_file_location("embedl_int8_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_manager_accepts_quantized_backbone_and_keeps_encoder_fp32():
    src = _read("fbxify/pose_estimation_manager.py")
    assert "quantized_backbone=None" in src
    assert "swap_in_quantized_backbone(model, self.quantized_dir, quantized_mode)" in src
    assert "if self.quantized_backbone:" in src and "model.backbone_dtype = torch.float32" in src


def test_normalize_quantized_mode():
    pytest.importorskip("torchvision")
    from fbxify.pose_estimation_manager import normalize_quantized_mode

    assert normalize_quantized_mode(None) is None
    assert normalize_quantized_mode("none") is None
    assert normalize_quantized_mode("0") is None
    assert normalize_quantized_mode("TRT") == "trt"
    assert normalize_quantized_mode("pt2") == "pt2"
    with pytest.raises(ValueError):
        normalize_quantized_mode("onnx")


def test_entry_points_expose_quantized_backbone():
    # The flag lives in the shared argparse helper; every entry point calls it and the legacy
    # value is mapped onto an encoder catalogue name.
    common = _read("fbxify/cli_common.py")
    assert "--quantized_backbone" in common
    assert 'encoder_name_for_quantized_mode(getattr(args, "quantized_backbone", "none"))' in common
    for rel in ("fbxify/cli.py", "fbxify/cli_pose_estimation.py", "fbxify/app.py"):
        src = _read(rel)
        assert "add_model_args(parser)" in src, rel
        assert "build_estimation_manager(args" in src, rel
    api = _read("fbxify/api.py")
    assert 'os.environ.get("FBXIFY_QUANTIZED_BACKBONE", "none")' in api
    compose = _read("fbxify/docker/docker-compose.yml")
    assert "FBXIFY_QUANTIZED_BACKBONE" in compose


def test_download_helper_knows_both_modes():
    from fbxify.checkpoint_download import EMBEDL_FILES, EMBEDL_REPO_ID, embedl_quantized_dir

    assert EMBEDL_REPO_ID == "embedl/sam-3d-body"
    assert "embedl_sam3dbody_int8.pt2" in EMBEDL_FILES["pt2"]
    assert "embedl_sam3dbody_int8.onnx.data" in EMBEDL_FILES["trt"]
    assert embedl_quantized_dir("/x/checkpoints").replace("\\", "/").endswith("/x/checkpoints/embedl-sam-3d-body")


def test_backbone_wrapper_casts_and_batches(monkeypatch):
    torch = pytest.importorskip("torch")
    mod = _load_embedl_module()

    calls = []

    class FakeRunner:
        input_dtype = torch.float32
        input_shape = (1, 3, 512, 512)

        def __call__(self, x):
            calls.append(tuple(x.shape))
            return torch.ones(1, 1280, 32, 32, dtype=x.dtype) * x.mean()

    def fake_init(self, quantized_dir, mode="trt", device=None, embed_dim=1280, patch_size=16, **kw):
        torch.nn.Module.__init__(self)
        self.mode, self.quantized_dir = mode, quantized_dir
        self.patch_size, self.embed_dim, self.embed_dims = patch_size, embed_dim, embed_dim
        self.device_ = torch.device("cpu")
        self._runner_dtype = torch.float32
        object.__setattr__(self, "_runner", FakeRunner())

    monkeypatch.setattr(mod.EmbedlInt8Backbone, "__init__", fake_init)
    bb = mod.EmbedlInt8Backbone("/nowhere", mode="trt")

    # Batch of 3 crops (body + 2 hands) in bf16 must be split per-crop and come back in bf16.
    x = torch.rand(3, 3, 512, 512).to(torch.bfloat16)
    y = bb(x)
    assert y.shape == (3, 1280, 32, 32)
    assert y.dtype == torch.bfloat16
    assert calls == [(1, 3, 512, 512)] * 3
    # .half()/.float() must not reach the runner (it is not a registered submodule).
    assert "_runner" not in dict(bb.named_modules())
    bb.half()
    assert bb._runner.input_dtype == torch.float32


def test_loader_refuses_checkpoint_without_backbone_weights():
    """A checkpoint that leaves the encoder randomly initialised must not load silently."""
    src = _read("sam_3d_body/build_models.py")
    assert "require_backbone_weights: bool = True" in src
    assert "_assert_backbone_weights_loaded(model, state_dict, checkpoint_path, model_cfg)" in src
    manager = _read("fbxify/pose_estimation_manager.py")
    assert "require_backbone_weights=not quantized_mode" in manager


def test_swap_rejects_non_dinov3():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    swap_in_quantized_backbone = _load_embedl_module().swap_in_quantized_backbone

    model = SimpleNamespace(cfg=SimpleNamespace(MODEL=SimpleNamespace(BACKBONE=SimpleNamespace(TYPE="vit_hmr"))),
                            backbone=torch.nn.Identity(), device=torch.device("cpu"))
    with pytest.raises(ValueError):
        swap_in_quantized_backbone(model, "/nowhere", "pt2")
