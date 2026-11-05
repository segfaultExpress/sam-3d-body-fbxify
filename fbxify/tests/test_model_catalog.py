"""Model catalogue: schema, overlay, resolution, fetch gating and plumbing.

Does not import SAM 3D Body or touch the network.

Run: python -m pytest fbxify/tests/test_model_catalog.py -v
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from fbxify import model_catalog as mc

ROOT = Path(__file__).resolve().parents[2]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _dirs(tmp_path: Path) -> dict:
    return {"CHECKPOINTS_DIR": str(tmp_path / "ckpt"), "CACHE_DIR": str(tmp_path / "cache")}


def _load(tmp_path: Path, user: dict | None = None, **kw) -> mc.ModelCatalog:
    paths = []
    if user is not None:
        upath = tmp_path / "models.json"
        upath.write_text(json.dumps({"schema": mc.SCHEMA, **user}), encoding="utf-8")
        paths.append(str(upath))
    return mc.ModelCatalog.load(user_paths=paths, dirs=_dirs(tmp_path), **kw)


# ----------------------------------------------------------------------------- shipped file


def test_shipped_catalog_is_valid_and_pinned(tmp_path):
    cat = _load(tmp_path)
    assert cat.validate() == []
    for section in mc.SECTIONS:
        defaults = [e for e in cat.section(section).values() if e.default]
        assert len(defaults) == 1, section
        assert defaults[0].enabled
    # Shipped downloadable entries carry a checksum and size for every file (required for defaults).
    for e in cat.all_entries():
        if e.kind in ("huggingface", "url"):
            assert e.builtin
            for f in e.files:
                assert f.sha256 and len(f.sha256) == 64, (e.name, f.path)
                assert isinstance(f.size, int) and f.size > 0, (e.name, f.path)
        if e.kind == "huggingface":
            assert e.source.get("revision"), e.name  # pinned commit


def test_shipped_defaults_and_opt_ins(tmp_path):
    cat = _load(tmp_path)
    assert cat.default("body_models").name == "vith"
    assert cat.default("encoders").name == "fp32"
    assert cat.default("detectors").name == "vitdet"
    assert cat.get("detectors", "yolo11m").enabled is False  # AGPL: opt in
    assert cat.get("detectors", "yolo11m").type == "ultralytics"
    assert cat.get("encoders", "embedl_int8_trt").requires_body == ["dinov3"]
    assert cat.names("body_models") == ["vith", "dinov3"]
    assert "yolo11m" not in cat.names("detectors")
    assert "yolo11m" in cat.names("detectors", enabled_only=False)


def test_paths_expand_checkpoints_and_cache_dirs(tmp_path):
    cat = _load(tmp_path)
    d = _dirs(tmp_path)
    assert cat.get("body_models", "dinov3").local_dir == os.path.normpath(os.path.join(d["CHECKPOINTS_DIR"], "sam-3d-body-dinov3"))
    assert cat.get("detectors", "vitdet").local_dir == os.path.normpath(os.path.join(d["CACHE_DIR"], "videt_checkpoint"))
    assert mc.expand_path("${CACHE_DIR}/x", d) == os.path.normpath(os.path.join(d["CACHE_DIR"], "x"))


# ----------------------------------------------------------------------------- user overlay


def test_user_overlay_adds_local_entries_and_moves_default(tmp_path):
    my_dir = tmp_path / "mnt" / "my-sam3d"
    (my_dir / "assets").mkdir(parents=True)
    (my_dir / "model.ckpt").write_bytes(b"x" * 10)
    (my_dir / "assets" / "mhr_model.pt").write_bytes(b"y" * 10)
    weights = tmp_path / "mnt" / "dancers.pt"
    weights.write_bytes(b"z" * 5)

    cat = _load(tmp_path, {
        "body_models": {
            "my_sam3d": {
                "display_name": "My fine-tune",
                "enabled": True, "default": True, "type": "sam_3d_body",
                "source": {"kind": "local", "path": str(my_dir)},
            }
        },
        "detectors": {
            "dancers_yolo": {
                "display_name": "Dancers YOLO", "enabled": True, "type": "ultralytics",
                "source": {"kind": "local", "path": str(weights)},
                "options": {"imgsz": 960},
            },
            "yolo11m": {"enabled": True},
            "vitdet": {"enabled": False},
        },
        "encoders": {"embedl_int8_pt2": None},
    })
    assert cat.validate() == []
    body = cat.get("body_models", "my_sam3d")
    assert body.default and not cat.get("body_models", "vith").default
    assert body.builtin is False and body.origin.endswith("models.json")
    assert body.status() == "ready-unverified"  # custom: no checksum needed
    entry, ckpt, mhr = cat.resolve_body(None)  # default now the user model
    assert entry.name == "my_sam3d"
    assert ckpt == str(my_dir / "model.ckpt") and mhr == str(my_dir / "assets" / "mhr_model.pt")

    det, kwargs = cat.resolve_detector("dancers_yolo")
    assert kwargs["weights"] == str(weights)
    assert kwargs["options"] == {"imgsz": 960}
    assert cat.get("detectors", "yolo11m").enabled is True       # patched shipped entry
    assert cat.get("detectors", "yolo11m").builtin is True        # source untouched -> still verified
    assert cat.get("detectors", "vitdet").enabled is False
    assert "embedl_int8_pt2" not in cat.section("encoders")      # null removes


def test_overlay_validation_errors(tmp_path):
    # Disabling the shipped default hands the default to the next enabled entry.
    cat = _load(tmp_path, {"body_models": {"vith": {"enabled": False}}})
    assert cat.default("body_models").name == "dinov3"
    cat = _load(tmp_path, {"detectors": {"vitdet": {"enabled": False}}})
    with pytest.raises(mc.CatalogError, match="no default entry"):
        cat.default("detectors")
    with pytest.raises(mc.CatalogError, match="source.kind"):
        _load(tmp_path, {"detectors": {"x": {"type": "ultralytics", "source": {"kind": "ftp"}}}})
    with pytest.raises(mc.CatalogError, match="unknown body model"):
        _load(tmp_path, {"encoders": {"e": {"type": "checkpoint", "requires_body": ["nope"]}}})
    # Shipped entry with a replaced source becomes custom: checksum optional, no error.
    cat = _load(tmp_path, {"detectors": {"vitdet": {"source": {"kind": "local", "path": str(tmp_path / "v")}}}})
    assert cat.get("detectors", "vitdet").builtin is False


def test_default_user_paths_env(monkeypatch, tmp_path):
    monkeypatch.delenv("FBXIFY_MODELS_JSON", raising=False)
    assert mc.default_user_paths(_dirs(tmp_path)) == [os.path.join(_dirs(tmp_path)["CHECKPOINTS_DIR"], "models.json")]
    monkeypatch.setenv("FBXIFY_MODELS_JSON", str(tmp_path / "custom.json"))
    assert mc.default_user_paths(_dirs(tmp_path)) == [str(tmp_path / "custom.json")]


# ----------------------------------------------------------------------------- resolution


def test_encoder_requires_body_and_legacy_mapping(tmp_path):
    cat = _load(tmp_path)
    with pytest.raises(mc.CatalogError, match="requires body model dinov3"):
        cat.resolve_encoder("embedl_int8_trt", body="vith")
    e, mode, qdir = cat.resolve_encoder("embedl_int8_trt", body="dinov3")
    assert mode == "trt" and qdir.endswith("embedl-sam-3d-body")
    e, mode, qdir = cat.resolve_encoder("fp32", body="vith")
    assert mode is None and qdir is None
    assert mc.encoder_name_for_quantized_mode(None) == "fp32"
    assert mc.encoder_name_for_quantized_mode("none") == "fp32"
    assert mc.encoder_name_for_quantized_mode("TRT") == "embedl_int8_trt"
    assert mc.encoder_name_for_quantized_mode("pt2") == "embedl_int8_pt2"
    with pytest.raises(mc.CatalogError):
        mc.encoder_name_for_quantized_mode("onnx")


def test_unknown_names_raise_and_describe_hides_paths(tmp_path):
    cat = _load(tmp_path)
    with pytest.raises(mc.CatalogError, match="unknown detector"):
        cat.get("detectors", "../../etc/passwd")
    payload = cat.describe(loaded={"body_model": "vith", "encoder": "fp32", "detector": "vitdet"})
    assert payload["schema"] == mc.SCHEMA
    vith = next(x for x in payload["body_models"] if x["name"] == "vith")
    assert vith["loaded"] is True and vith["status"] == "missing"
    text = json.dumps(payload)
    assert str(tmp_path) not in text and "local_dir" not in text


# ----------------------------------------------------------------------------- status + fetch


def test_status_tracks_missing_size_and_ready(tmp_path):
    cat = _load(tmp_path)
    det = cat.get("detectors", "yolo11m")
    det.enabled = True
    assert det.status() == "missing"
    p = Path(det.file_path("yolo11m.pt"))
    p.parent.mkdir(parents=True)
    p.write_bytes(b"short")
    assert det.status() == "size-mismatch"
    ok, msg = det.verify()
    assert not ok and "size mismatch" in msg


def test_fetch_never_downloads_local_or_disabled(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(mc, "_download_url", boom)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    cat = _load(tmp_path, {"detectors": {"mine": {
        "enabled": True, "type": "ultralytics", "source": {"kind": "local", "path": str(tmp_path / "nope.pt")}}}})
    r = mc.fetch_entry(cat.get("detectors", "mine"))
    assert r.status == "missing" and "not downloaded" in r.message
    # Hugging Face entries without a token: reported, not attempted.
    r = mc.fetch_entry(cat.get("body_models", "vith"))
    assert r.status == "missing" and "HF_TOKEN" in r.message
    # fetch_enabled touches enabled entries only (fp32 has no files; vitdet is url -> would call boom).
    cat.get("detectors", "vitdet").enabled = False
    cat.get("detectors", "vitdet").default = False
    results = mc.fetch_enabled(cat, log=lambda *a, **k: None)
    assert {r.entry for r in results} == {"mine", "fp32", "vith", "dinov3"}


def test_fetch_url_verifies_sha256(tmp_path, monkeypatch):
    payload = b"fake-weights" * 100
    good = hashlib.sha256(payload).hexdigest()

    def fake_download(url, dest, expected_size):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        Path(dest).write_bytes(payload)

    monkeypatch.setattr(mc, "_download_url", fake_download)
    user = {"detectors": {
        "ok": {"enabled": True, "type": "ultralytics",
               "source": {"kind": "url", "url": "https://example.invalid/ok.pt", "sha256": good, "size": len(payload)}},
        "bad": {"enabled": True, "type": "ultralytics",
                "source": {"kind": "url", "url": "https://example.invalid/bad.pt", "sha256": "0" * 64, "size": len(payload)}},
    }}
    cat = _load(tmp_path, user)
    assert mc.fetch_entry(cat.get("detectors", "ok")).status == "ready"
    r = mc.fetch_entry(cat.get("detectors", "bad"))
    assert r.status == "failed" and "sha256" in r.message
    assert os.path.exists(cat.get("detectors", "bad").file_path("bad.pt") + ".bad")
    assert cat.get("detectors", "bad").status() == "missing"


def test_cli_list_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CHECKPOINTS_DIR", _dirs(tmp_path)["CHECKPOINTS_DIR"])
    monkeypatch.setenv("CACHE_DIR", _dirs(tmp_path)["CACHE_DIR"])
    monkeypatch.delenv("FBXIFY_MODELS_JSON", raising=False)
    assert mc._cli(["list", "--all", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {r["name"] for r in rows} >= {"vitdet", "yolo11m", "fp32", "embedl_int8_trt", "vith", "dinov3"}


# ----------------------------------------------------------------------------- plumbing


def test_api_exposes_models_and_accepts_names_only():
    api = _read("fbxify/api.py")
    assert '@app.get("/models")' in api and '@app.get("/api/models")' in api
    assert '@app.post("/models/select")' in api
    assert "body_model: Optional[str] = Form(None)" in api
    assert "encoder: Optional[str] = Form(None)" in api
    assert "detector: Optional[str] = Form(None)" in api
    assert "_validate_model_names(body_model=body_model, encoder=encoder, detector=detector)" in api
    assert "PoseEstimationManager.from_catalog(" in api
    assert "fetch_enabled(catalog)" in api
    assert '"models": _loaded_models()' in api  # /health reports what is loaded


def test_manager_swaps_under_lock_and_frees_gpu():
    src = _read("fbxify/pose_estimation_manager.py")
    assert "def select(self, body_model=None, encoder=None, detector=None)" in src
    assert "with self.swap_lock:" in src
    assert "torch.cuda.empty_cache()" in src
    assert "def _release_body(self)" in src and "def _release_detector(self" in src
    assert "def from_catalog(cls, catalog" in src


def test_backend_and_ui_wire_the_dropdowns():
    backend = _read("fbxify/backend.py")
    assert "def list_models(self)" in backend
    assert "payload.update(_clean_model_names(body_model, encoder, detector))" in backend
    assert "self._select_models(body_model=body_model, encoder=encoder, detector=detector)" in backend
    ui = _read("fbxify/gradio_ui/entry_section.py")
    assert 'components[key] = gr.Dropdown(' in ui and "MODEL_SECTIONS" in ui
    assert 'gr.update(label=t.t("ui.body_model")' in ui
    app = _read("fbxify/app.py")
    assert "entry_components['body_model'], entry_components['encoder'], entry_components['detector']," in app
    assert "body_model=body_model,\n                encoder=encoder,\n                detector=detector," in app
    assert "backend.list_models()" in app
    for lang in ("en", "es", "fr", "ja"):
        data = json.loads(_read(f"fbxify/i18n/{lang}.json"))
        for key in ("body_model", "body_model_info", "encoder", "encoder_info", "detector", "detector_info"):
            assert key in data["ui"], (lang, key)


def test_detector_registry_has_opt_in_ultralytics():
    src = _read("tools/build_detector.py")
    assert 'elif name in ("ultralytics", "yolo"):' in src
    assert "def load_ultralytics(" in src and "pip install ultralytics" in src
    assert "def register_detector_type(" in src


def test_entrypoints_fetch_enabled_models_only():
    for rel in ("fbxify/docker/worker-entrypoint.sh", "fbxify/docker/standalone-entrypoint.sh"):
        src = _read(rel)
        assert "python -m fbxify.model_catalog fetch" in src, rel
        assert "FBXIFY_SKIP_MODEL_FETCH" in src, rel


def test_readme_documents_custom_models():
    readme = _read("README.md")
    assert "models.json" in readme
    assert "FBXIFY_MODELS_JSON" in readme
    assert "/models" in readme
