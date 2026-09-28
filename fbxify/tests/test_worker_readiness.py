"""Readiness: a missing mesh must fail /health and refuse pose work."""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

import fbxify.api as api_module
from fbxify.api import app
from fbxify import checkpoint_download as assets


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    assets_dir = tmp_path / "cache" / "mhr_assets"
    assets_dir.mkdir(parents=True)
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("FBXIFY_MHR_RUNTIME_ASSETS", str(assets_dir))
    monkeypatch.setenv("FBXIFY_MOUNTS_DIR", str(tmp_path / "mounts"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(assets, "MHR_LOD1_MIN_BYTES", 8)
    monkeypatch.setattr(assets, "MHR_COMPANION_MIN_BYTES", 8)
    monkeypatch.setattr("fbxify.cli_common.checkpoints_available", lambda model: False)
    api_module._manager = None
    api_module._tracking_manager = None
    with api_module._jobs_lock:
        api_module._jobs.clear()
    return assets_dir


def _seed(assets_dir):
    for name in assets._required_assets():
        (assets_dir / name).write_bytes(b"x" * 16)


def test_missing_assets_fail_health_and_reject_pose_before_a_job(isolated):
    with TestClient(app, raise_server_exceptions=False) as client:
        api_module._manager = None
        assert client.get("/live").status_code == 200
        health = client.get("/health")
        assert health.status_code == 503
        assert health.json()["ready"] is False
        posed = client.post(
            "/jobs/pose",
            files={"input_file": ("frame.png", io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32))},
            data={"tracking_mode": "count", "num_people": "1"},
        )
        assert posed.status_code == 503
        assert "not ready" in posed.json()["detail"]
        assert api_module._jobs == {}


def test_ready_worker_accepts_pose(isolated):
    _seed(isolated)
    with TestClient(app, raise_server_exceptions=False) as client:
        api_module._manager = object()
        assert client.get("/health").json()["ready"] is True
        manager = MagicMock()
        manager.estimation_manager.estimate_all_frames.return_value = {}
        manager.estimation_manager.save_estimation_results.return_value = None
        with patch.object(api_module, "_get_manager", return_value=manager), patch.object(
            api_module, "_get_tracking_manager", return_value=MagicMock()
        ):
            posed = client.post(
                "/jobs/pose",
                files={"input_file": ("frame.png", io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32))},
                data={"tracking_mode": "count", "num_people": "1"},
            )
        assert posed.status_code == 200
        assert "job_id" in posed.json()


def test_reload_retries_assets_and_refuses_until_they_validate(isolated):
    with TestClient(app, raise_server_exceptions=False) as client:
        denied = client.post("/reload")
        assert denied.status_code == 503
        assert api_module._manager is None
        _seed(isolated)
        with patch.object(api_module, "_get_manager", return_value=MagicMock()), patch.object(
            api_module, "_get_tracking_manager", return_value=MagicMock()
        ):
            allowed = client.post("/reload")
        assert allowed.status_code == 200
        assert allowed.json()["status"] == "ok"
