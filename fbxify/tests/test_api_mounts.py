"""Tests for the /mount API and mount_id integration with job endpoints.

Run with:  python -m pytest fbxify/tests/test_api_mounts.py -v
"""
import io
import json
import os
import textwrap
import time
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import fbxify.api as api_module
from fbxify.api import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_ESTIMATION = json.dumps({
    "version": "1.0",
    "source": "test.mp4",
    "num_people": 1,
    "fps": 30.0,
    "frames": {
        "0": {"person_0": {"joints": [[0, 0, 0]] * 17}},
    },
}).encode()


@pytest.fixture(autouse=True)
def _isolate_mounts(tmp_path, monkeypatch):
    """Point FBXIFY_MOUNTS_DIR to a per-test temp dir and reset global state."""
    mounts_dir = str(tmp_path / "mounts")
    os.makedirs(mounts_dir, exist_ok=True)
    monkeypatch.setenv("FBXIFY_MOUNTS_DIR", mounts_dir)

    # Reset global registries between tests
    with api_module._mounts_lock:
        api_module._mounts.clear()
    with api_module._jobs_lock:
        api_module._jobs.clear()

    yield


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


def _mock_managers():
    """Return a pair of (manager_mock, tracking_manager_mock)."""
    manager = MagicMock()
    tracking_manager = MagicMock()

    manager.estimation_manager.load_estimation_results.return_value = (
        {"0": {"person_0": {"joints": [[0, 0, 0]] * 17}}},
        {"source": "test.mp4", "num_people": 1, "fps": 30.0},
    )
    manager.estimation_manager.save_estimation_results.return_value = None
    tracking_manager.run.return_value = {"frame_assignments": {}}
    return manager, tracking_manager


def _mount_sample_file(client: TestClient, content: bytes = SAMPLE_ESTIMATION, filename: str = "estimation.json") -> str:
    """Upload a file via /mount and return the mount_id."""
    resp = client.post("/mount", files={"file": (filename, io.BytesIO(content))})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "mount_id" in data
    return data["mount_id"]


# ---------------------------------------------------------------------------
# Mount lifecycle tests
# ---------------------------------------------------------------------------

class TestMountLifecycle:
    def test_mount_upload(self, client):
        resp = client.post("/mount", files={"file": ("test.json", io.BytesIO(b'{"hello":"world"}'))})
        assert resp.status_code == 200
        data = resp.json()
        assert "mount_id" in data
        assert data["filename"] == "test.json"
        assert data["size_bytes"] == len(b'{"hello":"world"}')

        mounts_dir = os.environ["FBXIFY_MOUNTS_DIR"]
        file_path = os.path.join(mounts_dir, data["mount_id"], "test.json")
        assert os.path.isfile(file_path)

    def test_mount_upload_no_file(self, client):
        resp = client.post("/mount")
        assert resp.status_code == 422

    def test_list_mounts_empty(self, client):
        resp = client.get("/mounts")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_mounts_after_upload(self, client):
        mount_id = _mount_sample_file(client, b"data", "my_file.bin")
        resp = client.get("/mounts")
        assert resp.status_code == 200
        entries = resp.json()
        assert len(entries) == 1
        assert entries[0]["mount_id"] == mount_id
        assert entries[0]["filename"] == "my_file.bin"
        assert entries[0]["size_bytes"] == 4

    def test_delete_mount(self, client):
        mount_id = _mount_sample_file(client, b"to_delete", "del.txt")
        mounts_dir = os.environ["FBXIFY_MOUNTS_DIR"]
        mount_dir = os.path.join(mounts_dir, mount_id)
        assert os.path.isdir(mount_dir)

        resp = client.delete(f"/mounts/{mount_id}")
        assert resp.status_code == 200
        assert resp.json()["mount_id"] == mount_id

        assert not os.path.isdir(mount_dir)
        assert client.get("/mounts").json() == []

    def test_delete_mount_not_found(self, client):
        resp = client.delete("/mounts/nonexistent_id")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Mount usage in job endpoints (mocked processing)
# ---------------------------------------------------------------------------

def _patch_rerun_tracking_deps():
    """Context manager that patches the gradio import inside _run_rerun_tracking_job."""
    import sys
    from unittest.mock import MagicMock as _MagicMock

    fake_tracking_section = _MagicMock()
    from fbxify.tracking.tracking_config import TrackingConfig
    fake_tracking_section.build_tracking_config_from_gui.return_value = TrackingConfig()

    return patch.dict(sys.modules, {
        "fbxify.gradio_ui.tracking_section": fake_tracking_section,
    })


class TestRerunTrackingWithMount:
    @patch.object(api_module, "_get_tracking_manager")
    @patch.object(api_module, "_get_manager")
    def test_rerun_tracking_with_mount_id(self, mock_mgr, mock_tmgr, client):
        manager, tracking_manager = _mock_managers()
        mock_mgr.return_value = manager
        mock_tmgr.return_value = tracking_manager

        mount_id = _mount_sample_file(client)

        with _patch_rerun_tracking_deps():
            resp = client.post("/jobs/rerun_tracking", data={
                "estimation_mount_id": mount_id,
                "step_through": False,
                "debug_start_frame": 0,
            })
            assert resp.status_code == 200
            job_id = resp.json()["job_id"]

            for _ in range(50):
                status = client.get(f"/jobs/{job_id}").json()
                if status["status"] in ("completed", "failed"):
                    break
                time.sleep(0.1)

        assert status["status"] == "completed", status.get("error")
        assert len(status["output_files"]) == 1

    def test_rerun_tracking_with_upload_still_works(self, client):
        """Backward compatibility: uploading estimation_file directly still works."""
        with patch.object(api_module, "_get_manager") as mock_mgr, \
             patch.object(api_module, "_get_tracking_manager") as mock_tmgr, \
             _patch_rerun_tracking_deps():
            manager, tracking_manager = _mock_managers()
            mock_mgr.return_value = manager
            mock_tmgr.return_value = tracking_manager

            resp = client.post("/jobs/rerun_tracking", files={
                "estimation_file": ("estimation.json", io.BytesIO(SAMPLE_ESTIMATION)),
            }, data={
                "step_through": "false",
                "debug_start_frame": "0",
            })
            assert resp.status_code == 200
            job_id = resp.json()["job_id"]

            for _ in range(50):
                status = client.get(f"/jobs/{job_id}").json()
                if status["status"] in ("completed", "failed"):
                    break
                time.sleep(0.1)

            assert status["status"] == "completed", status.get("error")

    def test_rerun_tracking_neither_provided(self, client):
        resp = client.post("/jobs/rerun_tracking", data={
            "step_through": "false",
            "debug_start_frame": "0",
        })
        assert resp.status_code == 422

    @patch.object(api_module, "_get_tracking_manager")
    @patch.object(api_module, "_get_manager")
    def test_rerun_tracking_invalid_mount_id(self, mock_mgr, mock_tmgr, client):
        manager, tracking_manager = _mock_managers()
        mock_mgr.return_value = manager
        mock_tmgr.return_value = tracking_manager

        resp = client.post("/jobs/rerun_tracking", data={
            "estimation_mount_id": "does_not_exist",
            "step_through": "false",
            "debug_start_frame": "0",
        })
        assert resp.status_code == 404


class TestFbxWithMount:
    @patch.object(api_module, "_get_manager")
    def test_fbx_with_mount_id(self, mock_mgr, client):
        manager = MagicMock()

        class FakeResult:
            profile_name = "mhr"
            joint_to_bone_mappings = []
            root_motions = []
            frame_paths = []
            fps = 30.0
            mesh_obj_paths = None
            height_offset = 0
            metadata_extras = {}
            refinement_logs = None

        manager.process_from_estimation_json.return_value = FakeResult()
        manager.export_fbx_files.return_value = []
        mock_mgr.return_value = manager

        mount_id = _mount_sample_file(client)

        resp = client.post("/jobs/fbx", data={
            "pose_json_mount_id": mount_id,
            "profile_name": "mhr",
        })
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        for _ in range(50):
            status = client.get(f"/jobs/{job_id}").json()
            if status["status"] in ("completed", "failed"):
                break
            time.sleep(0.1)

        assert status["status"] == "completed", status.get("error")

    @patch.object(api_module, "_get_manager")
    def test_fbx_with_upload_still_works(self, mock_mgr, client):
        manager = MagicMock()

        class FakeResult:
            profile_name = "mhr"
            joint_to_bone_mappings = []
            root_motions = []
            frame_paths = []
            fps = 30.0
            mesh_obj_paths = None
            height_offset = 0
            metadata_extras = {}
            refinement_logs = None

        manager.process_from_estimation_json.return_value = FakeResult()
        manager.export_fbx_files.return_value = []
        mock_mgr.return_value = manager

        resp = client.post("/jobs/fbx", files={
            "pose_json_file": ("estimation.json", io.BytesIO(SAMPLE_ESTIMATION)),
        }, data={
            "profile_name": "mhr",
        })
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        for _ in range(50):
            status = client.get(f"/jobs/{job_id}").json()
            if status["status"] in ("completed", "failed"):
                break
            time.sleep(0.1)

        assert status["status"] == "completed", status.get("error")


class TestPoseWithMount:
    @patch.object(api_module, "_get_tracking_manager")
    @patch.object(api_module, "_get_manager")
    def test_pose_with_mount_id(self, mock_mgr, mock_tmgr, client, tmp_path):
        manager = MagicMock()
        tracking_manager = MagicMock()
        mock_mgr.return_value = manager
        mock_tmgr.return_value = tracking_manager

        manager.prepare_video.return_value = ([], None, 30.0)
        manager.estimation_manager.estimate_all_frames.return_value = {}
        manager.estimation_manager.save_estimation_results.return_value = None

        img_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        mount_id = _mount_sample_file(client, img_bytes, "frame.png")

        resp = client.post("/jobs/pose", data={
            "input_mount_id": mount_id,
            "tracking_mode": "count",
            "num_people": "1",
        })
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        for _ in range(50):
            status = client.get(f"/jobs/{job_id}").json()
            if status["status"] in ("completed", "failed"):
                break
            time.sleep(0.1)

        assert status["status"] == "completed", status.get("error")

    @patch.object(api_module, "_get_tracking_manager")
    @patch.object(api_module, "_get_manager")
    def test_pose_with_upload_still_works(self, mock_mgr, mock_tmgr, client):
        manager = MagicMock()
        tracking_manager = MagicMock()
        mock_mgr.return_value = manager
        mock_tmgr.return_value = tracking_manager

        manager.estimation_manager.estimate_all_frames.return_value = {}
        manager.estimation_manager.save_estimation_results.return_value = None

        img_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        resp = client.post("/jobs/pose", files={
            "input_file": ("frame.png", io.BytesIO(img_bytes)),
        }, data={
            "tracking_mode": "count",
            "num_people": "1",
        })
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Cleanup integration tests
# ---------------------------------------------------------------------------

class TestCleanupIntegration:
    def test_cleanup_default_does_not_remove_mounts(self, client):
        mount_id = _mount_sample_file(client)

        resp = client.post("/cleanup")
        assert resp.status_code == 200
        data = resp.json()
        assert data["unmounted"] == 0

        mounts = client.get("/mounts").json()
        assert len(mounts) == 1
        assert mounts[0]["mount_id"] == mount_id

    def test_cleanup_unmount_all(self, client):
        _mount_sample_file(client, b"file1", "a.json")
        _mount_sample_file(client, b"file2", "b.json")
        assert len(client.get("/mounts").json()) == 2

        resp = client.post("/cleanup", params={"unmount_all": "true"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["unmounted"] == 2
        assert client.get("/mounts").json() == []

        mounts_dir = os.environ["FBXIFY_MOUNTS_DIR"]
        remaining = [d for d in os.listdir(mounts_dir) if os.path.isdir(os.path.join(mounts_dir, d))]
        assert remaining == []

    def test_storage_includes_mount_info(self, client):
        _mount_sample_file(client, b"x" * 1024, "data.bin")

        resp = client.get("/storage")
        assert resp.status_code == 200
        data = resp.json()
        assert "mounts_count" in data
        assert data["mounts_count"] == 1
        assert "mounts_total_gb" in data
        assert "mounts_dir" in data


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_mount_survives_cleanup(self, client):
        """Mounts must persist even when normal cleanup removes all job dirs."""
        mount_id = _mount_sample_file(client, b"important", "keep_me.json")

        # Simulate a completed job by inserting one directly
        import tempfile as _tempfile
        job_dir = _tempfile.mkdtemp(prefix="fbxify_job_")
        with api_module._jobs_lock:
            api_module._jobs["fake_job"] = {
                "status": "completed",
                "output_dir": job_dir,
                "output_files": [],
            }

        resp = client.post("/cleanup")
        assert resp.status_code == 200
        assert resp.json()["purged_jobs"] >= 1

        mounts = client.get("/mounts").json()
        assert len(mounts) == 1
        assert mounts[0]["mount_id"] == mount_id

    def test_mount_large_filename_preserved(self, client):
        filename = "my-complex_estimation (v2) [final].json"
        mount_id = _mount_sample_file(client, b"data", filename)

        mounts = client.get("/mounts").json()
        assert mounts[0]["filename"] == filename

        mounts_dir = os.environ["FBXIFY_MOUNTS_DIR"]
        on_disk = os.path.join(mounts_dir, mount_id, filename)
        assert os.path.isfile(on_disk)
