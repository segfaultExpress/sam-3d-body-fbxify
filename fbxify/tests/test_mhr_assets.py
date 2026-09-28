"""MHR asset download, archive checks, and runtime-path repair.

These tests never touch the working cache and never download the release archive.
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pytest

from fbxify import checkpoint_download as assets


def _relax_sizes(monkeypatch) -> None:
    monkeypatch.setattr(assets, "MHR_LOD1_MIN_BYTES", 8)
    monkeypatch.setattr(assets, "MHR_COMPANION_MIN_BYTES", 8)


def _seed(directory: Path, monkeypatch) -> Path:
    _relax_sizes(monkeypatch)
    directory.mkdir(parents=True, exist_ok=True)
    for name in assets._required_assets():
        (directory / name).write_bytes(b"x" * 16)
    return directory


def _symlink(src: Path, dst: Path) -> None:
    try:
        os.symlink(src, dst, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")


def test_release_pin_matches_verified_v101_archive():
    assert assets.MHR_ASSETS_URL.endswith("/v1.0.1/assets.zip")
    assert assets.MHR_ASSETS_SHA256 == "e4f4f205cd87c0fa106577ba1de4fc763e4eb197c924461d2ef7e6944e9d6b94"
    assert assets.MHR_ASSETS_SIZE == 198_943_157
    assert assets.MHR_LOD1_SHA256 == "d66fbca815bcde6532f728f1f63071003c5d43ff44f56e263a0807baec1ae055"
    assert assets.MHR_LOD1_EXPECTED_BYTES == 7_884_560


def test_warm_cache_is_not_downloaded_or_rewritten(tmp_path, monkeypatch):
    dest = _seed(tmp_path / "mhr_assets", monkeypatch)
    before = (dest / "lod1.fbx").read_bytes()

    def _forbidden(url, path):
        raise AssertionError(f"download attempted: {url}")

    monkeypatch.setattr(assets, "urlretrieve", _forbidden)
    assert assets.download_mhr_assets_if_missing(str(dest)) is True
    assert (dest / "lod1.fbx").read_bytes() == before


def test_failed_download_keeps_existing_files(tmp_path, monkeypatch):
    dest = tmp_path / "mhr_assets"
    dest.mkdir()
    (dest / "keep.txt").write_text("safe", encoding="utf-8")
    monkeypatch.setenv("FBXIFY_ALLOW_ASSET_DOWNLOAD", "1")

    def _boom(url, path):
        raise OSError("HTTP Error 404: Not Found")

    monkeypatch.setattr(assets, "urlretrieve", _boom)
    assert assets.download_mhr_assets_if_missing(str(dest)) is False
    assert (dest / "keep.txt").read_text(encoding="utf-8") == "safe"
    assert not (dest / "lod1.fbx").exists()


def test_checksum_mismatch_does_not_publish(tmp_path, monkeypatch):
    dest = tmp_path / "mhr_assets"
    dest.mkdir()
    (dest / "keep.txt").write_text("safe", encoding="utf-8")
    monkeypatch.setenv("FBXIFY_ALLOW_ASSET_DOWNLOAD", "1")

    def _write_bad(url, path):
        Path(path).write_bytes(b"not a zip")

    monkeypatch.setattr(assets, "urlretrieve", _write_bad)
    assert assets.download_mhr_assets_if_missing(str(dest)) is False
    assert (dest / "keep.txt").read_text(encoding="utf-8") == "safe"
    assert list(dest.iterdir()) == [dest / "keep.txt"]


def test_rejects_zip_slip_and_writes_nothing(tmp_path, monkeypatch):
    _relax_sizes(monkeypatch)
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("assets/../../evil.txt", "nope")
        for name in assets._required_assets():
            handle.writestr(f"assets/{name}", b"x" * 16)
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(ValueError, match="unsafe path"):
        assets.extract_mhr_archive(str(archive), str(dest))
    assert not (tmp_path / "evil.txt").exists()
    assert list(dest.iterdir()) == []


def test_extracts_assets_prefix(tmp_path, monkeypatch):
    _relax_sizes(monkeypatch)
    archive = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("assets/", "")
        for name in assets._required_assets():
            handle.writestr(f"assets/{name}", b"x" * 16)
        handle.writestr("assets/lod0.fbx", b"y" * 16)
    dest = tmp_path / "out"
    assets.extract_mhr_archive(str(archive), str(dest))
    assert assets._assets_valid(str(dest))
    assert (dest / "lod0.fbx").read_bytes() == b"y" * 16
    assert not (dest / "assets").exists()


def test_stale_symlink_is_replaced_without_deleting_its_target(tmp_path, monkeypatch):
    cache = _seed(tmp_path / "cache", monkeypatch)
    old = tmp_path / "old-target"
    old.mkdir()
    (old / "sentinel.txt").write_text("keep", encoding="utf-8")
    runtime = tmp_path / "assets"
    _symlink(old, runtime)
    assets.ensure_runtime_assets_link(str(runtime), str(cache))
    assert os.path.realpath(runtime) == os.path.realpath(cache)
    assert (old / "sentinel.txt").read_text(encoding="utf-8") == "keep"
    assert assets._assets_valid(str(runtime))


def test_mounted_runtime_is_filled_without_being_deleted(tmp_path, monkeypatch):
    cache = _seed(tmp_path / "cache", monkeypatch)
    runtime = tmp_path / "mounted"
    runtime.mkdir()
    (runtime / "sentinel.txt").write_text("keep", encoding="utf-8")
    monkeypatch.setattr(
        assets,
        "_is_mounted",
        lambda path: os.path.abspath(path) == os.path.abspath(runtime),
    )
    monkeypatch.setenv("FBXIFY_MHR_RUNTIME_ASSETS", str(runtime))
    assert assets.ensure_mhr_assets(str(cache)) is True
    assert runtime.is_dir()
    assert not runtime.is_symlink()
    assert (runtime / "sentinel.txt").read_text(encoding="utf-8") == "keep"
    assert (runtime / "lod1.fbx").is_file()
    assert (cache / "lod1.fbx").is_file()


def test_entrypoints_check_stale_links_and_mounts():
    root = Path(__file__).resolve().parents[2]
    for rel in ("fbxify/docker/worker-entrypoint.sh", "fbxify/docker/standalone-entrypoint.sh"):
        text = (root / rel).read_text(encoding="utf-8")
        assert "link_mhr_runtime_assets" in text
        assert "replacing stale symlink" in text
        assert "is mounted; leaving it unchanged" in text
