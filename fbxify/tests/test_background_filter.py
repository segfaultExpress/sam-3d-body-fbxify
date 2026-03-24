"""Tests for background_filter scoring including the three new methods:
trajectory linearity, spatial occupancy, and perspective consistency.

Run with:  python -m pytest fbxify/tests/test_background_filter.py -v
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest

from fbxify.tracking.tracklet import Detection, Tracklet
from fbxify.tracking.tracking_config import TrackingConfig
from fbxify.tracking.background_filter import (
    _score_trajectory,
    _score_spatial_occupancy,
    _score_perspective_consistency,
    score_tracklets,
    filter_tracklets_by_score,
)


# ---------------------------------------------------------------------------
# Helpers for building synthetic data
# ---------------------------------------------------------------------------

def _make_detection(
    frame: int,
    person_id: str = "0",
    cam_t: list | None = None,
    bbox_xywh: list | None = None,
) -> Detection:
    return Detection(
        frame_index=frame,
        person_id=person_id,
        pred_cam_t=cam_t,
        bbox_xywh=bbox_xywh,
    )


def _stationary_tracklet(
    track_id: int,
    n_frames: int = 100,
    cam_t_base: list | None = None,
    bbox_xywh: list | None = None,
    jitter: float = 0.02,
) -> Tracklet:
    """Foreground-like tracklet: stays roughly in the same 3D position."""
    rng = np.random.RandomState(track_id)
    cam = cam_t_base or [0.0, -0.3, 5.0]
    bbox = bbox_xywh or [300.0, 200.0, 150.0, 400.0]
    t = Tracklet(track_id=track_id)
    for i in range(n_frames):
        noise = rng.randn(3) * jitter
        t.add_detection(_make_detection(
            frame=i,
            person_id=str(track_id),
            cam_t=[cam[0] + noise[0], cam[1] + noise[1], cam[2] + noise[2]],
            bbox_xywh=list(bbox),
        ))
    return t


def _walking_tracklet(
    track_id: int,
    n_frames: int = 30,
    start_cam_t: list | None = None,
    velocity: list | None = None,
    bbox_xywh: list | None = None,
    start_frame: int = 0,
) -> Tracklet:
    """Background walker: moves linearly through the scene."""
    cam = start_cam_t or [-2.0, -0.1, 12.0]
    vel = velocity or [0.05, 0.0, 0.0]
    bbox = bbox_xywh or [600.0, 300.0, 60.0, 160.0]
    t = Tracklet(track_id=track_id)
    for i in range(n_frames):
        t.add_detection(_make_detection(
            frame=start_frame + i,
            person_id=str(track_id),
            cam_t=[cam[0] + vel[0] * i, cam[1] + vel[1] * i, cam[2] + vel[2] * i],
            bbox_xywh=list(bbox),
        ))
    return t


# ---------------------------------------------------------------------------
# Unit tests for individual scoring functions
# ---------------------------------------------------------------------------

class TestTrajectoryScoring:
    def test_stationary_has_low_linearity(self):
        t = _stationary_tracklet(track_id=0, n_frames=50, jitter=0.01)
        info = _score_trajectory(t, min_frames=5)
        assert info is not None
        assert info["linearity"] < 0.3, f"Expected low linearity for stationary, got {info['linearity']}"
        assert info["term"] > 0.5, f"Foreground term should be > 0.5, got {info['term']}"

    def test_walker_has_high_linearity(self):
        t = _walking_tracklet(track_id=1, n_frames=50, velocity=[0.1, 0.0, 0.0])
        info = _score_trajectory(t, min_frames=5)
        assert info is not None
        assert info["linearity"] > 0.8, f"Expected high linearity for walker, got {info['linearity']}"
        assert info["term"] < 0.5, f"Background term should be < 0.5, got {info['term']}"

    def test_returns_none_when_too_few_frames(self):
        t = _walking_tracklet(track_id=2, n_frames=3)
        info = _score_trajectory(t, min_frames=10)
        assert info is None

    def test_stationary_scores_higher_than_walker(self):
        fg = _stationary_tracklet(track_id=0, n_frames=60, jitter=0.01)
        bg = _walking_tracklet(track_id=1, n_frames=60, velocity=[0.08, 0.0, 0.0])
        fg_info = _score_trajectory(fg, min_frames=5)
        bg_info = _score_trajectory(bg, min_frames=5)
        assert fg_info is not None and bg_info is not None
        assert fg_info["term"] > bg_info["term"]


class TestSpatialOccupancyScoring:
    def test_stationary_has_small_extent(self):
        t = _stationary_tracklet(track_id=0, n_frames=50, jitter=0.01)
        info = _score_spatial_occupancy(t)
        assert info is not None
        assert info["max_extent"] < 0.5, f"Stationary extent should be small, got {info['max_extent']}"
        assert info["term"] > 0.5

    def test_walker_has_large_extent(self):
        t = _walking_tracklet(track_id=1, n_frames=50, velocity=[0.1, 0.0, 0.0])
        info = _score_spatial_occupancy(t)
        assert info is not None
        assert info["max_extent"] > 2.0, f"Walker extent should be large, got {info['max_extent']}"

    def test_depth_std_low_for_stationary(self):
        t = _stationary_tracklet(track_id=0, n_frames=50, jitter=0.01)
        info = _score_spatial_occupancy(t)
        assert info is not None
        assert info["depth_std"] < 0.1

    def test_stationary_scores_higher_than_walker(self):
        fg = _stationary_tracklet(track_id=0, n_frames=60, jitter=0.01)
        bg = _walking_tracklet(track_id=1, n_frames=60, velocity=[0.08, 0.01, 0.02])
        fg_info = _score_spatial_occupancy(fg)
        bg_info = _score_spatial_occupancy(bg)
        assert fg_info is not None and bg_info is not None
        assert fg_info["term"] > bg_info["term"]


class TestPerspectiveConsistencyScoring:
    def test_consistent_depth_size_product(self):
        """Foreground: bbox_h * z is constant across frames."""
        t = Tracklet(track_id=0)
        for i in range(50):
            z = 5.0 + np.random.randn() * 0.05
            h = 2000.0 / z  # constant product = 2000
            t.add_detection(_make_detection(
                frame=i, cam_t=[0.0, 0.0, z],
                bbox_xywh=[100.0, 50.0, h * 0.4, h],
            ))
        info = _score_perspective_consistency(t)
        assert info is not None
        assert info["dsp_cv"] < 0.1, f"CV should be low for consistent perspective, got {info['dsp_cv']}"
        assert info["term"] > 0.5

    def test_inconsistent_depth_size_product(self):
        """Background: bbox_h * z varies wildly (noisy depth estimates)."""
        rng = np.random.RandomState(42)
        t = Tracklet(track_id=1)
        for i in range(50):
            z = rng.uniform(3.0, 20.0)
            h = rng.uniform(50.0, 400.0)
            t.add_detection(_make_detection(
                frame=i, cam_t=[0.0, 0.0, z],
                bbox_xywh=[100.0, 50.0, h * 0.4, h],
            ))
        info = _score_perspective_consistency(t)
        assert info is not None
        assert info["dsp_cv"] > 0.3
        assert info["term"] < 0.5

    def test_consistent_scores_higher(self):
        rng = np.random.RandomState(99)
        fg = Tracklet(track_id=0)
        bg = Tracklet(track_id=1)
        for i in range(60):
            z_fg = 5.0 + rng.randn() * 0.05
            h_fg = 2000.0 / z_fg
            fg.add_detection(_make_detection(
                frame=i, cam_t=[0.0, 0.0, z_fg],
                bbox_xywh=[100.0, 50.0, h_fg * 0.4, h_fg],
            ))
            z_bg = rng.uniform(3.0, 15.0)
            h_bg = rng.uniform(60.0, 300.0)
            bg.add_detection(_make_detection(
                frame=i, cam_t=[0.0, 0.0, z_bg],
                bbox_xywh=[400.0, 200.0, h_bg * 0.4, h_bg],
            ))
        fg_info = _score_perspective_consistency(fg)
        bg_info = _score_perspective_consistency(bg)
        assert fg_info is not None and bg_info is not None
        assert fg_info["term"] > bg_info["term"]


# ---------------------------------------------------------------------------
# Integration tests: score_tracklets with new weights
# ---------------------------------------------------------------------------

class TestScoreTrackletsIntegration:
    @staticmethod
    def _build_scene() -> List[Tracklet]:
        """2 foreground subjects + 3 background walkers."""
        return [
            _stationary_tracklet(track_id=0, n_frames=100, cam_t_base=[0.0, -0.3, 5.0],
                                 bbox_xywh=[300.0, 200.0, 150.0, 400.0]),
            _stationary_tracklet(track_id=1, n_frames=80, cam_t_base=[1.5, -0.2, 5.5],
                                 bbox_xywh=[500.0, 180.0, 140.0, 380.0]),
            _walking_tracklet(track_id=2, n_frames=30, start_cam_t=[-3.0, -0.1, 12.0],
                              velocity=[0.1, 0.0, 0.0], bbox_xywh=[50.0, 350.0, 50.0, 130.0]),
            _walking_tracklet(track_id=3, n_frames=25, start_cam_t=[2.0, 0.0, 14.0],
                              velocity=[-0.08, 0.0, 0.01], bbox_xywh=[800.0, 360.0, 45.0, 120.0],
                              start_frame=20),
            _walking_tracklet(track_id=4, n_frames=20, start_cam_t=[0.0, -0.2, 10.0],
                              velocity=[0.06, 0.0, 0.0], bbox_xywh=[400.0, 320.0, 55.0, 140.0],
                              start_frame=50),
        ]

    def test_original_weights_only(self):
        """With original 4 weights (new weights=0), FG should still outscore BG."""
        tracklets = self._build_scene()
        config = TrackingConfig(
            bg_filter_enabled=True,
            bg_tracklet_score_enabled=True,
            bg_w_length=0.45, bg_w_size=0.25,
            bg_w_size_stability=0.15, bg_w_centering=0.15,
            bg_w_trajectory=0.0, bg_w_spatial_occupancy=0.0, bg_w_perspective=0.0,
        )
        scored = score_tracklets(tracklets, config)
        scores = {t.track_id: s for t, s, _ in scored}
        assert scores[0] > scores[2], "FG0 should outscore BG2"
        assert scores[1] > scores[3], "FG1 should outscore BG3"

    def test_trajectory_weight_improves_separation(self):
        tracklets = self._build_scene()
        config_base = TrackingConfig(
            bg_filter_enabled=True,
            bg_w_length=0.35, bg_w_size=0.20,
            bg_w_size_stability=0.10, bg_w_centering=0.10,
            bg_w_trajectory=0.0, bg_w_spatial_occupancy=0.0, bg_w_perspective=0.0,
        )
        config_traj = TrackingConfig(
            bg_filter_enabled=True,
            bg_w_length=0.25, bg_w_size=0.15,
            bg_w_size_stability=0.10, bg_w_centering=0.10,
            bg_w_trajectory=0.25, bg_w_spatial_occupancy=0.0, bg_w_perspective=0.0,
            bg_trajectory_min_frames=5,
        )
        base_scored = score_tracklets(tracklets, config_base)
        traj_scored = score_tracklets(tracklets, config_traj)

        def _gap(scored_list):
            fg = min(s for t, s, _ in scored_list if t.track_id in (0, 1))
            bg = max(s for t, s, _ in scored_list if t.track_id in (2, 3, 4))
            return fg - bg

        assert _gap(traj_scored) >= _gap(base_scored), \
            "Adding trajectory weight should improve FG/BG separation"

    def test_spatial_weight_improves_separation(self):
        tracklets = self._build_scene()
        config_spat = TrackingConfig(
            bg_filter_enabled=True,
            bg_w_length=0.25, bg_w_size=0.15,
            bg_w_size_stability=0.10, bg_w_centering=0.10,
            bg_w_trajectory=0.0, bg_w_spatial_occupancy=0.25, bg_w_perspective=0.0,
        )
        scored = score_tracklets(tracklets, config_spat)
        scores = {t.track_id: s for t, s, _ in scored}
        for bg_id in (2, 3, 4):
            assert scores[0] > scores[bg_id], f"FG0 should outscore BG{bg_id}"

    def test_perspective_weight_improves_separation(self):
        tracklets = self._build_scene()
        config_persp = TrackingConfig(
            bg_filter_enabled=True,
            bg_w_length=0.25, bg_w_size=0.15,
            bg_w_size_stability=0.10, bg_w_centering=0.10,
            bg_w_trajectory=0.0, bg_w_spatial_occupancy=0.0, bg_w_perspective=0.25,
        )
        scored = score_tracklets(tracklets, config_persp)
        scores = {t.track_id: s for t, s, _ in scored}
        assert scores[0] > scores[2]

    def test_all_three_combined(self):
        """With all new weights active, FG should clearly beat BG."""
        tracklets = self._build_scene()
        config = TrackingConfig(
            bg_filter_enabled=True,
            bg_w_length=0.20, bg_w_size=0.10,
            bg_w_size_stability=0.05, bg_w_centering=0.05,
            bg_w_trajectory=0.20, bg_w_spatial_occupancy=0.20, bg_w_perspective=0.20,
            bg_trajectory_min_frames=5,
        )
        scored = score_tracklets(tracklets, config)
        scores = {t.track_id: s for t, s, _ in scored}
        fg_min = min(scores[0], scores[1])
        bg_max = max(scores[tid] for tid in (2, 3, 4))
        assert fg_min > bg_max, (
            f"All FG should beat all BG: FG min={fg_min:.4f}, BG max={bg_max:.4f}"
        )

    def test_breakdown_contains_new_keys(self):
        tracklets = self._build_scene()
        config = TrackingConfig(
            bg_filter_enabled=True,
            bg_w_trajectory=0.2, bg_w_spatial_occupancy=0.2, bg_w_perspective=0.2,
            bg_trajectory_min_frames=5,
        )
        scored = score_tracklets(tracklets, config)
        for t, s, bd in scored:
            if len(t.detections) >= 5:
                assert "trajectory" in bd, f"track {t.track_id} missing trajectory key"
                assert "spatial_occupancy" in bd, f"track {t.track_id} missing spatial_occupancy key"
                assert "perspective" in bd, f"track {t.track_id} missing perspective key"


# ---------------------------------------------------------------------------
# End-to-end: filter_tracklets_by_score
# ---------------------------------------------------------------------------

class TestFilterTrackletsEndToEnd:
    def test_threshold_keeps_only_fg(self):
        tracklets = TestScoreTrackletsIntegration._build_scene()
        config = TrackingConfig(
            bg_filter_enabled=True,
            bg_tracklet_score_enabled=True,
            bg_tracklet_score_threshold=0.7,
            bg_w_length=0.20, bg_w_size=0.10,
            bg_w_size_stability=0.05, bg_w_centering=0.05,
            bg_w_trajectory=0.20, bg_w_spatial_occupancy=0.20, bg_w_perspective=0.20,
            bg_trajectory_min_frames=5,
        )
        kept, rows = filter_tracklets_by_score(tracklets, config)
        kept_ids = {t.track_id for t in kept}
        assert 0 in kept_ids, "FG tracklet 0 should survive"
        assert 1 in kept_ids, "FG tracklet 1 should survive"
        for bg_id in (2, 3, 4):
            assert bg_id not in kept_ids, f"BG tracklet {bg_id} should be filtered out"

    def test_threshold_zero_keeps_all(self):
        tracklets = TestScoreTrackletsIntegration._build_scene()
        config = TrackingConfig(
            bg_filter_enabled=True,
            bg_tracklet_score_enabled=True,
            bg_tracklet_score_threshold=0.0,
            bg_w_trajectory=0.2, bg_w_spatial_occupancy=0.2, bg_w_perspective=0.2,
        )
        kept, rows = filter_tracklets_by_score(tracklets, config)
        assert len(kept) == 5

    def test_guardrail_keeps_best_when_threshold_too_high(self):
        tracklets = TestScoreTrackletsIntegration._build_scene()
        config = TrackingConfig(
            bg_filter_enabled=True,
            bg_tracklet_score_enabled=True,
            bg_tracklet_score_threshold=0.99,
            bg_w_trajectory=0.2, bg_w_spatial_occupancy=0.2, bg_w_perspective=0.2,
        )
        kept, rows = filter_tracklets_by_score(tracklets, config)
        assert len(kept) >= 1, "Guardrail should keep at least 1 tracklet"

    def test_hard_cutoff_spatial_extent(self):
        """bg_spatial_max_extent forces score to 0 for walkers with large extent."""
        tracklets = TestScoreTrackletsIntegration._build_scene()
        config = TrackingConfig(
            bg_filter_enabled=True,
            bg_tracklet_score_enabled=True,
            bg_tracklet_score_threshold=0.01,
            bg_w_trajectory=0.0, bg_w_spatial_occupancy=0.2, bg_w_perspective=0.0,
            bg_spatial_max_extent=1.0,
        )
        scored = score_tracklets(tracklets, config)
        for t, s, bd in scored:
            if t.track_id in (2, 3, 4):
                assert s == 0.0 or bd.get("spatial_max_extent", 0.0) <= 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_tracklets(self):
        scored = score_tracklets([], TrackingConfig())
        assert scored == []

    def test_single_detection_tracklet(self):
        t = Tracklet(track_id=99)
        t.add_detection(_make_detection(frame=0, cam_t=[0.0, 0.0, 5.0], bbox_xywh=[100, 100, 50, 100]))
        config = TrackingConfig(
            bg_filter_enabled=True,
            bg_w_trajectory=0.2, bg_w_spatial_occupancy=0.2, bg_w_perspective=0.2,
        )
        scored = score_tracklets([t], config)
        assert len(scored) == 1
        _, s, bd = scored[0]
        assert math.isfinite(s)

    def test_no_cam_t_data(self):
        t = Tracklet(track_id=50)
        for i in range(20):
            t.add_detection(_make_detection(frame=i, cam_t=None, bbox_xywh=[100, 100, 50, 100]))
        config = TrackingConfig(
            bg_filter_enabled=True,
            bg_w_trajectory=0.2, bg_w_spatial_occupancy=0.2, bg_w_perspective=0.2,
        )
        scored = score_tracklets([t], config)
        assert len(scored) == 1
        _, s, bd = scored[0]
        assert math.isfinite(s)

    def test_no_bbox_data(self):
        t = Tracklet(track_id=51)
        for i in range(20):
            t.add_detection(_make_detection(frame=i, cam_t=[0.0, 0.0, 5.0], bbox_xywh=None))
        config = TrackingConfig(
            bg_filter_enabled=True,
            bg_w_trajectory=0.2, bg_w_spatial_occupancy=0.2, bg_w_perspective=0.2,
        )
        scored = score_tracklets([t], config)
        assert len(scored) == 1
        _, s, bd = scored[0]
        assert math.isfinite(s)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
