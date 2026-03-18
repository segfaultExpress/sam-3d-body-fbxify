"""Tests for shape matching: _track_shape_average bidirectional window,
shape_avg_max_frames=0 (full tracklet), and the merge-path fix.

Run with:  python -m pytest fbxify/tests/test_shape_matching.py -v
"""
from __future__ import annotations

from typing import List

import numpy as np
import pytest

from fbxify.tracking.tracklet import Detection, Tracklet
from fbxify.tracking.tracking_config import TrackingConfig
from fbxify.tracking.mhr_tracker import MHRTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _det(frame: int, shape: List[float], cam_t: list | None = None) -> Detection:
    return Detection(
        frame_index=frame,
        person_id="0",
        shape_params=shape,
        pred_cam_t=cam_t or [0.0, 0.0, 5.0],
        pred_global_rots=[[0.0] * 6] * 24,
        bbox_xywh=[100.0, 100.0, 80.0, 200.0],
    )


def _asymmetric_tracklet(track_id: int, n: int = 60) -> Tracklet:
    """Tracklet whose first-half shape differs from second-half shape.

    First 30 frames: shape ~[1, 1, 1, ...]
    Last 30 frames:  shape ~[2, 2, 2, ...]
    """
    rng = np.random.RandomState(track_id)
    t = Tracklet(track_id=track_id)
    dim = 10
    for i in range(n):
        base = 1.0 if i < n // 2 else 2.0
        shape = (np.ones(dim) * base + rng.randn(dim) * 0.02).tolist()
        t.add_detection(_det(frame=i, shape=shape))
    return t


def _constant_tracklet(track_id: int, shape_val: float, n: int = 40,
                        jitter: float = 0.01, start_frame: int = 0,
                        cam_t: list | None = None) -> Tracklet:
    rng = np.random.RandomState(track_id + 1000)
    t = Tracklet(track_id=track_id)
    dim = 10
    for i in range(n):
        shape = (np.ones(dim) * shape_val + rng.randn(dim) * jitter).tolist()
        t.add_detection(_det(
            frame=start_frame + i, shape=shape,
            cam_t=cam_t or [0.0, 0.0, 5.0],
        ))
    return t


# ---------------------------------------------------------------------------
# Tests: _track_shape_average
# ---------------------------------------------------------------------------

class TestTrackShapeAverage:
    tracker = MHRTracker()
    cfg = TrackingConfig(
        shape_maturity_frames=5,
        shape_avg_max_frames=20,
        shape_spike_mad_factor=3.0,
        shape_avg_weight_decay=0.98,
    )

    def test_from_end_returns_recent_shape(self):
        t = _asymmetric_tracklet(0, n=60)
        avg = self.tracker._track_shape_average(t, self.cfg, from_start=False)
        assert avg is not None
        assert np.mean(avg) > 1.5, "Last-N avg should reflect second-half shape (~2.0)"

    def test_from_start_returns_early_shape(self):
        t = _asymmetric_tracklet(0, n=60)
        avg = self.tracker._track_shape_average(t, self.cfg, from_start=True)
        assert avg is not None
        assert np.mean(avg) < 1.5, "First-N avg should reflect first-half shape (~1.0)"

    def test_from_start_and_from_end_differ(self):
        t = _asymmetric_tracklet(0, n=60)
        avg_start = self.tracker._track_shape_average(t, self.cfg, from_start=True)
        avg_end = self.tracker._track_shape_average(t, self.cfg, from_start=False)
        assert avg_start is not None and avg_end is not None
        diff = np.mean(np.abs(np.array(avg_start) - np.array(avg_end)))
        assert diff > 0.5, f"Start and end averages should differ significantly, got diff={diff:.4f}"

    def test_max_frames_zero_uses_all(self):
        t = _constant_tracklet(0, shape_val=3.0, n=100)
        cfg_full = TrackingConfig(
            shape_maturity_frames=5,
            shape_avg_max_frames=0,
            shape_spike_mad_factor=3.0,
            shape_avg_weight_decay=0.98,
        )
        avg = self.tracker._track_shape_average(t, cfg_full, from_start=False)
        assert avg is not None
        assert abs(np.mean(avg) - 3.0) < 0.1, f"Full-tracklet avg should be ~3.0, got {np.mean(avg):.4f}"

    def test_max_frames_override(self):
        # 50 frames of shape ~1.0, then 10 frames of shape ~2.0
        rng = np.random.RandomState(7)
        t = Tracklet(track_id=0)
        dim = 10
        for i in range(50):
            shape = (np.ones(dim) * 1.0 + rng.randn(dim) * 0.01).tolist()
            t.add_detection(_det(frame=i, shape=shape))
        for i in range(50, 60):
            shape = (np.ones(dim) * 2.0 + rng.randn(dim) * 0.01).tolist()
            t.add_detection(_det(frame=i, shape=shape))

        avg_small = self.tracker._track_shape_average(
            t, self.cfg, from_start=False, max_frames_override=10,
        )
        avg_large = self.tracker._track_shape_average(
            t, self.cfg, from_start=False, max_frames_override=50,
        )
        assert avg_small is not None and avg_large is not None
        assert np.mean(avg_small) > np.mean(avg_large), \
            "Smaller window from end should be more biased toward recent (higher) values"

    def test_min_frames_override(self):
        t = _constant_tracklet(0, shape_val=1.0, n=4)
        avg_default = self.tracker._track_shape_average(t, self.cfg)
        assert avg_default is None, "Default maturity=5 should reject 4-frame tracklet"

        avg_low = self.tracker._track_shape_average(
            t, self.cfg, min_frames_override=2,
        )
        assert avg_low is not None, "min_frames_override=2 should accept 4-frame tracklet"

    def test_returns_none_for_empty_tracklet(self):
        t = Tracklet(track_id=99)
        avg = self.tracker._track_shape_average(t, self.cfg)
        assert avg is None


# ---------------------------------------------------------------------------
# Tests: _merge_similarity
# ---------------------------------------------------------------------------

class TestMergeSimilarity:
    tracker = MHRTracker()

    def test_merge_same_person_high_score(self):
        """Two tracklets of the same person (same shape) should have high merge score."""
        base = _constant_tracklet(0, shape_val=1.5, n=40, start_frame=0,
                                  cam_t=[0.0, 0.0, 5.0])
        candidate = _constant_tracklet(1, shape_val=1.5, n=40, start_frame=50,
                                       cam_t=[0.0, 0.0, 5.0])
        cfg = TrackingConfig(
            shape_maturity_frames=5,
            shape_avg_max_frames=20,
            merge_shape_window_base=0, merge_shape_window_candidate=0,
            min_similarity=0.3,
        )
        score = self.tracker._merge_similarity(base, candidate, cfg)
        assert score > 0.5, f"Same-person merge should score high, got {score:.4f}"

    def test_merge_different_persons_low_score(self):
        """Two tracklets with very different shapes and positions should score low."""
        base = _constant_tracklet(0, shape_val=1.0, n=40, start_frame=0,
                                  cam_t=[0.0, 0.0, 5.0])
        candidate = _constant_tracklet(1, shape_val=5.0, n=40, start_frame=50,
                                       cam_t=[3.0, 0.0, 12.0])
        cfg = TrackingConfig(
            shape_maturity_frames=5,
            shape_avg_max_frames=20,
            merge_shape_window_base=0, merge_shape_window_candidate=0,
        )
        score = self.tracker._merge_similarity(base, candidate, cfg)
        assert score < 0.3, f"Different-person merge should score low, got {score:.4f}"

    def test_merge_with_noisy_first_frame_improved(self):
        """The old merge used candidate.detections[0] raw. The new one averages,
        so a noisy first frame should no longer tank the score."""
        rng = np.random.RandomState(42)
        dim = 10
        target_shape = np.ones(dim) * 2.0

        base = _constant_tracklet(0, shape_val=2.0, n=40, start_frame=0)

        candidate = Tracklet(track_id=1)
        # First frame: very noisy
        noisy_shape = (target_shape + rng.randn(dim) * 2.0).tolist()
        candidate.add_detection(_det(frame=50, shape=noisy_shape))
        # Remaining frames: clean
        for i in range(1, 40):
            clean_shape = (target_shape + rng.randn(dim) * 0.01).tolist()
            candidate.add_detection(_det(frame=50 + i, shape=clean_shape))

        cfg = TrackingConfig(
            shape_maturity_frames=5,
            shape_avg_max_frames=20,
            merge_shape_window_base=0, merge_shape_window_candidate=0,
        )

        new_score = self.tracker._merge_similarity(base, candidate, cfg)

        # Compare against old-style: use base avg vs candidate first detection raw
        old_base_shape = self.tracker._track_shape_average(base, cfg)
        from fbxify.tracking.metrics import mean_abs_diff, similarity_from_distance
        old_dist = mean_abs_diff(old_base_shape, noisy_shape)
        old_shape_sim = similarity_from_distance(old_dist, cfg.shape_distance_threshold)

        assert new_score > old_shape_sim, (
            f"Averaged merge ({new_score:.4f}) should beat single-frame ({old_shape_sim:.4f})"
        )

    def test_merge_shape_window_config(self):
        """Narrow merge windows should restrict to boundary frames."""
        # Build base: 80 frames of shape ~1.0, then 10 frames of shape ~2.0
        rng = np.random.RandomState(55)
        dim = 10
        base = Tracklet(track_id=0)
        for i in range(80):
            shape = (np.ones(dim) * 1.0 + rng.randn(dim) * 0.01).tolist()
            base.add_detection(_det(frame=i, shape=shape))
        for i in range(80, 90):
            shape = (np.ones(dim) * 2.0 + rng.randn(dim) * 0.01).tolist()
            base.add_detection(_det(frame=i, shape=shape))

        # Candidate matches base's tail (shape ~2.0)
        candidate = _constant_tracklet(1, shape_val=2.0, n=40, start_frame=95)

        cfg_narrow = TrackingConfig(
            shape_maturity_frames=3,
            shape_avg_max_frames=50,
            merge_shape_window_base=10, merge_shape_window_candidate=10,
        )
        cfg_wide = TrackingConfig(
            shape_maturity_frames=3,
            shape_avg_max_frames=50,
            merge_shape_window_base=0, merge_shape_window_candidate=0,
        )
        score_narrow = self.tracker._merge_similarity(base, candidate, cfg_narrow)
        score_wide = self.tracker._merge_similarity(base, candidate, cfg_wide)
        # Narrow window (last-10 of base = 2.0) matches candidate (2.0) better
        # than wide window (last-50 of base is mostly ~1.0)
        assert score_narrow > score_wide, (
            f"Narrow merge window ({score_narrow:.4f}) should score > wide ({score_wide:.4f})"
        )

    def test_asymmetric_merge_windows(self):
        """Base and candidate can use different window sizes."""
        rng = np.random.RandomState(77)
        dim = 10
        # Base: 80 frames shape ~1.0 then 10 frames shape ~2.0
        base = Tracklet(track_id=0)
        for i in range(80):
            shape = (np.ones(dim) * 1.0 + rng.randn(dim) * 0.01).tolist()
            base.add_detection(_det(frame=i, shape=shape))
        for i in range(80, 90):
            shape = (np.ones(dim) * 2.0 + rng.randn(dim) * 0.01).tolist()
            base.add_detection(_det(frame=i, shape=shape))

        # Candidate: shape ~2.0
        candidate = _constant_tracklet(1, shape_val=2.0, n=40, start_frame=95)

        # Narrow base window (last-10 ≈ 2.0) + wide candidate (all 40 ≈ 2.0)
        cfg_narrow_base = TrackingConfig(
            shape_maturity_frames=3, shape_avg_max_frames=50,
            merge_shape_window_base=10, merge_shape_window_candidate=0,
        )
        # Wide base window (last-50 includes ~1.0) + narrow candidate (first-10 ≈ 2.0)
        cfg_wide_base = TrackingConfig(
            shape_maturity_frames=3, shape_avg_max_frames=50,
            merge_shape_window_base=0, merge_shape_window_candidate=10,
        )
        score_narrow_base = self.tracker._merge_similarity(base, candidate, cfg_narrow_base)
        score_wide_base = self.tracker._merge_similarity(base, candidate, cfg_wide_base)
        # Narrow base sees only the matching tail; wide base dilutes with mismatching head
        assert score_narrow_base > score_wide_base, (
            f"Narrow base ({score_narrow_base:.4f}) should beat wide base ({score_wide_base:.4f})"
        )


# ---------------------------------------------------------------------------
# Tests: _merge_tracklets integration
# ---------------------------------------------------------------------------

class TestMergeTrackletsIntegration:
    tracker = MHRTracker()

    def test_same_person_tracklets_merge(self):
        """Two tracklets of the same person with a gap should merge."""
        t1 = _constant_tracklet(0, shape_val=1.5, n=40, start_frame=0,
                                cam_t=[0.0, 0.0, 5.0])
        t2 = _constant_tracklet(1, shape_val=1.5, n=40, start_frame=45,
                                cam_t=[0.0, 0.0, 5.0])
        cfg = TrackingConfig(
            shape_maturity_frames=5,
            shape_avg_max_frames=20,
            merge_max_gap_frames=12,
            min_similarity=0.3,
            merge_shape_window_base=0, merge_shape_window_candidate=0,
        )
        result = self.tracker._merge_tracklets([t1, t2], cfg)
        assert len(result) == 1, f"Should merge into 1 tracklet, got {len(result)}"
        assert len(result[0].detections) == 80

    def test_different_persons_dont_merge(self):
        """Two tracklets of different persons should not merge."""
        t1 = _constant_tracklet(0, shape_val=1.0, n=40, start_frame=0,
                                cam_t=[0.0, 0.0, 5.0])
        t2 = _constant_tracklet(1, shape_val=5.0, n=40, start_frame=45,
                                cam_t=[3.0, 0.0, 12.0])
        cfg = TrackingConfig(
            shape_maturity_frames=5,
            shape_avg_max_frames=20,
            merge_max_gap_frames=12,
            min_similarity=0.4,
            merge_shape_window_base=0, merge_shape_window_candidate=0,
        )
        result = self.tracker._merge_tracklets([t1, t2], cfg)
        assert len(result) == 2, f"Should stay as 2 tracklets, got {len(result)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
