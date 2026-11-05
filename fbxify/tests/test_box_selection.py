"""Confidence-based tracking modes: selector semantics and plumbing.

Does not import SAM 3D Body or touch the network. The smoke test at the bottom is opt-in and
needs a running worker (see ``_smoke_env``).

Run: python -m pytest fbxify/tests/test_box_selection.py -v
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import pytest

from fbxify import box_selection as bs

ROOT = Path(__file__).resolve().parents[2]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _box(pid, h, conf, x=0.0, y=0.0, w=10.0):
    return (pid, x, y, w, h, conf, -1, -1, -1)


# ----------------------------------------------------------------------------- box_confidence


@pytest.mark.parametrize(
    "raw, expected",
    [
        (0.9, 0.9),
        ("0.75", 0.75),
        (1, 1.0),
        (0, 0.0),
        ("-1", None),  # MOT "not provided"
        (-1.0, None),
        ("", None),
        (None, None),
        ("nan", None),
        ("inf", None),
        ("abc", None),
    ],
)
def test_box_confidence_parses_mot_conf_column(raw, expected):
    assert bs.box_confidence(_box(0, 10, raw)) == expected


def test_box_confidence_short_tuple_is_none():
    assert bs.box_confidence((0, 1, 2, 3, 4)) is None
    assert bs.box_confidence(None) is None


# ----------------------------------------------------------------------------- select_boxes


def test_non_selecting_modes_keep_every_box():
    boxes = [_box(0, 10, 0.1), _box(1, 20, 0.9)]
    for mode in ("bbox", "inference", "inference_bbox", None, "something_else"):
        assert bs.select_boxes(boxes, mode, num_people=1) == boxes


def test_count_keeps_n_tallest_in_height_order():
    boxes = [_box(0, 10, 0.9), _box(1, 30, 0.1), _box(2, 20, 0.5)]
    out = bs.select_boxes(boxes, bs.MODE_COUNT, num_people=2)
    assert [b[0] for b in out] == [1, 2]


def test_count_without_a_cap_keeps_all():
    boxes = [_box(0, 10, 0.9), _box(1, 30, 0.1)]
    assert bs.select_boxes(boxes, bs.MODE_COUNT, num_people=0) == boxes
    assert bs.select_boxes(boxes, bs.MODE_COUNT, num_people=None) == boxes


def test_confidence_top_ranks_by_confidence_not_height():
    boxes = [_box(0, 50, 0.2), _box(1, 10, 0.95), _box(2, 30, 0.6)]
    out = bs.select_boxes(boxes, bs.MODE_CONFIDENCE_TOP, num_people=2)
    assert [b[0] for b in out] == [1, 2]


def test_confidence_top_puts_unscored_boxes_last():
    boxes = [_box(0, 50, "-1"), _box(1, 10, 0.3), _box(2, 30, None)]
    out = bs.select_boxes(boxes, bs.MODE_CONFIDENCE_TOP, num_people=2)
    assert out[0][0] == 1
    # the two unscored boxes tie on confidence; the taller one wins the second slot
    assert out[1][0] == 0


def test_confidence_top_without_any_confidence_falls_back_to_closest():
    boxes = [_box(0, 10, -1), _box(1, 30, -1), _box(2, 20, "")]
    out = bs.select_boxes(boxes, bs.MODE_CONFIDENCE_TOP, num_people=2)
    assert [b[0] for b in out] == [1, 2]


def test_confidence_filter_keeps_at_or_above_threshold_and_drops_unscored():
    boxes = [_box(0, 10, 0.5), _box(1, 10, 0.49), _box(2, 10, 0.9), _box(3, 10, -1)]
    out = bs.select_boxes(boxes, bs.MODE_CONFIDENCE_FILTER, min_confidence=0.5)
    assert [b[0] for b in out] == [0, 2]


def test_confidence_filter_default_threshold():
    boxes = [_box(0, 10, 0.4), _box(1, 10, 0.6)]
    out = bs.select_boxes(boxes, bs.MODE_CONFIDENCE_FILTER)
    assert [b[0] for b in out] == [1]
    assert bs.DEFAULT_MIN_CONFIDENCE == 0.5


def test_confidence_filter_without_any_confidence_leaves_frame_untouched():
    boxes = [_box(0, 10, -1), _box(1, 30, None)]
    assert bs.select_boxes(boxes, bs.MODE_CONFIDENCE_FILTER, min_confidence=0.9) == boxes


def test_confidence_filter_can_empty_a_frame():
    boxes = [_box(0, 10, 0.1)]
    assert bs.select_boxes(boxes, bs.MODE_CONFIDENCE_FILTER, min_confidence=0.5) == []


def test_select_boxes_handles_none_and_empty():
    assert bs.select_boxes(None, bs.MODE_COUNT, 1) is None
    assert bs.select_boxes([], bs.MODE_CONFIDENCE_TOP, 1) == []


def test_selection_keeps_tuples_intact():
    box = (7, 1.0, 2.0, 3.0, 40.0, "0.8", 0.1, 0.2, 0.3)
    out = bs.select_boxes([box, _box(1, 10, 0.1)], bs.MODE_CONFIDENCE_TOP, num_people=1)
    assert out == [box]


# ----------------------------------------------------------------------------- per frame


def test_select_boxes_per_frame_applies_to_every_frame_and_keeps_frames():
    bbox_dict = {
        1: [_box(0, 10, 0.9), _box(1, 10, 0.2)],
        2: [_box(0, 10, 0.1)],
        3: [],
    }
    out = bs.select_boxes_per_frame(bbox_dict, bs.MODE_CONFIDENCE_FILTER, min_confidence=0.5)
    assert set(out) == {1, 2, 3}
    assert [b[0] for b in out[1]] == [0]
    assert out[2] == []
    assert out[3] == []
    assert bs.selection_summary(bbox_dict, out) == (3, 1)


def test_select_boxes_per_frame_passthrough_for_other_modes():
    bbox_dict = {1: [_box(0, 10, 0.1)]}
    assert bs.select_boxes_per_frame(bbox_dict, "bbox", 1) is bbox_dict
    assert bs.select_boxes_per_frame(None, bs.MODE_COUNT, 1) is None


def test_mode_constants_are_consistent():
    assert set(bs.SELECTING_MODES) == {"count", "confidence_top", "confidence_filter"}
    assert set(bs.CONFIDENCE_MODES) == {"confidence_top", "confidence_filter"}
    assert set(bs.COUNT_MODES) == {"count", "confidence_top"}
    assert set(bs.ALL_MODES) >= set(bs.SELECTING_MODES) | {"bbox", "inference", "inference_bbox"}
    assert bs.NO_CONFIDENCE == -1.0


# ----------------------------------------------------------------------------- detector scores


def test_boxes_to_tuples_carries_detector_scores():
    pytest.importorskip("torchvision")  # pose_estimation_manager imports sam_3d_body
    from fbxify.pose_estimation_manager import _boxes_to_tuples

    boxes = [[0, 0, 10, 20], [5, 5, 15, 45]]
    with_scores = _boxes_to_tuples(boxes, [0.9, 0.3])
    assert [b[bs.CONF_INDEX] for b in with_scores] == [0.9, 0.3]
    assert with_scores[1][1:5] == (5.0, 5.0, 10.0, 40.0)

    without = _boxes_to_tuples(boxes)
    assert [b[bs.CONF_INDEX] for b in without] == [bs.NO_CONFIDENCE, bs.NO_CONFIDENCE]

    nan_score = _boxes_to_tuples(boxes, [float("nan"), 0.5])
    assert nan_score[0][bs.CONF_INDEX] == bs.NO_CONFIDENCE
    assert nan_score[1][bs.CONF_INDEX] == 0.5
    assert _boxes_to_tuples(None) == []
    assert _boxes_to_tuples([]) == []


def test_detector_runners_accept_return_scores():
    src = _read("tools/build_detector.py")
    for fn in ("run_ultralytics", "run_ultralytics_batch", "run_detectron2_vitdet", "run_detectron2_vitdet_batch"):
        assert fn in src
    assert src.count("return_scores: bool = False") >= 4
    assert "def run_human_detection_scored" in src
    assert "def run_human_detection_batch_scored" in src


# ----------------------------------------------------------------------------- plumbing


def test_manager_applies_selection_before_posing():
    src = _read("fbxify/pose_estimation_manager.py")
    assert "selection_mode: Optional[str] = None" in src
    assert "min_confidence: Optional[float] = None" in src
    assert "select_boxes_per_frame(" in src
    assert "_boxes_to_tuples(boxes_xyxy, scores)" in src


def test_fbxify_manager_forwards_mode_and_threshold():
    src = _read("fbxify/fbxify_manager.py")
    assert src.count("selection_mode=tracking_mode") >= 2
    assert src.count("min_confidence=min_confidence") >= 2


def test_api_accepts_and_validates_the_modes():
    src = _read("fbxify/api.py")
    assert "min_confidence: float = Form(" in src
    assert "_validate_tracking_mode(tracking_mode, num_people, min_confidence)" in src
    assert '"min_confidence": min_confidence' in src
    assert "selection_mode=tracking_mode" in src


def test_api_validator_rejects_bad_inputs():
    pytest.importorskip("fastapi")
    from fastapi import HTTPException

    from fbxify.api import _validate_tracking_mode

    _validate_tracking_mode("count", 2, 0.5)
    _validate_tracking_mode("confidence_top", 3, 0.5)
    _validate_tracking_mode("confidence_filter", 0, 0.0)
    _validate_tracking_mode("confidence_filter", 0, 1.0)
    _validate_tracking_mode("bbox", 0, 0.5)

    with pytest.raises(HTTPException) as exc:
        _validate_tracking_mode("nope", 1, 0.5)
    assert exc.value.status_code == 400
    with pytest.raises(HTTPException):
        _validate_tracking_mode("confidence_top", 0, 0.5)
    with pytest.raises(HTTPException):
        _validate_tracking_mode("confidence_filter", 1, 1.5)
    with pytest.raises(HTTPException):
        _validate_tracking_mode("confidence_filter", 1, -0.1)


def test_backends_forward_min_confidence():
    src = _read("fbxify/backend.py")
    assert "min_confidence" in src
    assert 'payload["min_confidence"]' in src
    assert "SELECTING_MODES" in src


def test_clis_expose_the_modes():
    for rel in ("fbxify/cli.py", "fbxify/cli_pose_estimation.py"):
        src = _read(rel)
        assert '"confidence_top", "confidence_filter"' in src, rel
        assert '"--min_confidence"' in src, rel
        assert "min_confidence=args.min_confidence" in src, rel
        assert "SELECTING_MODES" in src, rel


def test_app_builds_cli_commands_for_the_modes():
    src = _read("fbxify/app.py")
    assert "TRACKING_MODE_CONFIDENCE_TOP" in src
    assert "TRACKING_MODE_CONFIDENCE_FILTER" in src
    assert "--min_confidence" in src
    assert "min_confidence=min_confidence" in src


def test_gui_has_labels_only_no_helper_text():
    src = _read("fbxify/gradio_ui/entry_section.py")
    assert 'translator.t("ui.tracking.mode_confidence_top")' in src
    assert 'translator.t("ui.tracking.mode_confidence_filter")' in src
    assert 'translator.t("ui.min_confidence")' in src
    # No explanatory copy in the GUI: the documentation carries it.
    for marker in ("ui.tracking.mode_confidence_top_info", "ui.min_confidence_info", "confidence column"):
        assert marker not in src


def test_translations_have_the_two_labels_and_threshold():
    for lang in ("en", "es", "fr", "ja"):
        data = json.loads(_read(f"fbxify/i18n/{lang}.json"))
        tracking = data["ui"]["tracking"]
        assert tracking["mode_confidence_top"].strip(), lang
        assert tracking["mode_confidence_filter"].strip(), lang
        assert data["ui"]["min_confidence"].strip(), lang
    en = json.loads(_read("fbxify/i18n/en.json"))
    assert en["ui"]["tracking"]["mode_confidence_top"] == "Highest Confidence N People"
    assert en["ui"]["tracking"]["mode_confidence_filter"] == "Filter Confidence N People"


def test_toggle_tracking_inputs_shows_the_right_controls():
    pytest.importorskip("gradio")
    from fbxify.gradio_ui.entry_section import (
        TRACKING_MODE_CONFIDENCE_FILTER,
        TRACKING_MODE_CONFIDENCE_TOP,
        toggle_tracking_inputs,
    )

    def vis(update):
        return update.get("visible") if isinstance(update, dict) else getattr(update, "visible", None)

    top = toggle_tracking_inputs(TRACKING_MODE_CONFIDENCE_TOP)
    assert len(top) == 5
    bbox_file, num_people, _behavior, _group, min_conf = top
    assert vis(bbox_file) is True  # optional bbox file carries the conf column
    assert vis(num_people) is True
    assert vis(min_conf) is False

    flt = toggle_tracking_inputs(TRACKING_MODE_CONFIDENCE_FILTER)
    bbox_file, num_people, _behavior, _group, min_conf = flt
    assert vis(bbox_file) is True
    assert vis(num_people) is False
    assert vis(min_conf) is True

    count = toggle_tracking_inputs("count")
    _bbox_file, num_people, _behavior, _group, min_conf = count
    assert vis(num_people) is True
    assert vis(min_conf) is False


def test_readme_and_postman_document_the_modes():
    readme = _read("README.md")
    assert "## Choosing who gets posed" in readme
    for needle in ("confidence_top", "confidence_filter", "min_confidence", "frame,id,x,y,w,h,conf", "-1"):
        assert needle in readme
    assert "is a person" in readme  # stock-detector caveat
    postman = _read("fbxify/postman/fbxify-worker-api.postman_collection.json")
    assert postman.count('"key": "min_confidence"') == 2
    assert "confidence_top" in postman


# ----------------------------------------------------------------------------- opt-in smoke


def _smoke_env():
    """Worker URL, clip, cap and reference for the smoke test; all via environment, nothing in the repo.

    FBXIFY_SMOKE_URL        running worker, e.g. http://localhost:8000
    FBXIFY_SMOKE_CLIP       ~10 s clip with at least two people in frame
    FBXIFY_SMOKE_K          N for confidence_top (default 2)
    FBXIFY_SMOKE_MODE       confidence_top (default) or confidence_filter
    FBXIFY_SMOKE_MIN_CONF   threshold for confidence_filter (default 0.5)
    FBXIFY_SMOKE_BBOX       optional bbox file with a conf column
    FBXIFY_SMOKE_REFERENCE  optional pose_outputs JSON from a reference run (same clip, same people)
    FBXIFY_SMOKE_TOLERANCE  mean joint distance allowed against the reference (default 0.05)
    FBXIFY_SMOKE_API_KEY    optional X-API-Key
    """
    url = os.environ.get("FBXIFY_SMOKE_URL")
    clip = os.environ.get("FBXIFY_SMOKE_CLIP")
    if not url or not clip:
        pytest.skip("set FBXIFY_SMOKE_URL and FBXIFY_SMOKE_CLIP to run the worker smoke test")
    return {
        "url": url.rstrip("/"),
        "clip": clip,
        "k": int(os.environ.get("FBXIFY_SMOKE_K", "2")),
        "mode": os.environ.get("FBXIFY_SMOKE_MODE", bs.MODE_CONFIDENCE_TOP),
        "min_conf": float(os.environ.get("FBXIFY_SMOKE_MIN_CONF", "0.5")),
        "bbox": os.environ.get("FBXIFY_SMOKE_BBOX"),
        "reference": os.environ.get("FBXIFY_SMOKE_REFERENCE"),
        "tolerance": float(os.environ.get("FBXIFY_SMOKE_TOLERANCE", "0.05")),
        "api_key": os.environ.get("FBXIFY_SMOKE_API_KEY"),
    }


def _mean_joint_distance(a, b) -> float:
    total, n = 0.0, 0
    for ja, jb in zip(a, b):
        total += math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(ja, jb)))
        n += 1
    return total / n if n else 0.0


def test_smoke_confidence_mode_caps_people_per_frame():
    env = _smoke_env()
    requests = pytest.importorskip("requests")
    headers = {"X-API-Key": env["api_key"]} if env["api_key"] else {}

    data = {"tracking_mode": env["mode"], "num_people": str(env["k"]), "min_confidence": str(env["min_conf"])}
    files = {"input_file": open(env["clip"], "rb")}
    if env["bbox"]:
        files["bbox_file"] = open(env["bbox"], "rb")
    try:
        r = requests.post(f"{env['url']}/jobs/pose", data=data, files=files, headers=headers, timeout=60)
    finally:
        for f in files.values():
            f.close()
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    deadline = time.time() + float(os.environ.get("FBXIFY_SMOKE_TIMEOUT", "1800"))
    while True:
        status = requests.get(f"{env['url']}/jobs/{job_id}", headers=headers, timeout=30).json()
        if status["status"] in ("completed", "failed", "cancelled"):
            break
        assert time.time() < deadline, f"job still {status['status']} after timeout"
        time.sleep(3)
    assert status["status"] == "completed", status

    pose_name = next(os.path.basename(p) for p in status["output_files"] if p.endswith(".json"))
    result = requests.get(f"{env['url']}/jobs/{job_id}/files/{pose_name}", headers=headers, timeout=120).json()
    frames = result["frames"]
    assert frames, "no frames posed"

    per_frame = [len(people) for people in frames.values()]
    if env["mode"] == bs.MODE_CONFIDENCE_TOP:
        assert max(per_frame) <= env["k"], f"a frame posed more than K={env['k']} people: {max(per_frame)}"
    # both dancers present: at least two distinct people somewhere in the clip
    ids = {pid for people in frames.values() for pid in people}
    assert len(ids) >= 2, f"expected both dancers, saw ids {sorted(ids)}"

    if env["reference"]:
        ref = json.loads(Path(env["reference"]).read_text(encoding="utf-8"))["frames"]
        worst = 0.0
        for frame, people in frames.items():
            ref_people = ref.get(frame) or {}
            for pid, est in people.items():
                ref_est = ref_people.get(pid)
                if not ref_est or "pred_joint_coords" not in est or "pred_joint_coords" not in ref_est:
                    continue
                worst = max(worst, _mean_joint_distance(est["pred_joint_coords"], ref_est["pred_joint_coords"]))
        assert worst <= env["tolerance"], f"pose drifted from reference by {worst:.4f} > {env['tolerance']}"
