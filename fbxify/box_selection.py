"""Per-frame box selection: decide which detected / supplied boxes get posed.

Pose estimation is the expensive step, so after detection (or after a bbox file is loaded) each
frame's boxes go through one selector before SAM 3D Body sees them. The selector is picked by the
job's ``tracking_mode``:

``count`` (Closest N People)
    Keep the N tallest boxes. Box height is the proxy for distance to the camera.
``confidence_top`` (Highest Confidence N People)
    Keep the N boxes with the highest confidence. Boxes without a confidence sort last. A frame
    where no box carries a confidence falls back to Closest N, so a provider without scores still
    works.
``confidence_filter`` (Filter Confidence N People)
    Keep every box whose confidence is at least ``min_confidence``. A frame where no box carries
    a confidence is left untouched (there is nothing to filter on).
anything else (``bbox``, ``inference``, ``inference_bbox``)
    No selection; every box is posed.

Where the confidence comes from:

* bbox file: the 7th column of the MOTChallenge layout ``frame, id, x, y, w, h, conf, ...``.
  ``-1`` (the MOT convention for "not provided"), blanks and non-numbers mean "no confidence".
* detector: the detector's own score for the box (ViTDet / Ultralytics report one; a detector
  type registered without scores yields none).

A stock person detector's confidence means "this is a person". It does not rank the people you
care about above the rest; that ranking is only as good as whatever produced the confidence
column. See README, "Choosing who gets posed".
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

MODE_COUNT = "count"
MODE_CONFIDENCE_TOP = "confidence_top"
MODE_CONFIDENCE_FILTER = "confidence_filter"

# Modes that reduce the boxes of a frame before pose estimation.
SELECTING_MODES = (MODE_COUNT, MODE_CONFIDENCE_TOP, MODE_CONFIDENCE_FILTER)
# Modes that need the confidence column / detector scores to do their job.
CONFIDENCE_MODES = (MODE_CONFIDENCE_TOP, MODE_CONFIDENCE_FILTER)
# Modes that take a count of people (N).
COUNT_MODES = (MODE_COUNT, MODE_CONFIDENCE_TOP)
# All tracking_mode values accepted by the API / CLI.
ALL_MODES = ("bbox", "count", "inference", "inference_bbox", MODE_CONFIDENCE_TOP, MODE_CONFIDENCE_FILTER)

DEFAULT_MIN_CONFIDENCE = 0.5

# Index of the confidence in a box tuple (person_id, x, y, w, h, conf, x_world, y_world, z_world).
CONF_INDEX = 5
NO_CONFIDENCE = -1.0  # written to MOT files when a provider gives no score


def box_confidence(box: Sequence[Any]) -> Optional[float]:
    """Confidence of one box tuple, or None when absent / unparseable / negative (MOT "-1")."""
    if box is None or len(box) <= CONF_INDEX:
        return None
    raw = box[CONF_INDEX]
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value


def box_height(box: Sequence[Any]) -> float:
    try:
        return float(box[4])
    except (TypeError, ValueError, IndexError):
        return 0.0


def _closest(boxes: List[Sequence[Any]], n: int) -> List[Sequence[Any]]:
    # Stable: equal heights keep the provider's order.
    return sorted(boxes, key=box_height, reverse=True)[:n]


def select_boxes(
    boxes: Optional[Iterable[Sequence[Any]]],
    mode: Optional[str],
    num_people: Optional[int] = None,
    min_confidence: Optional[float] = None,
) -> Optional[List[Sequence[Any]]]:
    """Return the boxes of one frame that should be posed. Order of the survivors is the ranking."""
    if boxes is None:
        return None
    boxes = list(boxes)
    if mode not in SELECTING_MODES or not boxes:
        return boxes

    if mode in COUNT_MODES:
        n = int(num_people) if num_people is not None else 0
        if n <= 0:
            return boxes  # no cap requested

    if mode == MODE_COUNT:
        return _closest(boxes, n)

    confidences = [box_confidence(b) for b in boxes]
    if all(c is None for c in confidences):
        # Nothing to rank or filter on: behave like a provider without scores.
        return _closest(boxes, n) if mode == MODE_CONFIDENCE_TOP else boxes

    if mode == MODE_CONFIDENCE_TOP:
        order = sorted(
            range(len(boxes)),
            key=lambda i: (confidences[i] is None, -(confidences[i] or 0.0), -box_height(boxes[i])),
        )
        return [boxes[i] for i in order[:n]]

    # MODE_CONFIDENCE_FILTER
    threshold = DEFAULT_MIN_CONFIDENCE if min_confidence is None else float(min_confidence)
    return [b for b, c in zip(boxes, confidences) if c is not None and c >= threshold]


def select_boxes_per_frame(
    bbox_dict: Optional[Dict[int, List[Sequence[Any]]]],
    mode: Optional[str],
    num_people: Optional[int] = None,
    min_confidence: Optional[float] = None,
) -> Optional[Dict[int, List[Sequence[Any]]]]:
    """Apply select_boxes to every frame of a {frame_index: [box, ...]} dict."""
    if bbox_dict is None or mode not in SELECTING_MODES:
        return bbox_dict
    return {
        frame: select_boxes(boxes, mode, num_people=num_people, min_confidence=min_confidence) or []
        for frame, boxes in bbox_dict.items()
    }


def selection_summary(before: Optional[Dict[int, List]], after: Optional[Dict[int, List]]) -> Tuple[int, int]:
    """(boxes before, boxes after) for a log line."""
    if not before or after is None:
        return 0, 0
    return sum(len(v) for v in before.values()), sum(len(v) for v in after.values())
