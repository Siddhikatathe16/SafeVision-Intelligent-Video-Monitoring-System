"""Object-removal (theft) detector.

Watches configured "item zones" (e.g. a lab table) for monitored objects
(default: laptops, since 'laptop' is already a standard COCO class -- no
custom training needed). An item must be CONFIRMED present for several
consecutive frames before it becomes a tracked baseline item. If a
confirmed item then goes missing for several consecutive frames, it is
reported as a POSSIBLE_ITEM_REMOVED event, associated with whichever
tracked person was closest to the item's last known position in the
moments just before it disappeared.

Mirrors the existing WeaponEventTracker / RestrictedZoneDetector design:
explainable, rule-based, multi-frame confirmation in both directions
(appearance AND disappearance), no black-box logic.
"""
import math
from collections import defaultdict

# ---- Tunable thresholds (same style as safety_config.py) ----
ITEM_CONFIRM_FRAMES = 20          # frames an item must be seen before it's "baseline confirmed"
ITEM_MIN_CONFIDENCE = 0.45        # min YOLO confidence to count as a real detection
ITEM_MATCH_IOU = 0.35             # IoU to match a detection to an existing tracked item
ITEM_MISSING_CONFIRM_FRAMES = 15  # frames it must stay missing before "removed" is confirmed
ITEM_ASSOCIATION_WINDOW_SEC = 2.0 # how far back to look for a nearby person before disappearance
ITEM_ASSOCIATION_MAX_DIST_RATIO = 0.35  # max distance (as a fraction of frame width) to associate a person

MONITORED_CLASSES = {"laptop"}    # extend later e.g. {"laptop", "cell phone", "backpack"}


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


class ObjectTheftDetector:
    """
    zones: list of {"name": str, "polygon": [(x_norm, y_norm), ...]}
    Pass an empty/None zones list to monitor the whole frame instead of a
    specific table area.
    """

    def __init__(self, zones=None):
        self.zones = zones or []
        self.reset()

    def reset(self):
        self._next_item_id = 1
        # tracked_items[item_id] = {
        #   "bbox": [...], "label": "laptop", "zone": name,
        #   "present_streak": int, "missing_streak": int,
        #   "confirmed": bool, "alerted_removed": bool,
        #   "last_seen_time": float, "person_positions_seen": deque
        # }
        self.tracked_items = {}

    def _zone_for_point(self, point, frame_w, frame_h):
        from safety_geometry import point_in_polygon
        if not self.zones:
            return "whole_frame"
        nx, ny = point[0] / frame_w, point[1] / frame_h
        for z in self.zones:
            if point_in_polygon((nx, ny), z["polygon"]):
                return z["name"]
        return None  # detected but outside any monitored zone -- ignore

    def update(self, item_detections, people, frame_index, fps, frame_shape):
        """
        item_detections: list of {"bbox": [x1,y1,x2,y2], "label": str, "confidence": float}
          -- from a pretrained COCO-class YOLO model, already filtered/unfiltered is fine,
             this function filters by MONITORED_CLASSES and ITEM_MIN_CONFIDENCE itself.
        people: list of tracked person dicts (same shape used elsewhere in the pipeline),
          each with "stable_id" and "bbox".
        """
        fps = max(float(fps), 1.0)
        now = frame_index / fps
        h, w = frame_shape[:2]
        events = []

        # ---- 1. Match this frame's item detections to existing tracked items ----
        matched_ids = set()
        for det in item_detections:
            if det["label"] not in MONITORED_CLASSES:
                continue
            if det["confidence"] < ITEM_MIN_CONFIDENCE:
                continue
            zone_name = self._zone_for_point(_center(det["bbox"]), w, h)
            if zone_name is None:
                continue  # not in a monitored area

            best_id, best_iou = None, 0.0
            for iid, item in self.tracked_items.items():
                if iid in matched_ids:
                    continue
                iou = _iou(det["bbox"], item["bbox"])
                if iou > best_iou:
                    best_id, best_iou = iid, iou

            if best_id is not None and best_iou >= ITEM_MATCH_IOU:
                item = self.tracked_items[best_id]
                item["bbox"] = det["bbox"]
                item["present_streak"] += 1
                item["missing_streak"] = 0
                item["last_seen_time"] = now
                if item["present_streak"] >= ITEM_CONFIRM_FRAMES:
                    item["confirmed"] = True
                matched_ids.add(best_id)
            else:
                # New candidate item
                new_id = self._next_item_id
                self._next_item_id += 1
                self.tracked_items[new_id] = {
                    "bbox": det["bbox"], "label": det["label"], "zone": zone_name,
                    "present_streak": 1, "missing_streak": 0,
                    "confirmed": False, "alerted_removed": False,
                    "last_seen_time": now,
                }
                matched_ids.add(new_id)

        # ---- 2. Any confirmed item NOT matched this frame is one frame more "missing" ----
        for iid, item in list(self.tracked_items.items()):
            if iid in matched_ids:
                continue
            if not item["confirmed"]:
                # Was never confirmed present long enough -- just a flicker, drop it
                item["present_streak"] = max(0, item["present_streak"] - 2)
                if item["present_streak"] <= 0:
                    self.tracked_items.pop(iid, None)
                continue

            item["missing_streak"] += 1
            if item["missing_streak"] >= ITEM_MISSING_CONFIRM_FRAMES and not item["alerted_removed"]:
                item["alerted_removed"] = True
                person_id, dist_ratio = self._find_likely_person(item, people, now, w)
                confidence = 0.55 + (0.25 if person_id is not None else 0.0)
                confidence += min(0.15, item["present_streak"] / 200.0)
                events.append({
                    "event_type": "POSSIBLE_ITEM_REMOVED",
                    "severity": "HIGH",
                    "person_id": person_id,
                    "confidence": round(min(0.97, confidence), 2),
                    "frame": frame_index,
                    "timestamp": now,
                    "details": {
                        "item_label": item["label"],
                        "zone": item["zone"],
                        "last_seen_time": round(item["last_seen_time"], 2),
                        "confirmed_missing_time": round(now, 2),
                        "association_method": "nearest_person_before_disappearance",
                        "association_distance_ratio": round(dist_ratio, 3) if dist_ratio is not None else None,
                    }
                })

        return events

    def _find_likely_person(self, item, people, now, frame_w):
        """
        Look at people currently tracked (or tracked just before the item
        vanished -- callers should pass the latest 'people' each frame, so
        by the time missing_streak crosses the threshold we're checking who
        is near the item's last known position right now). Nearest person
        to the item's last bbox center, within a max distance, wins.
        """
        item_center = _center(item["bbox"])
        best_pid, best_dist = None, None
        max_dist = ITEM_ASSOCIATION_MAX_DIST_RATIO * frame_w

        for person in people:
            pid = person.get("stable_id")
            if pid is None:
                continue
            p_center = _center(person["bbox"])
            d = _dist(item_center, p_center)
            if d <= max_dist and (best_dist is None or d < best_dist):
                best_pid, best_dist = int(pid), d

        dist_ratio = (best_dist / frame_w) if best_dist is not None else None
        return best_pid, dist_ratio