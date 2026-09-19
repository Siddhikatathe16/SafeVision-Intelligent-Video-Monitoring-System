"""Integrated event layer for Crowd, Loitering, Zone, Movement and Fall.

The app supplies tracked people.  This module converts trajectories into
structured events and provides a single drawing/reporting interface.
"""
import json
import sys
import traceback
import cv2
import numpy as np

from safety_config import LOITERING_ZONE_NORM, MANUAL_RESTRICTED_ZONE_NORM
from safety_geometry import foot_point, normalize_polygon
from crowd_detector import CrowdDetector
from loitering_detector import LoiteringDetector
from restricted_zone_detector import RestrictedZoneDetector
from abnormal_movement_detector import AbnormalMovementDetector
from fall_detector import FallDetector


class SafetyEventPipeline:
    def __init__(self):
        self.crowd = CrowdDetector()
        self.loitering = LoiteringDetector()
        self.zone = RestrictedZoneDetector()
        self.movement = AbnormalMovementDetector()
        self.fall = FallDetector()
        self.restricted_zone_norm = normalize_polygon(MANUAL_RESTRICTED_ZONE_NORM)
        self.loitering_zone_norm = normalize_polygon(LOITERING_ZONE_NORM)
        self.final_events = []
        self.event_counter = 0
        self.restricted_zone_norm = normalize_polygon(self.zone.current_zone())
        self.reset()

    def reset(self):
        self.crowd.reset()
        self.loitering.reset()
        self.zone.reset()
        self.movement.reset()
        self.fall.reset()
        self.final_events = []
        self.event_counter = 0
        self.detector_errors = {}

    def _safe(self, name, fn, *args, default=None, **kwargs):
        """Run one detector; a bug in one detector must not abort the whole
        video. The FULL traceback is printed the first time each detector
        fails (so the file/line is never hidden) and every failure is counted
        in `detector_errors`, which is returned to the caller/API."""
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            n = self.detector_errors.get(name, 0) + 1
            self.detector_errors[name] = n
            if n == 1:
                print(f"\n[safety pipeline] detector '{name}' raised {exc!r} -- full traceback:", file=sys.stderr)
                traceback.print_exc()
            return default

    def set_zones(self, restricted_zone=None, loitering_zone=None):
        if restricted_zone is not None:
            zone = normalize_polygon(restricted_zone)
            zone = [[max(0.0,min(1.0,x)), max(0.0,min(1.0,y))] for x,y in zone]
            self.restricted_zone_norm = zone
            self.zone.set_manual_zone(zone if zone else None)
        if loitering_zone is not None and len(loitering_zone)>=3:
            zone = normalize_polygon(loitering_zone)
            self.loitering_zone_norm = [[max(0.0,min(1.0,x)), max(0.0,min(1.0,y))] for x,y in zone]

    @staticmethod
    def _decorate_event(event):
        return dict(event)

    def update(self, people, frame_index, fps, frame_shape, frame=None):
        h, w = frame_shape[:2]
        if frame is not None:
            self._safe("zone_auto_detect", self.zone.auto_update, frame, frame_index)
        current_zone_norm = self.zone.current_zone() or []
        current_zone_norm = normalize_polygon(current_zone_norm)
        self.restricted_zone_norm = current_zone_norm
        restricted_zone = [(int(x*w), int(y*h)) for x, y in current_zone_norm]
        loiter_norm = normalize_polygon(self.loitering_zone_norm)
        loiter_zone = [(int(x*w), int(y*h)) for x, y in loiter_norm]

        normalized_people = []
        for p in people:
            q = dict(p)
            # Always plain Python floats: detectors must never see numpy
            # scalars/arrays in a bbox (they are not iterable / not JSON-safe).
            q["bbox"] = [float(v) for v in q["bbox"]]
            q["foot_point"] = foot_point(q["bbox"])
            normalized_people.append(q)

        crowd_result = self._safe(
            "crowd", self.crowd.update,
            normalized_people, frame_index, fps, frame_shape, roi_polygon=None,
            default={"count": len(normalized_people), "density": 0.0, "event": None},
        )
        events = []
        if crowd_result.get("event"):
            events.append(self._decorate_event(crowd_result["event"]))

        events.extend(self._safe("restricted_zone", self.zone.update,
                                 normalized_people, frame_index, fps, restricted_zone, default=[]))
        events.extend(self._safe("loitering", self.loitering.update,
                                 normalized_people, frame_index, fps, loiter_zone, default=[]))
        events.extend(self._safe("abnormal_movement", self.movement.update,
                                 normalized_people, frame_index, fps, default=[]))
        events.extend(self._safe("fall", self.fall.update,
                                 normalized_people, frame_index, fps, default=[]))

        for event in events:
            self.event_counter += 1
            event["event_id"] = f"EVENT-{self.event_counter:04d}"
            event["frame"] = int(frame_index)
            event["timestamp"] = float(event.get("timestamp", frame_index / max(fps, 1.0)))
            self.final_events.append(dict(event))

        alert_person_ids = set()
        alert_types_by_person = {}
        for detector_state in (self.zone.state, self.loitering.state, self.movement.alert_state, self.fall.state):
            if hasattr(detector_state, "items"):
                for pid, state in detector_state.items():
                    if isinstance(state, dict) and state.get("alerted", False):
                        alert_person_ids.add(int(pid))
                        alert_types_by_person.setdefault(int(pid), set())
        # Add event types that fired on this frame.
        for event in events:
            pid = event.get("person_id")
            if pid is not None:
                pid = int(pid)
                alert_person_ids.add(pid)
                alert_types_by_person.setdefault(pid, set()).add(event["event_type"])

        # Detector states have different structures; explicitly map the
        # persistent flags to human-readable event types.
        for pid, state in self.zone.state.items():
            if state.get("alerted", False):
                alert_types_by_person.setdefault(int(pid), set()).add("ZONE_INTRUSION")
        for pid, state in self.loitering.state.items():
            if state.get("alerted", False):
                alert_types_by_person.setdefault(int(pid), set()).add("LOITERING")
        for pid in self.movement.alerted:
            alert_person_ids.add(int(pid))
            alert_types_by_person.setdefault(int(pid), set()).add("ABNORMAL_MOVEMENT")
        for pid, state in self.fall.state.items():
            if state.get("alerted", False):
                alert_person_ids.add(int(pid))
                alert_types_by_person.setdefault(int(pid), set()).add("POSSIBLE_FALL")

        alert_types_by_person = {k: sorted(v) for k, v in alert_types_by_person.items()}

        return {
            "events": events,
            "alert_person_ids": sorted(alert_person_ids),
            "alert_types_by_person": alert_types_by_person,
            "crowd_count": crowd_result["count"],
            "crowd_density": crowd_result["density"],
            "restricted_zone": restricted_zone,
            "loitering_zone": loiter_zone,
            "detector_errors": dict(self.detector_errors),
        }

    @staticmethod
    def draw_zone(frame, polygon, label):
        if not polygon:
            return
        pts = np.asarray(polygon, dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 120))
        cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)
        cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
        x, y = pts[0]
        cv2.putText(frame, label, (int(x), max(20, int(y) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

    def draw(self, frame, result):
        restricted_zone = result.get("restricted_zone")
        self.draw_zone(frame, restricted_zone, "RESTRICTED ZONE")
        # The logical loitering area is the full camera by default.  Do not
        # draw a full-frame polygon because it obscures CCTV footage.

        events = result.get("events", [])
        y = 28
        for event in events:
            pid = event.get("person_id")
            target = f"PERSON-{int(pid):03d}" if pid is not None else "ALL"
            text = f"ALERT {event['event_type']} -> {target}"
            cv2.rectangle(frame, (5, y - 20), (min(frame.shape[1] - 5, 560), y + 8), (0, 0, 180), -1)
            cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (255, 255, 255), 2, cv2.LINE_AA)
            y += 32
            if y > 150:
                break

        # Persistent status line is deliberately compact.
        status = (
            f"CROWD: {result.get('crowd_count', 0)}  "
            f"DENSITY: {result.get('crowd_density', 0.0):.2f}"
        )
        cv2.putText(frame, status, (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def report(self, fps):
        rows = []
        for event in self.final_events:
            row = dict(event)
            row["time_seconds"] = round(float(event["timestamp"]), 2)
            if event.get("person_id") is not None:
                row["person_label"] = f"PERSON-{int(event['person_id']):03d}"
            else:
                row["person_label"] = "ALL"
            rows.append(row)
        return rows

    def report_json(self, fps):
        return json.dumps(self.report(fps), indent=2)
