"""Restricted-zone detector.

Default mode is AUTO: a new video starts with no restricted zone.  A zone is
created only after a stable red barrier/ribbon geometry is observed over
multiple frames.  An old video's polygon is never reused.

Manual mode is available by supplying SAFETY_RESTRICTED_ZONE or POSTing a
polygon to /safety/config.
"""
from collections import defaultdict, deque
import cv2
import traceback
import numpy as np
import math

from safety_config import (
    ZONE_CONFIRM_FRAMES, ZONE_HOLD_FRAMES, MAX_HISTORY_FRAMES,
    ZONE_AUTO_ENABLED, ZONE_AUTO_WARMUP_FRAMES, ZONE_AUTO_SAMPLE_EVERY,
    ZONE_AUTO_MIN_STABLE_SAMPLES, ZONE_AUTO_MIN_LONG_LINES,
    ZONE_AUTO_MIN_LINE_LENGTH, ZONE_AUTO_MIN_AREA, ZONE_AUTO_MAX_AREA,
    ZONE_AUTO_MIN_X, ZONE_AUTO_MIN_Y, MANUAL_RESTRICTED_ZONE_NORM,
)
from safety_geometry import point_in_polygon, normalize_polygon

class RestrictedZoneDetector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = defaultdict(lambda: {
            "inside_streak": 0, "outside_streak": 0, "alerted": False
        })
        self.auto_candidates = []
        self.auto_zone_norm = None
        self.auto_failed = False
        # normalize_polygon(None) returns [] (not None) by design, so only
        # normalize when an env-configured manual zone actually exists --
        # otherwise manual_zone_norm must stay exactly None, or the "is not
        # None" check below would treat an empty [] as "manual zone is set"
        # and auto-detection would never run.
        self.manual_zone_norm = normalize_polygon(MANUAL_RESTRICTED_ZONE_NORM) if MANUAL_RESTRICTED_ZONE_NORM else None
        self.frame_shape = None

    @staticmethod
    def _red_mask(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0,100,70],np.uint8), np.array([12,255,255],np.uint8))
        m2 = cv2.inRange(hsv, np.array([170,100,70],np.uint8), np.array([179,255,255],np.uint8))
        return cv2.bitwise_or(m1,m2)

    def _candidate_from_frame(self, frame):
        h,w = frame.shape[:2]
        mask = self._red_mask(frame)
        roi = np.zeros_like(mask)
        roi[int(ZONE_AUTO_MIN_Y*h):, int(ZONE_AUTO_MIN_X*w):] = 255
        mask = cv2.bitwise_and(mask, roi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        edges = cv2.Canny(mask, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=28,
            minLineLength=ZONE_AUTO_MIN_LINE_LENGTH, maxLineGap=16
        )
        if lines is None:
            return None
        segments=[]
        # HoughLinesP normally returns (N,1,4) int32, but be defensive about
        # the array shape: flatten to (N,4) so every row is always a 4-vector
        # (indexing [:,0] on an already-flat array yields bare numpy.int32
        # scalars, which cannot be iterated).
        for line in np.asarray(lines).reshape(-1,4):
            x1,y1,x2,y2=(int(v) for v in line)
            length=math.hypot(x2-x1,y2-y1)
            angle=abs(math.degrees(math.atan2(y2-y1,x2-x1)))
            if angle>90: angle=180-angle
            # A ribbon/rope is long and oblique/horizontal.  Near-vertical
            # red objects (clothes/signs) are intentionally rejected.
            if length >= ZONE_AUTO_MIN_LINE_LENGTH and 7 <= angle <= 55:
                segments.append((length, angle, (x1,y1,x2,y2)))
        if len(segments) < ZONE_AUTO_MIN_LONG_LINES:
            return None
        segments=sorted(segments, reverse=True)[:12]
        pts=[]
        for _,_,(x1,y1,x2,y2) in segments:
            pts.extend([(x1,y1),(x2,y2)])
        hull=cv2.convexHull(np.asarray(pts,dtype=np.int32).reshape(-1,1,2)).reshape(-1,2)
        area=cv2.contourArea(hull)
        frac=area/(w*h)
        if frac < ZONE_AUTO_MIN_AREA or frac > ZONE_AUTO_MAX_AREA:
            return None
        M=cv2.moments(hull)
        if abs(M["m00"]) < 1:
            return None
        cx=(M["m10"]/M["m00"])/w
        cy=(M["m01"]/M["m00"])/h
        if cx < ZONE_AUTO_MIN_X or cy < ZONE_AUTO_MIN_Y:
            return None
        peri=cv2.arcLength(hull,True)
        approx=cv2.approxPolyDP(hull,0.04*peri,True).reshape(-1,2)
        if len(approx)>6:
            approx=cv2.approxPolyDP(hull,0.08*peri,True).reshape(-1,2)
        if len(approx)<3:
            return None
        return [(float(p[0])/w,float(p[1])/h) for p in np.asarray(approx).reshape(-1,2)]

    @staticmethod
    def _polygon_iou(a,b):
        if not a or not b:
            return 0.0
        # Raster-free convex polygon overlap approximation using masks.
        scale=600
        aa=np.array([[int(p[0]*scale),int(p[1]*scale)] for p in a],np.int32)
        bb=np.array([[int(p[0]*scale),int(p[1]*scale)] for p in b],np.int32)
        ma=np.zeros((scale,scale),np.uint8); mb=ma.copy()
        cv2.fillPoly(ma,[aa],1); cv2.fillPoly(mb,[bb],1)
        inter=np.logical_and(ma,mb).sum()
        union=np.logical_or(ma,mb).sum()
        return float(inter/union) if union else 0.0

    def auto_update(self, frame, frame_index):
        if self.manual_zone_norm:
            return self.manual_zone_norm
        if not ZONE_AUTO_ENABLED or self.auto_zone_norm is not None or getattr(self,"auto_failed",False):
            return self.auto_zone_norm
        if frame_index > ZONE_AUTO_WARMUP_FRAMES:
            return None
        if frame_index % max(1,ZONE_AUTO_SAMPLE_EVERY) != 0:
            return None
        try:
            candidate=self._candidate_from_frame(frame)
        except Exception:
            # Best-effort heuristic: log the full traceback ONCE, disable
            # auto-detection for this video and carry on without a zone.
            print("[restricted_zone] auto zone detection failed; continuing without a zone")
            traceback.print_exc()
            self.auto_failed=True
            return None
        if candidate is None:
            return None
        self.auto_candidates.append(candidate)
        self.auto_candidates=self.auto_candidates[-8:]
        # Require repeated geometrically similar candidates, not one red object.
        best=None; best_count=0
        for cand in self.auto_candidates:
            count=sum(self._polygon_iou(cand,other)>=0.45 for other in self.auto_candidates)
            if count>best_count:
                best,best_count=cand,count
        if best is not None and best_count>=ZONE_AUTO_MIN_STABLE_SAMPLES:
            self.auto_zone_norm=best
        return self.auto_zone_norm

    def set_manual_zone(self, zone_polygon):
        normalized = normalize_polygon(zone_polygon)
        self.manual_zone_norm = normalized if len(normalized) >= 3 else None
        if self.manual_zone_norm is not None:
            self.auto_zone_norm = None

    def clear_manual_zone(self):
        self.manual_zone_norm = None
        self.auto_zone_norm = None
        self.auto_candidates.clear()

    def current_zone(self):
        return normalize_polygon(self.manual_zone_norm if self.manual_zone_norm is not None else self.auto_zone_norm)

    def update(self, people, frame_index, fps, zone_polygon):
        if not zone_polygon or len(zone_polygon)<3:
            return []
        fps=max(float(fps),1.0)
        now=frame_index/fps
        events=[]; active_ids=set()
        for person in people:
            pid=person.get("stable_id")
            if pid is None: continue
            pid=int(pid); active_ids.add(pid)
            state=self.state[pid]
            inside=point_in_polygon(person["foot_point"],zone_polygon)
            if inside:
                state["inside_streak"]+=1; state["outside_streak"]=0
                if state["inside_streak"]>=ZONE_CONFIRM_FRAMES and not state["alerted"]:
                    state["alerted"]=True
                    events.append({
                        "event_type":"ZONE_INTRUSION","severity":"HIGH","person_id":pid,
                        "confidence":min(0.99,0.70+0.04*min(6,state["inside_streak"])),
                        "frame":frame_index,"timestamp":now,
                        "details":{"zone":"restricted_zone","confirmation_frames":state["inside_streak"]}
                    })
            else:
                state["outside_streak"]+=1
                if state["outside_streak"]>=ZONE_HOLD_FRAMES:
                    state["inside_streak"]=0; state["alerted"]=False
        for pid in list(self.state):
            if pid not in active_ids:
                self.state[pid]["outside_streak"]+=1
                if self.state[pid]["outside_streak"]>max(ZONE_HOLD_FRAMES*10,MAX_HISTORY_FRAMES):
                    self.state.pop(pid,None)
        return events
