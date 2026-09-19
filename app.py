from flask import Flask, request, jsonify, render_template, send_from_directory, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import os
import cv2
from flask_cors import CORS
import time
import pandas as pd
import base64
from PIL import Image
import io
import numpy as np
import subprocess
import sys
import traceback
import imageio_ffmpeg
import re
import time

from safety_event_pipeline import SafetyEventPipeline
from safety_config import WEAPON_MAX_PERSON_AREA_RATIO
from agentic_ai import AgenticSupervisor

# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    static_url_path="/static",
    static_folder="static"
)

CORS(app)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# General YOLO model:
# Detects common objects such as person, bicycle, car, bus, truck, etc.
GENERAL_MODEL_PATH = os.path.join(REPO_ROOT, "yolov8s.pt")

# Updated custom safety/weapon model:
# YOLO11n threat detector: ammo, firearm, grenade, knife, pistol, rocket.
WEAPON_MODEL_PATH = os.path.join(REPO_ROOT, "gun_knife_yolo11n.pt")
WEAPON_FALLBACK_MODEL_PATH = os.path.join(REPO_ROOT, "best.pt")
WEAPON_MODEL_URL = (
    "https://raw.githubusercontent.com/ChiragAgg5k/suraksha-ai/"
    "master/yolo11n_threat_detection.pt"
)

# Loaded models
general_model = None
weapon_model = None

try:
    # --------------------------------------------------------
    # GENERAL YOLO MODEL
    # --------------------------------------------------------
    if os.path.exists(GENERAL_MODEL_PATH):
        general_model = YOLO(GENERAL_MODEL_PATH)

        print("=" * 60)
        print("GENERAL YOLO MODEL LOADED")
        print("=" * 60)
        print(f"Model path: {GENERAL_MODEL_PATH}")
        print(f"Classes: {general_model.names}")
        print("=" * 60)
    else:
        print("WARNING: yolov8s.pt NOT FOUND")
        print("Place yolov8s.pt in the same folder as app.py.")

    # --------------------------------------------------------
    # CUSTOM WEAPON MODEL
    # --------------------------------------------------------
    # The project's trained best.pt explicitly contains Gun, Knife and
    # Handgun classes, so it is preferred for this CCTV task. The newer
    # threat checkpoint is retained as a fallback if best.pt is unavailable.
    if os.path.exists(WEAPON_MODEL_PATH):
        weapon_model = YOLO(WEAPON_MODEL_PATH)
        print("=" * 60)
        print("PROJECT WEAPON MODEL LOADED")
        print("=" * 60)
        print(f"Model path: {WEAPON_MODEL_PATH}")
        print(f"Classes: {weapon_model.names}")
        print("=" * 60)
    elif os.path.exists(WEAPON_FALLBACK_MODEL_PATH):
        WEAPON_MODEL_PATH = WEAPON_FALLBACK_MODEL_PATH
        weapon_model = YOLO(WEAPON_MODEL_PATH)
        print("=" * 60)
        print("FALLBACK YOLO11 THREAT MODEL LOADED")
        print("=" * 60)
        print(f"Model path: {WEAPON_MODEL_PATH}")
        print(f"Classes: {weapon_model.names}")
        print("=" * 60)
    else:
        print("WARNING: no weapon model found")
        print("Place best.pt beside app.py.")

except Exception as e:
    print("=" * 60)
    print("MODEL LOADING ERROR")
    print("=" * 60)
    print(str(e))
    print("=" * 60)


# ============================================================
# DETECTION HELPERS
# ============================================================

def _model_name(model_obj, cls_id):
    """Safely get a class name from a YOLO model."""
    names = model_obj.names
    if isinstance(names, dict):
        return names.get(cls_id, str(cls_id))
    return names[cls_id]


def _box_to_list(box):
    """Convert YOLO xyxy tensor to [x1, y1, x2, y2]."""
    return [float(v) for v in box.xyxy[0].tolist()]


def _iou(box_a, box_b):
    """Intersection over Union for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def _center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _center_distance_ratio(box_a, box_b):
    """Center distance normalized by the diagonal of box_a."""
    ax, ay = _center(box_a)
    bx, by = _center(box_b)
    distance = float(np.hypot(ax - bx, ay - by))
    diagonal = float(np.hypot(box_a[2] - box_a[0], box_a[3] - box_a[1]))
    return distance / max(diagonal, 1.0)


# ============================================================
# STABLE PERSON IDENTITY
# ============================================================

STABLE_ID_MAX_MISSED_FRAMES = 90
STABLE_ID_IOU_THRESHOLD = 0.15
STABLE_ID_CENTER_THRESHOLD = 0.65


class StablePersonIdentity:
    """
    One-to-one public IDs on top of ByteTrack.

    The previous implementation could assign the SAME stable ID to multiple
    people in one frame because its re-identification pass did not reserve an
    ID after it had been matched. This implementation explicitly reserves
    every public ID used in the current frame. Raw ByteTrack IDs are also
    mapped directly and remain the primary identity key.
    """

    def __init__(self):
        self.stable_tracks = {}
        self.raw_to_stable = {}
        self.next_generated_id = 1
        self.current_frame = -1
        self.used_this_frame = set()

    def reset(self):
        self.stable_tracks.clear()
        self.raw_to_stable.clear()
        self.next_generated_id = 1
        self.current_frame = -1
        self.used_this_frame.clear()

    def _new_id(self):
        while self.next_generated_id in self.stable_tracks:
            self.next_generated_id += 1
        stable_id = self.next_generated_id
        self.next_generated_id += 1
        return stable_id

    def assign(self, raw_id, bbox, frame_index):
        if frame_index != self.current_frame:
            self.current_frame = frame_index
            self.used_this_frame.clear()

        raw_id = int(raw_id) if raw_id is not None else -1

        # Existing ByteTrack identity: use its already assigned public ID.
        if raw_id >= 0:
            known = self.raw_to_stable.get(raw_id)
            if known is not None and known in self.stable_tracks:
                if known not in self.used_this_frame:
                    track = self.stable_tracks[known]
                    track.update({
                        "bbox": list(bbox),
                        "last_frame": frame_index,
                        "raw_id": raw_id,
                    })
                    self.used_this_frame.add(known)
                    return known

        # Re-identification is strictly one-to-one within a frame.
        best_id = None
        best_score = -1.0
        for stable_id, track in self.stable_tracks.items():
            if stable_id in self.used_this_frame:
                continue
            missed = frame_index - track["last_frame"]
            if missed > STABLE_ID_MAX_MISSED_FRAMES:
                continue

            previous_box = track["bbox"]
            iou = _iou(previous_box, bbox)
            center_ratio = _center_distance_ratio(previous_box, bbox)
            score = iou + max(0.0, 1.0 - center_ratio) * 0.25

            if (iou >= STABLE_ID_IOU_THRESHOLD or
                    center_ratio <= STABLE_ID_CENTER_THRESHOLD) and score > best_score:
                best_score = score
                best_id = stable_id

        if best_id is None:
            best_id = self._new_id()

        self.stable_tracks[best_id] = {
            "bbox": list(bbox),
            "last_frame": frame_index,
            "raw_id": raw_id,
        }
        if raw_id >= 0:
            self.raw_to_stable[raw_id] = best_id

        self.used_this_frame.add(best_id)

        # Remove identities that have been absent too long.
        stale = [
            sid for sid, track in self.stable_tracks.items()
            if frame_index - track["last_frame"] > STABLE_ID_MAX_MISSED_FRAMES
        ]
        for sid in stale:
            self.stable_tracks.pop(sid, None)
            for rid, mapped_sid in list(self.raw_to_stable.items()):
                if mapped_sid == sid:
                    self.raw_to_stable.pop(rid, None)

        return best_id


stable_person_identity = StablePersonIdentity()


# ============================================================
# WEAPON DETECTION / FINAL EVENT TRACKING
#
# The supplied project model best.pt is a YOLOv8s model trained specifically
# for public-safety weapons. Its documented classes include Gun, Knife and
# Handgun, which are the important classes for this CCTV footage.
#
# We READ/CAPTURE every video frame and WRITE every video frame. To keep the
# pipeline fast, the expensive weapon detector runs on every 2nd frame; the
# most recent weapon boxes are carried forward by the weapon tracker and
# rendered on the intermediate frame. High-resolution person crops are added
# periodically for small weapons.
WEAPON_CONFIDENCE = 0.25
WEAPON_INFERENCE_SIZE = 768
WEAPON_ROI_INFERENCE_SIZE = 960
WEAPON_DETECT_EVERY_N_FRAMES = 2
WEAPON_ROI_EVERY_N_FRAMES = 4
WEAPON_MAX_PERSON_ROIS = 4
WEAPON_PROGRESS_EVERY_N_FRAMES = 30
WEAPON_CONFIRM_OBSERVATIONS = 3
WEAPON_CONFIRM_WINDOW = 18
WEAPON_MIN_CONFIRM_CONFIDENCE = 0.45
WEAPON_PRIMARY_MIN_CONF = 0.25
WEAPON_FALLBACK_GUN_MIN_CONF = 0.62
WEAPON_FALLBACK_KNIFE_MIN_CONF = 0.38
WEAPON_MIN_GOOD_GEOMETRY_HITS = 2
WEAPON_MATCH_IOU = 0.10
WEAPON_MAX_FRAME_GAP = 8
WEAPON_HOLD_FRAMES = 10
CAPTURE_EVERY_N_FRAMES = 30

# The dedicated YOLO11n model is the primary weapon detector.  best.pt is
# only a fallback when the dedicated model is unavailable.
PRIMARY_WEAPON_CLASSES = {"gun", "knife"}
FALLBACK_WEAPON_CLASSES = {
    "gun","knife","handgun","automatic rifle","bazooka","grenade launcher",
    "smg","shotgun","sniper","sword","firearm","pistol","rocket","grenade"
}

def _is_weapon_class(name):
    return str(name).strip().lower() in (PRIMARY_WEAPON_CLASSES | FALLBACK_WEAPON_CLASSES)

def _canonical_weapon_class(name):
    name = str(name).strip().lower()
    if name in {"pistol", "firearm", "gun", "handgun"}:
        return "firearm"
    return name


class WeaponEventTracker:
    """Conservative temporal weapon confirmation and persistent drawing."""

    def __init__(self): self.reset()
    def reset(self):
        self.tracks={}; self.next_weapon_id=1; self.final_events=[]

    def _new_track(self,d,frame_index):
        t={"candidate_id":self.next_weapon_id,"class":str(d["class"]),
           "bbox":list(d["bbox"]),"velocity":[0,0,0,0],
           "confidence":float(d["confidence"]),"best_confidence":float(d["confidence"]),
           "person_id":d.get("associated_person_id"),"source":d.get("model_source","primary"),
           "first_frame":frame_index,"last_frame":frame_index,"hit_frames":[frame_index],
           "good_geometry_hits":1 if d.get("geometry_ok") else 0,
           "confirmed":False,"event_start_frame":None,"event_end_frame":None,
           "matched_this_frame":False,"newly_confirmed":False,
           "rel_centers":[d.get("relative_center",(0.5,0.5))]}
        self.next_weapon_id+=1; self.tracks[t["candidate_id"]]=t; return t

    def update(self,detections,frame_index):
        for t in self.tracks.values(): t["matched_this_frame"]=False; t["newly_confirmed"]=False
        confirmed=[]
        for d in detections:
            if not d.get("associated_person_id") and d.get("associated_person_id") != 0: continue
            cls=str(d["class"]).strip()
            best=None; best_score=-1
            for t in self.tracks.values():
                if t["matched_this_frame"] or t["class"].lower()!=cls.lower(): continue
                if frame_index-t["last_frame"]>WEAPON_MAX_FRAME_GAP: continue
                if t.get("person_id") is not None and d.get("associated_person_id") is not None and int(t["person_id"])!=int(d["associated_person_id"]):
                    continue
                iou=_iou(t["bbox"],d["bbox"]); cd=_center_distance_ratio(t["bbox"],d["bbox"])
                score=iou+max(0,1-cd)*0.35
                if (iou>=WEAPON_MATCH_IOU or cd<=0.45) and score>best_score:
                    best_score=score; best=t
            if best is None: best=self._new_track(d,frame_index)
            else:
                old=list(best["bbox"]); new=list(d["bbox"]); gap=max(1,frame_index-best["last_frame"])
                best["velocity"]=[(new[i]-old[i])/gap for i in range(4)]
                best["bbox"]=new; best["confidence"]=float(d["confidence"])
                best["best_confidence"]=max(best["best_confidence"],float(d["confidence"]))
                best["person_id"]=d.get("associated_person_id",best.get("person_id"))
                best["last_frame"]=frame_index; best["hit_frames"].append(frame_index)
                best["good_geometry_hits"]+=1 if d.get("geometry_ok") else 0
                best["rel_centers"].append(d.get("relative_center",(0.5,0.5)))
            best["matched_this_frame"]=True
            best["hit_frames"]=[f for f in best["hit_frames"] if frame_index-f<WEAPON_CONFIRM_WINDOW]
            # Requiring both repeated hits and geometry prevents a single
            # person/clothing hallucination from becoming an alert.
            enough=len(best["hit_frames"])>=WEAPON_CONFIRM_OBSERVATIONS
            geom=best["good_geometry_hits"]>=WEAPON_MIN_GOOD_GEOMETRY_HITS
            strong=best["best_confidence"]>=WEAPON_MIN_CONFIRM_CONFIDENCE
            if enough and geom and strong:
                if not best["confirmed"]:
                    best["confirmed"]=True; best["newly_confirmed"]=True
                    best["event_start_frame"]=best["hit_frames"][0]
                best["event_end_frame"]=frame_index
                d.update({"weapon_id":best["candidate_id"],"confirmed":True,
                          "weapon_status":"CONFIRMED","confirmation_count":len(best["hit_frames"]),
                          "associated_person_id":best.get("person_id"),
                          "newly_confirmed":best["newly_confirmed"]})
                confirmed.append(d)
        stale=[]
        for wid,t in self.tracks.items():
            if frame_index-t["last_frame"]>WEAPON_MAX_FRAME_GAP:
                if t["confirmed"]: self._finalize_track(t)
                stale.append(wid)
        for wid in stale: self.tracks.pop(wid,None)
        return confirmed

    def active_confirmed(self,frame_index):
        out=[]
        for t in self.tracks.values():
            if not t.get("confirmed") or frame_index-t["last_frame"]>WEAPON_HOLD_FRAMES: continue
            gap=frame_index-t["last_frame"]; v=t["velocity"]
            pred=[t["bbox"][i]+v[i]*min(gap,WEAPON_HOLD_FRAMES) for i in range(4)]
            out.append({"source":"weapon","class":t["class"],"class_id":-1,
                        "confidence":float(t["confidence"]),"bbox":pred,"track_id":-1,
                        "weapon_id":t["candidate_id"],"associated_person_id":t.get("person_id"),
                        "confirmed":True,"weapon_status":"CONFIRMED",
                        "newly_confirmed":bool(t.get("newly_confirmed",False))})
        return out

    def _finalize_track(self,t):
        if not t.get("confirmed") or t.get("event_start_frame") is None: return
        if any(e["weapon_id"]==t["candidate_id"] for e in self.final_events): return
        self.final_events.append({"weapon_id":t["candidate_id"],"type":str(t["class"]).upper(),
            "confidence":float(t["best_confidence"]),"person_id":t.get("person_id"),
            "start_frame":int(t["event_start_frame"]),"end_frame":int(t.get("event_end_frame",t["last_frame"]))})

    def finalize_all(self,final_frame_index):
        for t in list(self.tracks.values()):
            if t.get("confirmed"):
                t["event_end_frame"]=max(t.get("event_end_frame",t["last_frame"]),t["last_frame"],
                                          final_frame_index if final_frame_index-t["last_frame"]<=WEAPON_HOLD_FRAMES else t["last_frame"])
                self._finalize_track(t)
        self.tracks.clear()
        return sorted(self.final_events,key=lambda e:e["weapon_id"])

weapon_event_tracker = WeaponEventTracker()
safety_event_pipeline = SafetyEventPipeline()
agentic_supervisor = AgenticSupervisor()


def _weapon_detection_to_structured_event(detection, frame_index, fps):
    """Convert a newly-confirmed weapon detection into the same structured
    event schema the safety detectors use, so the Event Correlator and
    Agentic layer can treat weapon and safety events uniformly. Built here
    (with frame_index/fps already in scope) rather than trusting the
    tracker's own bookkeeping, since WeaponEventTracker.active_confirmed()
    does not carry frame/timestamp fields."""
    fps = max(float(fps), 1.0)
    return {
        "event_type": "WEAPON_CONFIRMED",
        "severity": "CRITICAL",
        "person_id": detection.get("associated_person_id"),
        "confidence": float(detection.get("confidence", 0.5)),
        "frame": int(frame_index),
        "timestamp": frame_index / fps,
        "details": {
            "weapon_id": detection.get("weapon_id"),
            "weapon_type": str(detection.get("class", "weapon")),
        },
    }


_agentic_error_count = 0


def _safe_agentic_ingest(event):
    """Feed one event to the Agentic AI layer. The agentic layer is an
    analysis add-on: if it ever fails, print the full traceback (once) and
    keep processing the video instead of aborting the whole request."""
    global _agentic_error_count
    try:
        return agentic_supervisor.ingest_event(event)
    except Exception as exc:
        _agentic_error_count += 1
        if _agentic_error_count == 1:
            print(f"\n[agentic] ingest_event raised {exc!r} for event {event!r} -- traceback:", file=sys.stderr)
            traceback.print_exc()
        return None


def _weapon_class_name(name):
    return str(name).strip().lower()


def _associate_weapon_with_person(weapon_detection, person_detections):
    if not person_detections: return None
    wx,wy=_center(weapon_detection["bbox"])
    best=None; best_score=-1
    for person in person_detections:
        px1,py1,px2,py2=map(float,person["bbox"])
        pw=max(1.0,px2-px1); ph=max(1.0,py2-py1)
        expanded=[px1-pw*0.18,py1-ph*0.12,px2+pw*0.18,py2+ph*0.12]
        inside=expanded[0]<=wx<=expanded[2] and expanded[1]<=wy<=expanded[3]
        dist=_center_distance_ratio(person["bbox"],weapon_detection["bbox"])
        score=(4.0 if inside else 0.0)+max(0,1-dist)
        if inside and score>best_score:
            best_score=score; best=person
    return best.get("stable_id") if best is not None else None

def _run_general_detection(frame, confidence=0.30, tracking=False):
    """Run general YOLO person detection, optionally with ByteTrack."""
    if general_model is None:
        return []
    common = {
        "source": frame,
        "imgsz": 640,
        "conf": float(confidence),
        "classes": [0],
        "verbose": False,
    }
    if tracking:
        results = general_model.track(
            tracker="bytetrack.yaml",
            persist=True,
            **common,
        )
    else:
        results = general_model.predict(**common)
    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "source": "general",
                "class": _model_name(general_model, cls_id),
                "class_id": cls_id,
                "confidence": float(box.conf[0]),
                "bbox": _box_to_list(box),
                "track_id": int(box.id[0]) if tracking and box.id is not None else -1,
            })
    return detections

def _weapon_geometry(person, weapon_bbox):
    """Validate that a weapon candidate is plausibly attached to a person.

    Returns (ok, relative_center).  This is a conservative geometric gate,
    not a classifier: the dedicated weapon model remains responsible for the
    actual class decision.
    """
    if person is None:
        return False, (0.5, 0.5)
    px1, py1, px2, py2 = map(float, person["bbox"])
    wx1, wy1, wx2, wy2 = map(float, weapon_bbox)
    pw = max(1.0, px2 - px1)
    ph = max(1.0, py2 - py1)
    wcx = (wx1 + wx2) * 0.5
    wcy = (wy1 + wy2) * 0.5
    rel_x = (wcx - px1) / pw
    rel_y = (wcy - py1) / ph

    # Candidate center must be inside a modestly expanded person box and
    # should be below the head/upper-torso area where weapon-like objects are
    # normally held.  This rejects many signs, bags and background objects.
    expanded = [px1 - 0.10 * pw, py1 - 0.05 * ph,
                px2 + 0.10 * pw, py2 + 0.10 * ph]
    inside = (expanded[0] <= wcx <= expanded[2] and
              expanded[1] <= wcy <= expanded[3])

    weapon_area = max(0.0, wx2 - wx1) * max(0.0, wy2 - wy1)
    person_area = pw * ph
    area_ratio = weapon_area / max(person_area, 1.0)

    # A very large candidate is unlikely to be a handheld weapon.
    size_ok = 0.00015 <= area_ratio <= WEAPON_MAX_PERSON_AREA_RATIO
    vertical_ok = 0.22 <= rel_y <= 1.05
    horizontal_ok = -0.10 <= rel_x <= 1.10

    return bool(inside and size_ok and vertical_ok and horizontal_ok), (rel_x, rel_y)

def _run_weapon_detection(frame, confidence=WEAPON_CONFIDENCE, person_detections=None, frame_index=0):
    """Primary: dedicated gun/knife model. Fallback: best.pt only if needed."""
    if weapon_model is None: return []
    sources=[frame]; offsets=[(0,0)]; sizes=[WEAPON_INFERENCE_SIZE]
    if person_detections and frame_index % WEAPON_ROI_EVERY_N_FRAMES == 0:
        h,w=frame.shape[:2]
        people=sorted(person_detections,key=lambda d:(d["bbox"][2]-d["bbox"][0])*(d["bbox"][3]-d["bbox"][1]),reverse=True)[:WEAPON_MAX_PERSON_ROIS]
        for person in people:
            x1,y1,x2,y2=map(int,person["bbox"]); pw=x2-x1; ph=y2-y1
            # Focus on upper/lower arm region but keep enough context for the detector.
            pad_x=int(pw*0.35); pad_y=int(ph*0.10)
            cx1,cy1=max(0,x1-pad_x),max(0,y1-pad_y); cx2,cy2=min(w,x2+pad_x),min(h,y2+pad_y)
            crop=frame[cy1:cy2,cx1:cx2]
            if crop.size and crop.shape[1]>=80 and crop.shape[0]>=80:
                sources.append(crop); offsets.append((cx1,cy1)); sizes.append(WEAPON_ROI_INFERENCE_SIZE)

    detections=[]
    is_primary = os.path.basename(WEAPON_MODEL_PATH).lower().startswith("gun_knife")
    allowed=PRIMARY_WEAPON_CLASSES if is_primary else FALLBACK_WEAPON_CLASSES
    for source,(ox,oy),size in zip(sources,offsets,sizes):
        results=weapon_model.predict(source=source,imgsz=size,conf=confidence,iou=0.45,max_det=30,verbose=False)
        for result in results:
            if result.boxes is None: continue
            for box in result.boxes:
                cls_name=_model_name(weapon_model,int(box.cls[0]))
                key=str(cls_name).strip().lower()
                if key not in allowed: continue
                conf_value=float(box.conf[0])
                if is_primary:
                    if conf_value < WEAPON_PRIMARY_MIN_CONF: continue
                else:
                    minc=WEAPON_FALLBACK_KNIFE_MIN_CONF if key=="knife" else WEAPON_FALLBACK_GUN_MIN_CONF
                    if conf_value < minc: continue
                b=_box_to_list(box); h,w=frame.shape[:2]
                bbox=[max(0,min(w-1,b[0]+ox)),max(0,min(h-1,b[1]+oy)),
                      max(0,min(w-1,b[2]+ox)),max(0,min(h-1,b[3]+oy))]
                if bbox[2]<=bbox[0] or bbox[3]<=bbox[1]: continue
                # Association is mandatory; no free-floating weapon alert.
                assoc=_associate_weapon_with_person({"bbox":bbox},person_detections or [])
                if assoc is None: continue
                person=next((p for p in person_detections if p.get("stable_id")==assoc),None)
                geom_ok,rel=_weapon_geometry(person,bbox)
                # Primary model can keep a candidate with weak geometry for
                # one frame, but confirmation requires good geometry hits.
                detections.append({"source":"weapon","class":cls_name,"class_id":int(box.cls[0]),
                    "confidence":conf_value,"bbox":bbox,"track_id":-1,
                    "associated_person_id":assoc,"geometry_ok":geom_ok,
                    "relative_center":rel,"model_source":"primary" if is_primary else "fallback"})
    # Class-aware NMS
    kept=[]
    for d in sorted(detections,key=lambda x:x["confidence"],reverse=True):
        if any(str(d["class"]).lower()==str(k["class"]).lower() and _iou(d["bbox"],k["bbox"])>=0.45 for k in kept):
            continue
        kept.append(d)
    return kept

def reset_tracking_state():
    """Reset all state at the start of a new video/webcam session."""
    stable_person_identity.reset()
    weapon_event_tracker.reset()
    safety_event_pipeline.reset()
    agentic_supervisor.reset()

    # Ultralytics stores tracker state in the model predictor when persist=True.
    # Resetting predictor forces a fresh tracker for the next session.
    if general_model is not None:
        try:
            general_model.predictor = None
        except Exception:
            pass


def run_combined_detection(frame, confidence=0.30):
    """
    Run General YOLO + dedicated gun/knife YOLO on one image.

    This endpoint is single-frame detection, so temporal confirmation is not
    applied. Video/webcam processing uses process_video_frame(), which does
    temporal confirmation.
    """
    detections = _run_general_detection(frame, confidence, tracking=False)
    detections.extend(_run_weapon_detection(frame, WEAPON_CONFIDENCE))
    return detections


def draw_detection(frame, detection, track_id=None):
    """Draw a detection with clear person/weapon status labels."""
    x1, y1, x2, y2 = map(int, detection["bbox"])
    source = detection["source"]
    cls_name = str(detection["class"])
    conf = detection["confidence"]

    if source == "weapon":
        # Only confirmed weapon detections are drawn on the processed video.
        if not detection.get("confirmed", False):
            return

        color = (0, 0, 255)
        weapon_id = detection.get("weapon_id")
        person_id = detection.get("associated_person_id")

        if weapon_id is not None:
            weapon_label = f"WEAPON-{int(weapon_id):03d}"
        else:
            weapon_label = "WEAPON"

        if person_id is not None:
            label = (
                f"ALERT {weapon_label} {cls_name.upper()} {conf:.2f}"
                f" -> PERSON-{int(person_id):03d}"
            )
        else:
            label = f"ALERT {weapon_label} {cls_name.upper()} {conf:.2f}"
    else:
        color = (0, 255, 0)
        public_id = detection.get("stable_id")
        is_threat = detection.get("is_threat", False)
        is_safety_event = detection.get("is_safety_event", False)
        safety_types = detection.get("safety_event_types", [])

        if cls_name.lower() == "person" and public_id is not None:
            raw_id = detection.get("track_id", -1)
            if is_threat:
                color = (0, 0, 255)  # red — linked to a CONFIRMED weapon
                label = f"ALERT PERSON-{int(public_id):03d} ARMED {conf:.2f}"
            elif is_safety_event:
                color = (0, 0, 255)
                types = ",".join(safety_types[:2]) if safety_types else "SAFETY"
                label = f"ALERT PERSON-{int(public_id):03d} {types} {conf:.2f}"
            else:
                label = f"PERSON-{int(public_id):03d} BT:{raw_id} {conf:.2f}"
        elif track_id is not None and track_id >= 0:
            label = f"{cls_name} #{track_id} {conf:.2f}"
        else:
            label = f"{cls_name} {conf:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Keep the label inside the frame where possible.
    text_y = max(y1 - 10, 20)
    cv2.putText(
        frame,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_combined_detections(frame, detections):
    """Draw detections returned by run_combined_detection()."""
    for detection in detections:
        draw_detection(frame, detection, detection.get("track_id", -1))
    return frame


# ============================================================
# TRACKED VIDEO FRAME PROCESSING
# ============================================================

def process_video_frame(frame, confidence=0.30, frame_index=0, fps=25.0):
    """Read every frame, track people every frame, and detect weapons frequently."""
    general_detections = _run_general_detection(frame, confidence=confidence, tracking=True)

    person_detections = []
    for detection in general_detections:
        if detection["class"].lower() == "person":
            stable_id = stable_person_identity.assign(
                detection.get("track_id", -1), detection["bbox"], frame_index
            )
            detection["stable_id"] = stable_id
            person_detections.append(detection)

    # Expensive weapon inference is sampled, while the weapon tracker carries
    # the last confirmed box through the intermediate frames.
    if frame_index % WEAPON_DETECT_EVERY_N_FRAMES == 0:
        weapon_detections = _run_weapon_detection(
            frame, WEAPON_CONFIDENCE, person_detections, frame_index
        )
        for weapon in weapon_detections:
            associated = _associate_weapon_with_person(weapon, person_detections)
            if associated is not None:
                weapon["associated_person_id"] = associated
        newly_confirmed_weapons = weapon_event_tracker.update(weapon_detections, frame_index)
        for weapon in newly_confirmed_weapons:
            if weapon.get("newly_confirmed"):
                _safe_agentic_ingest(
                    _weapon_detection_to_structured_event(weapon, frame_index, fps)
                )

    active_weapon_detections = weapon_event_tracker.active_confirmed(frame_index)

    # --------------------------------------------------------
    # SAFETY EVENT ANALYSIS (after stable person tracking)
    # --------------------------------------------------------
    safety_result = safety_event_pipeline.update(
        person_detections, frame_index=frame_index, fps=fps, frame_shape=frame.shape, frame=frame
    )

    # --------------------------------------------------------
    # AGENTIC AI LAYER: feed each structured event as it happens so the
    # Event Correlator / Risk Agent / Response Agent stay live, and surface
    # the worst currently-open incident for the on-frame banner.
    # --------------------------------------------------------
    touched_incidents = []
    for event in safety_result.get("events", []):
        incident = _safe_agentic_ingest(event)
        if incident is not None:
            touched_incidents.append(incident)
    safety_result["active_incident"] = None
    if touched_incidents:
        severity_rank = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}
        worst = max(touched_incidents, key=lambda inc: severity_rank.get(inc.get("risk_severity"), -1))
        safety_result["active_incident"] = worst

    alert_person_ids = set(safety_result.get("alert_person_ids", []))
    alert_types_by_person = safety_result.get("alert_types_by_person", {})
    for person in person_detections:
        pid = person.get("stable_id")
        person["is_safety_event"] = pid in alert_person_ids
        person["safety_event_types"] = alert_types_by_person.get(pid, [])

    # Update association using the currently tracked weapon position.
    for weapon in active_weapon_detections:
        associated = _associate_weapon_with_person(weapon, person_detections)
        if associated is not None:
            weapon["associated_person_id"] = associated
        track = weapon_event_tracker.tracks.get(weapon.get("weapon_id"))
        if track is not None and weapon.get("associated_person_id") is not None:
            track["person_id"] = weapon["associated_person_id"]

    threat_person_ids = {
        d.get("associated_person_id") for d in active_weapon_detections
        if d.get("associated_person_id") is not None
    }
    for detection in person_detections:
        detection["is_threat"] = detection.get("stable_id") in threat_person_ids

    combined = general_detections + active_weapon_detections
    people_count = len(person_detections)
    weapon_count = len(active_weapon_detections)

    for detection in combined:
        draw_detection(frame, detection, detection.get("track_id", -1))

    safety_event_pipeline.draw(frame, safety_result)
    _draw_incident_banner(frame, safety_result.get("active_incident"))

    return frame, combined, people_count, weapon_count, safety_result


def _draw_incident_banner(frame, incident):
    """Draw a full-width banner when the Agentic AI layer's live risk
    assessment for the currently active incident reaches HIGH/CRITICAL --
    this is the human-facing signal that several correlated signals (not
    just one detector) point to a real unfolding situation."""
    if not incident or incident.get("risk_severity") not in ("HIGH", "CRITICAL"):
        return
    is_critical = incident["risk_severity"] == "CRITICAL"
    color = (0, 0, 255) if is_critical else (0, 140, 255)
    h, w = frame.shape[:2]
    bar_h = 34
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    types = "+".join(incident.get("event_types", []))
    text = (
        f"AGENTIC AI: {incident['risk_severity']} INCIDENT {incident.get('incident_id', '')} "
        f"risk={incident.get('risk_score', 0):.0f}/100 [{types}]"
    )
    cv2.putText(frame, text, (10, bar_h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (255, 255, 255), 2, cv2.LINE_AA)


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

UPLOAD_FOLDER = os.path.join(
    REPO_ROOT,
    "uploads"
)

OUTPUT_FOLDER = os.path.join(
    REPO_ROOT,
    "outputs"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# STATIC / HTML ROUTES
# ============================================================

@app.route("/style.css")
def serve_style_css():

    return send_from_directory(
        app.static_folder,
        "style.css"
    )


@app.route("/", methods=["GET"])
def index():

    return render_template(
        "index.html"
    )


@app.route("/about.html", methods=["GET"])
def about():

    return render_template(
        "about.html"
    )


@app.route("/contact")
def contact():

    return render_template(
        "contact.html"
    )


@app.route("/results.html")
def results_partial():

    return render_template(
        "results.html"
    )


@app.route("/outputs/<filename>")
def serve_output(filename):

    return send_from_directory(
        OUTPUT_FOLDER,
        filename
    )


@app.route("/outputs/<path:filename>")
def serve_nested_output(filename):
    """Serve captured evidence frames and reports stored under outputs/."""
    return send_from_directory(OUTPUT_FOLDER, filename)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/safety/config", methods=["GET", "POST"])
def safety_config():
    """Read or update normalized restricted/loitering polygons at runtime."""
    try:
        if request.method == "POST":
            data = request.get_json(silent=True) or {}
            safety_event_pipeline.set_zones(
                restricted_zone=data.get("restricted_zone"),
                loitering_zone=data.get("loitering_zone"),
            )

        return jsonify({
            "restricted_zone": safety_event_pipeline.restricted_zone_norm,
            "loitering_zone": safety_event_pipeline.loitering_zone_norm,
            "coordinate_system": "normalized_0_to_1",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "server": "running",
        "general_model_loaded": general_model is not None,
        "weapon_model_loaded": weapon_model is not None,
        "general_model_path": GENERAL_MODEL_PATH if general_model is not None else None,
        "weapon_model_path": WEAPON_MODEL_PATH if weapon_model is not None else None,
        "general_classes": general_model.names if general_model is not None else None,
        "weapon_classes": weapon_model.names if weapon_model is not None else None
    }), 200


# ============================================================
# IMAGE DETECTION
# ============================================================

@app.route("/detect", methods=["POST"])
def detect():
    try:
        print("\n" + "=" * 60)
        print("NEW COMBINED DETECTION REQUEST")
        print("=" * 60)

        if "image" not in request.files:
            return jsonify({
                "error": "No file part. Please send multipart/form-data with field 'image'."
            }), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({"error": "Empty filename."}), 400

        if general_model is None and weapon_model is None:
            return jsonify({
                "error": "No detection model is loaded."
            }), 500

        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        try:
            confidence_threshold = float(
                request.form.get("confidence", 0.30)
            )
        except ValueError:
            confidence_threshold = 0.30

        confidence_threshold = max(
            0.05,
            min(confidence_threshold, 0.95)
        )

        img = cv2.imread(save_path)

        if img is None:
            return jsonify({
                "error": f"Could not read uploaded image: {filename}"
            }), 400

        detections = run_combined_detection(
            img,
            confidence_threshold
        )

        # Draw combined results.
        for detection in detections:
            draw_detection(img, detection)

        output_filename = "annotated_" + filename
        output_path = os.path.join(
            OUTPUT_FOLDER,
            output_filename
        )

        success = cv2.imwrite(
            output_path,
            img
        )

        if not success:
            return jsonify({
                "error": "Could not save annotated image."
            }), 500

        csv_url = None

        if detections:
            df = pd.DataFrame(detections)

            csv_filename = (
                "predictions_"
                + filename.rsplit(".", 1)[0]
                + ".csv"
            )

            csv_path = os.path.join(
                OUTPUT_FOLDER,
                csv_filename
            )

            df.to_csv(
                csv_path,
                index=False
            )

            csv_url = f"/outputs/{csv_filename}"

        return jsonify({
            "detections": detections,
            "count": len(detections),
            "filename": filename,
            "annotated_image": f"/outputs/{output_filename}",
            "csv_url": csv_url
        }), 200

    except Exception as e:
        print("\n" + "=" * 60)
        print("DETECTION ERROR")
        print(repr(e))
        print("=" * 60)

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# VIDEO DETECTION + GENERAL YOLO TRACKING + CUSTOM WEAPON YOLO
# ============================================================

ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}


def allowed_video(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_VIDEO_EXTENSIONS
    )


def get_accurate_fps(video_path, cap_fps, total_frames):
    """
    Cross-check OpenCV FPS against ffmpeg duration.
    This helps with phone/WhatsApp videos where FPS metadata
    can be unreliable.
    """
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        result = subprocess.run(
            [ffmpeg_exe, "-i", video_path],
            capture_output=True,
            text=True
        )

        stderr = result.stderr

        match = re.search(
            r"Duration:\s*(\d+):(\d+):(\d+\.\d+)",
            stderr
        )

        if match:
            hours, minutes, seconds = match.groups()

            duration_sec = (
                int(hours) * 3600
                + int(minutes) * 60
                + float(seconds)
            )

            if duration_sec > 0 and total_frames > 0:

                computed_fps = (
                    total_frames / duration_sec
                )

                if 1 <= computed_fps <= 120:
                    print(
                        f"Real duration: {duration_sec:.2f}s "
                        f"-> corrected fps: {computed_fps:.2f}"
                    )

                    return computed_fps

    except Exception as e:
        print(
            "Could not compute accurate fps via ffmpeg:",
            repr(e)
        )

    if cap_fps and 1 <= cap_fps <= 120:
        return cap_fps

    return 25.0


@app.route("/detect_video", methods=["POST"])
def detect_video():

    try:

        print("\n" + "=" * 60)
        print("NEW COMBINED VIDEO DETECTION REQUEST")
        print("=" * 60)

        if "video" not in request.files:
            return jsonify({
                "error":
                "No file part. Send multipart/form-data with field 'video'."
            }), 400

        file = request.files["video"]

        if (
            file.filename == ""
            or not allowed_video(file.filename)
        ):
            return jsonify({
                "error":
                "Invalid or empty video file. Use mp4, avi, mov, or mkv."
            }), 400

        if general_model is None and weapon_model is None:
            return jsonify({
                "error":
                "No detection model is loaded."
            }), 500

        filename = secure_filename(file.filename)

        input_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        file.save(input_path)

        print(
            f"Saved video: {input_path}"
        )

        try:
            confidence_threshold = float(
                request.form.get(
                    "confidence",
                    0.30
                )
            )
        except ValueError:
            confidence_threshold = 0.30

        confidence_threshold = max(
            0.05,
            min(confidence_threshold, 0.95)
        )

        # ----------------------------------------------------
        # VIDEO INFORMATION
        # ----------------------------------------------------

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            return jsonify({
                "error":
                f"Could not open video: {filename}"
            }), 400

        raw_fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        cap.release()

        fps = get_accurate_fps(
            input_path,
            raw_fps,
            total_frames
        )

        print(
            f"Video info: "
            f"{width}x{height} @ {fps:.2f}fps, "
            f"{total_frames} frames"
        )

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        timestamp = str(
            int(time.time())
        )

        base_name = os.path.splitext(
            filename
        )[0]

        output_filename = (
            "tracked_"
            + timestamp
            + "_"
            + base_name
            + ".mp4"
        )

        output_path = os.path.join(
            OUTPUT_FOLDER,
            output_filename
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        if not writer.isOpened():
            return jsonify({
                "error":
                "Could not create output video writer."
            }), 500

        max_people = 0
        max_weapons = 0
        frame_idx = 0
        frame_events = []
        progress_step = 30

        capture_dir = os.path.join(OUTPUT_FOLDER, f"captured_frames_{timestamp}")
        os.makedirs(capture_dir, exist_ok=True)

        # ----------------------------------------------------
        # IMPORTANT:
        # Reset ByteTrack state before processing this video.
        # ----------------------------------------------------

        reset_tracking_state()

        print(
            "Running Person Tracking every frame + project weapon model "
            f"every {WEAPON_DETECT_EVERY_N_FRAMES} frames + high-resolution person ROIs..."
        )

        cap = cv2.VideoCapture(
            input_path
        )

        try:

            while True:

                success, frame = cap.read()

                if not success:
                    break

                # --------------------------------------------
                # PROCESS FRAME
                # --------------------------------------------

                (
                    annotated_frame,
                    detections,
                    people_count,
                    weapon_count,
                    safety_result
                ) = process_video_frame(
                    frame,
                    confidence_threshold,
                    frame_index=frame_idx,
                    fps=fps,
                )

                # --------------------------------------------
                # STATISTICS
                # --------------------------------------------

                max_people = max(
                    max_people,
                    people_count
                )

                max_weapons = max(
                    max_weapons,
                    weapon_count
                )

                # No detection details are printed here.
                # The terminal shows only 30-frame progress and the final
                # weapon/person report after the complete video.

                frame_events.append({
                    "frame": frame_idx,
                    "people_count": people_count,
                    "weapon_count": weapon_count,
                    "detections": len(detections),
                    "safety_events": [e.get("event_type") for e in safety_result.get("events", [])],
                    "crowd_count": safety_result.get("crowd_count", people_count),
                    "crowd_density": safety_result.get("crowd_density", 0.0),
                })

                for event in safety_result.get("events", []):
                    event_type = str(event.get("event_type", "EVENT"))
                    pid = event.get("person_id")
                    pid_text = f"PERSON-{int(pid):03d}" if pid is not None else "ALL"
                    safe_type = re.sub(r"[^A-Za-z0-9_-]+", "_", event_type)
                    cv2.imwrite(
                        os.path.join(capture_dir, f"evidence_{safe_type}_{pid_text}_frame_{frame_idx:06d}.jpg"),
                        annotated_frame
                    )

                # --------------------------------------------
                # WRITE EVERY FRAME
                # --------------------------------------------

                writer.write(
                    annotated_frame
                )

                # Capture representative video frames for the final result.
                if frame_idx % CAPTURE_EVERY_N_FRAMES == 0:
                    cv2.imwrite(
                        os.path.join(capture_dir, f"frame_{frame_idx:06d}.jpg"),
                        annotated_frame
                    )

                # Save one evidence frame when a weapon becomes confirmed.
                for detection in detections:
                    if detection.get("source") == "weapon" and detection.get("newly_confirmed"):
                        wid = int(detection.get("weapon_id", 0))
                        cv2.imwrite(
                            os.path.join(capture_dir, f"evidence_WEAPON-{wid:03d}_frame_{frame_idx:06d}.jpg"),
                            annotated_frame
                        )

                frame_idx += 1

                if frame_idx % WEAPON_PROGRESS_EVERY_N_FRAMES == 0:
                    print(
                        f"Processed {frame_idx}/"
                        f"{total_frames} frames..."
                    )

        finally:

            cap.release()

            # IMPORTANT:
            # Release only AFTER ALL frames are written.
            writer.release()

        # Finalize all confirmed weapon events only after the complete video.
        final_weapon_events = weapon_event_tracker.finalize_all(
            max(0, frame_idx - 1)
        )

        print("\n" + "=" * 70)
        print("FINAL WEAPON DETECTION REPORT")
        print("=" * 70)

        if final_weapon_events:
            for event in final_weapon_events:
                start_seconds = event["start_frame"] / max(fps, 1.0)
                end_seconds = event["end_frame"] / max(fps, 1.0)
                duration = max(0.0, end_seconds - start_seconds)
                person_text = (
                    f"PERSON-{int(event['person_id']):03d}"
                    if event.get("person_id") is not None
                    else "UNASSIGNED"
                )

                print(
                    f"[WEAPON FINAL] "
                    f"{start_seconds:07.2f}s -> {end_seconds:07.2f}s "
                    f"(frames {event['start_frame']}-{event['end_frame']})"
                )
                print(
                    f"WEAPON ID   : WEAPON-{int(event['weapon_id']):03d}"
                )
                print(
                    f"TYPE        : {event['type']}"
                )
                print(
                    f"CONFIDENCE  : {event['confidence']:.2f}"
                )
                print(
                    f"PERSON ID   : {person_text}"
                )
                print(
                    f"DURATION    : {duration:.2f}s"
                )
                print(
                    f"STATUS      : FINAL ALERT"
                )
                print("-" * 70)
        else:
            print("NO CONFIRMED WEAPON DETECTED")

        final_safety_events = safety_event_pipeline.report(fps)

        print("\n" + "=" * 70)
        print("FINAL SAFETY EVENT REPORT")
        print("=" * 70)
        if final_safety_events:
            for event in final_safety_events:
                pid = event.get("person_id")
                person_text = f"PERSON-{int(pid):03d}" if pid is not None else "ALL"
                print(
                    f"[{event.get('event_type')}] {event.get('time_seconds', 0.0):07.2f}s | "
                    f"{person_text} | severity={event.get('severity', 'MEDIUM')} | "
                    f"confidence={float(event.get('confidence', 0.0)):.2f}"
                )
                if event.get("details"):
                    print(f"DETAILS     : {event['details']}")
                print("-" * 70)
        else:
            print("NO CROWD / LOITERING / ZONE / MOVEMENT / FALL EVENTS DETECTED")

        # Close out every remaining open incident now that the full video
        # (and every structured event it produced) has been processed.
        final_incidents = agentic_supervisor.finalize()

        print("\n" + "=" * 70)
        print("FINAL AGENTIC AI INCIDENT REPORT")
        print("=" * 70)
        if final_incidents:
            for incident in final_incidents:
                persons_text = (
                    ",".join(f"PERSON-{pid:03d}" for pid in incident["person_ids"])
                    if incident["person_ids"] else "SCENE-WIDE"
                )
                print(
                    f"[{incident['incident_id']}] {incident['start_time']:07.2f}s -> "
                    f"{incident['end_time']:07.2f}s | {persons_text}"
                )
                print(f"EVENT TYPES : {'+'.join(incident['event_types'])} ({incident['event_count']} events)")
                print(f"RISK        : {incident['risk_severity']} ({incident['risk_score']}/100)")
                print(f"EXPLANATION : {incident['risk_explanation']}")
                print(f"RESPONSE    : {', '.join(incident['response_action_labels'])}")
                print("-" * 70)
        else:
            print("NO CORRELATED INCIDENTS -- no events crossed the correlation window together")

        print("=" * 70)
        print(
            f"Video processing complete: "
            f"{frame_idx} frames written"
        )
        print(f"Captured frames: {capture_dir}")

        report_csv = os.path.join(capture_dir, "final_weapon_report.csv")
        pd.DataFrame(final_weapon_events).to_csv(report_csv, index=False) if final_weapon_events else pd.DataFrame(columns=[
            "weapon_id", "type", "confidence", "person_id", "start_frame", "end_frame"
        ]).to_csv(report_csv, index=False)

        safety_report_csv = os.path.join(capture_dir, "final_safety_event_report.csv")
        pd.DataFrame(final_safety_events).to_csv(safety_report_csv, index=False)

        incident_report_csv = os.path.join(capture_dir, "final_incident_report.csv")
        if final_incidents:
            incident_rows = [{
                "incident_id": i["incident_id"],
                "start_time": i["start_time"],
                "end_time": i["end_time"],
                "duration_seconds": i["duration_seconds"],
                "person_ids": ",".join(str(p) for p in i["person_ids"]) or "SCENE-WIDE",
                "event_types": "+".join(i["event_types"]),
                "event_count": i["event_count"],
                "risk_score": i["risk_score"],
                "risk_severity": i["risk_severity"],
                "response_actions": ",".join(i["response_actions"]),
            } for i in final_incidents]
            pd.DataFrame(incident_rows).to_csv(incident_report_csv, index=False)
        else:
            pd.DataFrame(columns=[
                "incident_id", "start_time", "end_time", "duration_seconds", "person_ids",
                "event_types", "event_count", "risk_score", "risk_severity", "response_actions"
            ]).to_csv(incident_report_csv, index=False)

        # ----------------------------------------------------
        # RE-ENCODE TO H.264 FOR BROWSER PLAYBACK
        # ----------------------------------------------------

        final_output_filename = (
            "tracked_"
            + timestamp
            + "_"
            + base_name
            + "_h264.mp4"
        )

        final_output_path = os.path.join(
            OUTPUT_FOLDER,
            final_output_filename
        )

        served_filename = output_filename

        try:

            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

            print(
                "Re-encoding to H.264 "
                "for browser playback..."
            )

            subprocess.run(
                [
                    ffmpeg_exe,
                    "-y",
                    "-i",
                    output_path,
                    "-vcodec",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    final_output_path
                ],
                check=True,
                capture_output=True
            )

            os.remove(
                output_path
            )

            print(
                "Re-encoding complete"
            )

            served_filename = (
                final_output_filename
            )

        except Exception as ffmpeg_error:

            print(
                "FFMPEG RE-ENCODE FAILED:",
                repr(ffmpeg_error)
            )

            print(
                "Falling back to original mp4v file."
            )

        return jsonify({

            "output_video":
            f"/outputs/{served_filename}",

            "frames_processed":
            frame_idx,

            "input_total_frames":
            total_frames,

            "fps":
            fps,

            "estimated_duration_seconds":
            (
                frame_idx / fps
                if fps > 0
                else None
            ),

            "max_people_count":
            max_people,

            "max_weapon_count":
            max_weapons,

            "weapon_events":
            final_weapon_events,

            "safety_events":
            final_safety_events,

            "incidents":
            final_incidents,

            "captured_frames":
            f"/outputs/{os.path.basename(capture_dir)}",

            "safety_report":
            f"/outputs/{os.path.basename(capture_dir)}/final_safety_event_report.csv",

            "incident_report":
            f"/outputs/{os.path.basename(capture_dir)}/final_incident_report.csv",

            "detector_errors":
            dict(safety_event_pipeline.detector_errors),

            "restricted_zone_normalized":
            safety_event_pipeline.restricted_zone_norm,

            "loitering_zone_normalized":
            safety_event_pipeline.loitering_zone_norm

        }), 200

    except Exception as e:

        # Always show WHERE it failed -- repr(e) alone hides the file/line.
        tb = traceback.format_exc()
        print("\n" + "=" * 60)
        print("VIDEO DETECTION ERROR")
        print(repr(e))
        print(tb)
        print("=" * 60)

        frames = traceback.extract_tb(e.__traceback__)
        where = f"{os.path.basename(frames[-1].filename)}:{frames[-1].lineno} in {frames[-1].name}" if frames else None
        return jsonify({
            "error": str(e),
            "error_type": type(e).__name__,
            "error_location": where,
        }), 500


# ============================================================
# LIVE WEBCAM DETECTION
# ============================================================

def generate_frames():
    """Generate annotated webcam frames using the same video pipeline."""
    camera_index = int(os.environ.get("CAMERA_INDEX", 0))
    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        raise RuntimeError(f"Could not open webcam index {camera_index}.")

    print(f"Webcam started: index {camera_index}")
    reset_tracking_state()

    confidence = 0.30
    frame_index = 0

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            if general_model is not None or weapon_model is not None:
                annotated_frame, _, _, _, _ = process_video_frame(
                    frame,
                    confidence=confidence,
                    frame_index=frame_index,
                    fps=30.0,
                )
                frame = annotated_frame

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                frame_index += 1
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

            frame_index += 1
            time.sleep(0.03)

    finally:
        camera.release()
        print("Webcam released")


@app.route("/live")
def live():

    if (
        general_model is None
        and weapon_model is None
    ):

        return jsonify({
            "error":
            "No detection model is loaded."
        }), 500

    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# LIVE FRAME DETECTION
# ============================================================

@app.route(
    "/detect_live_frame",
    methods=["POST"]
)
def detect_live_frame():

    try:

        data = request.get_json()

        if not data or "frame" not in data:
            return jsonify({
                "error":
                "No frame data"
            }), 400

        if (
            general_model is None
            and weapon_model is None
        ):
            return jsonify({
                "error":
                "No detection model is loaded."
            }), 500

        frame_data = data["frame"]

        confidence = float(
            data.get(
                "confidence",
                0.30
            )
        )

        confidence = max(
            0.05,
            min(confidence, 0.95)
        )

        # ----------------------------------------------------
        # Base64 -> Image
        # ----------------------------------------------------

        if "," in frame_data:
            frame_data = frame_data.split(
                ",",
                1
            )[1]

        image_data = base64.b64decode(
            frame_data
        )

        image_pil = Image.open(
            io.BytesIO(
                image_data
            )
        )

        image_np = np.array(
            image_pil
        )

        frame = cv2.cvtColor(
            image_np,
            cv2.COLOR_RGB2BGR
        )

        # ----------------------------------------------------
        # GENERAL + CUSTOM DETECTION
        # ----------------------------------------------------

        detections = run_combined_detection(
            frame,
            confidence
        )

        return jsonify({
            "detections": detections,
            "count": len(detections),
            "frame_processed": True
        })

    except Exception as e:

        print(
            "Live detection error:",
            repr(e)
        )

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# START FLASK SERVER
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("STARTING AI PUBLIC SAFETY VIDEO MONITOR")
    print("=" * 60)

    print(
        f"General model: {GENERAL_MODEL_PATH}"
    )
    print(
        f"Weapon model: {WEAPON_MODEL_PATH}"
    )
    print(
        f"Weapon confidence: {WEAPON_CONFIDENCE:.2f}"
    )
    print(
        "Weapon inference: every frame | progress log: every 30 frames"
    )
    print(
        f"Weapon inference size: {WEAPON_INFERENCE_SIZE}"
    )
    print(
        f"Weapon detector interval: every {WEAPON_DETECT_EVERY_N_FRAMES} frames"
    )


    if general_model is not None or weapon_model is not None:

        print(
            f"General classes: {general_model.names if general_model is not None else None}"
        )
        print(
            f"Weapon classes: {weapon_model.names if weapon_model is not None else None}"
        )

    else:

        print(
            "WARNING: Model is NOT loaded!"
        )


    print("=" * 60)
    print(
        "Server: http://127.0.0.1:8000"
    )
    print("=" * 60)


    app.run(

        host="127.0.0.1",

        port=8000,

        debug=False,

        use_reloader=False

    )