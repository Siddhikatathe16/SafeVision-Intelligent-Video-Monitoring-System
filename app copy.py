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
import imageio_ffmpeg
import re
import time

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

# Custom safety/weapon model:
# Detects project-specific weapon classes.
WEAPON_MODEL_PATH = os.path.join(REPO_ROOT, "gun_knife_yolo11n.pt")

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
    if os.path.exists(WEAPON_MODEL_PATH):
        weapon_model = YOLO(WEAPON_MODEL_PATH)

        print("=" * 60)
        print("CUSTOM WEAPON MODEL LOADED")
        print("=" * 60)
        print(f"Model path: {WEAPON_MODEL_PATH}")
        print(f"Classes: {weapon_model.names}")
        print("=" * 60)
    else:
        print("WARNING: gun_knife_yolo11n.pt NOT FOUND")
        print("Weapon detection will not be available.")

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
    Adds an application-level identity layer on top of ByteTrack.

    ByteTrack IDs can change after a long occlusion. This class tries to
    reconnect a new raw ByteTrack ID to the last known stable person by
    comparing the person's last bounding box with the current bounding box.

    The first raw ByteTrack ID becomes the public ID, so if a person starts
    as ByteTrack ID 52, the UI will keep showing PERSON-052 when possible.
    """

    def __init__(self):
        self.stable_tracks = {}
        self.raw_to_stable = {}
        self.next_generated_id = 1

    def reset(self):
        self.stable_tracks.clear()
        self.raw_to_stable.clear()
        self.next_generated_id = 1

    def _new_id(self, raw_id):
        # Always hand out the next small sequential number. Never reuse
        # ByteTrack's raw internal counter directly (that counter never
        # resets and climbs fast in crowded/occluded scenes — verified:
        # it produced PERSON-213 for a ~14-person crowd). The public ID
        # now only grows when a genuinely new stable identity is created.
        while self.next_generated_id in self.stable_tracks:
            self.next_generated_id += 1
        stable_id = self.next_generated_id
        self.next_generated_id += 1
        return stable_id

    def assign(self, raw_id, bbox, frame_index):
        # Existing raw tracker ID: use its known stable identity.
        if raw_id is not None and raw_id >= 0:
            raw_id = int(raw_id)
            known = self.raw_to_stable.get(raw_id)
            if known is not None and known in self.stable_tracks:
                track = self.stable_tracks[known]
                track["bbox"] = bbox
                track["last_frame"] = frame_index
                track["raw_id"] = raw_id
                return known

        # Try to reconnect this detection to an existing stable identity.
        best_id = None
        best_score = -1.0
        for stable_id, track in self.stable_tracks.items():
            missed = frame_index - track["last_frame"]
            if missed > STABLE_ID_MAX_MISSED_FRAMES:
                continue

            previous_box = track["bbox"]
            iou = _iou(previous_box, bbox)
            center_ratio = _center_distance_ratio(previous_box, bbox)

            # Score favors overlap, but also permits re-identification after
            # a short miss/occlusion when the person is still nearby.
            score = iou + max(0.0, 1.0 - center_ratio) * 0.35

            if (iou >= STABLE_ID_IOU_THRESHOLD or
                    center_ratio <= STABLE_ID_CENTER_THRESHOLD) and score > best_score:
                best_score = score
                best_id = stable_id

        if best_id is None:
            best_id = self._new_id(raw_id)

        self.stable_tracks[best_id] = {
            "bbox": bbox,
            "last_frame": frame_index,
            "raw_id": raw_id if raw_id is not None else -1,
        }

        if raw_id is not None and raw_id >= 0:
            self.raw_to_stable[int(raw_id)] = best_id

        # Remove identities that have been absent for too long.
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
# ROBUST GUN / KNIFE DETECTION
# ============================================================

# The downloaded YOLO11n gun/knife model was trained at 640 px.
# For CCTV footage, weapons are often tiny, so the detector uses overlapping
# tiles in addition to the full frame. The thresholds below apply ONLY to the
# weapon detector; person tracking and all other application logic are unchanged.
WEAPON_CONFIDENCE = 0.25
WEAPON_INFERENCE_SIZE = 960
WEAPON_TILE_OVERLAP = 0.20
WEAPON_NMS_IOU = 0.45
WEAPON_CONFIRM_FRAMES = 3
WEAPON_CONFIRM_WINDOW = 5
WEAPON_MATCH_IOU = 0.15
WEAPON_MAX_FRAME_GAP = 4


class WeaponTemporalConfirmation:
    """
    Confirm a gun/knife after three consistent observations inside a
    five-frame window. This is deliberately tolerant of a missed frame,
    which is important for small/partially occluded weapons in CCTV video.
    """

    def __init__(self):
        self.candidates = []
        self.next_candidate_id = 1

    def reset(self):
        self.candidates.clear()
        self.next_candidate_id = 1

    def update(self, detections, frame_index):
        for candidate in self.candidates:
            candidate["matched"] = False

        confirmed = []

        for detection in detections:
            cls_name = str(detection["class"]).lower()
            if cls_name not in {"gun", "knife"}:
                continue

            best = None
            best_iou = 0.0

            for candidate in self.candidates:
                if candidate["class"] != cls_name:
                    continue
                if candidate["matched"]:
                    continue
                if frame_index - candidate["last_frame"] > WEAPON_MAX_FRAME_GAP:
                    continue

                overlap = _iou(candidate["bbox"], detection["bbox"])
                if overlap >= WEAPON_MATCH_IOU and overlap > best_iou:
                    best = candidate
                    best_iou = overlap

            if best is None:
                best = {
                    "id": self.next_candidate_id,
                    "class": cls_name,
                    "bbox": detection["bbox"],
                    "last_frame": frame_index,
                    "hits": 1,
                    "first_frame": frame_index,
                    "matched": True,
                }
                self.next_candidate_id += 1
                self.candidates.append(best)
            else:
                best["bbox"] = detection["bbox"]
                best["last_frame"] = frame_index
                best["hits"] += 1
                best["matched"] = True

            # Keep only hits inside the confirmation window.
            if frame_index - best["first_frame"] >= WEAPON_CONFIRM_WINDOW:
                best["first_frame"] = frame_index
                best["hits"] = 1

            detection["weapon_candidate_id"] = best["id"]
            detection["confirmation_count"] = min(
                best["hits"], WEAPON_CONFIRM_FRAMES
            )
            detection["confirmed"] = best["hits"] >= WEAPON_CONFIRM_FRAMES
            detection["weapon_status"] = (
                "CONFIRMED" if detection["confirmed"] else "CHECK"
            )

            if detection["confirmed"]:
                confirmed.append(detection)

        self.candidates = [
            c for c in self.candidates
            if frame_index - c["last_frame"] <= WEAPON_MAX_FRAME_GAP
        ]

        return confirmed


weapon_confirmation = WeaponTemporalConfirmation()


def reset_tracking_state():
    """Reset all state at the start of a new video/webcam session."""
    stable_person_identity.reset()
    weapon_confirmation.reset()

    # Ultralytics stores tracker state in the model predictor when persist=True.
    # Resetting predictor forces a fresh tracker for the next session.
    if general_model is not None:
        try:
            general_model.predictor = None
        except Exception:
            pass


def _weapon_class_name(name):
    return str(name).strip().lower()


def _associate_weapon_with_person(weapon_detection, person_detections):
    """Associate a confirmed weapon with the most plausible nearby person."""
    if not person_detections:
        return None

    wx, wy = _center(weapon_detection["bbox"])
    best = None
    best_score = -1.0

    for person in person_detections:
        px1, py1, px2, py2 = person["bbox"]
        expanded = [
            px1 - (px2 - px1) * 0.20,
            py1 - (py2 - py1) * 0.20,
            px2 + (px2 - px1) * 0.20,
            py2 + (py2 - py1) * 0.20,
        ]

        inside = (
            expanded[0] <= wx <= expanded[2]
            and expanded[1] <= wy <= expanded[3]
        )
        distance = _center_distance_ratio(person["bbox"], weapon_detection["bbox"])

        score = (2.0 if inside else 0.0) + max(0.0, 1.0 - distance)
        if score > best_score:
            best_score = score
            best = person

    if best is None:
        return None

    return best.get("stable_id")


def _run_general_detection(frame, confidence=0.30, tracking=False):
    """Run general YOLO, optionally with ByteTrack."""
    if general_model is None:
        return []

    if tracking:
        results = general_model.track(
            source=frame,
            imgsz=640,
            conf=confidence,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )
    else:
        results = general_model.predict(
            source=frame,
            imgsz=640,
            conf=confidence,
            verbose=False,
        )

    detections = []
    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = _model_name(general_model, cls_id)
            conf = float(box.conf[0])
            bbox = _box_to_list(box)

            raw_track_id = -1
            if tracking and box.id is not None:
                raw_track_id = int(box.id[0])

            detections.append({
                "source": "general",
                "class": cls_name,
                "class_id": cls_id,
                "confidence": conf,
                "bbox": bbox,
                "track_id": raw_track_id,
            })

    return detections


def _weapon_iou(a, b):
    return _iou(a, b)


def _weapon_boxes_are_same(a, b):
    """Recognize duplicate predictions from the full frame and overlapping tiles."""
    if a["class"].lower() != b["class"].lower():
        return False

    box_a = a["bbox"]
    box_b = b["bbox"]
    if _weapon_iou(box_a, box_b) >= WEAPON_NMS_IOU:
        return True

    # A small object can have different box sizes in different tiles.
    # Treat close centers as duplicates as well.
    ax, ay = _center(box_a)
    bx, by = _center(box_b)
    distance = float(np.hypot(ax - bx, ay - by))
    diag_a = float(np.hypot(box_a[2] - box_a[0], box_a[3] - box_a[1]))
    diag_b = float(np.hypot(box_b[2] - box_b[0], box_b[3] - box_b[1]))
    reference = max(1.0, max(diag_a, diag_b))
    return distance <= reference * 0.30


def _merge_weapon_detections(detections):
    """Class-aware greedy NMS for detections collected from overlapping views."""
    if not detections:
        return []

    kept = []
    for detection in sorted(
        detections, key=lambda d: float(d["confidence"]), reverse=True
    ):
        duplicate = any(
            _weapon_boxes_are_same(detection, existing)
            for existing in kept
        )
        if not duplicate:
            kept.append(detection)

    return kept


def _make_weapon_tiles(frame):
    """Create four overlapping tiles while retaining original coordinates."""
    h, w = frame.shape[:2]
    if w < 2 or h < 2:
        return []

    # Each tile is 60% of the frame with 20% overlap.
    tile_w = max(2, int(w * 0.60))
    tile_h = max(2, int(h * 0.60))
    step_x = max(1, int(tile_w * (1.0 - WEAPON_TILE_OVERLAP)))
    step_y = max(1, int(tile_h * (1.0 - WEAPON_TILE_OVERLAP)))

    xs = [0, max(0, w - tile_w)]
    ys = [0, max(0, h - tile_h)]

    # De-duplicate coordinates for small frames.
    tiles = []
    seen = set()
    for y0 in ys:
        for x0 in xs:
            x1 = min(w, x0 + tile_w)
            y1 = min(h, y0 + tile_h)
            key = (x0, y0, x1, y1)
            if key in seen:
                continue
            seen.add(key)
            tiles.append((frame[y0:y1, x0:x1], x0, y0))

    return tiles


def _run_weapon_detection(frame, confidence=WEAPON_CONFIDENCE):
    """
    Dedicated gun/knife detection optimized for small CCTV weapons.

    The detector sees the original frame AND four overlapping high-resolution
    tiles. Tile coordinates are mapped back to the original frame and duplicate
    predictions are removed. Only the gun and knife classes are accepted.
    """
    if weapon_model is None:
        return []

    # Never let the UI confidence slider make the weapon detector stricter than
    # the dedicated weapon threshold.
    threshold = WEAPON_CONFIDENCE

    sources = [frame]
    metadata = [(0, 0)]
    for tile, x_offset, y_offset in _make_weapon_tiles(frame):
        sources.append(tile)
        metadata.append((x_offset, y_offset))

    all_detections = []

    # Run all views as one Ultralytics batch. This is faster than calling the
    # model separately for every tile and keeps inference settings identical.
    results = weapon_model.predict(
        source=sources,
        imgsz=WEAPON_INFERENCE_SIZE,
        conf=threshold,
        iou=0.45,
        max_det=50,
        verbose=False,
    )

    for result_index, result in enumerate(results):
        if result.boxes is None:
            continue

        x_offset, y_offset = metadata[result_index]

        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = _model_name(weapon_model, cls_id)
            cls_lower = _weapon_class_name(cls_name)

            # This checkpoint should contain exactly gun and knife. Ignore
            # anything else rather than turning an unexpected class into an
            # alert.
            if cls_lower not in {"gun", "knife"}:
                continue

            conf_value = float(box.conf[0])
            if conf_value < threshold:
                continue

            local_box = _box_to_list(box)
            bbox = [
                local_box[0] + x_offset,
                local_box[1] + y_offset,
                local_box[2] + x_offset,
                local_box[3] + y_offset,
            ]

            # Clip coordinates to the original frame.
            h, w = frame.shape[:2]
            bbox = [
                max(0.0, min(float(w - 1), bbox[0])),
                max(0.0, min(float(h - 1), bbox[1])),
                max(0.0, min(float(w - 1), bbox[2])),
                max(0.0, min(float(h - 1), bbox[3])),
            ]

            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue

            all_detections.append({
                "source": "weapon",
                "class": cls_name,
                "class_id": cls_id,
                "confidence": conf_value,
                "bbox": bbox,
                "track_id": -1,
            })

    # Merge the full-frame and tiled predictions.
    return _merge_weapon_detections(all_detections)

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
        status = detection.get("weapon_status", "DETECTED")
        if status == "CONFIRMED":
            prefix = "ALERT"
            color = (0, 0, 255)
            count_text = ""
        elif status == "CHECK":
            prefix = "CHECK"
            color = (0, 165, 255)
            count_text = (
                f" {detection.get('confirmation_count', 1)}"
                f"/{WEAPON_CONFIRM_FRAMES}"
            )
        else:
            prefix = "WEAPON"
            color = (0, 0, 255)
            count_text = ""

        person_id = detection.get("associated_person_id")
        if person_id is not None:
            label = (
                f"{prefix} {cls_name} {conf:.2f}"
                f"{count_text} -> PERSON-{int(person_id):03d}"
            )
        else:
            label = f"{prefix} {cls_name} {conf:.2f}{count_text}"
    else:
        color = (0, 255, 0)
        public_id = detection.get("stable_id")
        is_threat = detection.get("is_threat", False)

        if cls_name.lower() == "person" and public_id is not None:
            raw_id = detection.get("track_id", -1)
            if is_threat:
                color = (0, 0, 255)  # red — linked to a CONFIRMED weapon
                label = f"ALERT PERSON-{int(public_id):03d} ARMED {conf:.2f}"
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

def process_video_frame(frame, confidence=0.30, frame_index=0):
    """
    Process one video/webcam frame.

    Pipeline:
      1. General YOLO + ByteTrack for people/general objects.
      2. Stable identity layer for person IDs.
      3. Dedicated YOLO11 gun/knife detector at 960px.
      4. Three-frame spatial confirmation for gun/knife.
      5. Associate confirmed weapons with the nearest tracked person.
    """
    general_detections = _run_general_detection(
        frame,
        confidence=confidence,
        tracking=True,
    )

    person_detections = []
    for detection in general_detections:
        if detection["class"].lower() == "person":
            stable_id = stable_person_identity.assign(
                detection.get("track_id", -1),
                detection["bbox"],
                frame_index,
            )
            detection["stable_id"] = stable_id
            person_detections.append(detection)

    weapon_detections = _run_weapon_detection(
        frame,
        confidence=WEAPON_CONFIDENCE,
    )

    confirmed_weapons = weapon_confirmation.update(
        weapon_detections,
        frame_index,
    )

    # Associate both candidates and confirmed detections with a person when
    # possible. This helps the later event/risk engine know who is involved.
    for weapon in weapon_detections:
        associated = _associate_weapon_with_person(
            weapon,
            person_detections,
        )
        if associated is not None:
            weapon["associated_person_id"] = associated

    # Flag which tracked people are currently carrying a CONFIRMED threat.
    # CHECK-status weapons do not turn the person's box red yet — only a
    # confirmed alert does, to avoid flashing red on single-frame noise.
    threat_person_ids = {
        d.get("associated_person_id")
        for d in weapon_detections
        if d.get("confirmed", False) and d.get("associated_person_id") is not None
    }
    for detection in person_detections:
        detection["is_threat"] = detection.get("stable_id") in threat_person_ids

    combined = general_detections + weapon_detections

    people_count = len(person_detections)
    weapon_count = len(confirmed_weapons)

    for detection in combined:
        if detection["source"] == "weapon":
            draw_detection(frame, detection)
        else:
            draw_detection(
                frame,
                detection,
                detection.get("track_id", -1),
            )

    return frame, combined, people_count, weapon_count


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


# ============================================================
# HEALTH CHECK
# ============================================================

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

        frame_events = []
        max_people = 0
        max_weapons = 0
        frame_idx = 0

        # ----------------------------------------------------
        # IMPORTANT:
        # Reset ByteTrack state before processing this video.
        # ----------------------------------------------------

        reset_tracking_state()

        print(
            "Running General YOLO + ByteTrack "
            "+ Stable IDs + Gun/Knife YOLO..."
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
                    weapon_count
                ) = process_video_frame(
                    frame,
                    confidence_threshold,
                    frame_index=frame_idx,
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

                frame_events.append({
                    "frame": frame_idx,
                    "people_count": people_count,
                    "weapon_count": weapon_count,
                    "detections": len(detections)
                })

                # --------------------------------------------
                # WRITE EVERY FRAME
                # --------------------------------------------

                writer.write(
                    annotated_frame
                )

                frame_idx += 1

                if frame_idx % 30 == 0:
                    print(
                        f"Processed {frame_idx}/"
                        f"{total_frames} frames..."
                    )

        finally:

            cap.release()

            # IMPORTANT:
            # Release only AFTER ALL frames are written.
            writer.release()

        print(
            f"Video processing complete: "
            f"{frame_idx} frames written"
        )

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

            "frame_events":
            frame_events[-50:]

        }), 200

    except Exception as e:

        print("\n" + "=" * 60)
        print("VIDEO DETECTION ERROR")
        print(repr(e))
        print("=" * 60)

        return jsonify({
            "error": str(e)
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
                annotated_frame, _, _, _ = process_video_frame(
                    frame,
                    confidence=confidence,
                    frame_index=frame_index,
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
        f"Weapon confirmation frames: {WEAPON_CONFIRM_FRAMES}"
    )
    print(
        f"Weapon inference size: {WEAPON_INFERENCE_SIZE}"
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