"""Final configurable safety-detection thresholds."""
import os
import json

# Crowd
CROWD_COUNT_THRESHOLD = int(os.getenv("CROWD_COUNT_THRESHOLD", "8"))
CROWD_DENSITY_THRESHOLD = float(os.getenv("CROWD_DENSITY_THRESHOLD", "0.10"))
CROWD_BUILDUP_DELTA = int(os.getenv("CROWD_BUILDUP_DELTA", "4"))
CROWD_BUILDUP_WINDOW_SEC = float(os.getenv("CROWD_BUILDUP_WINDOW_SEC", "4.0"))
CROWD_CONFIRM_FRAMES = int(os.getenv("CROWD_CONFIRM_FRAMES", "3"))

# Loitering
LOITER_SECONDS = float(os.getenv("LOITER_SECONDS", "5.0"))
LOITER_MAX_SPEED_BODY_HEIGHTS_SEC = float(os.getenv("LOITER_MAX_SPEED_BODY_HEIGHTS_SEC", "0.18"))
LOITER_MAX_DISPLACEMENT_BODY_HEIGHTS = float(os.getenv("LOITER_MAX_DISPLACEMENT_BODY_HEIGHTS", "0.65"))
LOITER_SMOOTH_WINDOW_SEC = float(os.getenv("LOITER_SMOOTH_WINDOW_SEC", "0.75"))
LOITER_MIN_SLOW_RATIO = float(os.getenv("LOITER_MIN_SLOW_RATIO", "0.80"))
LOITER_CONFIRM_FRAMES = int(os.getenv("LOITER_CONFIRM_FRAMES", "8"))

# Restricted zone
ZONE_CONFIRM_FRAMES = int(os.getenv("ZONE_CONFIRM_FRAMES", "3"))
ZONE_HOLD_FRAMES = int(os.getenv("ZONE_HOLD_FRAMES", "8"))
ZONE_AUTO_ENABLED = os.getenv("ZONE_AUTO_ENABLED", "1").lower() not in {"0","false","no"}
ZONE_AUTO_WARMUP_FRAMES = int(os.getenv("ZONE_AUTO_WARMUP_FRAMES", "96"))
ZONE_AUTO_SAMPLE_EVERY = int(os.getenv("ZONE_AUTO_SAMPLE_EVERY", "8"))
ZONE_AUTO_MIN_STABLE_SAMPLES = int(os.getenv("ZONE_AUTO_MIN_STABLE_SAMPLES", "3"))
ZONE_AUTO_MIN_LONG_LINES = int(os.getenv("ZONE_AUTO_MIN_LONG_LINES", "5"))
ZONE_AUTO_MIN_LINE_LENGTH = int(os.getenv("ZONE_AUTO_MIN_LINE_LENGTH", "75"))
ZONE_AUTO_MIN_AREA = float(os.getenv("ZONE_AUTO_MIN_AREA", "0.015"))
ZONE_AUTO_MAX_AREA = float(os.getenv("ZONE_AUTO_MAX_AREA", "0.30"))
ZONE_AUTO_MIN_X = float(os.getenv("ZONE_AUTO_MIN_X", "0.50"))
ZONE_AUTO_MIN_Y = float(os.getenv("ZONE_AUTO_MIN_Y", "0.22"))

# Explicit manual zone: ONLY used if SAFETY_RESTRICTED_ZONE is supplied.
def _parse_polygon(value):
    if not value:
        return None
    try:
        pts = json.loads(value)
        if not isinstance(pts, list) or len(pts) < 3:
            return None
        out=[]
        for p in pts:
            if not isinstance(p,(list,tuple)) or len(p)!=2:
                return None
            out.append([max(0.0,min(1.0,float(p[0]))), max(0.0,min(1.0,float(p[1])))])
        return out
    except Exception:
        return None

MANUAL_RESTRICTED_ZONE_NORM = _parse_polygon(os.getenv("SAFETY_RESTRICTED_ZONE"))

DEFAULT_LOITERING_ZONE = [[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]]
LOITERING_ZONE_NORM = _parse_polygon(os.getenv("SAFETY_LOITERING_ZONE")) or DEFAULT_LOITERING_ZONE

# Abnormal movement
ABNORMAL_SPEED_BODY_HEIGHTS_SEC = float(os.getenv("ABNORMAL_SPEED_BODY_HEIGHTS_SEC", "4.2"))
ABNORMAL_ACCEL_BODY_HEIGHTS_SEC2 = float(os.getenv("ABNORMAL_ACCEL_BODY_HEIGHTS_SEC2", "2.6"))
ABNORMAL_SPEED_PX_SEC = float(os.getenv("ABNORMAL_SPEED_PX_SEC", "550.0"))
ABNORMAL_ACCEL_PX_SEC2 = float(os.getenv("ABNORMAL_ACCEL_PX_SEC2", "700.0"))
ABNORMAL_REVERSE_ANGLE_DEG = float(os.getenv("ABNORMAL_REVERSE_ANGLE_DEG", "135.0"))
ABNORMAL_MIN_SPEED_PX_SEC = float(os.getenv("ABNORMAL_MIN_SPEED_PX_SEC", "140.0"))
MOVEMENT_WINDOW = int(os.getenv("MOVEMENT_WINDOW", "12"))
ABNORMAL_CONFIRM_FRAMES = int(os.getenv("ABNORMAL_CONFIRM_FRAMES", "4"))

# Fall
FALL_MIN_DROP_RATIO = float(os.getenv("FALL_MIN_DROP_RATIO", "0.18"))
FALL_HORIZONTAL_RATIO = float(os.getenv("FALL_HORIZONTAL_RATIO", "0.90"))
FALL_MAX_TALL_RATIO = float(os.getenv("FALL_MAX_TALL_RATIO", "0.78"))
FALL_HEIGHT_DROP_RATIO = float(os.getenv("FALL_HEIGHT_DROP_RATIO", "0.62"))
FALL_VERIFY_SECONDS = float(os.getenv("FALL_VERIFY_SECONDS", "0.75"))
FALL_CONFIRM_FRAMES = int(os.getenv("FALL_CONFIRM_FRAMES", "5"))
FALL_MIN_PERSON_HEIGHT = int(os.getenv("FALL_MIN_PERSON_HEIGHT", "45"))

MAX_HISTORY_FRAMES = int(os.getenv("MAX_HISTORY_FRAMES", "90"))

# Weapon
WEAPON_PRIMARY_MODEL = os.getenv("WEAPON_PRIMARY_MODEL", "gun_knife_yolo11n.pt")
WEAPON_FALLBACK_MODEL = os.getenv("WEAPON_FALLBACK_MODEL", "best.pt")
WEAPON_USE_FALLBACK = os.getenv("WEAPON_USE_FALLBACK", "1").lower() not in {"0","false","no"}
WEAPON_DETECT_EVERY_N_FRAMES = int(os.getenv("WEAPON_DETECT_EVERY_N_FRAMES", "2"))
WEAPON_ROI_EVERY_N_FRAMES = int(os.getenv("WEAPON_ROI_EVERY_N_FRAMES", "4"))
WEAPON_MAX_PERSON_ROIS = int(os.getenv("WEAPON_MAX_PERSON_ROIS", "4"))
WEAPON_INFERENCE_SIZE = int(os.getenv("WEAPON_INFERENCE_SIZE", "768"))
WEAPON_ROI_INFERENCE_SIZE = int(os.getenv("WEAPON_ROI_INFERENCE_SIZE", "960"))
WEAPON_PRIMARY_MIN_CONF = float(os.getenv("WEAPON_PRIMARY_MIN_CONF", "0.25"))
WEAPON_FALLBACK_GUN_MIN_CONF = float(os.getenv("WEAPON_FALLBACK_GUN_MIN_CONF", "0.62"))
WEAPON_FALLBACK_KNIFE_MIN_CONF = float(os.getenv("WEAPON_FALLBACK_KNIFE_MIN_CONF", "0.38"))
WEAPON_CONFIRM_OBSERVATIONS = int(os.getenv("WEAPON_CONFIRM_OBSERVATIONS", "3"))
WEAPON_CONFIRM_WINDOW = int(os.getenv("WEAPON_CONFIRM_WINDOW", "18"))
WEAPON_MIN_CONFIRM_CONFIDENCE = float(os.getenv("WEAPON_MIN_CONFIRM_CONFIDENCE", "0.45"))
WEAPON_MIN_GOOD_GEOMETRY_HITS = int(os.getenv("WEAPON_MIN_GOOD_GEOMETRY_HITS", "2"))
WEAPON_MAX_FRAME_GAP = int(os.getenv("WEAPON_MAX_FRAME_GAP", "8"))
WEAPON_HOLD_FRAMES = int(os.getenv("WEAPON_HOLD_FRAMES", "10"))
WEAPON_MATCH_IOU = float(os.getenv("WEAPON_MATCH_IOU", "0.10"))
WEAPON_MAX_PERSON_AREA_RATIO = float(os.getenv("WEAPON_MAX_PERSON_AREA_RATIO", "0.10"))
WEAPON_MIN_PERSON_AREA_RATIO = float(os.getenv("WEAPON_MIN_PERSON_AREA_RATIO", "0.00015"))
WEAPON_REQUIRE_PERSON_ASSOCIATION = True

CAPTURE_EVERY_N_FRAMES = 30
WEAPON_PROGRESS_EVERY_N_FRAMES = 30

def normalized_polygon_to_pixels(polygon, width, height):
    return [(int(round(x*width)), int(round(y*height))) for x,y in polygon]


# ============================================================
# Agentic AI layer: Event Correlation -> Supervisor -> Risk -> Response
# ============================================================

# How close in time (seconds) two events must be to be considered part of
# the same real-world incident. Also used as the "no new evidence" gap that
# closes an open incident.
AGENTIC_CORRELATION_WINDOW_SECONDS = float(os.getenv("AGENTIC_CORRELATION_WINDOW_SECONDS", "6.0"))

# Transparent, explainable per-event-type risk weights (0-100 scale before
# severity/confidence/bonus adjustments). Unknown event types fall back to
# AGENTIC_DEFAULT_EVENT_WEIGHT so a future detector never breaks scoring.
AGENTIC_DEFAULT_EVENT_WEIGHT = float(os.getenv("AGENTIC_DEFAULT_EVENT_WEIGHT", "10"))
AGENTIC_RISK_WEIGHTS = {
    "WEAPON_CONFIRMED": float(os.getenv("AGENTIC_WEIGHT_WEAPON", "48")),
    "POSSIBLE_FALL": float(os.getenv("AGENTIC_WEIGHT_FALL", "26")),
    "ZONE_INTRUSION": float(os.getenv("AGENTIC_WEIGHT_ZONE", "20")),
    "ABNORMAL_MOVEMENT": float(os.getenv("AGENTIC_WEIGHT_MOVEMENT", "15")),
    "HIGH_DENSITY": float(os.getenv("AGENTIC_WEIGHT_HIGH_DENSITY", "16")),
    "CROWD_BUILDUP": float(os.getenv("AGENTIC_WEIGHT_CROWD_BUILDUP", "12")),
    "LOITERING": float(os.getenv("AGENTIC_WEIGHT_LOITERING", "10")),
}

AGENTIC_SEVERITY_MULTIPLIER = {"CRITICAL": 1.30, "HIGH": 1.15, "MEDIUM": 1.0, "LOW": 0.80}

# Bonuses reward *correlated* signals over an isolated one-off detection,
# matching the "crowd buildup + abnormal movement + fall = one incident"
# design intent rather than scoring each event in isolation.
AGENTIC_MULTI_SIGNAL_BONUS = float(os.getenv("AGENTIC_MULTI_SIGNAL_BONUS", "10"))
AGENTIC_WEAPON_CORRELATION_BONUS = float(os.getenv("AGENTIC_WEAPON_CORRELATION_BONUS", "15"))

# Score -> severity band thresholds (0-100 scale).
AGENTIC_RISK_CRITICAL_THRESHOLD = float(os.getenv("AGENTIC_RISK_CRITICAL_THRESHOLD", "75"))
AGENTIC_RISK_HIGH_THRESHOLD = float(os.getenv("AGENTIC_RISK_HIGH_THRESHOLD", "50"))
AGENTIC_RISK_MEDIUM_THRESHOLD = float(os.getenv("AGENTIC_RISK_MEDIUM_THRESHOLD", "25"))

# Approved response actions per severity band. Deliberately a fixed,
# auditable table (no free-form LLM action generation) so the Response
# Agent can only ever execute actions a human has pre-approved.
AGENTIC_BASE_ACTIONS = {
    "LOW": ["LOG_INCIDENT"],
    "MEDIUM": ["LOG_INCIDENT", "FLAG_FOR_REVIEW"],
    "HIGH": ["LOG_INCIDENT", "FLAG_FOR_REVIEW", "DISPATCH_ALERT", "RETAIN_EVIDENCE"],
    "CRITICAL": ["LOG_INCIDENT", "FLAG_FOR_REVIEW", "DISPATCH_ALERT", "RETAIN_EVIDENCE",
                 "NOTIFY_SECURITY_TEAM", "REQUIRE_HUMAN_VERIFICATION"],
}

AGENTIC_ACTION_LABELS = {
    "LOG_INCIDENT": "Log incident to record",
    "FLAG_FOR_REVIEW": "Flag for operator review",
    "DISPATCH_ALERT": "Dispatch real-time alert to monitoring staff",
    "RETAIN_EVIDENCE": "Retain evidence frames/clip",
    "NOTIFY_SECURITY_TEAM": "Notify on-site security team",
    "REQUIRE_HUMAN_VERIFICATION": "Require human verification before further action",
}
