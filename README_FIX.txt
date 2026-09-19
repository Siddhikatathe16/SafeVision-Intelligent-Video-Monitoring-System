PUBLIC SAFETY SAFETY-EVENT FIX

Changes:
- Reduced false ABNORMAL_MOVEMENT alerts by smoothing trajectory and normalizing speed by person body height.
- Fall detection now requires a real downward transition plus sustained horizontal/low posture; it no longer requires both conditions in one exact frame.
- Loitering zone defaults to the full camera view instead of silently reusing the restricted zone.
- Restricted zone remains configurable through /safety/config.
- Weapon/person pipeline in app.py is preserved.

Run:
python app.py

The Ultralytics model files (yolov8s.pt and best.pt) must remain beside app.py.


V4 RESTRICTED-ZONE CALIBRATION
- The polygon is stored in normalized 0..1 coordinates, so it is resolution-independent.
- It is configurable through SAFETY_RESTRICTED_ZONE or POST /safety/config.
- The supplied default is calibrated to the supplied metro CCTV camera only.
- For a different camera/view, configure a new polygon; the detector does not automatically infer a restricted area from arbitrary video.


V5 FIX -- "TypeError: 'numpy.int32' object is not iterable" during /detect_video
- Cause: restricted_zone_detector auto-zone code assumed cv2.HoughLinesP always returns (N,1,4)
  and did `for line in lines[:,0]: map(int, line)`. With a flat (N,4) array that yields bare
  numpy.int32 scalars, which are not iterable, and the whole request aborted.
- Fix: line/hull/polygon arrays are reshaped defensively; auto-zone detection is best-effort
  (failure is logged and the video continues without a zone).
- Fault isolation: every detector in SafetyEventPipeline (and the agentic layer) is wrapped so one
  failing detector can no longer abort the video. The FULL traceback is printed once per detector,
  failures are counted in the JSON field `detector_errors`, and the dashboard shows a warning.
- The /detect_video error handler now prints the full traceback and returns error_type + error_location.
- New self-tests: ZONE-HOUGH-SHAPES, PIPELINE-FAULT-ISOLATION.
