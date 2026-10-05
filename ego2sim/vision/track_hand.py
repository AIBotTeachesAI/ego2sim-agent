"""Hand tracking with MediaPipe HandLandmarker (Tasks API, CPU delegate)."""
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

WRIST = 0  # MediaPipe 21-landmark scheme


def make_landmarker(model_path):
    base = mp_python.BaseOptions(
        model_asset_path=model_path,
        delegate=mp_python.BaseOptions.Delegate.CPU,
    )
    opts = vision.HandLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(opts)


def track_hand_video(video_path, model_path):
    """Returns landmarks (N,21,3), normalized x,y in [0,1]; NaN where no hand."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    landmarker = make_landmarker(model_path)
    out, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect_for_video(img, int((i / fps) * 1000))
        if res.hand_landmarks:
            out.append([[p.x, p.y, p.z] for p in res.hand_landmarks[0]])
        else:
            out.append(np.full((21, 3), np.nan).tolist())
        i += 1
    cap.release()
    landmarker.close()
    return np.array(out, dtype=np.float64)
