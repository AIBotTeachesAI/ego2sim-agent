"""Vision pipeline (flow step 3): video + calibration -> trajectory.json + metrics.json."""
import argparse
import json
import os
import time

import cv2
import numpy as np

from .events import detect_events
from .fill_gap import fill_hand_gaps
from .homography import compute_homography, load_calibration, pixels_to_table
from .overlay import render_overlay
from .track_cup import track_cup_video
from .track_hand import WRIST, track_hand_video

DEFAULT_MODEL = "models/hand_landmarker.task"


def _xy(v):
    return None if np.isnan(v).any() else [round(float(v[0]), 5), round(float(v[1]), 5)]


def _num(x):
    return None if np.isnan(x) else round(float(x), 5)


def run(video, calib_path, out_dir, model_path=DEFAULT_MODEL, overlay=True):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)

    calib = load_calibration(calib_path)
    H = compute_homography(calib)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    res = np.array([cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)])
    cap.release()
    if [int(v) for v in res] != list(calib["resolution"]):
        raise ValueError(f"Video is {res}, calibration expects {calib['resolution']}")

    cup_px, cup_area = track_cup_video(video, calib["cup_hsv"])
    lm = track_hand_video(video, model_path)
    if len(cup_px) != len(lm):
        raise ValueError(f"Frame count mismatch: cup {len(cup_px)} vs hand {len(lm)}")
    n = len(lm)

    filled, source, gaps = fill_hand_gaps(lm, cup_px / res)
    wrist_px = filled[:, WRIST, :2] * res
    cup_t = pixels_to_table(cup_px, H)
    wrist_t = pixels_to_table(wrist_px, H)
    dist = np.linalg.norm(wrist_t - cup_t, axis=1)
    events = detect_events(gaps, dist)

    detected = ~np.isnan(lm[:, WRIST, 0])
    cup_ok = ~np.isnan(cup_px[:, 0])

    frames = [{
        "i": i,
        "t": round(i / fps, 4),
        "cup_px": _xy(cup_px[i]),
        "cup_table": _xy(cup_t[i]),
        "cup_area_px": round(float(cup_area[i]), 1),
        "wrist_px": _xy(wrist_px[i]),
        "wrist_table": _xy(wrist_t[i]),
        "hand_source": str(source[i]),
        "hand_cup_dist": _num(dist[i]),
    } for i in range(n)]

    metrics = {
        "n_frames": n,
        "hand_detect_rate": round(float(detected.mean()), 4),
        "cup_track_rate": round(float(cup_ok.mean()), 4),
        "hand_frames_filled": int((~detected).sum()),
        "n_gaps": len(gaps),
        "processing_time_s": round(time.time() - t0, 2),
    }

    traj = {
        "video": os.path.basename(video),
        "fps": round(float(fps), 4),
        "resolution": [int(v) for v in res],
        "table_units": calib["units"],
        "homography": H.tolist(),
        "events": events,
        "gaps": [list(g) for g in gaps],
        "metrics": metrics,
        "frames": frames,
    }

    with open(os.path.join(out_dir, "trajectory.json"), "w") as f:
        json.dump(traj, f, indent=1)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"events": events, **metrics}, f, indent=2)
    np.save(os.path.join(out_dir, "hand_landmarks_filled.npy"), filled)  # used by overlay
    if overlay:
        render_overlay(video, out_dir)
    return traj


def main():
    p = argparse.ArgumentParser(description="Ego2Sim vision pipeline")
    p.add_argument("--video", required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--no-overlay", action="store_true")
    a = p.parse_args()
    traj = run(a.video, a.calib, a.out, a.model, overlay=not a.no_overlay)
    print(json.dumps({"events": traj["events"], **traj["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
