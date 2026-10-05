"""Tracking overlay video: hand skeleton (color = tracking source), cup + trail, table outline, phase."""
import argparse
import json
import os

import cv2
import numpy as np

CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
SRC_COLOR = {"mediapipe": (0, 220, 0), "interp": (0, 230, 255),
             "cup_offset": (0, 150, 255), "hold": (160, 160, 160)}
TRAIL = 30


def phase_of(i, ev):
    if i < ev["grasp_start"]:
        return "approach"
    if i < ev["grasp_end"]:
        return "grasp (hand occluded)"
    if ev["release_frame"] is None or i < ev["release_frame"]:
        return "carry"
    return "released"


def table_outline_px(H):
    corners = np.array([[[0, 0]], [[1, 0]], [[1, 1]], [[0, 1]]], dtype=np.float64)
    return cv2.perspectiveTransform(corners, np.linalg.inv(np.array(H))).reshape(-1, 2).astype(np.int32)


def render_overlay(video, out_dir):
    with open(os.path.join(out_dir, "trajectory.json")) as f:
        traj = json.load(f)
    lm = np.load(os.path.join(out_dir, "hand_landmarks_filled.npy"))
    ev, frames = traj["events"], traj["frames"]
    w, h = traj["resolution"]
    outline = table_outline_px(traj["homography"])

    cap = cv2.VideoCapture(video)
    out_path = os.path.join(out_dir, "overlay.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), traj["fps"], (w, h))
    trail = []

    for i, fr in enumerate(frames):
        ok, img = cap.read()
        if not ok:
            break

        cv2.polylines(img, [outline], True, (255, 200, 80), 2, cv2.LINE_AA)

        color = SRC_COLOR.get(fr["hand_source"], (255, 255, 255))
        pts = (lm[i, :, :2] * [w, h]).astype(int)
        for a, b in CONN:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, 3, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, tuple(p), 5, color, -1, cv2.LINE_AA)

        if fr["cup_px"] is not None:
            c = tuple(int(v) for v in fr["cup_px"])
            trail.append(c)
            trail = trail[-TRAIL:]
            cv2.polylines(img, [np.array(trail, np.int32)], False, (60, 60, 255), 3, cv2.LINE_AA)
            cv2.circle(img, c, 16, (40, 40, 255), 3, cv2.LINE_AA)

        ct = fr["cup_table"]
        cup_txt = f"cup table ({ct[0]:.2f}, {ct[1]:.2f})" if ct else "cup: lost"
        txt = f"frame {i}  |  {phase_of(i, ev)}  |  hand: {fr['hand_source']}  |  {cup_txt}"
        cv2.rectangle(img, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(img, txt, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(img)

    cap.release()
    writer.release()
    return out_path


def main():
    p = argparse.ArgumentParser(description="Ego2Sim tracking overlay")
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True, help="folder containing trajectory.json")
    a = p.parse_args()
    print("Saved", render_overlay(a.video, a.out))


if __name__ == "__main__":
    main()
