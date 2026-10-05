"""3-panel composite (flow step 8): human video | tracking overlay | robot replay, frame-synced."""
import argparse
import os

import cv2
import numpy as np

PANEL_H = 540
LABEL_H = 50


def _open(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    return cap


def _panel(frame, label):
    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (int(round(w * PANEL_H / h)) // 2 * 2, PANEL_H), interpolation=cv2.INTER_AREA)
    bar = np.zeros((LABEL_H, frame.shape[1], 3), np.uint8)
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(bar, label, ((frame.shape[1] - tw) // 2, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, frame])


def make_composite(video, run_dir, sim_view="wide"):
    sources = [
        (video, "1. Human video"),
        (os.path.join(run_dir, "overlay.mp4"), "2. OpenCV 5 tracking"),
        (os.path.join(run_dir, f"replay_{sim_view}.mp4"), "3. Franka Panda (MuJoCo)"),
    ]
    caps = [_open(p) for p, _ in sources]
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    out_path = os.path.join(run_dir, f"composite_{sim_view}.mp4")
    writer, n = None, 0

    while True:
        frames = [c.read() for c in caps]
        if not all(ok for ok, _ in frames):
            break
        row = np.hstack([_panel(f, lbl) for (_, f), (_, lbl) in zip(frames, sources)])
        if writer is None:
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (row.shape[1], row.shape[0]))
        writer.write(row)
        n += 1

    for c in caps:
        c.release()
    writer.release()
    return out_path, n


def main():
    p = argparse.ArgumentParser(description="Ego2Sim 3-panel composite")
    p.add_argument("--video", required=True)
    p.add_argument("--run", required=True, help="folder with overlay.mp4 and replay_*.mp4")
    p.add_argument("--view", default="wide", choices=["wide", "closeup"])
    a = p.parse_args()
    path, n = make_composite(a.video, a.run, a.view)
    print(f"Saved {path} ({n} frames)")


if __name__ == "__main__":
    main()
