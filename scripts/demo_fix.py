"""One miss -> OpenCV check -> fix, rendered as a two-panel MuJoCo video.

python scripts/demo_fix.py --take take01 --dx 8 --dy -6
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from ego2sim.agent.motion import HOVER_Z, Motion  # noqa: E402
from ego2sim.agent.scene import VerifyScene       # noqa: E402
from ego2sim.agent.video import DemoVideo         # noqa: E402

PLACE_NOISE_M = 0.005   # realistic placing error on the re-place


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", default="take01")
    ap.add_argument("--dx", type=float, default=8.0, help="injected miss, cm (+x = forward)")
    ap.add_argument("--dy", type=float, default=-6.0, help="injected miss, cm (+y = robot's left)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the placing noise")
    a = ap.parse_args()

    with open(f"outputs/{a.take}/sim_metrics.json") as f:
        target = np.array(json.load(f)["target_cup_robot_m"][:2])
    out = f"outputs/{a.take}_loop"
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    miss = np.array([a.dx, a.dy]) / 100
    miss_cm = np.linalg.norm(miss) * 100

    scene = VerifyScene()
    video = DemoVideo(scene, target, f"{out}/demo_fix.mp4")
    arm = Motion()

    # 1. place with an injected miss
    s1 = f"1. Robot places the cup  (injected miss: {miss_cm:.0f} cm)"
    arm.start_holding([0.45, 0.0, HOVER_Z], s1)
    arm.place_at(target + miss, s1)
    video.add_motion(arm.take_frames())

    # 2. OpenCV check
    m1, img1 = video.add_check(75, "2. OpenCV check")
    cv2.imwrite(f"{out}/check_1.png", img1)

    # 3. decide (simple rule) and re-place, picking the cup where OpenCV saw it
    move = -np.array(m1["error_xy_cm"]) / 100
    detected = np.array(m1["detected_m"])
    dest = detected + move + rng.normal(0, PLACE_NOISE_M, 2)
    s3 = f"3. Fix (rule): move cup ({move[0]*100:+.1f}, {move[1]*100:+.1f}) cm"
    arm.cup = np.array([*scene.data.qpos[scene.cup_q:scene.cup_q + 2], scene.data.qpos[scene.cup_q + 2]])
    arm.pick_from(detected, s3)
    arm.place_at(dest, s3)
    video.add_motion(arm.take_frames())

    # 4. check again
    m2, img2 = video.add_check(75, "4. OpenCV check")
    cv2.imwrite(f"{out}/check_2.png", img2)
    verdict = "cup on target after 1 fix" if m2["success"] else "still off target"
    video.add_hold(45, f"Done: {verdict}  ({m1['error_cm']:.1f} cm -> {m2['error_cm']:.1f} cm)", img2)
    video.close()

    result = {"target_m": target.round(4).tolist(), "injected_miss_cm": [a.dx, a.dy],
              "check_1": {k: m1[k] for k in ("detected_m", "error_xy_cm", "error_cm", "cup_conf", "success")},
              "fix_move_cm": (move * 100).round(2).tolist(),
              "check_2": {k: m2[k] for k in ("detected_m", "error_xy_cm", "error_cm", "cup_conf", "success")},
              "frames": video.n}
    with open(f"{out}/demo_fix.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Saved {out}/demo_fix.mp4")


if __name__ == "__main__":
    main()
