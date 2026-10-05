"""End to end for one take: video -> vision (step 3) -> replay (step 4) -> composites (step 8).

Conventions: data/raw/<take>.mp4 (normalized with scripts/prepare_video.py), data/calibration/<take>.json, outputs/<take>/
Use --traj to skip vision and start from an existing trajectory.json (e.g. downloaded from AWS).
"""
import argparse
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ego2sim.sim.replay import replay                 # noqa: E402
from ego2sim.vision.pipeline import run as run_vision  # noqa: E402
from make_composite import make_composite             # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Ego2Sim end-to-end pipeline")
    p.add_argument("--take", required=True, help="e.g. take01")
    p.add_argument("--traj", help="existing trajectory.json; skips the vision step")
    a = p.parse_args()

    video = os.path.join(ROOT, "data/raw", f"{a.take}.mp4")
    calib = os.path.join(ROOT, "data/calibration", f"{a.take}.json")
    out = os.path.join(ROOT, "outputs", a.take)
    os.makedirs(out, exist_ok=True)
    timings = {}

    t = time.time()
    if a.traj:
        if os.path.abspath(a.traj) != os.path.join(out, "trajectory.json"):
            shutil.copy(a.traj, os.path.join(out, "trajectory.json"))
        with open(os.path.join(out, "trajectory.json")) as f:
            traj = json.load(f)
        print("[1/3] vision: skipped, using", a.traj)
    else:
        print("[1/3] vision ...")
        traj = run_vision(video, calib, out)
    timings["vision_s"] = round(time.time() - t, 2)

    t = time.time()
    print("[2/3] sim replay ...")
    _, sim_metrics = replay(traj, out)
    timings["sim_s"] = round(time.time() - t, 2)

    t = time.time()
    print("[3/3] composites ...")
    for view in ("wide", "closeup"):
        if os.path.exists(os.path.join(out, "overlay.mp4")):
            make_composite(video, out, view)
        else:
            print("  no overlay.mp4 in", out, "- skipping composites")
            break
    timings["composite_s"] = round(time.time() - t, 2)

    summary = {"take": a.take, "events": traj["events"], "vision": traj["metrics"],
               "sim": {k: v for k, v in sim_metrics.items() if k != "bounds"}, "timings": timings}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
