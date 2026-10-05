"""Cloud run (flow steps 1-4, 8): upload to S3 -> S3-triggered Lambda runs OpenCV 5 analysis
-> download trajectory -> local MuJoCo replay -> composites.

Usage: python scripts/run_cloud.py --take take01
Needs: data/raw/<take>.mp4 (from prepare_video.py), data/calibration/<take>.json, BUCKET in .env
"""
import argparse
import json
import os
import sys
import time

import boto3
from botocore.exceptions import ClientError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ego2sim.sim.replay import replay        # noqa: E402
from make_composite import make_composite    # noqa: E402

RESULTS = ["trajectory.json", "metrics.json", "overlay.mp4", "hand_landmarks_filled.npy"]
DONE_MARKER = RESULTS[-1]   # the handler uploads this last


def load_env():
    env = {}
    with open(os.path.join(ROOT, ".env")) as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                env[k] = v
    return env


def exists(s3, bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def main():
    ap = argparse.ArgumentParser(description="Ego2Sim cloud run")
    ap.add_argument("--take", required=True)
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()

    env = load_env()
    bucket = env["BUCKET"]
    s3 = boto3.client("s3", region_name=env.get("AWS_REGION", "us-west-2"))

    video = os.path.join(ROOT, "data/raw", f"{a.take}.mp4")
    calib = os.path.join(ROOT, "data/calibration", f"{a.take}.json")
    run_id = f"{a.take}-{int(time.time())}"
    out = os.path.join(ROOT, "outputs", f"{a.take}_cloud")
    os.makedirs(out, exist_ok=True)

    print(f"[1/4] upload -> s3://{bucket}/input/{run_id}/")
    s3.upload_file(calib, bucket, f"input/{run_id}/calibration.json")   # first: does not trigger
    t0 = time.time()
    s3.upload_file(video, bucket, f"input/{run_id}/video.mp4")          # second: triggers Lambda

    print("[2/4] waiting for Lambda (OpenCV 5 analysis) ", end="", flush=True)
    while True:
        if exists(s3, bucket, f"output/{run_id}/{DONE_MARKER}"):
            break
        if exists(s3, bucket, f"output/{run_id}/error.json"):
            err = json.loads(s3.get_object(Bucket=bucket, Key=f"output/{run_id}/error.json")["Body"].read())
            print("\nCloud run FAILED:", err["error"])
            sys.exit(1)
        if time.time() - t0 > a.timeout:
            print("\nTimed out waiting for results")
            sys.exit(1)
        print(".", end="", flush=True)
        time.sleep(5)
    cloud_latency = round(time.time() - t0, 1)
    print(f" done in {cloud_latency}s")

    print("[3/4] download results")
    for name in RESULTS:
        s3.download_file(bucket, f"output/{run_id}/{name}", os.path.join(out, name))
    with open(os.path.join(out, "trajectory.json")) as f:
        traj = json.load(f)

    print("[4/4] local MuJoCo replay + composites")
    _, sim_metrics = replay(traj, out)
    for view in ("wide", "closeup"):
        make_composite(video, out, view)

    summary = {"run_id": run_id, "cloud_latency_s": cloud_latency, "events": traj["events"],
               "vision": traj["metrics"], "sim": {k: v for k, v in sim_metrics.items() if k != "bounds"}}
    with open(os.path.join(out, "cloud_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
