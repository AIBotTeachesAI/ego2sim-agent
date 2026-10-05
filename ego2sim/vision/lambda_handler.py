"""AWS Lambda entry point (flow steps 2-3): S3 upload -> vision pipeline -> results back to S3.

Expected layout:  input/<run_id>/calibration.json   (upload first)
                  input/<run_id>/video.mov          (upload second; this triggers the run)
Results go to:    output/<run_id>/...
"""
import json
import os
import shutil
import time
import traceback
import urllib.parse

import boto3

from .pipeline import run

s3 = boto3.client("s3")
MODEL = os.environ.get(
    "HAND_MODEL",
    os.path.join(os.environ.get("LAMBDA_TASK_ROOT", "."), "models", "hand_landmarker.task"),
)
OUTPUTS = ["trajectory.json", "metrics.json", "overlay.mp4", "hand_landmarks_filled.npy"]


def process(bucket, video_key):
    parts = video_key.split("/")
    if len(parts) != 3 or parts[0] != "input":
        raise ValueError(f"Unexpected key {video_key}; expected input/<run_id>/<video>")
    run_id, video_name = parts[1], parts[2]
    work = f"/tmp/ego2sim_{run_id}"
    out = os.path.join(work, "out")
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    try:
        video = os.path.join(work, video_name)
        calib = os.path.join(work, "calibration.json")
        s3.download_file(bucket, video_key, video)
        s3.download_file(bucket, f"input/{run_id}/calibration.json", calib)

        traj = run(video, calib, out, MODEL)

        for name in OUTPUTS:
            s3.upload_file(os.path.join(out, name), bucket, f"output/{run_id}/{name}")

        result = {"run_id": run_id, "status": "ok", "events": traj["events"],
                  **traj["metrics"], "total_s": round(time.time() - t0, 2)}
        print(json.dumps({"ego2sim_run": result}))   # one structured line per run -> CloudWatch
        return result
    except Exception as e:
        err = {"run_id": run_id, "status": "error", "error": repr(e),
               "trace": traceback.format_exc(), "total_s": round(time.time() - t0, 2)}
        print(json.dumps({"ego2sim_run": err}))
        s3.put_object(Bucket=bucket, Key=f"output/{run_id}/error.json",
                      Body=json.dumps(err, indent=2).encode())
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)       # Lambda reuses /tmp between runs


def handler(event, context):
    results = []
    for rec in event.get("Records", []):
        bucket = rec["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(rec["s3"]["object"]["key"])
        results.append(process(bucket, key))
    return {"statusCode": 200, "results": results}
