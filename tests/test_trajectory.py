"""trajectory.json is the handoff between the cloud vision step and the local robot replay."""
import json
import math
import os

import pytest

SAMPLE = {
    "video": "take01.mp4", "fps": 30.0, "n_frames": 2,
    "frames": [
        {"i": 0, "t": 0.0, "cup_table": [0.44, 0.51], "cup_conf": 0.97,
         "wrist_table": [0.20, 0.55], "hand_source": "mediapipe"},
        {"i": 1, "t": 0.033, "cup_table": [0.44, 0.51], "cup_conf": 0.97,
         "wrist_table": [0.21, 0.55], "hand_source": "interp"},
    ],
    "events": {"grasp_frame": 0, "release_frame": 1},
}


def _is_xy(v):
    return isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(c, (int, float)) and math.isfinite(c) for c in v)


def check_trajectory(traj):
    """The fields the robot replay depends on."""
    assert isinstance(traj.get("frames"), list) and traj["frames"], "frames missing or empty"
    for fr in traj["frames"]:
        assert _is_xy(fr.get("cup_table")), f"bad cup_table in frame {fr.get('i')}"
        assert _is_xy(fr.get("wrist_table")), f"bad wrist_table in frame {fr.get('i')}"
    assert isinstance(traj.get("events"), dict), "events missing"
    if "n_frames" in traj:
        assert traj["n_frames"] == len(traj["frames"])


def test_sample_is_valid():
    check_trajectory(SAMPLE)


def test_missing_wrist_is_rejected():
    bad = json.loads(json.dumps(SAMPLE))
    del bad["frames"][1]["wrist_table"]
    with pytest.raises(AssertionError):
        check_trajectory(bad)


def test_non_finite_cup_is_rejected():
    bad = json.loads(json.dumps(SAMPLE))
    bad["frames"][0]["cup_table"] = [float("nan"), 0.5]
    with pytest.raises(AssertionError):
        check_trajectory(bad)


@pytest.mark.parametrize("path", ["outputs/take01/trajectory.json", "outputs/take01_cloud/trajectory.json"])
def test_pipeline_output_if_present(path):
    if not os.path.exists(path):
        pytest.skip(f"{path} not found (run the pipeline first)")
    with open(path) as f:
        check_trajectory(json.load(f))
