"""MuJoCo replay (flow step 4): trajectory.json -> Panda + cup replay videos."""
import argparse
import json
import os

import mediapy as media
import mujoco
import numpy as np

from .ik import PANDA_XML, solve_trajectory
from .map_to_robot import map_trajectory

FINGER_OPEN, FINGER_CLOSED = 0.04, 0.0
BLEND_FRAMES = 10       # smooth cup hand-off into the gripper
REOPEN_FRAMES = 20      # gripper reopening after release
CAMERAS = {"closeup": 0.6, "wide": 1.7}   # camera distance per view
RENDER_H, RENDER_W = 480, 640


def make_spec(xml=PANDA_XML):
    """Panda + free-floating red cylinder cup, built in code (menagerie folder untouched)."""
    spec = mujoco.MjSpec.from_file(xml)
    cup = spec.worldbody.add_body(name="cup", pos=[0.45, 0.0, 0.05])
    cup.add_freejoint()
    cup.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.02, 0.045, 0.0],
                 rgba=[0.9, 0.1, 0.1, 1.0], mass=0.05)
    return spec


def build_scene(xml=PANDA_XML):
    return make_spec(xml).compile()


def make_camera(distance):
    cam = mujoco.MjvCamera()
    cam.lookat = [0.45, 0.0, 0.15]
    cam.distance = distance
    cam.azimuth = 135
    cam.elevation = -30
    return cam


def finger_value(i, ev):
    gs, ge, rel = ev["grasp_start"], ev["grasp_end"], ev["release_frame"]
    if i < gs:
        return FINGER_OPEN
    if i < ge:
        return FINGER_OPEN * (1 - (i - gs) / (ge - gs))
    if rel is None or i < rel:
        return FINGER_CLOSED
    return FINGER_CLOSED + min((i - rel) / REOPEN_FRAMES, 1.0) * (FINGER_OPEN - FINGER_CLOSED)


def cup_position(i, ev, cup_r, wrist_r, resting):
    """Tracked before grasp -> blended into gripper -> follows gripper -> frozen at release."""
    gs, rel = ev["grasp_start"], ev["release_frame"]
    if i < gs:
        return None if np.isnan(cup_r[i]).any() else cup_r[i]
    if rel is None or i < rel:
        grasp_pt = wrist_r[i]
        if i < gs + BLEND_FRAMES:
            t = min(max((i - gs + 1) / BLEND_FRAMES, 0.0), 1.0)
            return cup_r[gs - 1] * (1 - t) + grasp_pt * t
        return grasp_pt
    return resting


def replay(traj, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ev = traj["events"]
    wrist_r, cup_r, bounds = map_trajectory(traj)
    joints, ee_err = solve_trajectory(wrist_r)

    model = build_scene()
    data = mujoco.MjData(model)
    jid = lambda n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
    f1, f2 = jid("finger_joint1"), jid("finger_joint2")
    cup_q = model.jnt_qposadr[model.body_jntadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cup")]]

    rel = ev["release_frame"]
    resting = cup_r[rel] if rel is not None else cup_r[-1]

    renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    cams = {k: make_camera(d) for k, d in CAMERAS.items()}
    frames = {k: [] for k in cams}
    cup_path = []

    for i, q in enumerate(joints):
        data.qpos[:7] = q[:7]
        data.qpos[f1] = data.qpos[f2] = finger_value(i, ev)
        mujoco.mj_forward(model, data)

        p = cup_position(i, ev, cup_r, wrist_r, resting)
        if p is not None:
            data.qpos[cup_q:cup_q + 3] = p
            data.qpos[cup_q + 3:cup_q + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        cup_path.append(data.qpos[cup_q:cup_q + 3].copy())

        for k, cam in cams.items():
            renderer.update_scene(data, camera=cam)
            frames[k].append(renderer.render())

    renderer.close()
    videos = {}
    for k, fr in frames.items():
        videos[k] = os.path.join(out_dir, f"replay_{k}.mp4")
        media.write_video(videos[k], fr, fps=traj["fps"])

    np.save(os.path.join(out_dir, "joint_trajectory.npy"), joints)
    sim_metrics = {
        "ik_pos_err_mm_mean": round(float(np.nanmean(ee_err)) * 1000, 3),
        "ik_pos_err_mm_max": round(float(np.nanmax(ee_err)) * 1000, 3),
        "final_cup_robot_m": [round(float(v), 4) for v in cup_path[-1]],
        "target_cup_robot_m": [round(float(v), 4) for v in resting],
        "bounds": bounds,
    }
    with open(os.path.join(out_dir, "sim_metrics.json"), "w") as f:
        json.dump(sim_metrics, f, indent=2)
    return videos, sim_metrics


def main():
    p = argparse.ArgumentParser(description="Ego2Sim MuJoCo replay")
    p.add_argument("--traj", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    with open(a.traj) as f:
        traj = json.load(f)
    videos, m = replay(traj, a.out)
    print(json.dumps(m, indent=2))
    for v in videos.values():
        print("Saved", v)


if __name__ == "__main__":
    main()
