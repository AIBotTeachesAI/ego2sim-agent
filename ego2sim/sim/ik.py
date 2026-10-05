"""Inverse kinematics with mink: wrist targets (robot meters) -> Panda joint trajectory."""
import mink
import mujoco
import numpy as np

PANDA_DIR = "third_party/mujoco_menagerie/franka_emika_panda"
PANDA_XML = f"{PANDA_DIR}/panda.xml"

TCP_OFFSET = 0.1034                          # flange-to-fingertip (m)
DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])   # gripper pointing down (w,x,y,z)
EE_BODY = "hand"                             # Panda has no named sites; use the hand body


def solve_trajectory(wrist_targets, xml=PANDA_XML, iters=5, dt=0.02):
    """
    wrist_targets: (N,3) fingertip targets in robot meters (NaN rows = hold previous pose).
    Returns joints (N,nq), ee_err_m (N,) = |fingertip actual - target|.
    """
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    cfg = mink.Configuration(model)
    cfg.update(data.qpos)
    task = mink.FrameTask(frame_name=EE_BODY, frame_type="body",
                          position_cost=1.0, orientation_cost=0.5)

    joints, errs = [], []
    for tgt in np.asarray(wrist_targets, dtype=np.float64):
        if np.isnan(tgt).any():
            joints.append(cfg.q.copy())
            errs.append(np.nan)
            continue
        flange = tgt + np.array([0.0, 0.0, TCP_OFFSET])
        task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(wxyz=DOWN_QUAT), flange))
        for _ in range(iters):
            vel = mink.solve_ik(cfg, [task], dt=dt, solver="daqp", damping=1e-3)
            cfg.integrate_inplace(vel, dt=dt)
        joints.append(cfg.q.copy())
        actual = cfg.get_transform_frame_to_world(EE_BODY, "body").translation()
        errs.append(float(np.linalg.norm(actual - flange)))

    return np.array(joints), np.array(errs)
