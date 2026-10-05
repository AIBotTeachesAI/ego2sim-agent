"""Verification scene (flow step 5): Panda + cup + top-down camera + 4 green corner markers."""
import mujoco
import numpy as np

from ego2sim.sim.ik import PANDA_XML
from ego2sim.sim.replay import make_spec

CUP_Z = 0.20                      # cup centre height used by the replay
CUP_HALF_H = 0.045
PLANE_Z = CUP_Z + CUP_HALF_H      # markers sit in the plane of the cup's top face (what the camera sees)
TABLE_Z = CUP_Z - CUP_HALF_H      # table surface = bottom of the cup
MARKERS = {                       # known robot-frame positions (m), order: near-left, near-right, far-right, far-left
    "m0": (0.30, -0.25), "m1": (0.70, -0.25), "m2": (0.70, 0.25), "m3": (0.30, 0.25),
}
CAM_NAME = "topdown"
RETREAT_BASE_YAW = 2.8     # rad, joint1 limit is +/-2.8973
CAM_POS = (0.50, 0.0, 1.05)       # straight down: image right = +x, image up = +y
IMG_W, IMG_H = 960, 720
HIDDEN_Z = -5.0                   # below the floor: invisible
OBSTACLE_HALF = 0.03


def make_verify_model():
    spec = make_spec()
    wb = spec.worldbody
    # visual context only (no collisions): floor, a table under the cup, overhead light
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1], rgba=[0.12, 0.12, 0.14, 1],
                contype=0, conaffinity=0)
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.50, 0.0, TABLE_Z - 0.01], size=[0.27, 0.33, 0.01],
                rgba=[0.22, 0.21, 0.20, 1], contype=0, conaffinity=0)
    wb.add_light(pos=[0.5, 0.0, 1.6], dir=[0, 0, -1], diffuse=[0.5, 0.5, 0.5], castshadow=False)
    post_h = PLANE_Z - 0.002 - TABLE_Z
    for name, (x, y) in MARKERS.items():
        # grey post from the table up to the cup-top plane, green cap = the marker OpenCV sees
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[x, y, TABLE_Z + post_h / 2],
                    size=[0.012, post_h / 2, 0], rgba=[0.55, 0.55, 0.55, 1], contype=0, conaffinity=0)
        b = wb.add_body(name=name, pos=[x, y, PLANE_Z])
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.015, 0.015, 0.002],
                   rgba=[0.1, 0.85, 0.1, 1.0], contype=0, conaffinity=0)
    # scenario props, parked out of sight until a scenario places them
    occ = wb.add_body(name="occluder", pos=[0, 0, HIDDEN_Z])      # thin grey card lying on part of the cup
    occ.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.03, 0.0015], rgba=[0.62, 0.60, 0.56, 1],
                 contype=0, conaffinity=0)
    obs = wb.add_body(name="obstacle", pos=[0, 0, HIDDEN_Z])      # blue object left in the workspace
    obs.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[OBSTACLE_HALF, OBSTACLE_HALF, OBSTACLE_HALF],
                 rgba=[0.15, 0.35, 0.90, 1], contype=0, conaffinity=0)
    spec.worldbody.add_camera(name=CAM_NAME, pos=list(CAM_POS), quat=[1, 0, 0, 0])
    model = spec.compile()
    model.vis.global_.offwidth, model.vis.global_.offheight = IMG_W, IMG_H
    return model


def home_arm_q():
    """Panda 'home' keyframe (arm + fingers): rotated backwards: the retreat pose before the camera check."""
    m = mujoco.MjModel.from_xml_path(PANDA_XML)
    q = m.key_qpos[0].copy() if m.nkey else np.zeros(m.nq)
    q[0] = RETREAT_BASE_YAW   # swing the arm around its base so it points backwards, out of the camera view
    return q


class VerifyScene:
    def __init__(self):
        self.model = make_verify_model()
        self.data = mujoco.MjData(self.model)
        cup_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cup")
        self.cup_q = self.model.jnt_qposadr[self.model.body_jntadr[cup_body]]
        self.home = home_arm_q()
        self.renderer = mujoco.Renderer(self.model, height=IMG_H, width=IMG_W)
        self.occ_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "occluder")
        self.obs_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obstacle")
        self.occluder_offset = None     # (dx, dy) of the card relative to the cup centre, or None
        self.obstacle_xy = None         # (x, y) of the blue object on the table, or None

    def set_state(self, arm_q, cup_xy, cup_z=CUP_Z):
        self.data.qpos[:len(arm_q)] = arm_q
        self.data.qpos[self.cup_q:self.cup_q + 3] = [cup_xy[0], cup_xy[1], cup_z]
        self.data.qpos[self.cup_q + 3:self.cup_q + 7] = [1, 0, 0, 0]
        if self.occluder_offset is not None:    # the card rides on top of the cup
            dx, dy = self.occluder_offset
            self.model.body_pos[self.occ_id] = [cup_xy[0] + dx, cup_xy[1] + dy, cup_z + CUP_HALF_H + 0.0015]
        else:
            self.model.body_pos[self.occ_id] = [0, 0, HIDDEN_Z]
        if self.obstacle_xy is not None:
            self.model.body_pos[self.obs_id] = [self.obstacle_xy[0], self.obstacle_xy[1], TABLE_Z + OBSTACLE_HALF]
        else:
            self.model.body_pos[self.obs_id] = [0, 0, HIDDEN_Z]
        mujoco.mj_forward(self.model, self.data)

    def render_topdown(self):
        """RGB image from the top-down camera."""
        self.renderer.update_scene(self.data, camera=CAM_NAME)
        return self.renderer.render()

    def close(self):
        self.renderer.close()
