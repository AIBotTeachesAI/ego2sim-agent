"""Robot arm motion for place / pick / re-place (kinematic, same as the replay): IK waypoints -> frames.

Each frame is {"q": 7 arm joints, "finger": finger opening, "cup": cup centre xyz, "stage": caption}.
While the gripper holds the cup, the cup rides with the fingertips (no contact physics, as in the replay).
"""
import mink
import mujoco
import numpy as np

from ego2sim.sim.ik import DOWN_QUAT, EE_BODY, PANDA_XML, TCP_OFFSET

from .scene import CUP_Z, home_arm_q

HOVER_Z = 0.32                          # fingertip height when carrying / approaching
FINGER_OPEN, FINGER_GRIP = 0.04, 0.02   # cup radius is 2 cm, so 0.02 = fingers touching the cup


def _ease(n):
    t = np.linspace(0.0, 1.0, n + 1)[1:]
    return t * t * (3 - 2 * t)          # smooth start and stop


class Motion:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(PANDA_XML)
        data = mujoco.MjData(self.model)
        if self.model.nkey:
            mujoco.mj_resetDataKeyframe(self.model, data, 0)
        self.home = data.qpos.copy()            # arm forward: IK seed
        self.retreat = home_arm_q()[:7]         # arm swung back, out of the top-down view
        self.cfg = mink.Configuration(self.model)
        self.task = mink.FrameTask(frame_name=EE_BODY, frame_type="body",
                                   position_cost=1.0, orientation_cost=0.5)
        self.q = self.retreat.copy()
        self.finger = FINGER_OPEN
        self.cup = np.array([0.45, 0.0, CUP_Z])
        self.tip = None
        self.frames = []

    # --- low level ---------------------------------------------------------
    def _seed(self, q7):
        q = self.home.copy()
        q[:7] = q7
        self.cfg.update(q)

    def _solve(self, tip, iters):
        flange = np.asarray(tip, dtype=np.float64) + np.array([0.0, 0.0, TCP_OFFSET])
        self.task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(wxyz=DOWN_QUAT), flange))
        for _ in range(iters):
            vel = mink.solve_ik(self.cfg, [self.task], dt=0.02, solver="daqp", damping=1e-3)
            self.cfg.integrate_inplace(vel, dt=0.02)
        return self.cfg.q[:7].copy()

    def _emit(self, stage):
        self.frames.append({"q": self.q.copy(), "finger": float(self.finger),
                            "cup": self.cup.copy(), "stage": stage})

    def take_frames(self):
        f, self.frames = self.frames, []
        return f

    # --- motion primitives -------------------------------------------------
    def start_holding(self, tip, stage, n=15):
        """Arm hovering at `tip` with the cup already in the gripper."""
        self._seed(self.home[:7])
        self.q = self._solve(tip, 400)
        self.tip = np.asarray(tip, dtype=np.float64)
        self.finger, self.cup = FINGER_GRIP, self.tip.copy()
        for _ in range(n):
            self._emit(stage)

    def hold(self, n, stage):
        for _ in range(n):
            self._emit(stage)

    def move_to(self, tip, n, stage, carry=False):
        """Straight-line fingertip move (Cartesian), gripper pointing down."""
        start, tip = self.tip.copy(), np.asarray(tip, dtype=np.float64)
        for t in _ease(n):
            p = start + t * (tip - start)
            self.q = self._solve(p, 8)
            if carry:
                self.cup = p.copy()
            self._emit(stage)
        self.tip = tip

    def gripper(self, to, n, stage):
        f0 = self.finger
        for t in _ease(n):
            self.finger = f0 + t * (to - f0)
            self._emit(stage)

    def retreat_arm(self, n, stage):
        """Joint-space swing back, out of the camera view."""
        q0 = self.q.copy()
        for t in _ease(n):
            self.q = q0 + t * (self.retreat - q0)
            self._emit(stage)
        self.tip = None

    def approach(self, tip, n, stage):
        """Joint-space swing from the retreat pose to hover above `tip`."""
        self._seed(self.home[:7])
        q_goal = self._solve(tip, 400)
        q0 = self.q.copy()
        for t in _ease(n):
            self.q = q0 + t * (q_goal - q0)
            self._emit(stage)
        self._seed(q_goal)
        self.tip = np.asarray(tip, dtype=np.float64)

    # --- composite actions -------------------------------------------------
    def place_at(self, xy, stage):
        """Carry the held cup over `xy`, lower it, let go, lift, retreat."""
        hover, down = [xy[0], xy[1], HOVER_Z], [xy[0], xy[1], CUP_Z]
        self.move_to(hover, 45, stage, carry=True)
        self.move_to(down, 25, stage, carry=True)
        self.gripper(FINGER_OPEN, 12, stage)
        self.move_to(hover, 20, stage)
        self.retreat_arm(40, stage)

    def pick_from(self, xy, stage):
        """Swing in over the cup at `xy`, lower, grip, lift it."""
        hover, down = [xy[0], xy[1], HOVER_Z], [xy[0], xy[1], CUP_Z]
        self.approach(hover, 45, stage)
        self.move_to(down, 25, stage)
        self.gripper(FINGER_GRIP, 12, stage)
        self.move_to(hover, 20, stage, carry=True)
