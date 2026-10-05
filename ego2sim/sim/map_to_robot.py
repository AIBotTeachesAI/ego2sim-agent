"""Map table-frame trajectories into Panda task space (meters)."""
import numpy as np

# Target box in front of the Panda base (meters), well inside its ~0.85 m reach
DST_X = (0.35, 0.55)    # forward
DST_Y = (-0.15, 0.15)   # side to side
FIXED_Z = 0.20          # gripper height above table


def _col(frames, key):
    return np.array([f[key] if f[key] is not None else [np.nan, np.nan] for f in frames], dtype=np.float64)


def map_trajectory(traj, dst_x=DST_X, dst_y=DST_Y, z=FIXED_Z):
    """
    Returns wrist_robot (N,3), cup_robot (N,3), and the bounds used.
    Source bounds = min/max of the wrist path in table units (both wrist and cup use the same map).
    """
    wrist_t = _col(traj["frames"], "wrist_table")
    cup_t = _col(traj["frames"], "cup_table")

    sx = (float(np.nanmin(wrist_t[:, 0])), float(np.nanmax(wrist_t[:, 0])))
    sy = (float(np.nanmin(wrist_t[:, 1])), float(np.nanmax(wrist_t[:, 1])))

    def remap(v, s, d):
        return d[0] + (v - s[0]) / (s[1] - s[0]) * (d[1] - d[0])

    def to_robot(p):
        out = np.full((len(p), 3), np.nan)
        out[:, 0] = remap(p[:, 0], sx, dst_x)
        out[:, 1] = remap(p[:, 1], sy, dst_y)
        out[:, 2] = z
        return out

    bounds = {"src_x": sx, "src_y": sy, "dst_x": dst_x, "dst_y": dst_y, "z": z}
    return to_robot(wrist_t), to_robot(cup_t), bounds
