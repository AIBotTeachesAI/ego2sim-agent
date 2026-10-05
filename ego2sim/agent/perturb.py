"""Injected errors and test scenarios (all disclosed in the report): seeded, reproducible."""
import numpy as np

OFFSET_RANGE_M = (0.08, 0.12)   # 8-12 cm initial miss, random direction
SLIP_BIAS_M = 0.04              # 'slip' scenario: every re-place lands ~4 cm off in the same direction
OCCLUDER_OFFSET = (0.0, 0.026)  # 'occluded' scenario: a card lying over ~60% of the cup top
SCENARIOS = ("clean", "occluded", "obstructed", "slip", "slip_repair")


def sample_offset(seed, lo=OFFSET_RANGE_M[0], hi=OFFSET_RANGE_M[1]):
    rng = np.random.default_rng(seed)
    r, th = rng.uniform(lo, hi), rng.uniform(0, 2 * np.pi)
    return np.array([r * np.cos(th), r * np.sin(th)])


def make_scenario(name, seed, target, offset):
    """Props and faults for one trial. Returns {occluder_offset, obstacle_xy, slip_bias}."""
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name}; available: {SCENARIOS}")
    target, offset = np.asarray(target, float), np.asarray(offset, float)
    sc = {"occluder_offset": None, "obstacle_xy": None, "slip_bias": np.zeros(2)}
    if name == "occluded":
        sc["occluder_offset"] = OCCLUDER_OFFSET
    elif name == "obstructed":
        # blue object on the table beside the cup's path (not on the cup, the target or a marker)
        u = offset / np.linalg.norm(offset)
        perp = np.array([-u[1], u[0]])
        for sign in (1, -1):
            p = target + 0.5 * offset + sign * 0.09 * perp
            if 0.36 <= p[0] <= 0.66 and -0.19 <= p[1] <= 0.19:
                break
        sc["obstacle_xy"] = np.clip(p, [0.36, -0.19], [0.66, 0.19])
    elif name in ("slip", "slip_repair"):     # slip_repair: same fault, but the operator can fix it
        th = np.random.default_rng(seed + 2000).uniform(0, 2 * np.pi)
        sc["slip_bias"] = SLIP_BIAS_M * np.array([np.cos(th), np.sin(th)])
    return sc
