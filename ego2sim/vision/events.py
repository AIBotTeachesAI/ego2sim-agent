"""Detect grasp and release frames from hand-gap structure and hand-to-cup distance."""
import numpy as np


def detect_events(gaps, hand_cup_dist, margin=0.02, sustain=10):
    """
    gaps: [(start, end, method)] from fill_hand_gaps. The first long ('cup_offset') gap is the grasp.
    hand_cup_dist: (N,) wrist-to-cup distance in table units.
    Release = first frame after the grasp where distance stays above
              (carry baseline + margin) for `sustain` consecutive frames.
    """
    long_gaps = [g for g in gaps if g[2] == "cup_offset"]
    if not long_gaps:
        raise ValueError("No long occlusion gap found: cannot locate the grasp")
    gs, ge = long_gaps[0][0], long_gaps[0][1]

    d = np.asarray(hand_cup_dist, dtype=np.float64)
    baseline = float(np.nanmedian(d[gs:ge]))
    thr = baseline + margin
    above = d > thr

    release = None
    for i in range(ge, len(d) - sustain + 1):
        if above[i:i + sustain].all():
            release = i
            break

    return {
        "grasp_start": int(gs),
        "grasp_end": int(ge),
        "release_frame": None if release is None else int(release),
        "carry_baseline": round(baseline, 4),
        "release_threshold": round(thr, 4),
    }
