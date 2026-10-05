"""Fill hand-tracking gaps: interpolate short gaps, use cup displacement for long (grasp) gaps."""
import numpy as np


def _gaps(missing):
    """Yield (start, end_exclusive) runs where missing is True."""
    i, n = 0, len(missing)
    while i < n:
        if missing[i]:
            j = i
            while j < n and missing[j]:
                j += 1
            yield i, j
            i = j
        else:
            i += 1


def fill_hand_gaps(landmarks, cup_norm, short_max=5):
    """
    landmarks: (N,21,3) normalized, NaN where missing.
    cup_norm:  (N,2) cup centroid normalized to [0,1], NaN where missing.
    Returns filled (N,21,3), source (N,) strings, gaps [(start, end, method)].
    """
    lm = landmarks.copy()
    missing = np.isnan(lm[:, 0, 0])
    source = np.where(missing, "", "mediapipe").astype(object)
    gaps = []

    for s, e in _gaps(missing):
        n = e - s
        if s == 0 and e == len(lm):
            raise ValueError("No hand detected in any frame")
        if s == 0:                                   # leading gap: back-fill
            lm[s:e] = lm[e]
            source[s:e], method = "hold", "hold_start"
        elif e == len(lm):                           # trailing gap: hold last
            lm[s:e] = lm[s - 1]
            source[s:e], method = "hold", "hold_end"
        elif n <= short_max:                         # short gap: linear interp
            before, after = lm[s - 1], lm[e]
            for j in range(s, e):
                t = (j - s + 1) / (n + 1)
                lm[j] = before * (1 - t) + after * t
            source[s:e], method = "interp", "interp"
        else:                                        # long gap: follow the cup
            last_good, cup_start = lm[s - 1], cup_norm[s - 1]
            for j in range(s, e):
                if not np.isnan(cup_norm[j]).any() and not np.isnan(cup_start).any():
                    d = cup_norm[j] - cup_start
                    lm[j] = last_good + np.array([d[0], d[1], 0.0])
                    source[j] = "cup_offset"
                else:
                    lm[j] = last_good
                    source[j] = "hold"
            method = "cup_offset"
        gaps.append((s, e, method))

    return lm, source, gaps
