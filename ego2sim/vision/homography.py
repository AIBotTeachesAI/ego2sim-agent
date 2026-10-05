"""Tabletop plane calibration: image pixels -> normalized table coordinates (OpenCV 5)."""
import json

import cv2
import numpy as np


def load_calibration(path):
    with open(path) as f:
        return json.load(f)


def compute_homography(calib):
    src = np.array(calib["image_points"], dtype=np.float32)
    dst = np.array(calib["table_points"], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    if H is None:
        raise ValueError("Homography failed: check the 4 calibration points")
    return H


def pixels_to_table(points_px, H):
    """(N,2) pixel coords -> (N,2) table coords. Rows with NaN pass through as NaN."""
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    out = np.full_like(pts, np.nan)
    valid = ~np.isnan(pts).any(axis=1)
    if valid.any():
        mapped = cv2.perspectiveTransform(pts[valid].reshape(-1, 1, 2), H)
        out[valid] = mapped.reshape(-1, 2)
    return out
