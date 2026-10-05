"""Color-based cup tracking: HSV segmentation + largest contour centroid (OpenCV 5)."""
import cv2
import numpy as np


def cup_mask(frame_bgr, hsv_cfg):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = None
    for r in hsv_cfg["ranges"]:
        m = cv2.inRange(hsv, tuple(r["lower"]), tuple(r["upper"]))
        mask = m if mask is None else (mask | m)
    return mask


def track_cup_frame(frame_bgr, hsv_cfg):
    """Returns (cx, cy, area) in pixels, or None if no cup found."""
    mask = cup_mask(frame_bgr, hsv_cfg)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < hsv_cfg.get("min_area_px", 200):
        return None
    M = cv2.moments(c)
    return (M["m10"] / M["m00"], M["m01"] / M["m00"], area)


def track_cup_video(video_path, hsv_cfg):
    """Returns cup_px (N,2) with NaN where missing, and area (N,) with 0 where missing."""
    cap = cv2.VideoCapture(video_path)
    pts, areas = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = track_cup_frame(frame, hsv_cfg)
        pts.append([np.nan, np.nan] if r is None else [r[0], r[1]])
        areas.append(0.0 if r is None else r[2])
    cap.release()
    return np.array(pts, dtype=np.float64), np.array(areas, dtype=np.float64)
