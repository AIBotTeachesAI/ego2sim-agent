"""OpenCV 5 verification (flow step 5): top-down sim image -> cup position in robot meters -> error vs target."""
import cv2
import numpy as np

from .scene import MARKERS

MARKER_ORDER = ["m0", "m1", "m2", "m3"]            # near-left, near-right, far-right, far-left
MARKER_XY = np.array([MARKERS[k] for k in MARKER_ORDER], dtype=np.float32)
CUP_RADIUS_M = 0.02
GREEN = ((40, 100, 80), (85, 255, 255))
CUP_TOP = [((0, 120, 150), (10, 255, 255)),        # bright red only: the cup's top face,
           ((170, 120, 150), (180, 255, 255))]     # not its darker side wall


def _blobs(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        a = cv2.contourArea(c)
        if a >= min_area:
            M = cv2.moments(c)
            out.append((M["m10"] / M["m00"], M["m01"] / M["m00"], a))
    return sorted(out, key=lambda b: -b[2])


def detect_markers(bgr):
    """Pixel centroids of the 4 markers, in MARKER_ORDER. Image right = +x, image up = +y."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blobs = _blobs(cv2.inRange(hsv, *GREEN), 50)[:4]
    if len(blobs) < 4:
        raise ValueError(f"found {len(blobs)} markers, need 4")
    pts = np.array([b[:2] for b in blobs], dtype=np.float32)
    by_y = pts[np.argsort(-pts[:, 1])]                 # bottom of image (near, -y) first
    bottom = by_y[:2][np.argsort(by_y[:2, 0])]
    top = by_y[2:][np.argsort(by_y[2:, 0])]
    return np.array([bottom[0], bottom[1], top[1], top[0]], dtype=np.float32)


def detect_cup(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = None
    for lo, hi in CUP_TOP:
        m = cv2.inRange(hsv, lo, hi)
        mask = m if mask is None else (mask | m)
    blobs = _blobs(mask, 30)
    return blobs[0] if blobs else None


def detect_obstructions(bgr, markers_px, min_area=400):
    """Unexpected coloured objects inside the workspace (anything saturated that is not the cup or a marker)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    colourful = cv2.inRange(hsv, (0, 90, 60), (180, 255, 255))
    known = cv2.inRange(hsv, *GREEN) | cv2.inRange(hsv, (0, 90, 40), (12, 255, 255)) | \
        cv2.inRange(hsv, (165, 90, 40), (180, 255, 255))
    mask = cv2.bitwise_and(colourful, cv2.bitwise_not(known))
    region = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(region, [np.asarray(markers_px, np.int32)], 255)
    mask = cv2.bitwise_and(mask, region)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        a = cv2.contourArea(c)
        if a >= min_area:
            x, y, w, h = cv2.boundingRect(c)
            out.append({"bbox_px": [int(x), int(y), int(w), int(h)], "area_px": round(float(a), 1)})
    return sorted(out, key=lambda o: -o["area_px"])


def _px_to_robot(pts_px, H):
    return cv2.perspectiveTransform(np.asarray(pts_px, np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)


def _direction(err_m, dead_cm=0.5):
    parts = []
    for axis, v in (("x", err_m[0]), ("y", err_m[1])):
        if abs(v) * 100 >= dead_cm:
            parts.append(f"{'+' if v > 0 else '-'}{axis}")
    return ", ".join(parts) if parts else "on target"


def measure(rgb, target_xy, tol_cm=2.0):
    """OpenCV measurement of where the cup is, versus where it should be."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mk = detect_markers(bgr)
    H, _ = cv2.findHomography(mk, MARKER_XY)

    px_d = [np.linalg.norm(mk[i] - mk[j]) for i, j in ((0, 1), (1, 2), (2, 3), (3, 0))]
    m_d = [np.linalg.norm(MARKER_XY[i] - MARKER_XY[j]) for i, j in ((0, 1), (1, 2), (2, 3), (3, 0))]
    px_per_m = float(np.mean(np.array(px_d) / np.array(m_d)))
    expected_area = np.pi * (CUP_RADIUS_M * px_per_m) ** 2

    target = np.asarray(target_xy, dtype=np.float64)[:2]
    obstructions = detect_obstructions(bgr, mk)
    out = {"target_m": [round(float(v), 4) for v in target], "markers_px": mk.round(1).tolist(),
           "homography": H.tolist(), "px_per_m": round(px_per_m, 1),
           "obstructions": obstructions, "workspace_clear": len(obstructions) == 0}

    cup = detect_cup(bgr)
    if cup is None:
        out.update(cup_px=None, detected_m=None, error_xy_cm=None, error_cm=None,
                   error_dir="cup not found", cup_conf=0.0, success=False)
        return out

    detected = _px_to_robot([cup[:2]], H)[0]
    err = detected - target
    conf = min(cup[2] / expected_area, expected_area / cup[2])
    error_cm = float(np.linalg.norm(err) * 100)
    out.update(
        cup_px=[round(cup[0], 1), round(cup[1], 1)],
        cup_area_px=round(cup[2], 1),
        detected_m=[round(float(v), 4) for v in detected],
        error_xy_cm=[round(float(v) * 100, 2) for v in err],
        error_cm=round(error_cm, 2),
        error_dir=_direction(err),
        cup_conf=round(float(conf), 3),
        success=bool(conf >= 0.5 and error_cm < tol_cm),
    )
    return out


def draw_overlay(rgb, m):
    """Top-down image annotated with what OpenCV measured."""
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    H = np.array(m["homography"])
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (x, y) in enumerate(m["markers_px"]):
        cv2.circle(img, (int(x), int(y)), 24, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, MARKER_ORDER[i], (int(x) + 28, int(y) + 7), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

    tgt_px = cv2.perspectiveTransform(np.array([[m["target_m"]]], np.float64), np.linalg.inv(H))[0, 0]
    tx, ty = int(tgt_px[0]), int(tgt_px[1])
    cv2.drawMarker(img, (tx, ty), (0, 255, 0), cv2.MARKER_CROSS, 44, 3, cv2.LINE_AA)
    cv2.putText(img, "target", (tx - 30, ty - 30), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    for o in m.get("obstructions", []):
        x, y, w, h = o["bbox_px"]
        cv2.rectangle(img, (x - 6, y - 6), (x + w + 6, y + h + 6), (0, 140, 255), 3, cv2.LINE_AA)
        cv2.putText(img, "unexpected object", (x - 6, y - 14), font, 0.7, (0, 140, 255), 2, cv2.LINE_AA)

    if m["cup_px"] is not None:
        cx, cy = int(m["cup_px"][0]), int(m["cup_px"][1])
        cv2.circle(img, (cx, cy), 34, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(img, "cup (OpenCV)", (cx - 60, cy + 58), font, 0.7, (80, 80, 255), 2, cv2.LINE_AA)
        if m["error_cm"] >= 0.5 and not m["success"]:
            cv2.arrowedLine(img, (cx, cy), (tx, ty), (0, 220, 255), 4, cv2.LINE_AA, tipLength=0.12)
            mx, my = (cx + tx) // 2, (cy + ty) // 2
            cv2.putText(img, f"correction {m['error_cm']:.1f} cm", (mx + 14, my - 10), font, 0.7, (0, 220, 255), 2, cv2.LINE_AA)
        verdict = "OK" if m["success"] else "FAIL"
        txt = f"error {m['error_cm']:.1f} cm ({m['error_dir']})  conf {m['cup_conf']:.2f}  -> {verdict}"
    else:
        txt = "cup not found -> FAIL"
    if not m.get("workspace_clear", True):
        txt += " | OBJECT IN WORKSPACE"
    cv2.rectangle(img, (0, 0), (img.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(img, "OpenCV: " + txt, (14, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img
