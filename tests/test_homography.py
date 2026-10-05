"""OpenCV 5 homography: image pixels <-> table coordinates, as used by the vision pipeline."""
import cv2
import numpy as np

# take01 calibration: four table corners clicked in the image, mapped to a unit square
IMAGE_POINTS = np.array([[199, 952], [1879, 970], [1852, 179], [272, 158]], dtype=np.float64)
TABLE_POINTS = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)


def to_table(H, pts):
    return cv2.perspectiveTransform(np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)


def test_opencv_is_version_5():
    assert cv2.__version__.startswith("5."), cv2.__version__


def test_corners_map_to_unit_square():
    H, _ = cv2.findHomography(IMAGE_POINTS, TABLE_POINTS)
    assert np.allclose(to_table(H, IMAGE_POINTS), TABLE_POINTS, atol=1e-6)


def test_round_trip_pixel_table_pixel():
    H, _ = cv2.findHomography(IMAGE_POINTS, TABLE_POINTS)
    pixels = np.array([[969, 540], [400, 300], [1500, 800], [1000, 200]], dtype=np.float64)
    back = to_table(np.linalg.inv(H), to_table(H, pixels))
    assert np.allclose(back, pixels, atol=1e-6)


def test_image_centre_lands_inside_table():
    H, _ = cv2.findHomography(IMAGE_POINTS, TABLE_POINTS)
    x, y = to_table(H, [[969, 540]])[0]
    assert 0.0 < x < 1.0 and 0.0 < y < 1.0
