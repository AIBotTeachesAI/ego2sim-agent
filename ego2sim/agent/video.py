"""Two-panel video: robot arm (MuJoCo) | top-down camera + OpenCV check, with a caption bar."""
import cv2
import mujoco
import numpy as np

from .verify import draw_overlay, measure

PANEL_W, PANEL_H, CAPTION_H = 640, 480, 60
FONT = cv2.FONT_HERSHEY_SIMPLEX


class DemoVideo:
    def __init__(self, scene, target_xy, path, fps=30):
        self.scene, self.target = scene, np.asarray(target_xy, dtype=np.float64)[:2]
        self.side = mujoco.Renderer(scene.model, height=PANEL_H, width=PANEL_W)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat = [0.50, 0.02, 0.25]
        self.cam.distance = 1.0
        self.cam.azimuth = 150
        self.cam.elevation = -25
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                      (2 * PANEL_W, PANEL_H + CAPTION_H))
        # markers never move: one reference view gives the homography for the live target cross
        scene.set_state(scene.home, self.target)
        H = np.array(measure(scene.render_topdown(), self.target)["homography"])
        self.target_px = cv2.perspectiveTransform(np.array([[self.target]]), np.linalg.inv(H))[0, 0]
        self.n = 0

    def _set(self, fr):
        q = self.scene.home.copy()
        q[:7] = fr["q"]
        q[7] = q[8] = fr["finger"]
        self.scene.set_state(q, fr["cup"][:2], fr["cup"][2])

    def _left(self):
        self.side.update_scene(self.scene.data, camera=self.cam)
        img = cv2.cvtColor(self.side.render(), cv2.COLOR_RGB2BGR)
        cv2.putText(img, "robot arm (MuJoCo)", (12, PANEL_H - 14), FONT, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
        return img

    def _live_right(self, rgb):
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        tx, ty = int(self.target_px[0]), int(self.target_px[1])
        cv2.drawMarker(img, (tx, ty), (0, 160, 0), cv2.MARKER_CROSS, 44, 2, cv2.LINE_AA)
        cv2.putText(img, "target", (tx - 30, ty - 30), FONT, 0.7, (0, 160, 0), 2, cv2.LINE_AA)
        cv2.putText(img, "top-down camera (live)", (14, 30), FONT, 0.75, (200, 200, 200), 2, cv2.LINE_AA)
        return img

    def _write(self, left, right_bgr, caption):
        right = cv2.resize(right_bgr, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
        bar = np.zeros((CAPTION_H, 2 * PANEL_W, 3), np.uint8)
        if len(caption) <= 80:
            cv2.putText(bar, caption, (16, 40), FONT, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        else:                                   # long decision reasons: two smaller lines
            words, line1 = caption.split(), ""
            while words and len(line1) + len(words[0]) < 112:
                line1 += words.pop(0) + " "
            line2 = " ".join(words)
            if len(line2) > 112:
                line2 = line2[:109] + "..."
            cv2.putText(bar, line1.strip(), (16, 25), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(bar, line2, (16, 50), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        frame = np.vstack([bar, np.hstack([left, right])])
        cv2.line(frame, (PANEL_W, CAPTION_H), (PANEL_W, CAPTION_H + PANEL_H), (90, 90, 90), 1)
        self.writer.write(frame)
        self.n += 1

    def add_motion(self, frames):
        for fr in frames:
            self._set(fr)
            self._write(self._left(), self._live_right(self.scene.render_topdown()), fr["stage"])

    def add_check(self, n, label, tol_cm=2.0):
        """OpenCV check of the current scene (arm already retreated). Returns the measurement."""
        rgb = self.scene.render_topdown()
        meas = measure(rgb, self.target, tol_cm)
        overlay = draw_overlay(rgb, meas)
        if meas["cup_px"] is None:
            caption = f"{label}: cup not found -> FAIL"
        else:
            verdict = "OK" if meas["success"] else "FAIL"
            caption = f"{label}: error {meas['error_cm']:.1f} cm ({meas['error_dir']}) -> {verdict}"
        left = self._left()
        for _ in range(n):
            self._write(left, overlay, caption)
        return meas, overlay

    def add_hold(self, n, caption, right_bgr):
        left = self._left()
        for _ in range(n):
            self._write(left, right_bgr, caption)

    def close(self):
        self.writer.release()
        self.side.close()
