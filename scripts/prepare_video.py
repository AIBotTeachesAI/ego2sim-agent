"""Normalize a phone video to 8-bit H.264 so every machine decodes identical pixels.

iPhone HDR video is 10-bit HEVC. Different FFmpeg builds (macOS vs Linux) convert 10-bit to 8-bit
differently, shifting colors enough to break HSV cup tracking. We decode once on the Mac (the decode
the HSV thresholds were tuned on) and re-encode as 8-bit yuv420p H.264, which decodes the same everywhere.
"""
import argparse
import subprocess

import cv2
import imageio_ffmpeg


def normalize(src, dst, crf=12):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", repr(fps), "-i", "-",
           "-an", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-colorspace", "smpte170m", "-color_range", "tv",
           "-movflags", "+faststart", dst]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        p.stdin.write(fr.tobytes())
        n += 1
    cap.release()
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    return n


def main():
    ap = argparse.ArgumentParser(description="Normalize phone video to 8-bit H.264")
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    print(f"Wrote {a.dst} ({normalize(a.src, a.dst)} frames)")


if __name__ == "__main__":
    main()
