from __future__ import annotations

import struct
import time

import cv2
import numpy as np

MAGIC = b"DFWS"
VERSION = 1
HEADER_FMT = "<4sHHHHIIffQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def encode_rgb_to_jpeg_bytes(image_rgb: np.ndarray, quality: int = 85) -> bytes:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG")
    return buf.tobytes()


def encode_rgb_to_png_bytes(image_rgb: np.ndarray, compression: int = 3) -> bytes:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), int(compression)])
    if not ok:
        raise RuntimeError("Failed to encode heatmap as PNG")
    return buf.tobytes()


def build_packet(
    frame_rgb: np.ndarray,
    heatmap_rgb: np.ndarray,
    score: float,
    defect: bool,
    latency_ms: float,
) -> bytes:
    frame_bytes = encode_rgb_to_jpeg_bytes(frame_rgb)
    heatmap_bytes = encode_rgb_to_png_bytes(heatmap_rgb)

    height, width = frame_rgb.shape[:2]
    flags = 1 if defect else 0

    header = struct.pack(
        HEADER_FMT,
        MAGIC,
        VERSION,
        flags,
        width,
        height,
        len(frame_bytes),
        len(heatmap_bytes),
        float(score),
        float(latency_ms),
        int(time.time_ns()),
    )
    return header + frame_bytes + heatmap_bytes
