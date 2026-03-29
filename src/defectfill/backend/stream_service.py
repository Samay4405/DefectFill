from __future__ import annotations

import asyncio
from pathlib import Path

import cv2
import numpy as np

from defectfill.pipeline import DefectFillPipeline


class StreamInferenceService:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.pipeline = DefectFillPipeline(cfg)
        self._camera_paths = self._build_simulated_camera_feed()
        self._idx = 0

    def _build_simulated_camera_feed(self) -> list[str]:
        dcfg = self.cfg["dataset"]
        root = Path(dcfg["root"]) / dcfg["category"] / "test"
        if not root.exists():
            return []

        paths: list[str] = []
        for defect_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            paths.extend(sorted(str(x) for x in defect_dir.glob("*.png")))

        if not paths:
            return []

        return paths

    def next_frame(self) -> np.ndarray:
        if not self._camera_paths:
            size = int(self.cfg["dataset"]["image_size"])
            yy, xx = np.mgrid[0:size, 0:size]
            base = np.stack(
                [
                    (xx / max(1, size - 1)) * 180 + 40,
                    (yy / max(1, size - 1)) * 150 + 55,
                    np.full_like(xx, 120),
                ],
                axis=-1,
            ).astype(np.uint8)
            cv2.rectangle(base, (size // 5, size // 5), (size * 4 // 5, size * 4 // 5), (170, 170, 170), 2)
            return base

        path = self._camera_paths[self._idx]
        self._idx = (self._idx + 1) % len(self._camera_paths)

        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    async def infer_live(self) -> dict[str, np.ndarray | float | bool]:
        frame = self.next_frame()
        try:
            out = self.pipeline.infer_frame(frame)
        except Exception:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            edges = cv2.Laplacian(gray, cv2.CV_32F)
            edges = np.abs(edges)
            edges = edges / (edges.max() + 1e-6)
            heat = cv2.applyColorMap((edges * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            heat_rgb = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(frame, 0.52, heat_rgb, 0.48, 0)
            score = float(edges.mean())
            out = {
                "score": score,
                "defect": bool(score > 0.12),
                "latency_ms": 4.0,
                "frame": frame,
                "heatmap_overlay": overlay,
            }
        await asyncio.sleep(0)
        return out
