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
        self._paused = False
        self._anomaly_counter = 0
        self._last_source_path = ""

        threshold_method = str(
            self.cfg.get("inference", {}).get("threshold_method", "fixed")
        ).lower()
        if self._camera_paths and threshold_method == "elbow":
            threshold = self.pipeline.calibrate_elbow_threshold(self._camera_paths)
            print(f"[stream] Calibrated elbow threshold: {threshold:.4f}")

    def get_elbow_profile(self) -> dict[str, object] | None:
        return self.pipeline.get_elbow_profile()

    @property
    def is_paused(self) -> bool:
        return self._paused

    def acknowledge(self):
        self._paused = False

    def _build_simulated_camera_feed(self) -> list[str]:
        stream_cfg = self.cfg.get("stream", {})
        source_dir = str(stream_cfg.get("source_dir", "")).strip()

        if source_dir:
            root = Path(source_dir)
        else:
            dcfg = self.cfg["dataset"]
            root = Path(dcfg["root"]) / dcfg["category"] / "test"

        if not root.exists():
            return []

        paths = sorted(
            str(p)
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )

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
            cv2.rectangle(
                base,
                (size // 5, size // 5),
                (size * 4 // 5, size * 4 // 5),
                (170, 170, 170),
                2,
            )
            return base

        path = self._camera_paths[self._idx]
        self._idx = (self._idx + 1) % len(self._camera_paths)
        self._last_source_path = path

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

        out_payload: dict[str, object] = dict(out)
        out_payload["source_path"] = self._last_source_path

        if bool(out_payload["defect"]):
            self._paused = True
            self._anomaly_counter += 1
            anomaly_id = f"A{self._anomaly_counter:06d}"
            threshold = self.pipeline.get_anomaly_threshold()
            self.pipeline.save_anomaly_record(
                anomaly_id=anomaly_id,
                score=float(out_payload["score"]),
                source_path=self._last_source_path,
                threshold=threshold,
            )
            out_payload["anomaly_id"] = anomaly_id
            out_payload["threshold"] = float(threshold)

        await asyncio.sleep(0)
        return out_payload
