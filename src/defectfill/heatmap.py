from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf


class PiecewiseHeatmapGenerator:
    def __init__(self, thresholds: list[float]):
        self.thresholds = sorted(thresholds)

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        s_min = float(scores.min())
        s_max = float(scores.max())
        return (scores - s_min) / (s_max - s_min + 1e-8)

    def make_piecewise_map(self, patch_scores: tf.Tensor, patch_hw: tuple[int, int], output_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        ph, pw = patch_hw
        oh, ow = output_hw
        score_map = tf.reshape(patch_scores, [ph, pw]).numpy()
        score_map = self._normalize(score_map)
        up = cv2.resize(score_map, (ow, oh), interpolation=cv2.INTER_CUBIC)

        levels = np.zeros_like(up)
        for i, t in enumerate(self.thresholds, start=1):
            levels[up >= t] = i / len(self.thresholds)

        fg = (up >= self.thresholds[0]).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

        return levels, fg

    @staticmethod
    def overlay(image_rgb: np.ndarray, piecewise_map: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
        heat = (piecewise_map * 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

        alpha = np.clip(piecewise_map[..., None] * 0.75, 0.1, 0.8)
        blended = (1.0 - alpha) * image_rgb + alpha * (heat / 255.0)

        fg3 = np.repeat((fg_mask > 0)[..., None], 3, axis=-1)
        out = np.where(fg3, blended, image_rgb)
        return np.clip(out, 0.0, 1.0)
