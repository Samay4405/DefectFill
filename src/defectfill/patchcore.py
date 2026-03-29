from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf


@dataclass
class PatchCoreConfig:
    coreset_ratio: float = 0.1
    knn_k: int = 1


class PatchCoreTF:
    """TensorFlow-native PatchCore memory bank and anomaly scoring."""

    def __init__(self, cfg: PatchCoreConfig):
        self.cfg = cfg
        self.memory_bank: tf.Tensor | None = None

    @staticmethod
    @tf.function(jit_compile=True)
    def pairwise_l2(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        # d(i,j)^2 = ||a_i||^2 + ||b_j||^2 - 2*a_i.b_j
        a2 = tf.reduce_sum(tf.square(a), axis=1, keepdims=True)
        b2 = tf.reduce_sum(tf.square(b), axis=1, keepdims=True)
        ab = tf.matmul(a, b, transpose_b=True)
        d2 = tf.maximum(a2 - 2.0 * ab + tf.transpose(b2), 0.0)
        return tf.sqrt(d2 + 1e-8)

    def build_memory_bank(self, patch_features: tf.Tensor) -> tf.Tensor:
        flat = tf.reshape(patch_features, [-1, tf.shape(patch_features)[-1]])
        flat = tf.math.l2_normalize(flat, axis=-1)

        if self.cfg.coreset_ratio >= 1.0:
            self.memory_bank = flat
            return flat

        n = tf.shape(flat)[0]
        m = tf.cast(tf.maximum(1, tf.cast(tf.cast(n, tf.float32) * self.cfg.coreset_ratio, tf.int32)), tf.int32)
        idx = tf.random.shuffle(tf.range(n))[:m]
        self.memory_bank = tf.gather(flat, idx)
        return self.memory_bank

    def save(self, path: str):
        if self.memory_bank is None:
            raise RuntimeError("Memory bank has not been built.")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), self.memory_bank.numpy())

    def load(self, path: str):
        mb = np.load(path)
        self.memory_bank = tf.convert_to_tensor(mb, dtype=tf.float32)

    @tf.function(jit_compile=True)
    def score_patches(self, patch_features: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        if self.memory_bank is None:
            raise RuntimeError("Memory bank has not been loaded.")

        q = tf.math.l2_normalize(tf.reshape(patch_features, [-1, tf.shape(patch_features)[-1]]), axis=-1)
        d = self.pairwise_l2(q, self.memory_bank)

        values, _ = tf.math.top_k(-d, k=self.cfg.knn_k)
        nn_dist = -values
        min_dist = tf.reduce_mean(nn_dist, axis=-1)

        b = tf.shape(patch_features)[0]
        n = tf.shape(patch_features)[1]
        patch_scores = tf.reshape(min_dist, [b, n])
        image_scores = tf.reduce_max(patch_scores, axis=1)

        return image_scores, patch_scores
