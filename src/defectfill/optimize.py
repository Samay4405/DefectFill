from __future__ import annotations

import time
from pathlib import Path

import tensorflow as tf


def benchmark_latency_ms(fn, input_tensor: tf.Tensor, warmup: int = 5, runs: int = 30) -> float:
    for _ in range(warmup):
        _ = fn(input_tensor)

    start = time.perf_counter()
    for _ in range(runs):
        _ = fn(input_tensor)
    end = time.perf_counter()

    return ((end - start) / runs) * 1000.0


def export_distance_tflite(memory_bank: tf.Tensor, output_path: str):
    class DistanceModule(tf.Module):
        def __init__(self, mb: tf.Tensor):
            super().__init__()
            self.mb = tf.Variable(mb, trainable=False, dtype=tf.float32)

        @tf.function(input_signature=[tf.TensorSpec([None, None], tf.float32)])
        def __call__(self, patches: tf.Tensor):
            p = tf.math.l2_normalize(patches, axis=-1)
            m = tf.math.l2_normalize(self.mb, axis=-1)
            p2 = tf.reduce_sum(tf.square(p), axis=1, keepdims=True)
            m2 = tf.reduce_sum(tf.square(m), axis=1, keepdims=True)
            ab = tf.matmul(p, m, transpose_b=True)
            d2 = tf.maximum(p2 - 2.0 * ab + tf.transpose(m2), 0.0)
            d = tf.sqrt(d2 + 1e-8)
            return {"distances": d}

    module = DistanceModule(memory_bank)
    concrete = module.__call__.get_concrete_function()

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], module)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tflite_model)
