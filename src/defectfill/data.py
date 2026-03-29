from __future__ import annotations

from pathlib import Path

import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def _decode_and_resize(path: tf.Tensor, image_size: int) -> tf.Tensor:
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [image_size, image_size], method="bicubic")
    return tf.cast(image, tf.float32) / 255.0


def list_mvtec_images(root: str, category: str, split: str, defect_type: str = "good") -> list[str]:
    base = Path(root) / category / split / defect_type
    if not base.exists():
        raise FileNotFoundError(f"Missing path: {base}")
    return sorted(str(p) for p in base.glob("*.png"))


def make_dataset(paths: list[str], image_size: int, batch_size: int, shuffle: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(paths)
    if shuffle:
        ds = ds.shuffle(buffer_size=max(len(paths), 64), reshuffle_each_iteration=True)
    ds = ds.map(lambda p: _decode_and_resize(p, image_size), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds
