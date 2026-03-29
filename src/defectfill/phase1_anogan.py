from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from .phase1_synthesis import SyntheticDefectGenerator


@dataclass
class AnoGANConfig:
    latent_dim: int = 256
    attention_channels: int = 64


class AttentionFusion(tf.keras.layers.Layer):
    def __init__(self, channels: int):
        super().__init__()
        self.q = tf.keras.layers.Conv2D(channels, 1, padding="same")
        self.k = tf.keras.layers.Conv2D(channels, 1, padding="same")
        self.v = tf.keras.layers.Conv2D(channels, 1, padding="same")
        self.o = tf.keras.layers.Conv2D(channels, 1, padding="same")

    def call(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        q = self.q(x)
        k = self.k(y)
        v = self.v(y)
        attn = tf.nn.softmax((q * k) / tf.math.sqrt(tf.cast(tf.shape(q)[-1], tf.float32)), axis=-1)
        fused = attn * v + x
        return self.o(fused)


class AnoGANSynthesizer:
    """AnoGAN-like latent synthesis with attention fusion and SD inpainting refinement."""

    def __init__(self, base_synth: SyntheticDefectGenerator, cfg: AnoGANConfig):
        self.base_synth = base_synth
        self.cfg = cfg

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Input((base_synth.cfg.image_size, base_synth.cfg.image_size, 3)),
                tf.keras.layers.Conv2D(32, 3, strides=2, activation="relu", padding="same"),
                tf.keras.layers.Conv2D(64, 3, strides=2, activation="relu", padding="same"),
                tf.keras.layers.Conv2D(128, 3, strides=2, activation="relu", padding="same"),
            ],
            name="anogan_encoder",
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Input((base_synth.cfg.image_size // 8, base_synth.cfg.image_size // 8, 128)),
                tf.keras.layers.Conv2DTranspose(128, 3, strides=2, activation="relu", padding="same"),
                tf.keras.layers.Conv2DTranspose(64, 3, strides=2, activation="relu", padding="same"),
                tf.keras.layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same"),
                tf.keras.layers.Conv2D(3, 1, activation="sigmoid", padding="same"),
            ],
            name="anogan_decoder",
        )

        self.attn = AttentionFusion(cfg.attention_channels)

    @tf.function(jit_compile=True)
    def _latent_transform(self, image: tf.Tensor) -> tf.Tensor:
        z = self.encoder(image[tf.newaxis, ...], training=False)
        z_noise = z + tf.random.normal(tf.shape(z), stddev=0.07)
        fused = self.attn(z, z_noise)
        out = self.decoder(fused, training=False)[0]
        return tf.clip_by_value(out, 0.0, 1.0)

    def synthesize(self, image: tf.Tensor, prompt: str, negative_prompt: str) -> tuple[tf.Tensor, tf.Tensor]:
        # Create coarse anomalous texture with AnoGAN-style latent perturbation.
        coarse = self._latent_transform(image)
        # Reuse existing controlled masking + inpainting refinement to sharpen defect realism.
        refined, mask = self.base_synth.synthesize(coarse, prompt, negative_prompt)
        return refined, mask
