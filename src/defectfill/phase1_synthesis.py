from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


@dataclass
class SynthesisConfig:
    image_size: int = 384
    inpaint_steps: int = 25
    guidance_scale: float = 7.5
    scratch_density: float = 0.25
    dent_density: float = 0.15


class SyntheticDefectGenerator:
    """Synthetic defect generation with a Keras-compatible diffusion backend."""

    def __init__(self, cfg: SynthesisConfig):
        self.cfg = cfg
        self._diffusion = self._load_diffusion_backend()

    def _load_diffusion_backend(self):
        try:
            import keras_cv

            if hasattr(keras_cv.models, "StableDiffusionInpaint"):
                return keras_cv.models.StableDiffusionInpaint(img_height=self.cfg.image_size, img_width=self.cfg.image_size)
            if hasattr(keras_cv.models, "StableDiffusion"):
                return keras_cv.models.StableDiffusion(img_height=self.cfg.image_size, img_width=self.cfg.image_size)
        except Exception:
            return None
        return None

    @staticmethod
    def _rand_stroke_mask(image_size: int, density: float) -> np.ndarray:
        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        strokes = max(1, int(image_size * density * 0.2))
        for _ in range(strokes):
            x1, y1 = np.random.randint(0, image_size, size=2)
            length = np.random.randint(image_size // 12, image_size // 4)
            angle = np.random.rand() * 2 * np.pi
            x2 = int(np.clip(x1 + length * np.cos(angle), 0, image_size - 1))
            y2 = int(np.clip(y1 + length * np.sin(angle), 0, image_size - 1))
            thickness = np.random.randint(1, 3)
            cv2.line(mask, (x1, y1), (x2, y2), color=255, thickness=thickness)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        return (mask > 20).astype(np.float32)

    @staticmethod
    def _rand_dent_mask(image_size: int, density: float) -> np.ndarray:
        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        dents = max(1, int(image_size * density * 0.03))
        for _ in range(dents):
            x, y = np.random.randint(0, image_size, size=2)
            radius = np.random.randint(image_size // 40, image_size // 18)
            cv2.circle(mask, (x, y), radius, color=255, thickness=-1)
        mask = cv2.GaussianBlur(mask, (9, 9), 0)
        return (mask > 24).astype(np.float32)

    def _controlled_latent_perturbation(self, image: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        image_f = tf.cast(image, tf.float32)
        pooled = tf.nn.avg_pool2d(tf.expand_dims(image_f, 0), ksize=8, strides=8, padding="SAME")
        noise = tf.random.normal(tf.shape(pooled), stddev=0.08)
        sinusoid = tf.sin(tf.linspace(0.0, 18.0, tf.shape(pooled)[1]))
        sinusoid = sinusoid[tf.newaxis, :, tf.newaxis, tf.newaxis]
        latent = pooled + noise + 0.04 * sinusoid
        up = tf.image.resize(latent[0], [self.cfg.image_size, self.cfg.image_size], method="bicubic")
        mask3 = tf.repeat(mask[..., tf.newaxis], 3, axis=-1)
        perturbed = image_f * (1.0 - mask3) + tf.clip_by_value(up, 0.0, 1.0) * mask3
        return tf.clip_by_value(perturbed, 0.0, 1.0)

    def _run_diffusion_inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str,
        negative_prompt: str,
    ) -> np.ndarray:
        if self._diffusion is None:
            return image

        try:
            if self._diffusion.__class__.__name__.lower().endswith("inpaint"):
                out = self._diffusion.text_to_image(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=image,
                    mask=mask[..., None],
                    num_steps=self.cfg.inpaint_steps,
                    guidance_scale=self.cfg.guidance_scale,
                )
            else:
                synth = self._diffusion.text_to_image(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    batch_size=1,
                    num_steps=self.cfg.inpaint_steps,
                    guidance_scale=self.cfg.guidance_scale,
                )[0]
                blend = mask[..., None]
                out = (1.0 - blend) * image + blend * synth
            return np.clip(out, 0.0, 1.0)
        except Exception:
            return image

    def synthesize(
        self,
        image: tf.Tensor,
        prompt: str,
        negative_prompt: str,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Returns synthetic image and binary defect mask."""
        scratch = self._rand_stroke_mask(self.cfg.image_size, self.cfg.scratch_density)
        dent = self._rand_dent_mask(self.cfg.image_size, self.cfg.dent_density)
        mask = np.clip(scratch + dent, 0.0, 1.0).astype(np.float32)

        latent_aug = self._controlled_latent_perturbation(image, tf.convert_to_tensor(mask, dtype=tf.float32))
        latent_aug_np = latent_aug.numpy()
        inpainted = self._run_diffusion_inpaint(latent_aug_np, mask, prompt, negative_prompt)

        return tf.convert_to_tensor(inpainted, dtype=tf.float32), tf.convert_to_tensor(mask, dtype=tf.float32)

    def batch_synthesize_to_disk(
        self,
        images: tf.Tensor,
        output_dir: str,
        base_name: str,
        prompt: str,
        negative_prompt: str,
        count: int,
    ) -> list[str]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        out_paths: list[str] = []
        for i in range(images.shape[0]):
            img = images[i]
            for j in range(count):
                synth_img, mask = self.synthesize(img, prompt, negative_prompt)
                img_path = output / f"{base_name}_{i:04d}_{j:02d}.png"
                mask_path = output / f"{base_name}_{i:04d}_{j:02d}_mask.png"
                cv2.imwrite(str(img_path), (synth_img.numpy()[..., ::-1] * 255).astype(np.uint8))
                cv2.imwrite(str(mask_path), (mask.numpy() * 255).astype(np.uint8))
                out_paths.append(str(img_path))
        return out_paths
