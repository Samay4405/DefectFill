from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from .data import list_mvtec_images, make_dataset
from .feature_extractor import DinoConfig, DinoV2PatchExtractor
from .heatmap import PiecewiseHeatmapGenerator
from .optimize import benchmark_latency_ms, export_distance_tflite
from .patchcore import PatchCoreConfig, PatchCoreTF
from .phase1_anogan import AnoGANConfig, AnoGANSynthesizer
from .phase1_synthesis import SynthesisConfig, SyntheticDefectGenerator


@dataclass
class PipelineArtifacts:
    memory_bank_path: str
    tflite_path: str | None


class DefectFillPipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg

        img_size = cfg["dataset"]["image_size"]
        patch_cfg = cfg["patchcore"]

        self.synth = SyntheticDefectGenerator(
            SynthesisConfig(
                image_size=img_size,
                inpaint_steps=cfg["synthesis"]["inpaint_steps"],
                guidance_scale=cfg["synthesis"]["guidance_scale"],
                scratch_density=cfg["synthesis"]["scratch_density"],
                dent_density=cfg["synthesis"]["dent_density"],
            )
        )
        anogan_cfg = cfg["synthesis"].get("anogan", {})
        self.anogan = AnoGANSynthesizer(
            self.synth,
            AnoGANConfig(
                latent_dim=int(anogan_cfg.get("latent_dim", 256)),
                attention_channels=int(anogan_cfg.get("attention_channels", 64)),
            ),
        )

        self.extractor = DinoV2PatchExtractor(
            DinoConfig(
                image_size=img_size,
                layer_names=patch_cfg["layer_names"] or None,
            )
        )

        self.patchcore = PatchCoreTF(
            PatchCoreConfig(
                coreset_ratio=patch_cfg["coreset_ratio"],
                knn_k=patch_cfg["knn_k"],
            )
        )

        self.heatmap = PiecewiseHeatmapGenerator(cfg["inference"]["heatmap_thresholds"])
        self._memory_loaded = False
        self._anomaly_threshold: float | None = None
        self._elbow_profile: dict[str, object] | None = None

    def run_phase1_synthesis(self):
        dcfg = self.cfg["dataset"]
        scfg = self.cfg["synthesis"]

        normal_paths = list_mvtec_images(dcfg["root"], dcfg["category"], split="train", defect_type="good")
        ds = make_dataset(normal_paths, dcfg["image_size"], dcfg["batch_size"], shuffle=False)

        out_dir = Path(scfg["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        idx = 0
        mode = scfg.get("mode", "anogan")
        for batch in ds:
            output = Path(out_dir)
            output.mkdir(parents=True, exist_ok=True)
            for i in range(batch.shape[0]):
                for j in range(int(scfg["num_augmented_per_image"])):
                    if mode == "anogan":
                        synth_img, mask = self.anogan.synthesize(
                            batch[i],
                            scfg["diffusion_prompt"],
                            scfg["diffusion_negative_prompt"],
                        )
                    else:
                        synth_img, mask = self.synth.synthesize(
                            batch[i],
                            scfg["diffusion_prompt"],
                            scfg["diffusion_negative_prompt"],
                        )

                    img_path = output / f"{dcfg['category']}_synth_{idx:04d}_{i:04d}_{j:02d}.png"
                    mask_path = output / f"{dcfg['category']}_synth_{idx:04d}_{i:04d}_{j:02d}_mask.png"
                    cv2.imwrite(str(img_path), (synth_img.numpy()[..., ::-1] * 255).astype(np.uint8))
                    cv2.imwrite(str(mask_path), (mask.numpy() * 255).astype(np.uint8))
            idx += 1

    def build_memory_bank(self) -> PipelineArtifacts:
        dcfg = self.cfg["dataset"]
        pcfg = self.cfg["patchcore"]
        ocfg = self.cfg["optimization"]

        normal_paths = list_mvtec_images(dcfg["root"], dcfg["category"], split="train", defect_type="good")
        ds = make_dataset(normal_paths, dcfg["image_size"], dcfg["batch_size"], shuffle=False)

        all_patches = []
        for batch in ds:
            p = self.extractor.flatten_patches(batch)
            all_patches.append(p)
        all_patches_t = tf.concat(all_patches, axis=0)

        self.patchcore.build_memory_bank(all_patches_t)
        self.patchcore.save(pcfg["memory_bank_path"])
        self._memory_loaded = True

        tflite_path = None
        if ocfg.get("export_tflite", False):
            export_distance_tflite(self.patchcore.memory_bank, ocfg["tflite_path"])
            tflite_path = ocfg["tflite_path"]

        return PipelineArtifacts(memory_bank_path=pcfg["memory_bank_path"], tflite_path=tflite_path)

    def _ensure_memory_loaded(self):
        if self._memory_loaded:
            return
        memory_path = self.cfg["patchcore"]["memory_bank_path"]
        if not Path(memory_path).exists():
            self.build_memory_bank()
        self.patchcore.load(memory_path)
        self._memory_loaded = True

    def set_anomaly_threshold(self, threshold: float):
        self._anomaly_threshold = float(threshold)

    def get_anomaly_threshold(self) -> float:
        if self._anomaly_threshold is not None:
            return float(self._anomaly_threshold)
        return float(self.cfg["inference"].get("defect_score_threshold", 0.5))

    def score_paths(self, paths: list[str]) -> list[float]:
        self._ensure_memory_loaded()
        if not paths:
            return []

        dcfg = self.cfg["dataset"]
        ds = make_dataset(paths, dcfg["image_size"], dcfg["batch_size"], shuffle=False)

        scores: list[float] = []
        for batch in ds:
            (image_scores, _patch_scores), _hw = self._infer_step(batch)
            scores.extend(float(v) for v in image_scores.numpy().tolist())
        return scores

    def calibrate_elbow_threshold(self, paths: list[str]) -> float:
        scores = self.score_paths(paths)
        if not scores:
            raise ValueError("No images available to calibrate elbow threshold.")

        ordered = np.sort(np.asarray(scores, dtype=np.float32))
        points = np.stack([np.linspace(0.0, 1.0, ordered.size, dtype=np.float32), ordered], axis=1)
        distances = np.zeros_like(ordered, dtype=np.float32)
        if ordered.size == 1:
            threshold = float(ordered[0])
            elbow_idx = 0
        else:
            start = points[0]
            end = points[-1]
            direction = end - start
            length = float(np.linalg.norm(direction))

            if length < 1e-8:
                threshold = float(np.median(ordered))
                elbow_idx = int(np.argmin(np.abs(ordered - np.median(ordered))))
            else:
                offsets = points - start
                distances = np.abs(direction[0] * offsets[:, 1] - direction[1] * offsets[:, 0]) / length
                elbow_idx = int(np.argmax(distances))
                threshold = float(ordered[elbow_idx])

        self.set_anomaly_threshold(threshold)
        self._elbow_profile = {
            "threshold": threshold,
            "elbow_index": elbow_idx,
            "points": [[float(p[0]), float(p[1])] for p in points],
            "distances": [float(d) for d in distances.tolist()],
        }
        return threshold

    def get_elbow_profile(self) -> dict[str, object] | None:
        return self._elbow_profile

    def save_anomaly_record(
        self,
        *,
        anomaly_id: str,
        score: float,
        source_path: str,
        threshold: float,
    ):
        output_dir = Path(self.cfg["inference"].get("output_dir", "./artifacts/inference"))
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "anomaly_records.csv"

        exists = csv_path.exists()
        with csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(["anomaly_id", "score", "threshold", "source_path", "timestamp_ns"])
            writer.writerow([anomaly_id, float(score), float(threshold), source_path, int(time.time_ns())])

    @tf.function(jit_compile=True)
    def _infer_step(self, x: tf.Tensor):
        patch_map = self.extractor.extract_dense_features(x)
        b = tf.shape(patch_map)[0]
        h = tf.shape(patch_map)[1]
        w = tf.shape(patch_map)[2]
        c = tf.shape(patch_map)[3]
        patches = tf.reshape(patch_map, [b, h * w, c])
        scores = self.patchcore.score_patches(patches)
        return scores, tf.stack([h, w])

    def infer_frame(self, frame_rgb: np.ndarray) -> dict[str, np.ndarray | float | bool]:
        self._ensure_memory_loaded()
        dcfg = self.cfg["dataset"]
        score_threshold = self.get_anomaly_threshold()

        resized = cv2.resize(frame_rgb, (dcfg["image_size"], dcfg["image_size"]), interpolation=cv2.INTER_AREA)
        tensor = tf.convert_to_tensor(resized, dtype=tf.float32)[tf.newaxis, ...] / 255.0

        t0 = time.perf_counter()
        (image_scores, patch_scores), hw = self._infer_step(tensor)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        image_score = float(image_scores.numpy()[0])
        patch_h = int(hw.numpy()[0])
        patch_w = int(hw.numpy()[1])

        piecewise, fg = self.heatmap.make_piecewise_map(
            patch_scores=patch_scores[0],
            patch_hw=(patch_h, patch_w),
            output_hw=(dcfg["image_size"], dcfg["image_size"]),
        )

        overlay = self.heatmap.overlay(tensor[0].numpy(), piecewise, fg)
        overlay_u8 = (overlay * 255).astype(np.uint8)

        return {
            "score": image_score,
            "defect": bool(image_score >= score_threshold),
            "latency_ms": float(latency_ms),
            "frame": resized,
            "heatmap_overlay": overlay_u8,
        }

    def infer_folder(self, defect_type: str = "good", use_elbow_threshold: bool = False) -> dict[str, float]:
        dcfg = self.cfg["dataset"]
        icfg = self.cfg["inference"]
        pcfg = self.cfg["patchcore"]

        self.patchcore.load(pcfg["memory_bank_path"])

        paths = list_mvtec_images(dcfg["root"], dcfg["category"], split="test", defect_type=defect_type)
        if use_elbow_threshold:
            threshold = self.calibrate_elbow_threshold(paths)
            print(f"Calibrated elbow threshold: {threshold:.4f}")
        ds = make_dataset(paths, dcfg["image_size"], batch_size=1, shuffle=False)

        output_dir = Path(icfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        scores: dict[str, float] = {}

        @tf.function(jit_compile=True)
        def infer_step(x):
            patches = self.extractor.flatten_patches(x)
            return self.patchcore.score_patches(patches), patches

        latency_batch = next(iter(ds.take(1)))
        latency = benchmark_latency_ms(lambda t: infer_step(t), latency_batch)
        print(f"Average latency: {latency:.2f} ms/image")
        if latency > icfg["max_latency_ms"]:
            print(
                f"Warning: latency {latency:.2f} ms exceeds target {icfg['max_latency_ms']} ms. "
                "Try smaller image_size, smaller DINOv2 preset, and TFLite distance scoring."
            )

        for i, batch in enumerate(ds):
            (image_scores, patch_scores), patches = infer_step(batch)
            image_score = float(image_scores.numpy()[0])

            h = int(tf.shape(patches)[1] ** 0.5)
            w = h
            piecewise, fg = self.heatmap.make_piecewise_map(
                patch_scores=patch_scores[0],
                patch_hw=(h, w),
                output_hw=(dcfg["image_size"], dcfg["image_size"]),
            )

            img = batch[0].numpy()
            overlay = self.heatmap.overlay(img, piecewise, fg)

            out = (overlay * 255).astype(np.uint8)
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            out_path = output_dir / f"{defect_type}_{i:04d}_score_{image_score:.4f}.png"
            cv2.imwrite(str(out_path), out)

            scores[str(paths[i])] = image_score

        return scores
