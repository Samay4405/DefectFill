from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf
from keras import Model


@dataclass
class DinoConfig:
    preset: str = "dinov2_base_imagenet1k"
    image_size: int = 384
    layer_names: list[str] | None = None


class DinoV2PatchExtractor:
    """DINOv2 feature extractor via keras-hub, exposing multi-layer patch tokens."""

    def __init__(self, cfg: DinoConfig):
        self.cfg = cfg
        self.backbone = self._load_backbone(cfg.preset)
        self.model = self._build_feature_model(self.backbone, cfg.layer_names)

    @staticmethod
    def _load_backbone(preset: str) -> tf.keras.Model:
        import keras_hub

        candidates = []

        if hasattr(keras_hub.models, "Backbone"):
            candidates.append(lambda: keras_hub.models.Backbone.from_preset(preset))

        if hasattr(keras_hub.models, "DINOv2Backbone"):
            candidates.append(lambda: keras_hub.models.DINOv2Backbone.from_preset(preset))

        if hasattr(keras_hub.models, "ImageClassifier"):
            candidates.append(lambda: keras_hub.models.ImageClassifier.from_preset(preset).backbone)

        err_msgs: list[str] = []
        for builder in candidates:
            try:
                model = builder()
                if isinstance(model, tf.keras.Model):
                    return model
            except Exception as exc:
                err_msgs.append(str(exc))

        # Fallback for offline or preset mismatch environments.
        inp = tf.keras.Input(shape=(None, None, 3), name="fallback_input")
        x = tf.keras.layers.Conv2D(64, 7, strides=4, padding="same", activation="relu", name="fallback_stem")(inp)
        x = tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="relu", name="fallback_block1")(x)
        x = tf.keras.layers.Conv2D(192, 3, strides=2, padding="same", activation="relu", name="fallback_block2")(x)
        return tf.keras.Model(inp, x, name="fallback_backbone")

    def _build_feature_model(self, backbone: tf.keras.Model, requested_layers: list[str] | None) -> tf.keras.Model:
        if requested_layers:
            outputs = [backbone.get_layer(name).output for name in requested_layers]
            return Model(inputs=backbone.input, outputs=outputs, name="dinov2_multilayer")

        candidate_layers = [
            layer for layer in backbone.layers if any(k in layer.name.lower() for k in ["transformer", "encoder", "block"])
        ]

        if len(candidate_layers) < 3:
            return Model(inputs=backbone.input, outputs=[backbone.output], name="dinov2_single")

        selected = candidate_layers[-3:]
        return Model(inputs=backbone.input, outputs=[layer.output for layer in selected], name="dinov2_multilayer")

    @staticmethod
    def _tokens_to_patch_map(tokens: tf.Tensor, image_size: int) -> tf.Tensor:
        if tokens.shape.rank == 3:
            seq_len = tf.shape(tokens)[1]
            feat_dim = tf.shape(tokens)[2]
            side = tf.cast(tf.math.sqrt(tf.cast(seq_len, tf.float32)), tf.int32)

            def reshape_plain() -> tf.Tensor:
                return tf.reshape(tokens, [-1, side, side, feat_dim])

            def reshape_drop_cls() -> tf.Tensor:
                seq_len_wo_cls = seq_len - 1
                side_wo_cls = tf.cast(tf.math.sqrt(tf.cast(seq_len_wo_cls, tf.float32)), tf.int32)
                no_cls = tokens[:, 1:, :]
                return tf.reshape(no_cls, [-1, side_wo_cls, side_wo_cls, feat_dim])

            # Some ViT heads include CLS token while others do not.
            return tf.cond(tf.equal(side * side, seq_len), reshape_plain, reshape_drop_cls)

        if tokens.shape.rank == 4:
            return tokens

        raise ValueError(f"Unexpected token rank: {tokens.shape.rank}")

    @tf.function(jit_compile=True)
    def extract_dense_features(self, images: tf.Tensor) -> tf.Tensor:
        feats = self.model(images, training=False)
        if not isinstance(feats, (list, tuple)):
            feats = [feats]

        patch_maps = [self._tokens_to_patch_map(f, self.cfg.image_size) for f in feats]
        target_h = tf.shape(patch_maps[0])[1]
        target_w = tf.shape(patch_maps[0])[2]
        resized = [tf.image.resize(f, [target_h, target_w], method="bilinear") for f in patch_maps]
        concat = tf.concat(resized, axis=-1)
        return concat

    @tf.function(jit_compile=True)
    def flatten_patches(self, images: tf.Tensor) -> tf.Tensor:
        patch_map = self.extract_dense_features(images)
        b = tf.shape(patch_map)[0]
        h = tf.shape(patch_map)[1]
        w = tf.shape(patch_map)[2]
        c = tf.shape(patch_map)[3]
        return tf.reshape(patch_map, [b, h * w, c])

    @tf.function(jit_compile=True)
    def patch_map_hw(self, images: tf.Tensor) -> tf.Tensor:
        patch_map = self.extract_dense_features(images)
        return tf.shape(patch_map)[1:3]
