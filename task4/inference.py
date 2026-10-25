"""Load one local champion at startup; classify uploaded bytes in memory."""

import hashlib
import json
import math
from pathlib import Path

import tensorflow as tf

from visionguard.preprocessing import PREPROCESSING_ID, decode_resize_bytes


class VisionGuardPredictor:
    """Construct once at app startup and reuse for every request. No network calls."""

    def __init__(self, model_dir):
        directory = Path(model_dir)
        self.metadata = json.loads((directory / "metadata.json").read_text())
        if self.metadata["preprocessing"] != PREPROCESSING_ID:
            raise ValueError("Model preprocessing does not match the inference module.")
        path = directory / "model.keras"
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.metadata["sha256"]:
            raise ValueError("Local model hash does not match its metadata.")
        self.threshold = float(self.metadata["threshold"])
        if not 0 <= self.threshold <= 1 or self.metadata["positive_class"] != "AI":
            raise ValueError("Invalid model decision settings.")
        self.image_size = int(self.metadata["image_size"])
        self.model = tf.keras.models.load_model(path, compile=False)
        if tuple(self.model.input_shape[1:]) != (self.image_size, self.image_size, 3):
            raise ValueError("Model input shape does not match its metadata.")
        if tuple(self.model.output_shape[1:]) != (1,):
            raise ValueError("Expected one AI probability per image.")

    def predict_bytes(self, contents):
        pixels = decode_resize_bytes(contents, self.image_size)
        batch = tf.cast(pixels[None, ...], tf.float32)
        # No augmentation and no extra scaling; scaling is inside the trained model.
        probability = float(self.model(batch, training=False).numpy()[0, 0])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Model returned an invalid probability.")
        label = "AI" if probability >= self.threshold else "Real"
        return {
            "label": label, "p_ai": probability,
            "confidence": probability if label == "AI" else 1 - probability,
            "threshold": self.threshold, "confidence_is_calibrated": False,
            "model_name": self.metadata["model_name"], "model_version": self.metadata["model_version"],
        }
