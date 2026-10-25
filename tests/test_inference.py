import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import tensorflow as tf

from task4.inference import VisionGuardPredictor
from visionguard.preprocessing import PREPROCESSING_ID


class InferenceTests(unittest.TestCase):
    def package(self, directory):
        root = Path(directory)
        (root / "model.keras").write_bytes(b"mock model")
        metadata = {"sha256": hashlib.sha256(b"mock model").hexdigest(),
                    "threshold": .5, "positive_class": "AI", "image_size": 160,
                    "model_name": "test", "model_version": "2",
                    "preprocessing": PREPROCESSING_ID}
        (root / "metadata.json").write_text(json.dumps(metadata))
        return root

    def test_reuses_model_and_keeps_raw_letterboxed_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.package(folder)
            model = Mock(input_shape=(None, 160, 160, 3), output_shape=(None, 1))
            model.return_value = tf.constant([[.8]])
            contents = tf.io.encode_png(tf.ones([40, 80, 3], tf.uint8) * 255).numpy()
            with patch("task4.inference.tf.keras.models.load_model", return_value=model) as load:
                predictor = VisionGuardPredictor(root)
                first = predictor.predict_bytes(contents)
                model.return_value = tf.constant([[.2]])
                second = predictor.predict_bytes(contents)
                load.assert_called_once()
                self.assertEqual(model.call_count, 2)
            self.assertEqual((first["label"], second["label"]), ("AI", "Real"))
            self.assertAlmostEqual(first["confidence"], .8, places=6)
            self.assertAlmostEqual(second["confidence"], .8, places=6)
            batch = model.call_args.args[0].numpy()
            self.assertEqual(batch.shape, (1, 160, 160, 3))
            self.assertEqual(batch.dtype, np.float32)
            self.assertTrue(np.all(batch[0, 0] == 0))
            self.assertTrue(np.all(batch[0, 80] == 255))
            self.assertFalse(model.call_args.kwargs["training"])

    def test_model_hash_mismatch_fails_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.package(folder)
            (root / "model.keras").write_bytes(b"changed model")
            with patch("task4.inference.tf.keras.models.load_model") as load:
                with self.assertRaisesRegex(ValueError, "hash"):
                    VisionGuardPredictor(root)
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
