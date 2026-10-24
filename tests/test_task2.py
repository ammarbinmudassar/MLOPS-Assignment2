import unittest

import numpy as np
import pandas as pd

from task2.common import setup
setup()
from task2.data import check_splits, decode_resize, read_split
from task2.models import backbone, transfer, unfreeze_last_block
import tensorflow as tf


class Task2Tests(unittest.TestCase):
    def test_saved_manifests_have_no_exact_duplicate_leakage(self):
        frames = {name: read_split(name) for name in ["train", "validation", "test"]}
        check_splits(frames)
        self.assertEqual(sum(map(len, frames.values())), 974)

    def test_overlap_is_rejected(self):
        frame = pd.DataFrame({"sha256": ["a", "b"], "label": [0, 1]})
        with self.assertRaisesRegex(ValueError, "leakage"):
            check_splits({"train": frame, "validation": frame})

    def test_letterbox_preserves_shape_and_padding(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "image.png"
            tf.io.write_file(str(path), tf.io.encode_png(tf.ones([40, 80, 3], tf.uint8) * 255))
            image, label = decode_resize(str(path), 1)
            self.assertEqual(tuple(image.shape), (160, 160, 3))
            self.assertEqual(float(label), 1)
            self.assertTrue(np.all(image.numpy()[0] == 0))
            self.assertTrue(np.all(image.numpy()[80] == 255))

    def test_transfer_preprocessing_and_batchnorm_policy(self):
        model = transfer(weights=None)
        preprocessing = model.get_layer("mobilenet_preprocess")
        np.testing.assert_allclose(preprocessing(tf.constant([0., 127.5, 255.])), [-1., 0., 1.])
        self.assertFalse(backbone(model).trainable)
        unfreeze_last_block(model)
        for layer in backbone(model).layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                self.assertFalse(layer.trainable)
            elif layer.trainable:
                self.assertTrue(layer.name.startswith(("block_16_", "Conv_1", "out_relu")))

    def test_threshold_respects_false_accusation_cost(self):
        from task2.evaluate import choose_threshold
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([.1, .6, .5, .9])
        best, _ = choose_threshold(labels, probabilities)
        self.assertEqual(best["fp"], 0)
        self.assertEqual(best["fn"], 1)
        self.assertEqual(best["weighted_error"], 1)


if __name__ == "__main__":
    unittest.main()
