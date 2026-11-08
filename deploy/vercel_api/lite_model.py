"""TensorFlow-free CPU inference for the exported champion derivative."""
from io import BytesIO
from pathlib import Path
from threading import Lock
import hashlib
import json
import math

import numpy as np
from PIL import Image


def _resize_axis(pixels, target, axis):
    source = np.moveaxis(pixels, axis, 0)
    length = source.shape[0]
    scale = np.float32(target / length)
    kernel_scale = min(scale, np.float32(1))
    radius = np.float32(1) / kernel_scale
    resized = np.empty((target, *source.shape[1:]), dtype=np.float32)
    for index in range(target):
        center = np.float32(index + .5) / scale
        first = max(0, int(np.ceil(center - radius - .5)))
        last = min(length, int(np.floor(center + radius - .5)) + 1)
        positions = np.arange(first, last, dtype=np.float32) + np.float32(.5)
        weights = np.maximum(np.float32(0), np.float32(1) - np.abs(positions - center) * kernel_scale)
        weights /= weights.sum()
        resized[index] = np.sum(source[first:last] * weights.reshape((-1,) + (1,) * (source.ndim - 1)), axis=0)
    return np.moveaxis(resized, 0, axis)


def prepare_image(contents, size=160):
    with Image.open(BytesIO(contents)) as image:
        image.seek(0)
        pixels = np.asarray(image.convert('RGB'), dtype=np.float32)
    height, width = pixels.shape[:2]
    ratio = max(np.float32(height / size), np.float32(width / size))
    resized_h = max(1, int(np.floor(np.float32(height) / ratio)))
    resized_w = max(1, int(np.floor(np.float32(width) / ratio)))
    pixels = _resize_axis(pixels, resized_h, 0)
    pixels = _resize_axis(pixels, resized_w, 1)
    pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - resized_h) // 2, (size - resized_w) // 2
    canvas[top:top + resized_h, left:left + resized_w] = pixels
    return canvas.astype(np.float32)[None]


class LitePredictor:
    def __init__(self, directory=None):
        from ai_edge_litert.interpreter import Interpreter
        directory = Path(directory or Path(__file__).parent / 'model')
        self.metadata = json.loads((directory / 'metadata.json').read_text())
        path = directory / 'model.tflite'
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.metadata['sha256']:
            raise ValueError('Lite model hash mismatch')
        self.threshold = float(self.metadata['threshold'])
        self.interpreter = Interpreter(model_path=str(path), num_threads=2)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        expected = (1, self.metadata['image_size'], self.metadata['image_size'], 3)
        if tuple(self.input['shape']) != expected or self.input['dtype'] != np.float32:
            raise ValueError('Unexpected Lite model input')
        self.lock = Lock()

    def predict_array(self, pixels):
        with self.lock:
            self.interpreter.set_tensor(self.input['index'], pixels)
            self.interpreter.invoke()
            probability = float(self.interpreter.get_tensor(self.output['index'])[0, 0])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('Invalid model probability')
        return probability

    def predict_bytes(self, contents):
        probability = self.predict_array(prepare_image(contents, self.metadata['image_size']))
        label = 'AI' if probability >= self.threshold else 'Real'
        return {'label': label, 'p_ai': probability,
                'confidence': probability if label == 'AI' else 1 - probability,
                'threshold': self.threshold, 'confidence_is_calibrated': False,
                'model_name': self.metadata['model_name'],
                'model_version': self.metadata['model_version']}
