"""Export a float16 LiteRT derivative without modifying the registered champion."""
from pathlib import Path
import hashlib
import json
import tempfile
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'models/champion'
    output = ROOT / 'deploy/vercel_api/model'
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((source / 'metadata.json').read_text())
    model = tf.keras.models.load_model(source / 'model.keras', compile=False)
    with tempfile.TemporaryDirectory() as temporary:
        model.export(temporary, verbose=False)
        converter = tf.lite.TFLiteConverter.from_saved_model(temporary)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        contents = converter.convert()
    (output / 'model.tflite').write_bytes(contents)
    metadata.update(source_model_version=metadata['model_version'],
                    source_sha256=metadata['sha256'],
                    model_version=metadata['model_version'] + '-fp16',
                    sha256=hashlib.sha256(contents).hexdigest(),
                    quantization='float16 weights; float32 inputs and outputs',
                    preprocessing='pillow_rgb_numpy_antialias_letterbox_uint8_v1')
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    report = {'source_bytes': (source / 'model.keras').stat().st_size,
              'lite_bytes': len(contents), 'sha256': metadata['sha256'],
              'runtime_ops': 'TFLITE_BUILTINS only; no TensorFlow Flex dependency'}
    (ROOT / 'outputs/task7/export.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
