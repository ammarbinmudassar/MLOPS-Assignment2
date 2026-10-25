"""Load a registered Keras model by alias and classify one image."""

import argparse
import csv
from pathlib import Path

from task2.common import ROOT, setup, write_json

import mlflow
import mlflow.keras
import tensorflow as tf

from task2.data import decode_resize
from task3.dagshub_config import configure_dagshub
from task3.register_models import MODEL_NAME, OUT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, nargs="?", help="Image path; omit to use the first saved test image")
    args = parser.parse_args()
    image = args.image
    expected_label = None
    if image is None:
        with (ROOT / "outputs/task2/splits/test.csv").open() as source:
            first = next(csv.DictReader(source))
        image = ROOT / "dataset" / first["path"]
        expected_label = first["class"]
    if not image.is_file():
        raise SystemExit(f"Image not found: {image}")
    configure_dagshub()
    setup()
    client = mlflow.MlflowClient()
    resolved = client.get_model_version_by_alias(MODEL_NAME, "champion")
    threshold = float(resolved.tags["decision_threshold"])
    uri = f"models:/{MODEL_NAME}@champion"
    model = mlflow.keras.load_model(uri)
    # Use the exact training decoder and letterbox; the model scales pixels itself.
    pixels, _ = decode_resize(str(image), 0)
    batch = tf.cast(pixels[None, ...], tf.float32)
    probability = float(model.predict(batch, verbose=0)[0, 0])
    prediction = "AI" if probability >= threshold else "Real"
    confidence = probability if prediction == "AI" else 1 - probability
    after = client.get_model_version_by_alias(MODEL_NAME, "champion")
    if after.version != resolved.version:
        raise RuntimeError("Champion alias changed during loading; rerun to get a consistent version.")
    result = {"model_uri": uri, "model_version": resolved.version,
              "source_run_id": resolved.run_id, "image": str(image.relative_to(ROOT)) if image.is_relative_to(ROOT) else str(image),
              "expected_label": expected_label, "prediction": prediction,
              "p_ai": probability, "confidence": confidence, "threshold": threshold,
              "confidence_is_calibrated": False}
    write_json(OUT / "prediction.json", result)
    print("Loaded:", uri)
    print("Resolved version:", resolved.version)
    print("Image:", result["image"])
    if expected_label:
        print("Dataset label:", expected_label)
    print("Prediction:", prediction)
    print(f"AI probability: {probability:.6f}")
    print(f"Confidence: {confidence:.2%} (uncalibrated)")


if __name__ == "__main__":
    main()
