"""Task 4.1 evidence: reuse one loaded model for the first AI and Real test images."""

import csv
from unittest.mock import patch

from task2.common import ROOT, setup, write_json
import tensorflow as tf
from task4.inference import VisionGuardPredictor


def main():
    setup()
    with (ROOT / "outputs/task2/splits/test.csv").open() as source:
        rows = list(csv.DictReader(source))
    examples = [next(row for row in rows if row["class"] == label) for label in ("AI", "Real")]
    results = []
    # This also verifies that startup and inference need no network connection.
    with patch("socket.socket.connect", side_effect=AssertionError("Inference attempted network access")), \
            patch("task4.inference.tf.keras.models.load_model", wraps=tf.keras.models.load_model) as load_model:
        predictor = VisionGuardPredictor(ROOT / "models/champion")
        print(f"Loaded local model once: {predictor.metadata['model_name']} version {predictor.metadata['model_version']}")
        for row in examples:
            result = predictor.predict_bytes((ROOT / "dataset" / row["path"]).read_bytes())
            results.append({"image": row["path"], "dataset_label": row["class"], **result})
            print(f"\nImage: {row['path']}")
            print(f"Dataset label: {row['class']}")
            print(f"Prediction: {result['label']} | Confidence: {result['confidence']:.2%}")
            print(f"AI probability: {result['p_ai']:.6f}")
        assert load_model.call_count == 1
        write_json(ROOT / "outputs/task4/4_1/verification.json", {
            "model_load_count": load_model.call_count, "images_classified": len(results),
            "network_connections_blocked": True, "all_predictions_completed": True,
        })
    print("\nTwo images used the same loaded model; network calls were blocked.")
    print("Confidence is an uncalibrated model score.")
    write_json(ROOT / "outputs/task4/4_1/predictions.json", results)


if __name__ == "__main__":
    main()
