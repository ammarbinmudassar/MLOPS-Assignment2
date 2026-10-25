"""Task 3.4: attach verified artifacts to the completed VisionGuard runs."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys
import tempfile

from task2.common import ROOT, write_json, setup, git_commit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, roc_auc_score

from task2.data import make_dataset, read_split
from task3.dagshub_config import configure_dagshub

OUT = ROOT / "outputs/task3/3_4"
TRAINING = ROOT / "outputs/task3/3_3"


def figures(predictions, destination):
    labels, probabilities = predictions.label, predictions.p_ai
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(labels, probabilities >= .5, labels=[0, 1]),
        display_labels=["Real", "AI"],
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Test confusion matrix — threshold 0.5")
    fig.tight_layout()
    fig.savefig(destination / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    false_positive, true_positive, _ = roc_curve(labels, probabilities)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(false_positive, true_positive, label=f"AUC = {roc_auc_score(labels, probabilities):.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=.5)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Test ROC curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination / "roc_curve.png", dpi=160)
    plt.close(fig)


def main():
    configure_dagshub()
    setup()
    OUT.mkdir(parents=True, exist_ok=True)
    package_list = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True,
    ).stdout
    (OUT / "packages.txt").write_text(package_list)
    runs = json.loads((TRAINING / "runs.json").read_text())
    client = mlflow.MlflowClient()
    split_folder = ROOT / "outputs/task2/splits"
    example_frame = read_split("test").iloc[:3]
    example = next(iter(make_dataset(example_frame)))[0].numpy()
    checks = []
    for saved in runs:
        run_id, method = saved["run_id"], saved["method"]
        run = client.get_run(run_id)
        assert run.info.status == "FINISHED"
        destination = OUT / method
        destination.mkdir(exist_ok=True)
        predictions = pd.read_csv(TRAINING / method / "test_predictions.csv")
        assert predictions.path.tolist() == read_split("test").path.tolist()
        figures(predictions, destination)
        for name in ("confusion_matrix.png", "roc_curve.png"):
            client.log_artifact(run_id, str(destination / name), "evaluation")
        client.log_artifact(run_id, str(TRAINING / method / "test_predictions.csv"), "evaluation")
        client.log_artifacts(run_id, str(split_folder), "splits")
        client.log_artifact(run_id, str(OUT / "packages.txt"), "environment")
        client.log_artifact(run_id, str(ROOT / "dataset.dvc"), "provenance")
        model_path = TRAINING / method / "best.keras"
        model = tf.keras.models.load_model(model_path)
        expected = model.predict(example, verbose=0)
        np.testing.assert_allclose(expected.ravel(), predictions.p_ai.iloc[:3], atol=1e-6, rtol=1e-5)
        with tempfile.TemporaryDirectory() as temporary:
            export = Path(temporary) / "model"
            mlflow.keras.save_model(
                model, str(export), signature=infer_signature(example, expected),
                pip_requirements=[f"tensorflow=={version('tensorflow')}",
                                  f"keras=={version('keras')}", f"numpy=={version('numpy')}"],
            )
            client.log_artifacts(run_id, str(export), "model")
        model_uri = f"runs:/{run_id}/model"
        restored = mlflow.keras.load_model(model_uri)
        np.testing.assert_allclose(restored.predict(example, verbose=0), expected, atol=1e-6, rtol=1e-5)
        # Download every required artifact and compare its bytes with the source.
        sources = {
            "evaluation/confusion_matrix.png": destination / "confusion_matrix.png",
            "evaluation/roc_curve.png": destination / "roc_curve.png",
            "environment/packages.txt": OUT / "packages.txt",
            "model/data/model.keras": model_path,
            **{f"splits/{name}.csv": split_folder / f"{name}.csv"
               for name in ("train", "validation", "test")},
        }
        hashes = {}
        with tempfile.TemporaryDirectory() as temporary:
            for artifact, original in sources.items():
                downloaded = Path(client.download_artifacts(run_id, artifact, temporary))
                # Re-saving a Keras ZIP changes its metadata; compare model predictions
                # above and record the downloaded model hash instead of byte equality.
                digest = hashlib.sha256(downloaded.read_bytes()).hexdigest()
                if artifact != "model/data/model.keras":
                    assert digest == hashlib.sha256(original.read_bytes()).hexdigest()
                hashes[artifact] = digest
        client.set_tag(run_id, "task_3_4_artifacts_verified", "true")
        client.set_tag(run_id, "model_reload_verified", "true")
        client.set_tag(run_id, "artifact_code_commit", git_commit())
        client.set_tag(run_id, "model_logging", "Task 3.4: MLflow Keras run artifact")
        assert client.get_run(run_id).info.status == "FINISHED"
        checks.append({"method": method, "run_id": run_id,
                       "status": "FINISHED", "model_uri": model_uri,
                       "remote_model_prediction_check": "passed",
                       "artifact_sha256": hashes})
        print(f"Verified all Task 3.4 artifacts and model reload: {method}", flush=True)
        del model, restored
        tf.keras.backend.clear_session()
    write_json(OUT / "verification.json", checks)


if __name__ == "__main__":
    main()
