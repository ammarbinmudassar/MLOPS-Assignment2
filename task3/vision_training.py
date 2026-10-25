"""Task 3.3: compare manual and TensorFlow autologging on VisionGuard."""

import argparse
import hashlib
import json
import time

from task2.common import ROOT, SEED, BATCH_SIZE, IMAGE_SIZE, git_commit, setup, write_json

import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from task2.data import read_split, check_splits, make_dataset
from task2.models import baseline
from task3.dagshub_config import configure_dagshub
from task3.local_lab import save_plot

OUT = ROOT / "outputs/task3/3_3"


class ManualMetrics(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({key: float(value) for key, value in (logs or {}).items()}, step=epoch)


def verify_data():
    """Fail before training if an image or the saved split has changed."""
    frames = {name: read_split(name) for name in ("train", "validation", "test")}
    check_splits(frames)
    for frame in frames.values():
        for row in frame.itertuples():
            with (ROOT / "dataset" / row.path).open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if digest != row.sha256:
                raise ValueError(f"Image changed since audit: {row.path}")
    dvc = yaml.safe_load((ROOT / "dataset.dvc").read_text())
    data_version = dvc["outs"][0]["md5"]
    # The three manifests identify exactly which subset of the DVC data is used.
    split_version = hashlib.sha256(b"".join(
        (ROOT / "outputs/task2/splits" / f"{name}.csv").read_bytes()
        for name in ("train", "validation", "test")
    )).hexdigest()
    return frames, data_version, split_version


def train(method, epochs, frames, data_version, split_version):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    # Only the autolog run uses MLflow's injected training callbacks.
    mlflow.tensorflow.autolog(disable=True)
    if method == "autolog":
        mlflow.tensorflow.autolog(
            log_models=False, checkpoint=False, log_datasets=False,
            log_every_epoch=True, exclusive=False,
        )
    destination = OUT / method
    destination.mkdir(parents=True, exist_ok=True)
    model = baseline()
    training = make_dataset(frames["train"], training=True)
    validation = make_dataset(frames["validation"])
    stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True,
    )
    callbacks = [stopping, tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=.5, patience=2, min_lr=1e-7,
    ), tf.keras.callbacks.TerminateOnNaN()]
    if method == "manual":
        callbacks.append(ManualMetrics())
    with mlflow.start_run(run_name=f"3.3_visionguard_{method}") as run:
        mlflow.set_tags({
            "task": "3.3", "logging_method": method, "git_commit": git_commit(),
            "dvc_data_version": data_version, "dvc_file": "dataset.dvc",
            "split_sha256": split_version, "seed": str(SEED),
            "model_family": "baseline_cnn", "test_usage": "final evaluation only",
            "model_logging": "local checkpoint; remote model attachment reserved for 3.4",
        })
        # These describe our pipeline, including details autolog cannot infer.
        mlflow.log_params({
            "architecture": "baseline_cnn", "initial_learning_rate": .001,
            "train_batch_size": BATCH_SIZE, "image_size": IMAGE_SIZE,
            "augmentation": "horizontal_flip,rotation_0.04,zoom_0.1,contrast_0.1",
            "seed": SEED, "decision_threshold": .5,
            "train_samples": len(frames["train"]),
            "validation_samples": len(frames["validation"]),
            "test_samples": len(frames["test"]),
        })
        if method == "manual":
            mlflow.log_params({"epochs": epochs, "optimizer": "Adam",
                               "early_stopping_patience": 4, "restore_best_weights": True})
            lines = []
            model.summary(print_fn=lambda line: lines.append(line))
            mlflow.log_text("\n".join(lines), "model_summary.txt")
        started = time.monotonic()
        history = model.fit(training, validation_data=validation, epochs=epochs,
                            callbacks=callbacks, verbose=2).history
        elapsed = time.monotonic() - started
        mlflow.tensorflow.autolog(disable=True)
        # Test results and project-specific provenance are explicit in BOTH modes.
        # Autolog's fit hook does not compute held-out precision/F1/ROC-AUC.
        probabilities = model.predict(make_dataset(frames["test"]), verbose=0).ravel()
        labels = frames["test"].label.to_numpy()
        predicted = (probabilities >= .5).astype(int)
        metrics = {
            "test_accuracy": accuracy_score(labels, predicted),
            "test_precision": precision_score(labels, predicted, zero_division=0),
            "test_recall": recall_score(labels, predicted, zero_division=0),
            "test_f1": f1_score(labels, predicted, zero_division=0),
            "test_roc_auc": roc_auc_score(labels, probabilities),
            "epochs_run": len(history["loss"]),
            "best_epoch": int(np.argmin(history["val_loss"])) + 1,
            "training_seconds": elapsed,
        }
        mlflow.log_metrics(metrics)
        model.save(destination / "best.keras")
        pd.DataFrame(history).to_csv(destination / "history.csv", index=False)
        save_plot(history, destination / "training_curves.png")
        mlflow.log_artifact(str(destination / "history.csv"), "tables")
        mlflow.log_artifact(str(destination / "training_curves.png"), "plots")
        predictions = frames["test"][["path", "label"]].copy()
        predictions["p_ai"] = probabilities
        predictions.to_csv(destination / "test_predictions.csv", index=False)
        result = {"method": method, "run_id": run.info.run_id,
                  "experiment_id": run.info.experiment_id, **metrics}
        write_json(destination / "result.json", result)
        return result


def verify(runs):
    client = mlflow.MlflowClient()
    checks = []
    for saved in runs:
        run = client.get_run(saved["run_id"])
        assert run.info.status == "FINISHED"
        points = {}
        for key in ("loss", "accuracy", "auc", "val_loss", "val_accuracy", "val_auc"):
            history = client.get_metric_history(run.info.run_id, key)
            steps = {point.step for point in history}
            assert set(range(saved["epochs_run"])).issubset(steps), (key, steps)
            points[key] = len(history)
        for tag in ("git_commit", "dvc_data_version", "seed", "logging_method"):
            assert run.data.tags.get(tag)
        for key in ("test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"):
            assert np.isclose(run.data.metrics[key], saved[key])
        artifacts = [item.path for item in client.list_artifacts(run.info.run_id)]
        checks.append({"run_id": run.info.run_id, "method": saved["method"],
                       "status": run.info.status, "metric_points": points,
                       "params": run.data.params, "tags": run.data.tags,
                       "root_artifacts": artifacts, "test_metrics_verified": True})
    write_json(OUT / "verification.json", checks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    if (OUT / "runs.json").exists():
        raise SystemExit("Task 3.3 results already exist; do not overwrite completed runs.")
    configure_dagshub()
    setup()
    frames, data_version, split_version = verify_data()
    mlflow.set_experiment("23L-0630_A2_Ops")
    runs = [train(method, args.epochs, frames, data_version, split_version)
            for method in ("manual", "autolog")]
    write_json(OUT / "runs.json", runs)
    verify(runs)
    pd.DataFrame(runs).to_csv(OUT / "comparison.csv", index=False)
    print(pd.DataFrame(runs).to_string(index=False))
    print("Verified both hosted VisionGuard runs, epoch metrics, test metrics and provenance tags.")


if __name__ == "__main__":
    main()
