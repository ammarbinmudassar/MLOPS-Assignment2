"""Task 3.5: six controlled VisionGuard experiments, selected on validation."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path
import tempfile
import time

from task2.common import ROOT, SEED, BATCH_SIZE, IMAGE_SIZE, setup, write_json, git_commit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from task2.data import make_dataset
from task2.models import transfer, compile_binary, unfreeze_last_layers
from task3.attach_artifacts import figures
from task3.dagshub_config import configure_dagshub
from task3.local_lab import save_plot
from task3.vision_training import verify_data

OUT = ROOT / "outputs/task3/3_5"
HEAD_EPOCHS = 8
ADAPT_EPOCHS = 8
REFERENCE = dict(learning_rate=1e-4, augmentation=True, dropout=.3, unfrozen_layers=0)
CONFIGS = [
    dict(name="reference", **REFERENCE),
    dict(name="lower_lr", **{**REFERENCE, "learning_rate": 3e-5}),
    dict(name="no_augmentation", **{**REFERENCE, "augmentation": False}),
    dict(name="higher_dropout", **{**REFERENCE, "dropout": .5}),
    dict(name="unfreeze_4", **{**REFERENCE, "unfrozen_layers": 4}),
    dict(name="unfreeze_8", **{**REFERENCE, "unfrozen_layers": 8}),
]


def validate_configs():
    assert len(CONFIGS) >= 6
    assert len({config["name"] for config in CONFIGS}) == len(CONFIGS)
    keys = tuple(REFERENCE)
    for config in CONFIGS[1:]:
        changes = [key for key in keys if config[key] != REFERENCE[key]]
        assert len(changes) == 1, (config["name"], changes)


class EpochMetrics(tf.keras.callbacks.Callback):
    def __init__(self, phase, offset):
        self.phase, self.offset = phase, offset

    def on_epoch_end(self, epoch, logs=None):
        metrics = {key: float(value) for key, value in (logs or {}).items()}
        mlflow.log_metrics(metrics, step=self.offset + epoch)
        mlflow.log_metrics({f"{self.phase}_{key}": value for key, value in metrics.items()}, step=epoch)


def fit(model, training, validation, destination, phase, offset):
    history = model.fit(
        training, validation_data=validation, epochs=HEAD_EPOCHS if phase == "head" else ADAPT_EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(str(destination / f"{phase}.keras"),
                                               monitor="val_loss", save_best_only=True),
            tf.keras.callbacks.TerminateOnNaN(), EpochMetrics(phase, offset),
        ], verbose=2,
    ).history
    pd.DataFrame(history).to_csv(destination / f"{phase}_history.csv", index=False)
    return history


def classifier_metrics(labels, probabilities, prefix):
    predicted = probabilities >= .5
    return {f"{prefix}_accuracy": float(accuracy_score(labels, predicted)),
            f"{prefix}_precision": float(precision_score(labels, predicted, zero_division=0)),
            f"{prefix}_recall": float(recall_score(labels, predicted, zero_division=0)),
            f"{prefix}_f1": float(f1_score(labels, predicted, zero_division=0)),
            f"{prefix}_roc_auc": float(roc_auc_score(labels, probabilities))}


def train(config, frames, data_version, split_version):
    destination = OUT / config["name"]
    result_file = destination / "result.json"
    client = mlflow.MlflowClient()
    if result_file.exists():
        saved = json.loads(result_file.read_text())
        if client.get_run(saved["run_id"]).info.status == "FINISHED":
            assert (destination / "best.keras").exists()
            print("Reusing completed experiment:", config["name"], flush=True)
            return saved
    destination.mkdir(parents=True, exist_ok=True)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    model = transfer(dropout=config["dropout"])
    training = make_dataset(frames["train"], training=True, augment=config["augmentation"])
    validation = make_dataset(frames["validation"])
    changed = next((key for key in REFERENCE if config[key] != REFERENCE[key]), "reference")
    with mlflow.start_run(run_name=f"3.5_{config['name']}") as run:
        mlflow.log_params({**{key: config[key] for key in REFERENCE},
                           "backbone": "MobileNetV2", "pretrained_weights": "ImageNet",
                           "head_learning_rate": .001, "head_max_epochs": HEAD_EPOCHS,
                           "adapt_max_epochs": ADAPT_EPOCHS, "train_batch_size": BATCH_SIZE,
                           "image_size": IMAGE_SIZE, "seed": SEED, "threshold": .5,
                           "early_stopping_patience": 3, "batchnorm_frozen": True})
        mlflow.set_tags({"task": "3.5", "git_commit": git_commit(), "seed": str(SEED),
                         "dvc_data_version": data_version, "split_sha256": split_version,
                         "comparison_reference": "3.5_reference", "changed_setting": changed,
                         "selection_rule": "lowest validation loss; ROC-AUC breaks ties",
                         "test_usage": "only validation-selected winner evaluated"})
        started = time.monotonic()
        head = fit(model, training, validation, destination, "head", 0)
        if config["unfrozen_layers"]:
            unfreeze_last_layers(model, config["unfrozen_layers"], config["learning_rate"])
        else:
            compile_binary(model, config["learning_rate"])
        adapt = fit(model, training, validation, destination, "adapt", len(head["loss"]))
        # Keep the best validation checkpoint across BOTH phases.
        phase = "head" if min(head["val_loss"]) <= min(adapt["val_loss"]) else "adapt"
        model = tf.keras.models.load_model(destination / f"{phase}.keras")
        validation_loss = float(model.evaluate(validation, verbose=0, return_dict=True)["loss"])
        probabilities = model.predict(validation, verbose=0).ravel()
        metrics = {"selection_val_loss": validation_loss,
                   **classifier_metrics(frames["validation"].label, probabilities, "selection_val"),
                   "epochs_run": len(head["loss"]) + len(adapt["loss"]),
                   "training_seconds": time.monotonic() - started}
        mlflow.log_metrics(metrics)
        mlflow.set_tag("selected_phase", phase)
        model.save(destination / "best.keras")
        combined = {key: head[key] + adapt[key] for key in head}
        pd.DataFrame(combined).to_csv(destination / "history.csv", index=False)
        save_plot(combined, destination / "training_curves.png")
        predictions = frames["validation"][["path", "label"]].copy()
        predictions["p_ai"] = probabilities
        predictions.to_csv(destination / "validation_predictions.csv", index=False)
        mlflow.log_artifact(str(destination / "training_curves.png"), "plots")
        for file in ("head_history.csv", "adapt_history.csv", "history.csv", "validation_predictions.csv"):
            mlflow.log_artifact(str(destination / file), "tables")
        mlflow.log_artifacts(str(ROOT / "outputs/task2/splits"), "splits")
        mlflow.log_artifact(str(ROOT / "requirements.txt"), "environment")
        example = next(iter(validation))[0].numpy()[:3]
        expected = model.predict(example, verbose=0)
        with tempfile.TemporaryDirectory() as temporary:
            export = Path(temporary) / "model"
            mlflow.keras.save_model(model, str(export), signature=infer_signature(example, expected),
                                    pip_requirements=[f"tensorflow=={version('tensorflow')}",
                                                      f"keras=={version('keras')}", f"numpy=={version('numpy')}"])
            mlflow.log_artifacts(str(export), "model")
        result = {**config, "changed_setting": changed, "selected_phase": phase,
                  "run_id": run.info.run_id, "experiment_id": run.info.experiment_id,
                  "model_uri": f"runs:/{run.info.run_id}/model", **metrics}
        write_json(result_file, result)
        print("COMPLETED", config["name"], metrics, flush=True)
    return result


def select_and_evaluate(runs, frames):
    lock_file = OUT / "selection.json"
    winner = min(runs, key=lambda row: (row["selection_val_loss"], -row["selection_val_roc_auc"], row["name"]))
    lock = {"name": winner["name"], "run_id": winner["run_id"], "model_uri": winner["model_uri"],
            "selection_rule": "minimum validation binary cross-entropy; ROC-AUC breaks ties",
            "decision_threshold": .5, "test_consulted_for_selection": False}
    if lock_file.exists():
        assert json.loads(lock_file.read_text()) == lock
    else:
        write_json(lock_file, lock)  # Lock selection BEFORE opening the test set.
    result_file = OUT / "champion_test_metrics.json"
    if result_file.exists():
        return winner, json.loads(result_file.read_text())
    model = tf.keras.models.load_model(OUT / winner["name"] / "best.keras")
    probabilities = model.predict(make_dataset(frames["test"]), verbose=0).ravel()
    test = frames["test"][["path", "label"]].copy()
    test["p_ai"] = probabilities
    test.to_csv(OUT / "champion_test_predictions.csv", index=False)
    metrics = classifier_metrics(test.label, probabilities, "test")
    figures(test, OUT)
    client = mlflow.MlflowClient()
    client.log_batch(winner["run_id"], metrics=[
        mlflow.entities.Metric(key, value, int(time.time() * 1000), 0) for key, value in metrics.items()
    ])
    for file in ("confusion_matrix.png", "roc_curve.png", "champion_test_predictions.csv"):
        client.log_artifact(winner["run_id"], str(OUT / file), "evaluation")
    # Check the actual hosted champion model, not just the local checkpoint.
    example = next(iter(make_dataset(frames["test"].iloc[:3])))[0].numpy()
    restored = mlflow.keras.load_model(winner["model_uri"])
    np.testing.assert_allclose(restored.predict(example, verbose=0), model.predict(example, verbose=0),
                               atol=1e-6, rtol=1e-5)
    client.set_tag(winner["run_id"], "model_reload_verified", "true")
    client.set_tag(winner["run_id"], "task_3_5_selected", "true")
    write_json(result_file, metrics)
    return winner, metrics


def comparison(runs, winner):
    table = pd.DataFrame(runs).sort_values("selection_val_loss")
    table.to_csv(OUT / "comparison.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#237a57" if name == winner["name"] else "#527aab" for name in table.name]
    for ax, metric, label in zip(axes, ("selection_val_loss", "selection_val_roc_auc"),
                                  ("Validation loss (lower is better)", "Validation ROC-AUC (higher is better)")):
        ax.barh(table.name, table[metric], color=colors)
        ax.set_xlabel(label)
        ax.invert_yaxis()
    fig.suptitle("Six controlled MobileNetV2 experiments — green = validation-selected winner")
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png", dpi=160)
    plt.close(fig)
    client = mlflow.MlflowClient()
    for row in runs:
        for file in ("comparison.csv", "comparison.png", "selection.json"):
            client.log_artifact(row["run_id"], str(OUT / file), "comparison")


def verify(runs):
    client = mlflow.MlflowClient()
    checks = []
    for saved in runs:
        run = client.get_run(saved["run_id"])
        assert run.info.status == "FINISHED"
        for metric in ("loss", "accuracy", "auc", "val_loss", "val_accuracy", "val_auc"):
            history = client.get_metric_history(saved["run_id"], metric)
            assert {point.step for point in history} == set(range(saved["epochs_run"]))
        for key in ("selection_val_loss", "selection_val_accuracy", "selection_val_roc_auc", "selection_val_f1"):
            assert np.isclose(run.data.metrics[key], saved[key])
        assert "model/data/model.keras" in {file.path for file in client.list_artifacts(saved["run_id"], "model/data")}
        checks.append({"name": saved["name"], "run_id": saved["run_id"], "status": run.info.status,
                       "epochs_verified": saved["epochs_run"], "validation_metrics_verified": True,
                       "remote_model_present": True, "changed_setting": saved["changed_setting"]})
    write_json(OUT / "verification.json", checks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=[config["name"] for config in CONFIGS],
                        help="Run one configuration; automatically finalize when all six exist")
    args = parser.parse_args()
    validate_configs()
    configure_dagshub()
    setup()
    OUT.mkdir(parents=True, exist_ok=True)
    frames, data_version, split_version = verify_data()
    mlflow.set_experiment("23L-0630_A2_Ops")
    selected_configs = [config for config in CONFIGS if not args.run or config["name"] == args.run]
    for config in selected_configs:
        train(config, frames, data_version, split_version)
    missing = [config["name"] for config in CONFIGS if not (OUT / config["name"] / "result.json").exists()]
    if missing:
        print("Remaining experiments:", ", ".join(missing), flush=True)
        return
    runs = [json.loads((OUT / config["name"] / "result.json").read_text()) for config in CONFIGS]
    write_json(OUT / "runs.json", runs)
    winner, metrics = select_and_evaluate(runs, frames)
    comparison(runs, winner)
    verify(runs)
    print("VERIFIED SIX FINISHED RUNS. Validation-selected winner:", winner["name"], flush=True)
    print("Champion test results:", metrics, flush=True)


if __name__ == "__main__":
    main()
