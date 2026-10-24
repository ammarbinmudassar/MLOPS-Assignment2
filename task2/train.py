"""Train both approaches and run the required Keras Tuner exercises."""

import argparse
import contextlib
import io
import platform
import subprocess
import time
from datetime import datetime
from importlib.metadata import version

from task2.common import OUT, ROOT, SEED, IMAGE_SIZE, BATCH_SIZE, git_commit, setup, write_json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
from tensorboard.plugins.hparams import api as hparams_api  # Keras Tuner runtime dependency.

from task2.data import make_dataset, prepare_splits, read_split, save_augmented_batch
from task2.models import (
    backbone, baseline, transfer, unfreeze_last_block, unfreeze_last_layers,
)

RESUME = False


def callbacks(directory):
    directory.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(directory / "best.keras"), monitor="val_loss", save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7,
        ),
        tf.keras.callbacks.CSVLogger(str(directory / "epochs.csv")),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


def describe_model(model, name):
    destination = OUT / "summaries"
    destination.mkdir(parents=True, exist_ok=True)
    lines = []
    model.summary(print_fn=lambda line: lines.append(line), expand_nested=True)
    (destination / f"{name}.txt").write_text("\n".join(lines) + "\n")
    rows = []
    for layer in model.layers:
        rows.append({
            "layer": layer.name, "type": type(layer).__name__,
            "output_shape": str(tuple(layer.output.shape)),
            "parameters": layer.count_params(),
            "trainable_parameters": sum(int(np.prod(w.shape)) for w in layer.trainable_weights),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / f"{name}_layers.csv", index=False)
    if name == "baseline":
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis("off")
        table = ax.table(cellText=frame.values, colLabels=frame.columns,
                         loc="center", cellLoc="left", colWidths=[.15, .18, .27, .14, .2])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)
        ax.set_title("Task 2.2 — baseline CNN architecture", pad=20)
        fig.tight_layout()
        fig.savefig(OUT / "figures/baseline_architecture.png", dpi=160)
        plt.close(fig)


def save_trainable_layers(model, name):
    rows = []
    for layer in model.layers:
        nested = layer.layers if isinstance(layer, tf.keras.Model) else [layer]
        for item in nested:
            rows.append({"layer": item.name, "trainable": item.trainable,
                         "trainable_parameters": sum(int(np.prod(w.shape)) for w in item.trainable_weights)})
    pd.DataFrame(rows).to_csv(OUT / "summaries" / f"{name}_trainable_layers.csv", index=False)


def save_curves(frame, name):
    best = int(frame["val_loss"].idxmin()) + 1
    epochs = np.arange(1, len(frame) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, label in zip(axes, ["loss", "accuracy"], ["Binary cross-entropy", "Accuracy"]):
        ax.plot(epochs, frame[metric], label="Training")
        ax.plot(epochs, frame[f"val_{metric}"], label="Validation")
        ax.axvline(best, color="black", linestyle="--", label=f"Best loss: epoch {best}")
        ax.set(xlabel="Epoch", ylabel=label)
        ax.legend()
        ax.grid(alpha=.2)
    fig.suptitle(name.replace("_", " "))
    fig.tight_layout()
    fig.savefig(OUT / "figures" / f"{name}_curves.png", dpi=160)
    plt.close(fig)
    return best


def fit_stage(model, name, train, validation, epochs):
    destination = OUT / "models" / name
    result_path = OUT / "histories" / f"{name}.json"
    if RESUME and result_path.exists() and (destination / "best.keras").exists():
        import json
        print(f"Reusing completed stage: {name}", flush=True)
        return tf.keras.models.load_model(destination / "best.keras"), json.loads(result_path.read_text())
    start = time.monotonic()
    print(f"\n=== {name}: maximum {epochs} epochs ===", flush=True)
    history = model.fit(train, validation_data=validation, epochs=epochs,
                        callbacks=callbacks(destination), verbose=2)
    frame = pd.DataFrame(history.history)
    history_dir = OUT / "histories"
    history_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(history_dir / f"{name}.csv", index=False)
    best = save_curves(frame, name)
    result = {
        "best_epoch": best, "epochs_run": len(frame),
        "best_validation_loss": float(frame.val_loss.min()),
        "seconds": round(time.monotonic() - start, 2),
    }
    write_json(history_dir / f"{name}.json", result)
    return tf.keras.models.load_model(destination / "best.keras"), result


def trial_table(tuner, name):
    # Fail early at module import if TensorBoard is absent, before expensive training.
    _ = hparams_api
    rows = []
    for trial_id, trial in tuner.oracle.trials.items():
        row = {"trial_id": trial_id, "status": trial.status, **trial.hyperparameters.values,
               "best_val_loss": trial.score, "best_step_zero_based": trial.best_step}
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("best_val_loss")
    frame.to_csv(OUT / "tuning" / f"{name}_trials.csv", index=False)
    with contextlib.redirect_stdout(io.StringIO()) as capture:
        tuner.results_summary(num_trials=len(rows))
    (OUT / "tuning" / f"{name}_results_summary.txt").write_text(capture.getvalue())
    print(frame.to_string(index=False), flush=True)
    return frame


def toy_tuning(epochs):
    print("\n=== Fashion-MNIST Keras Tuner toy exercise ===", flush=True)
    (images, labels), _ = tf.keras.datasets.fashion_mnist.load_data()
    # A fixed, class-stratified subset keeps this demonstration small.
    from sklearn.model_selection import train_test_split
    subset, _ = train_test_split(np.arange(len(labels)), train_size=12000,
                                 stratify=labels, random_state=SEED)
    tr, va = train_test_split(subset, test_size=2000, stratify=labels[subset], random_state=SEED)
    x_train = images[tr].astype("float32") / 255
    x_val = images[va].astype("float32") / 255

    def build(hp):
        tf.keras.utils.set_random_seed(SEED)
        model = tf.keras.Sequential([
            tf.keras.Input((28, 28)), tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(hp.Choice("units", [32, 64]), activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice("learning_rate", [1e-3, 3e-4])),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"], jit_compile=False)
        return model

    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    train = tf.data.Dataset.from_tensor_slices((x_train, labels[tr])).shuffle(10000, seed=SEED).batch(64).with_options(options)
    val = tf.data.Dataset.from_tensor_slices((x_val, labels[va])).batch(64).with_options(options)
    tuner = kt.RandomSearch(build, objective="val_loss", max_trials=4, seed=SEED,
                            directory=str(OUT / "tuner_state"), project_name="fashion", overwrite=False)
    tuner.search(train, validation_data=val, epochs=epochs, verbose=2,
                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)])
    frame = trial_table(tuner, "fashion")
    write_json(OUT / "tuning/fashion_best.json", tuner.get_best_hyperparameters(1)[0].values)
    return frame


def vision_tuning(head_model, train, validation, epochs):
    # All trials start from the SAME learned frozen-stage classifier weights.
    initial_head = head_model.get_layer("p_ai").get_weights()

    class VisionHyperModel(kt.HyperModel):
        def build(self, hp):
            tf.keras.utils.set_random_seed(SEED)
            rate = hp.Choice("learning_rate", [1e-5, 1e-4])
            dropout = hp.Choice("dropout", [0.2, 0.5])
            count = hp.Choice("unfrozen_layers", [4, 8])
            model = transfer(dropout=dropout)
            model.get_layer("p_ai").set_weights(initial_head)
            unfreeze_last_layers(model, count, rate)
            return model

        def fit(self, hp, model, *args, **kwargs):
            history = model.fit(*args, **kwargs)
            key = f"lr_{hp.get('learning_rate')}_drop_{hp.get('dropout')}_layers_{hp.get('unfrozen_layers')}"
            pd.DataFrame(history.history).to_csv(OUT / "histories" / f"trial_{key}.csv", index=False)
            return history

    tuner = kt.RandomSearch(
        VisionHyperModel(), objective="val_loss", max_trials=8,
        seed=SEED, directory=str(OUT / "tuner_state"), project_name="vision", overwrite=False,
    )
    tuner.search(train, validation_data=validation, epochs=epochs, verbose=2,
                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
                            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=.5)])
    frame = trial_table(tuner, "vision")
    if len(frame) < 8 or not frame.status.eq("COMPLETED").all():
        raise RuntimeError("Eight completed VisionGuard trials are required.")
    hp = tuner.get_best_hyperparameters(1)[0]
    write_json(OUT / "tuning/vision_best.json", hp.values)
    best = tuner.get_best_models(1)[0]
    best.save(OUT / "models/tuned_transfer.keras")
    save_trainable_layers(best, "transfer_tuned")
    fig, ax = plt.subplots(figsize=(10, 4))
    ordered = frame.sort_values("trial_id")
    ax.bar(ordered.trial_id.astype(str), ordered.best_val_loss)
    ax.set(xlabel="Trial ID", ylabel="Best validation loss", title="Task 2.7 — eight VisionGuard trials")
    fig.tight_layout()
    fig.savefig(OUT / "figures/tuner_trials.png", dpi=160)
    plt.close(fig)
    effects = []
    for parameter in ["learning_rate", "dropout", "unfrozen_layers"]:
        means = frame.groupby(parameter).best_val_loss.mean()
        effects.append({"parameter": parameter, "range_of_mean_val_loss": float(means.max() - means.min()),
                        "group_means": {str(key): float(value) for key, value in means.items()}})
    write_json(OUT / "tuning/parameter_effects.json", effects)
    return best, float(frame.best_val_loss.min())


def main():
    global RESUME
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn-epochs", type=int, default=20)
    parser.add_argument("--head-epochs", type=int, default=12)
    parser.add_argument("--fine-epochs", type=int, default=12)
    parser.add_argument("--trial-epochs", type=int, default=5)
    parser.add_argument("--toy-epochs", type=int, default=3)
    parser.add_argument("--resume", action="store_true", help="Reuse completed stages after interruption")
    args = parser.parse_args()
    RESUME = args.resume
    if RESUME and (OUT / "run_config.json").exists():
        import json
        previous = json.loads((OUT / "run_config.json").read_text())
        for key in ["cnn_epochs", "head_epochs", "fine_epochs", "trial_epochs", "toy_epochs"]:
            if previous[key] != getattr(args, key):
                raise SystemExit(f"Cannot resume with a changed training budget: {key}")
    if (OUT / "evaluation/metrics.csv").exists():
        raise SystemExit("Test evaluation already exists. Use a new output directory for a new experiment.")
    setup()
    for folder in ["figures", "models", "summaries", "histories", "tuning", "logs"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    frames = prepare_splits()
    train = make_dataset(frames["train"], training=True)
    validation = make_dataset(frames["validation"])
    save_augmented_batch(train)
    config_path = OUT / "run_config.json"
    write_json(OUT / ("resume_config.json" if RESUME and config_path.exists() else "run_config.json"), {
        **vars(args), "seed": SEED, "image_size": IMAGE_SIZE, "batch_size": BATCH_SIZE,
        "started_at": datetime.now().astimezone().isoformat(),
        "git_commit": git_commit(), "python": platform.python_version(),
        "tensorflow": version("tensorflow"), "keras": version("keras"),
        "keras_tuner": version("keras-tuner"),
        "gpus": [str(device) for device in tf.config.list_physical_devices("GPU")],
        "selection": "Validation loss only; test images are never read during training or tuning.",
        "tracking": "Local CSV/JSON evidence; Task 3 MLflow/DagsHub logging is not implemented here.",
    })
    packages = subprocess.check_output([str(ROOT / ".venv/bin/python"), "-m", "pip", "freeze"], text=True)
    (OUT / "packages.txt").write_text(packages)
    tf.keras.utils.set_random_seed(SEED)
    cnn = baseline()
    describe_model(cnn, "baseline")
    cnn, cnn_result = fit_stage(cnn, "baseline", train, validation, args.cnn_epochs)
    del cnn
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    mobile = transfer()
    describe_model(mobile, "transfer")
    save_trainable_layers(mobile, "transfer_frozen")
    mobile, head_result = fit_stage(mobile, "transfer_frozen", train, validation, args.head_epochs)
    unfreeze_last_block(mobile)
    save_trainable_layers(mobile, "transfer_last_block")
    mobile, fine_result = fit_stage(mobile, "transfer_last_block", train, validation, args.fine_epochs)
    del mobile
    tf.keras.backend.clear_session()
    head = tf.keras.models.load_model(OUT / "models/transfer_frozen/best.keras")
    tuned, tuned_loss = vision_tuning(head, train, validation, args.trial_epochs)
    # Choose the transfer candidate before touching the test set.
    candidates = {
        "frozen_head": (head_result["best_validation_loss"], "models/transfer_frozen/best.keras"),
        "last_block": (fine_result["best_validation_loss"], "models/transfer_last_block/best.keras"),
        "tuned": (tuned_loss, "models/tuned_transfer.keras"),
    }
    chosen = min(candidates, key=lambda key: candidates[key][0])
    write_json(OUT / "selection.json", {
        "transfer_candidate": chosen, "transfer_path": candidates[chosen][1],
        "candidate_validation_losses": {key: value[0] for key, value in candidates.items()},
        "baseline_path": "models/baseline/best.keras", "test_used": False,
    })
    del head, tuned
    tf.keras.backend.clear_session()
    toy_tuning(args.toy_epochs)
    print("\nTraining complete. Run python -m task2.evaluate for the locked test evaluation.", flush=True)


if __name__ == "__main__":
    main()
