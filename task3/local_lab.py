"""Task 3.1: compare two Keras runs using a local MLflow tracking server."""

import argparse
import json
from importlib.metadata import version

from task2.common import ROOT, SEED, git_commit, setup

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

OUT = ROOT / "outputs/task3/3_1"
EXPERIMENT = "23L-0630_A2_Ops"


class LogEpoch(tf.keras.callbacks.Callback):
    """Send the epoch's measured results to the currently active MLflow run."""

    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({key: float(value) for key, value in logs.items()}, step=epoch + 1)


def dataset(images, labels, training):
    data = tf.data.Dataset.from_tensor_slices((images, labels))
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    data = data.with_options(options)
    if training:
        data = data.shuffle(len(labels), seed=SEED, reshuffle_each_iteration=True)
    return data.batch(64).prefetch(1)


def build_model(learning_rate):
    model = tf.keras.Sequential([
        tf.keras.Input((28, 28), name="image"),
        tf.keras.layers.Rescaling(1 / 255.0),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"],
                  jit_compile=False)
    return model


def save_plot(history, destination):
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ["loss", "accuracy"]):
        ax.plot(epochs, history[metric], label="Training")
        ax.plot(epochs, history[f"val_{metric}"], label="Validation")
        ax.set(xlabel="Epoch", ylabel=metric.title())
        ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    args = parser.parse_args()
    setup()
    OUT.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    (images, labels), _ = tf.keras.datasets.fashion_mnist.load_data()
    subset, _ = train_test_split(np.arange(len(labels)), train_size=12000,
                                 stratify=labels, random_state=SEED)
    train, validation = train_test_split(subset, test_size=2000,
                                        stratify=labels[subset], random_state=SEED)
    images = images.astype("float32")
    summaries = []
    for learning_rate in [0.001, 0.0001]:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        name = f"3.1_fashion_lr_{learning_rate}"
        run_dir = OUT / name
        run_dir.mkdir(exist_ok=True)
        model = build_model(learning_rate)
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params({
                "learning_rate": learning_rate, "hidden_units": 64, "epochs": 5,
                "batch_size": 64, "optimizer": "Adam", "seed": SEED,
                "dataset": "Fashion-MNIST", "train_samples": len(train),
                "validation_samples": len(validation),
            })
            mlflow.set_tags({"task": "3.1", "logging": "manual_callback",
                             "git_commit": git_commit(), "purpose": "local_tool_lab"})
            history = model.fit(
                dataset(images[train], labels[train], True),
                validation_data=dataset(images[validation], labels[validation], False),
                epochs=5, callbacks=[LogEpoch()], verbose=2,
            ).history
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            save_plot(history, run_dir / "training_curves.png")
            mlflow.log_artifact(str(run_dir / "training_curves.png"), "plots")
            mlflow.log_artifact(str(run_dir / "history.csv"), "tables")
            lines = []
            model.summary(print_fn=lambda line: lines.append(line))
            mlflow.log_text("\n".join(lines), "model_summary.txt")
            example = images[train[:3]]
            probabilities = model.predict(example, verbose=0)
            logged_model = mlflow.keras.log_model(
                model, name="model", input_example=example,
                signature=infer_signature(example, probabilities),
                pip_requirements=[f"tensorflow=={version('tensorflow')}",
                                  f"keras=={version('keras')}", f"numpy=={version('numpy')}"],
            )
            # Verify the stored MLflow model can actually be loaded and used.
            restored = mlflow.keras.load_model(logged_model.model_uri)
            np.testing.assert_allclose(
                restored.predict(example, verbose=0), probabilities, rtol=1e-5, atol=1e-6,
            )
            mlflow.set_tag("model_reload_verified", "true")
            summaries.append({"name": name, "run_id": run.info.run_id,
                              "experiment_id": run.info.experiment_id,
                              "learning_rate": learning_rate,
                              "final_val_loss": history["val_loss"][-1],
                              "final_val_accuracy": history["val_accuracy"][-1],
                              "model_uri": logged_model.model_uri})
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "comparison.csv", index=False)
    (OUT / "runs.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print("\nCOMPARISON\n", summary.to_string(index=False))
    print(f"Open {args.tracking_uri} and select experiment {EXPERIMENT}.")


if __name__ == "__main__":
    main()

