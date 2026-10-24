"""Select thresholds on validation data, then evaluate each final model once."""

import json

from task2.common import CLASS_NAMES, OUT, ROOT, git_commit, setup, write_json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)

from task2.data import make_dataset, read_split
from task2.train import save_curves


def metric_values(labels, probabilities, threshold):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold), "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "weighted_error": int(2 * fp + fn), "sample_count": len(labels),
    }


def choose_threshold(labels, probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    # Every distinct decision boundary, including predict-all/predict-none.
    candidates = np.unique(np.r_[0.0, probabilities, np.nextafter(probabilities.max(), np.inf), 1.0])
    table = pd.DataFrame([metric_values(labels, probabilities, value) for value in candidates])
    table["distance_from_half"] = abs(table.threshold - .5)
    ordered = table.sort_values(
        ["weighted_error", "f1", "distance_from_half"], ascending=[True, False, True],
    )
    return ordered.iloc[0].to_dict(), table


def save_error_grid(frame, name):
    # Most confident errors are interesting, but do not represent their frequency.
    errors = frame.loc[frame.label != frame.predicted_label].sort_values(
        "confidence", ascending=False,
    ).head(10).copy()
    errors["figure_id"] = np.arange(1, len(errors) + 1)
    errors["hypothesis"] = "Pending visual review; a hypothesis is not a verified cause."
    errors.to_csv(OUT / "evaluation" / f"{name}_errors.csv", index=False)
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, record) in zip(axes.flat, errors.iterrows()):
        with Image.open(ROOT / "dataset" / record["path"]) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(
            f"#{int(record.figure_id)} True: {CLASS_NAMES[int(record.label)]}\n"
            f"Predicted: {CLASS_NAMES[int(record.predicted_label)]} ({record.confidence:.1%})",
            fontsize=10,
        )
    fig.suptitle(f"{name}: {len(errors)} most confident test errors — hypotheses in CSV")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / f"{name}_errors.png", dpi=170)
    plt.close(fig)
    return len(errors)


def save_evaluation_figures(predictions, results):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, frame in predictions.items():
        fpr, tpr, _ = roc_curve(frame.label, frame.p_ai)
        auc = results.loc[results.model == name, "roc_auc"].iloc[0]
        ax.plot(fpr, tpr, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=.5)
    ax.set(xlabel="False-positive rate (Real called AI)", ylabel="True-positive rate (AI recall)",
           title="Task 2.5 — held-out test ROC curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "figures/roc_curves.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, frame) in zip(axes, predictions.items()):
        matrix = confusion_matrix(frame.label, frame.predicted_label, labels=[0, 1])
        ax.imshow(matrix, cmap="Blues", vmin=0)
        for row in range(2):
            for column in range(2):
                ax.text(column, row, str(matrix[row, column]), ha="center", va="center",
                        color="white" if matrix[row, column] > matrix.max() / 2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Real", "AI"], yticklabels=["Real", "AI"],
               xlabel="Predicted", ylabel="Actual", title=name)
    fig.suptitle("Task 2.5 — test confusion matrices at validation-selected thresholds")
    fig.tight_layout()
    fig.savefig(OUT / "figures/confusion_matrices.png", dpi=170)
    plt.close(fig)


def main():
    setup()
    destination = OUT / "evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    # Avoid repeatedly consulting the test set to improve the model.
    if (destination / "metrics.csv").exists():
        raise SystemExit("Test results already exist. Read the saved predictions; do not retune on test.")
    selection = json.loads((OUT / "selection.json").read_text())
    validation = read_split("validation")
    test = read_split("test")
    validation_ds = make_dataset(validation)
    paths = {"baseline": selection["baseline_path"], "transfer": selection["transfer_path"]}
    chosen = {}
    for name, path in paths.items():
        model = tf.keras.models.load_model(OUT / path)
        probabilities = model.predict(validation_ds, verbose=0).ravel().astype(np.float64)
        best, thresholds = choose_threshold(validation.label.to_numpy(), probabilities)
        thresholds.to_csv(destination / f"{name}_validation_thresholds.csv", index=False)
        frame = validation.copy()
        frame["p_ai"] = probabilities
        frame.to_csv(destination / f"{name}_validation_predictions.csv", index=False)
        chosen[name] = best
        del model
        tf.keras.backend.clear_session()
    # Champion selection also uses validation data only.
    champion = min(chosen, key=lambda name: (chosen[name]["weighted_error"], -chosen[name]["roc_auc"]))
    write_json(destination / "locked_protocol.json", {
        "git_commit": git_commit(), "positive_class": "AI", "false_positive_cost": 2,
        "false_negative_cost": 1, "threshold_source": "validation only",
        "champion": champion, "validation_metrics": chosen,
        "test_predictions_per_model": 1,
        "rationale": "Wrongly accusing a real image of being AI can harm trust; cost ratio is a product assumption.",
    })
    results, predictions = [], {}
    test_ds = make_dataset(test)
    for name, path in paths.items():
        model = tf.keras.models.load_model(OUT / path)
        probabilities = model.predict(test_ds, verbose=0).ravel().astype(np.float64)
        threshold = chosen[name]["threshold"]
        result = {"model": name, **metric_values(test.label.to_numpy(), probabilities, threshold)}
        results.append(result)
        frame = test.copy()
        frame["p_ai"] = probabilities
        frame["predicted_label"] = (probabilities >= threshold).astype(int)
        frame["confidence"] = np.where(frame.predicted_label == 1, probabilities, 1 - probabilities)
        frame.to_csv(destination / f"{name}_test_predictions.csv", index=False)
        predictions[name] = frame
        save_error_grid(frame, name)
        del model
        tf.keras.backend.clear_session()
    results = pd.DataFrame(results)
    results.to_csv(destination / "metrics.csv", index=False)
    save_evaluation_figures(predictions, results)
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in paths:
        thresholds = pd.read_csv(destination / f"{name}_validation_thresholds.csv")
        ax.plot(thresholds.threshold, thresholds.weighted_error / len(validation), label=name)
        ax.axvline(chosen[name]["threshold"], linestyle="--", alpha=.5)
    ax.set(xlabel="AI probability threshold", ylabel="Validation cost per image: (2 FP + FN) / N",
           title="Threshold choice uses validation data only")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "figures/validation_thresholds.png", dpi=170)
    plt.close(fig)
    # Plot the selected tuner trial's history, without additional training or test calls.
    if selection["transfer_candidate"] == "tuned":
        hp = json.loads((OUT / "tuning/vision_best.json").read_text())
        key = f"lr_{hp['learning_rate']}_drop_{hp['dropout']}_layers_{hp['unfrozen_layers']}"
        save_curves(pd.read_csv(OUT / "histories" / f"trial_{key}.csv"), "transfer_tuned")
    print(results.to_string(index=False))
    print("Validation-selected champion:", champion)
    print("Raw sigmoid confidence has not been calibrated.")


if __name__ == "__main__":
    main()
