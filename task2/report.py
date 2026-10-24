"""Build a measured results note from saved experiment evidence."""

import hashlib
import json
from pathlib import Path

from task2.common import OUT, write_json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def markdown_table(frame):
    def cell(value):
        return str(value).replace("|", "/").replace("\n", " ")
    lines = ["| " + " | ".join(map(cell, frame.columns)) + " |",
             "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines += ["| " + " | ".join(map(cell, row)) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def save_table(frame, name, title):
    display = frame.copy()
    for column in display.select_dtypes("floating"):
        display[column] = display[column].map(lambda value: f"{value:.5f}")
    fig, ax = plt.subplots(figsize=(max(8, len(frame.columns) * 1.5), max(2, len(frame) * .42 + 1.2)))
    ax.axis("off")
    table = ax.table(cellText=display.values, colLabels=display.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.7)
    ax.set_title(title, pad=18)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / f"{name}.png", dpi=170)
    plt.close(fig)


def main():
    metrics = pd.read_csv(OUT / "evaluation/metrics.csv")
    protocol = json.loads((OUT / "evaluation/locked_protocol.json").read_text())
    selection = json.loads((OUT / "selection.json").read_text())
    config = json.loads((OUT / "run_config.json").read_text())
    counts = pd.read_csv(OUT / "splits/counts.csv")
    toy = pd.read_csv(OUT / "tuning/fashion_trials.csv")
    vision = pd.read_csv(OUT / "tuning/vision_trials.csv")
    effects = json.loads((OUT / "tuning/parameter_effects.json").read_text())
    strongest = max(effects, key=lambda value: value["range_of_mean_val_loss"])
    model_manifest = []
    for name, relative in [("baseline", selection["baseline_path"]), ("transfer", selection["transfer_path"])]:
        path = OUT / relative
        model_manifest.append({"approach": name, "path": relative,
                               "bytes": path.stat().st_size,
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    write_json(OUT / "model_manifest.json", model_manifest)
    display_metrics = metrics[["model", "threshold", "accuracy", "precision", "recall", "f1", "roc_auc", "fp", "fn"]].round(5)
    save_table(display_metrics, "test_metrics_table", "Task 2.5 — test metrics (AI is positive)")
    save_table(toy.drop(columns=["status", "best_step_zero_based"]), "fashion_trials_table", "Task 2.7 — Fashion-MNIST toy tuning")
    save_table(vision.drop(columns=["status", "best_step_zero_based"]), "vision_trials_table", "Task 2.7 — all eight VisionGuard trials")
    stages = []
    for name in ["baseline", "transfer_frozen", "transfer_last_block"]:
        result = json.loads((OUT / "histories" / f"{name}.json").read_text())
        stages.append({"stage": name, **result})
    trainables = []
    for name in ["transfer_frozen", "transfer_last_block", "transfer_tuned"]:
        frame = pd.read_csv(OUT / "summaries" / f"{name}_trainable_layers.csv")
        weighted = frame.loc[frame.trainable_parameters > 0]
        trainables.append(f"- **{name}:** " + ", ".join(weighted.layer) + ".")
    explanations = []
    for name in ["baseline", "transfer"]:
        errors = pd.read_csv(OUT / "evaluation" / f"{name}_errors.csv")
        error_table = errors[["figure_id", "path", "class", "predicted_label", "confidence", "hypothesis"]].copy()
        error_table["predicted_label"] = error_table.predicted_label.map({0: "Real", 1: "AI"})
        error_table["confidence"] = error_table.confidence.map(lambda value: f"{value:.1%}")
        explanations.append(f"### {name}: {len(errors)} errors\n\n![{name} error examples](figures/{name}_errors.png)\n\n" + markdown_table(error_table))
    validation_rows = pd.DataFrame([
        {"model": name, **{key: value[key] for key in ["threshold", "accuracy", "f1", "roc_auc", "weighted_error"]}}
        for name, value in protocol["validation_metrics"].items()
    ]).round(5)
    text = f"""# Task 2 — measured results

Project: `23L-0630_A2_Ops`. Started: {config['started_at']}.
TensorFlow {config['tensorflow']}; Keras {config['keras']}; Keras Tuner {config['keras_tuner']}.
Initial training source commit: `{config['git_commit']}`. Evaluation source commit:
`{protocol['git_commit']}`. See `resume_config.json` for the resumed-stage revision.

These are actual local experiment outputs, not MLflow/DagsHub evidence. Full
methods, layer justifications and screenshot instructions are in
`../../docs/TASK2_GUIDE.md`. Raw image provenance has not been independently verified.

## 2.1 — fixed splits and augmentation

One extra exact duplicate was excluded without deleting original files. The split
uses seed 42 and has no overlapping exact content hashes.

{markdown_table(counts)}

![Augmented training batch](figures/augmented_batch.png)

Figure: 160 × 160 RGB training samples after padding and random augmentation.
Validation and test images receive no random transformations. See the guide for
the decoding, caching, batching and prefetching explanation.

## 2.2 — baseline architecture

![CNN layer shapes and parameter counts](figures/baseline_architecture.png)

Figure: three convolution/normalization/ReLU/pooling blocks followed by global
average pooling, a 32-unit dense layer, dropout and one sigmoid output. Exact
layer counts are also in `summaries/baseline_layers.csv` and `summaries/baseline.txt`.

## 2.3 and 2.4 — training stages and best epochs

{markdown_table(pd.DataFrame(stages).round(5))}

![Baseline curves](figures/baseline_curves.png)
![Frozen-head curves](figures/transfer_frozen_curves.png)
![Last-block fine-tuning curves](figures/transfer_last_block_curves.png)

Figures: training and validation loss/accuracy for each stage. Dashed lines mark
minimum validation-loss epochs. Checkpoints restore those epochs, not necessarily
the final epoch. Accuracy during training uses threshold 0.5; the final evaluation
uses the validation-selected thresholds below.

Layers with trainable parameters (the full lists also include frozen layers):

{chr(10).join(trainables)}

Transfer candidate selected before testing: **{selection['transfer_candidate']}**.
Candidate minimum validation losses: `{selection['candidate_validation_losses']}`.

## 2.5 — threshold selection and one held-out evaluation

Validation results used for decisions:

{markdown_table(validation_rows)}

AI is positive. Thresholds minimize `2 FP + FN` on validation images; a false
accusation of AI is assigned twice the cost of missing an AI image. This is a
declared product choice, not a measured harm ratio. Ties prefer higher F1 and then
a threshold nearer 0.5. **{protocol['champion']}** was chosen using validation cost
and validation ROC-AUC, before looking at test results.

Test metrics on 147 previously held-out images:

{markdown_table(display_metrics)}

![Validation threshold costs](figures/validation_thresholds.png)
![Test confusion matrices](figures/confusion_matrices.png)
![Test ROC curves](figures/roc_curves.png)

Figures: validation-only threshold costs, test confusion matrices (Real then AI),
and test ROC curves. All test measures come from one saved prediction pass per
final model. Precision/recall/F1 refer to the AI class. Scores are uncalibrated;
neither model should be presented as reliable proof of an image's origin.

## 2.6 — error analysis

The following are the most confident wrong test predictions. Their selection
highlights failure modes rather than estimating their prevalence. Each explanation
is a hypothesis from visual inspection; it does not establish the model's causal
reason for predicting that label.

{chr(10).join(explanations)}

## 2.7 — Keras Tuner

Toy search: four combinations, three epochs maximum per trial, 10,000 training and
2,000 validation Fashion-MNIST images. Best settings:
`{json.loads((OUT / 'tuning/fashion_best.json').read_text())}`.

{markdown_table(toy.round(6))}

VisionGuard: all eight combinations, up to five epochs each, selecting minimum
validation loss. Best settings:
`{json.loads((OUT / 'tuning/vision_best.json').read_text())}`.

{markdown_table(vision.round(6))}

![Eight trial comparison](figures/tuner_trials.png)

Figure: minimum validation loss in each VisionGuard trial. The largest range in
mean validation loss across tested settings was for **{strongest['parameter']}**
({strongest['range_of_mean_val_loss']:.5f}). Group means:
`{strongest['group_means']}`. This is descriptive evidence from one small grid,
not a causal importance ranking; parameter interactions and augmentation noise remain.

Repeated model, epoch and threshold choices can overfit the same validation set.
The locked test set provides a final comparison; its results must not be used to
choose another threshold or configuration. No claim of universal performance on
new image generators or real-world image sources is supported by this split.

## Troubleshooting and remaining work

The original CNN run completed, then the official weight download failed with a
DNS error. Downloading the official weights and resuming preserved the completed
CNN rather than training it again. Logs for both attempts are in `logs/`.
The first three tuner attempts failed before training because TensorBoard was
missing. Its dependency was installed, failed tuner state was retained separately,
and a fresh eight-trial search was run. Failed attempts are not counted as trials.

Models are saved locally under `models/` and Git-ignored. `model_manifest.json`
records selected paths, sizes and SHA-256 hashes. DVC publication, MLflow/DagsHub
tracking, a serving application and the optional Grad-CAM bonus are not included.
Capture actual terminal screenshots with username, date and project context using
`python -m task2.show_results`; the PNG plots are figures, not terminal screenshots.

AI assistance was used to implement, test, execute and document this Task 2 work.
"""
    (OUT / "RESULTS.md").write_text(text)
    print("Saved measured results:", OUT / "RESULTS.md")


if __name__ == "__main__":
    main()
