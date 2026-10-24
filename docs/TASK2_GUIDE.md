# Task 2 — model development

Project: `23L-0630_A2_Ops`.

This guide explains the implementation. Measured outcomes are saved separately in
`outputs/task2/RESULTS.md` after the experiments finish. Use the explanations to
understand the choices and adapt them into your own report; disclose AI assistance
as the assignment requires. None of these local outputs is a DagsHub/MLflow run.
Task 3 still requires its own tracking instrumentation and hosted evidence.

## Files and commands

| File | Responsibility |
|---|---|
| `task2/data.py` | Duplicate exclusion, split lists, decoding, padding and augmentation |
| `task2/models.py` | CNN, MobileNetV2, preprocessing and trainable-layer policies |
| `task2/train.py` | Training, callbacks, curves, summaries and both tuning exercises |
| `task2/evaluate.py` | Validation thresholds, final test metrics, ROC, matrices and error grids |
| `task2/show_results.py` | Display saved evidence without training or evaluating again |
| `tests/test_task2.py` | Focused checks for leakage, preprocessing and threshold policy |

Run from the assignment folder with its `.venv` installed:

```bash
bash run_gpu.sh python -u -m task2.train
# Only after an interrupted run, with the SAME epoch budgets:
bash run_gpu.sh python -u -m task2.train --resume
# Once training and tuning have finished:
bash run_gpu.sh python -u -m task2.evaluate
# Display previously saved evidence, with no new predictions:
.venv/bin/python -m task2.show_results
# Verify implementation checks:
.venv/bin/python -m unittest discover -s tests -v
```

The default training budget is 20 CNN epochs, 12 frozen-head epochs, 12 last-block
epochs, five epochs per VisionGuard tuning trial and three per toy trial. Early
stopping can shorten a stage. These are small coursework experiments, not an
exhaustive benchmark. Model files and tuner checkpoints are local and Git-ignored;
use DVC to publish the chosen model in the later versioning/tracking work.

## 2.1 — input pipeline

The audit contains 975 readable files. One extra exact duplicate is excluded from
the split manifests while the original dataset remains unchanged. The remaining
974 files are stratified into 681 training, 146 validation and 147 test images.
`Real = 0` and `AI = 1`. A check rejects overlapping SHA-256 hashes across splits.
Near-duplicates and images from the same web source can remain; this is a limitation
of the available provenance and the exact-duplicate check.

`tf.data` starts from explicit paths and labels in each CSV manifest. It reads file
bytes, decodes three RGB channels, resizes while preserving aspect ratio, and adds
black padding to produce 160 × 160 images. This avoids stretching faces and objects,
but padding and downsampling may introduce shortcuts or remove fine synthesis cues.
The chosen size reduces computation on the 6 GB laptop GPU and has official
MobileNetV2 pretrained weights. Image size is not itself tuned in this task.

Decoded/resized images are cached in memory as uint8 so epochs need not repeatedly
read and decode large originals. Only training data is shuffled, using seed 42, then
batched in groups of 32. Training batches receive horizontal flips, rotations up to
0.04 of a full turn, 10% zoom and 10% contrast changes. Values are clipped to [0,255].
Validation and test inputs receive no random transformations. One batch is prefetched
to overlap input preparation with model computation; CPU threads are bounded for the laptop.

**Figure caption:** An augmented training batch after aspect-preserving resize and
padding. These transformations vary orientation, scale and contrast without changing
the AI/Real labels; their effect on generalization remains an empirical question.

**Evidence:** `splits/*.csv`, `splits/counts.csv`, `figures/augmented_batch.png`.

## 2.2 — baseline CNN

The CNN starts with a rescaling layer that maps [0,255] pixels to [0,1]. Three 3 × 3
convolution blocks learn 16, 32 and 64 feature maps. Each convolution uses no bias
because a following BatchNormalization layer already has a learned offset. Batch
normalization stabilizes activations, ReLU supplies nonlinearity, and 2 × 2 max
pooling reduces each spatial dimension by half.

Global average pooling summarizes each feature map with one value. Compared with
flattening all spatial positions, this keeps the classifier small. A 32-unit ReLU
dense layer combines the features, dropout removes 40% of its activations during
training to discourage reliance on a few units, and one sigmoid output represents
the AI score. This score has not been calibrated as a probability of correctness.

Binary cross-entropy matches a two-class sigmoid output. Adam starts at 0.001 and
adapts updates per parameter. Training logs accuracy at the default 0.5 threshold
and an approximate Keras ROC-AUC; final ROC-AUC is calculated with scikit-learn from
saved predictions. Accuracy alone does not show the kinds of errors the model makes.

**Evidence:** `summaries/baseline.txt`, `summaries/baseline_layers.csv`,
`figures/baseline_architecture.png`. Capture the actual saved model-summary text in
your terminal for the assignment's requested summary screenshot.

## 2.3 — transfer learning

MobileNetV2 is loaded with ImageNet weights and without its original classifier.
A rescaling layer implements its required preprocessing exactly: `x / 127.5 - 1`,
which maps [0,255] to [-1,1]. Using [0,1] instead would change the pretrained model's
input distribution. Keeping rescaling inside the saved model helps later inference
use the same transformation, provided serving still performs identical RGB decoding,
resize/padding and pixel-range handling.

The replacement head uses global average pooling, dropout 0.3 and one sigmoid unit.
First the entire backbone is frozen and only the head is trained at learning rate
0.001. The best frozen-stage checkpoint initializes stage two. Only the final
inverted residual block (`block_16_*`) and final `Conv_1` projection are unfrozen;
earlier feature extractors stay fixed. This stage uses learning rate 0.00001, 100
times smaller, to limit destructive changes to pretrained features.

Backbone BatchNormalization layers remain frozen and the backbone is called with
`training=False`, retaining its moving statistics on this small dataset. The model
is recompiled whenever trainability changes. Separate CSV files record the exact
layer names and trainable parameter counts for both stages.

**Figure captions:** Frozen-stage curves show classifier-head learning with the
ImageNet backbone fixed. Fine-tuning curves show subsequent adaptation of its final
block; dashed markers identify each stage's minimum validation-loss epoch.

**Evidence:** `figures/transfer_frozen_curves.png`,
`figures/transfer_last_block_curves.png`, `summaries/transfer_*_trainable_layers.csv`.

## 2.4 — training discipline

Python, NumPy and TensorFlow seeds are fixed at 42 and deterministic TensorFlow
operations are enabled. Runs also record software versions, GPU details, timestamps
and source commits. Exact replay is still sensitive to hardware, software and the
resume boundary; a resumed stage is not claimed to reproduce an uninterrupted
random-augmentation stream bit for bit.

EarlyStopping monitors validation loss with patience four and restores the best
weights. ModelCheckpoint independently saves the best model to disk. ReduceLROnPlateau
halves the learning rate after two unimproved validation-loss epochs, down to 1e-7.
These callbacks serve different roles: stop wasted training, preserve the best
state, and allow smaller updates when progress stalls. The checkpoint is explicitly
loaded after each stage. Curves mark the best epoch, numbered from one.

**Evidence:** `histories/*.csv`, `histories/*.json`, training curves and actual logs
in `logs/`. The measured results file gives the best epochs.

## 2.5 — evaluation and threshold choice

AI is the positive class. A false positive is a real image called AI-generated; a
false negative is an AI image called real. For this product, falsely accusing a
real image is assigned twice the cost because it can damage trust in authentic
work. This 2:1 ratio is an explicit product assumption, not an experimentally
established harm estimate. An alternative application could justify a different ratio.

Each model's threshold is chosen ONLY on validation predictions by minimizing
`2 × false positives + false negatives`. Ties prefer higher F1, then a threshold
closer to 0.5. The transfer candidate is chosen by validation loss among the frozen
head, final-block model and best tuner trial. The champion approach is selected by
validation cost, with validation ROC-AUC as a tie-breaker. These decisions are saved
before any test predictions are requested.

One prediction pass per final approach produces all test metrics and figures.
Accuracy is the fraction correct; precision is the fraction of AI accusations that
are correct; recall is the fraction of AI images detected; F1 balances precision and
recall. ROC-AUC measures ranking across thresholds and is not accuracy. Confusion
matrices display Real then AI for both rows and columns. Test predictions are saved
so figures can be inspected without rerunning inference. Evaluation refuses to
overwrite a completed result table, discouraging repeated tuning against the test set.

**Evidence:** `evaluation/metrics.csv`, `evaluation/locked_protocol.json`,
`evaluation/*_validation_thresholds.csv`, `figures/validation_thresholds.png`,
`figures/confusion_matrices.png`, `figures/roc_curves.png`.

## 2.6 — error analysis

Error grids show up to ten most confident wrong predictions for each final model,
using saved test predictions. Their IDs correspond to the error CSVs and the measured
results explanation. Selecting confident errors highlights failure modes; it does
not estimate how frequent each failure mode is. If fewer than ten errors exist, only
real errors are shown, never fabricated examples.

Inspect subjects, artistic style, source text/watermarks, resolution and framing.
An image can look like artwork while being in the Real folder; folder labels are
the assignment's ground truth, not independent verification of provenance. Visual
explanations are hypotheses, not demonstrated causes. Attribution or controlled
experiments would be needed to establish what caused each decision.

**Evidence:** `figures/*_errors.png`, `evaluation/*_errors.csv`, per-image hypotheses
in `outputs/task2/RESULTS.md` after visual review.

## 2.7 — Keras Tuner

The toy exercise uses a fixed, stratified 12,000-image Fashion-MNIST subset, with
10,000 for training and 2,000 for validation. A tiny dense network searches hidden
units {32,64} and learning rate {0.001,0.0003} using RandomSearch over all four
combinations. The official Fashion-MNIST test set is not evaluated for this exercise.

VisionGuard runs eight RandomSearch trials over learning rate {0.00001,0.0001},
dropout {0.2,0.5} and the last {4,8} backbone layers. BatchNormalization remains
frozen, so the requested number of tail layers differs from the count of layers
with trainable weights. Each trial starts with the same ImageNet backbone and
classifier weights from the best frozen-head checkpoint, and trains for up to five
epochs. All eight combinations are tried; trials use the same split, augmentation
policy and starting seed. A trial's objective is its best validation loss.

The strongest apparent setting is identified by comparing the range of mean
validation loss across each parameter's values in the balanced eight-trial table.
This small-grid effect is descriptive, includes interactions and training noise,
and does not establish universal hyperparameter importance. Check actual values in
`tuning/parameter_effects.json` before making a claim about which mattered most.

Repeatedly choosing configurations using the same validation set can fit its
particular examples. Eight trials, checkpoint selection and threshold selection
all consult that set, making the best validation score optimistic. A locked test
set provides a final independent comparison; nested validation or repeated splits
would give stronger evidence but are outside this small experiment.

**Evidence:** `tuning/fashion_trials.csv`, `tuning/fashion_best.json`,
`tuning/vision_trials.csv`, `tuning/vision_best.json`, tuner summaries and
`figures/tuner_trials.png`. Tuner state and model weights are kept locally but
excluded from Git because they are large generated artifacts.

## What went wrong

The first transfer-learning attempt failed to resolve the official Google Storage
weight-download host. The completed CNN checkpoint was preserved. The weights were
downloaded from the official URL after retrying, and the resume command continued
without retraining the CNN. Original and resumed logs are saved separately.

Keras Tuner initially failed before any trial training because TensorBoard was
absent. TensorFlow 2.21 did not install that dependency automatically. TensorBoard
was added explicitly, the failed state and log were retained, and a fresh search
was started. Failed attempts are not counted toward the eight completed trials.

The GPU is unavailable inside the restricted execution session, so training uses
the same unrestricted GPU access verified in the user's terminal. `run_gpu.sh`
sets the CUDA library search path required by this particular local environment.
An image also emits a malformed ICC-profile warning; it remains decodable, so this
warning is not counted as corruption in the audit.

## Screenshot checklist

Keep address-bar or terminal-prompt context, username, date and project name visible.
Generated PNG figures are figures, not fabricated terminal screenshots.

| ID | Capture or include |
|---|---|
| 2.1 | Augmented batch and split counts; explain the pipeline |
| 2.2 | Terminal model summary plus the layer-shape/parameter table |
| 2.3 | Frozen and final-block training curves; exact trainable-layer lists |
| 2.4 | Curves with best epochs marked and callback explanation |
| 2.5 | Both-model metric table, ROC curves, confusion matrices and threshold rationale |
| 2.6 | Ten real error examples and a hypothesis for each |
| 2.7 | Toy and eight-trial results, best settings and discussion |

## References

- TensorFlow transfer learning: https://www.tensorflow.org/tutorials/images/transfer_learning
- MobileNetV2 and preprocessing: https://keras.io/api/applications/mobilenet/
- Keras Tuner: https://keras.io/keras_tuner/

Optional Grad-CAM bonus is not part of this implementation.
