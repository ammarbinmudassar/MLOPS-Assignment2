# Task 3.3 — VisionGuard training instrumentation

Code: `task3/vision_training.py`

Run from the assignment folder:

```bash
bash run_gpu.sh python -u -m task3.vision_training
```

Two fresh baseline CNN runs use the saved Task 2 splits, seed 42, image size 160,
batch size 32 and initial learning rate 0.001. Each runs for up to 20 epochs with
early stopping on validation loss. Logging method is the only intended change.
Test metrics use a fixed 0.5 decision threshold and do not select the model.

Both methods explicitly record the Git commit, DVC dataset hash, seed,
augmentation, actual dataset batch size and final test metrics. The DVC hash
identifies all 975 original images; the split hash identifies the 974 unique
images used after excluding the extra duplicate. Image hashes are checked
against the saved audit before training.

Manual logging uses our own callback. TensorFlow autologging injects its own
callback; it also records optimizer configuration, fit settings, early-stopping
information, a model summary and TensorBoard logs. Its batch-size field may be
`None` for this prefetched dataset, so `train_batch_size=32` is recorded explicitly.
Model upload and autolog checkpoints are disabled for this subsection; trained
Keras checkpoints are saved locally for Task 3.4.

## Evidence

Open https://dagshub.com/ammarbinmudassar/MLOPS-Assignment2 and its **Experiments**
page. In `23L-0630_A2_Ops`, find:

- `3.3_visionguard_manual`
- `3.3_visionguard_autolog`

For each run, capture parameters and final test metrics, training/validation
metric curves, and tags showing `git_commit`, `dvc_data_version` and `seed`.
If they do not fit on one screen, use separate screenshots.

## Short report explanation

Manual logging records the values chosen in my code. Autologging records epoch
metrics, optimizer settings and training details automatically. I added test
metrics and Git/DVC/seed tags explicitly to both methods.

Autologging does not infer the project's data version or custom test metrics.
Manual logging misses anything I forget to add.
