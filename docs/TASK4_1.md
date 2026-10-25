# Task 4.1 — local inference

`task4/inference.py` provides `VisionGuardPredictor`. Create one instance at app
startup and reuse its `predict_bytes(uploaded_bytes)` method. It returns a label,
confidence, AI probability, threshold and model version. Uploaded bytes stay in
memory. The module contains no DagsHub or MLflow calls.

`task4/prepare_model.py` downloads an immutable champion version once into the
ignored `models/champion/` directory. Startup checks its SHA-256 and metadata.
The prepared package is version 2 of `23L-0630_VisionGuard`.

```bash
# Already prepared; needed again only when downloading a new champion:
.venv/bin/python -m task4.prepare_model

# Reuse one local model for the first AI and first Real test images:
.venv/bin/python -m task4.demo_inference
```

Training and inference share `visionguard/preprocessing.py`: RGB decoding,
160x160 letterboxing with antialiasing, uint8 quantization and float32 input.
Inference applies no augmentation. Scaling to [-1,1] remains inside the model.

## Evidence

Screenshot this terminal output:

```bash
cat outputs/task4/4_1/console.txt
```

The AI sample was classified as AI (55.34%). The Real sample was wrongly
classified as AI (86.23%). These are the first images of each class in the saved
test split, without selecting examples based on their predictions. Both decisions
agree with the saved Task 3.5 test predictions. CPU/GPU scores differ slightly.

The demo completed with network connections blocked and verified one model
load for two predictions. All 13 existing and new unit tests passed.

## Short explanation

The model loads once at startup because loading it for every request adds delay
and memory overhead. Training and inference use the same preprocessing so the
model receives images in the format it learned.
