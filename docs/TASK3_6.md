# Task 3.6 — model registry

Registered model: `23L-0630_VisionGuard`.

The setup script registers the existing `unfreeze_4` candidate and the
validation-selected `unfreeze_8` winner as two versions. The `champion` alias
points to the winner. Re-running setup reuses matching source-run versions.

```bash
.venv/bin/python -m task3.register_models
bash run_gpu.sh python -m task3.predict_champion
```

The prediction script loads `models:/23L-0630_VisionGuard@champion` and classifies
the first saved test image. To classify another image, append its quoted path.
It uses the exact training decoder and letterbox, with model-internal scaling
and the registered 0.5 threshold.

## Evidence

In DagsHub's MLflow interface, open **Models**, then `23L-0630_VisionGuard`.
Capture the two versions and `champion` alias. For terminal evidence, run:

```bash
cat outputs/task3/3_6/registration.txt
cat outputs/task3/3_6/prediction.txt
```

## Short explanation

I registered two trained model versions and assigned `champion` to the
validation-selected winner. A separate script loads the model by this alias
and classifies one image.

For deployment, resolve `champion` once during build or deployment and download
that exact version into the server's local model directory. Record its version
and file hash, then load it once at application startup. Requests use the loaded
local model, so they do not contact DagsHub for each prediction.

Official alias reference: https://mlflow.org/docs/latest/ml/model-registry/workflow/

The saved `registry.json` and `prediction.json` provide measured completion
evidence; file creation alone does not prove a successful registry or prediction.
