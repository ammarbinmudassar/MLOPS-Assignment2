# Task 3.1 — completed local MLflow lab

Experiment: **23L-0630_A2_Ops**. MLflow version: **3.17.2**.

I trained the same small Fashion-MNIST network twice, changing only its learning
rate. Both used seed 42, 64 hidden units, batches of 64, five epochs, 10,000 training
images and 2,000 validation images. MLflow recorded parameters, per-epoch metrics,
plots, model summaries and trained models.

| Learning rate | Final validation accuracy | Final validation loss |
|---|---|---|
| 0.001 | 84.25% | 0.4531 |
| 0.0001 | 78.60% | 0.6533 |

The larger learning rate performed better within this five-epoch budget; this does
not establish that it is always the better setting.

Both runs are FINISHED. All four metrics have five epoch points per run; plot and
history artifacts exist. Both logged models are READY and were loaded back with
matching predictions. See `verification.json`, `runs.json` and `comparison.csv`.
Code revision: `dd8dbce3827939bdab2c712f49d190382b46ccdc`.

The first attempt failed while logging an input example through MLflow's Keras
flavor. It remains visible as a FAILED run. The fixed code saves that example as a
separate artifact; use the two FINISHED runs for the comparison.

## Screenshot mapping

| Evidence | File |
|---|---|
| First run, learning rate 0.001 | `screenshots/run_lr_0_001.jpg` |
| Second run, learning rate 0.0001 | `screenshots/run_lr_0_0001.jpg` |
| Both runs compared | `screenshots/comparison.jpg` |
| Per-epoch metric curves | `screenshots/epoch_metrics.jpg` |
| Logged curve artifact | `screenshots/artifacts.jpg` |
| Saved model | `screenshots/logged_model.jpg` |

Open http://127.0.0.1:5000. To start the interface again from the assignment folder,
run `bash scripts/start_mlflow.sh` and keep that terminal running. Definitions of
experiment, run, parameter, metric, artifact and tag are in `docs/TASK3_1.md`.
