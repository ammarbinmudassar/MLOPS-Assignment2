# Task 3.1 — local MLflow

Two tiny Keras models use the same Fashion-MNIST split, initial seed, architecture,
batch size and five-epoch budget. Only the learning rate changes: 0.001 vs 0.0001.
Each run records parameters, per-epoch loss/accuracy, validation loss/accuracy,
a training-curve plot, a history CSV, tags and the trained MLflow Keras model.
The model is loaded back from MLflow and its predictions checked against the original.
The input example is logged separately because the installed MLflow Keras flavor
has a directory-creation error when `input_example` is passed to `log_model`.
The first failed attempt is retained in MLflow; compare the two FINISHED runs.

## Open the local interface

From the assignment folder, open a terminal and run:

```bash
bash scripts/start_mlflow.sh
```

Leave that terminal running. Open **http://127.0.0.1:5000** in your browser.
Choose **23L-0630_A2_Ops**. Select the two `3.1_fashion_lr_...` runs and click
**Compare**. Compare `learning_rate`, `val_loss` and `val_accuracy`, then open
each run to inspect its metric curves, artifacts and logged model.

The command starts a local web server; opening a browser alone does not start it.
Press Ctrl+C in that terminal to stop it. Runs remain saved in `.mlflow/mlflow.db`
and `.mlflow/artifacts`, so restarting the same command shows them again.
This interface is local to your computer; it is not a deployed URL or DagsHub.

## Run the toy exercise again, only if needed

Keep the MLflow server running in the first terminal. In a second terminal:

```bash
bash run_gpu.sh python -u -m task3.local_lab
```

Each execution creates two NEW runs; it does not replace the existing runs.

## Simple report explanation

I trained the same small Fashion-MNIST network twice, changing only learning rate.
MLflow recorded the settings, per-epoch metrics, training plots and models. Its
comparison view lets me check how this change affected validation performance.

| Term | Meaning |
|---|---|
| Experiment | A group of related runs. |
| Run | One recorded training execution. |
| Parameter | A chosen setting, such as learning rate. |
| Metric | A measured result, such as validation accuracy. |
| Artifact | A saved file, such as a plot or model. |
| Tag | A descriptive label, such as task or code commit. |

## Evidence to submit

1. Screenshot of each run showing its learning rate and metrics.
2. Screenshot of both runs in the comparison view.
3. Screenshot showing the plot/history artifacts and logged model.
4. The short explanation and six definitions above.

Show the address bar and experiment/run name in screenshots. All artifacts must
belong to the actual runs; the locally saved PNGs are also in `outputs/task3/3_1/`.

The completed results, run IDs and saved interface screenshots are listed in
`outputs/task3/3_1/RESULTS.md`. Final validation accuracies were 84.25% at 0.001
and 78.60% at 0.0001, using the same five-epoch budget.
