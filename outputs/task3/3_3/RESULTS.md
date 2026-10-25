# Task 3.3 results

Both remote runs are FINISHED. Each has 20 points for loss, accuracy, AUC,
validation loss, validation accuracy and validation AUC. Final test metrics and
Git/DVC/seed tags were checked directly against the hosted MLflow server.

| Logging | Test accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Manual | 63.27% | 0.6134 | 0.9012 | 0.7300 | 0.7495 |
| Autolog | 63.27% | 0.6134 | 0.9012 | 0.7300 | 0.7495 |

These runs compare logging methods using the same baseline CNN, initialization,
splits, augmentation and fixed threshold 0.5. Their identical results are expected.
They are fresh instrumentation runs, not the tuned Task 2 champion.

- [Manual run](https://dagshub.com/ammarbinmudassar/MLOPS-Assignment2.mlflow/#/experiments/0/runs/a479985fb89442e485a12d4f0f7e80f4)
- [Autolog run](https://dagshub.com/ammarbinmudassar/MLOPS-Assignment2.mlflow/#/experiments/0/runs/df4e67511af74ed6ac35cb357f65ae69)

Training code commit: `f681e5df099b6c4712d3fa64856c40c17a5767a1`.
DVC dataset version: `b2cec67839f742df4653f66d4bcbbafb.dir`.
Seed: 42. Data and DVC cache remain local; no DVC remote upload was performed.

Saved files: `manual/` and `autolog/` each contain history, training curves,
test predictions, results and an ignored local Keras checkpoint. `verification.json`
records the remote fields checked. Existing unit tests: 11 passed.

## Short explanation

Manual logging records the values chosen in my code. Autologging records epoch
metrics, optimizer settings and training details automatically. I added test
metrics and Git/DVC/seed tags explicitly to both methods.

Autologging does not infer the project's data version or custom test metrics.
Manual logging misses anything I forget to add.

## Screenshots to submit

For BOTH linked runs, capture parameters and final test metrics, the training
and validation metric curves, and tags showing Git commit, DVC data version and
seed. Use extra screenshots if the fields do not fit on one screen.
