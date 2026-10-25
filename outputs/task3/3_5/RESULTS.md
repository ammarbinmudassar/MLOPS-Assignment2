# Task 3.5 — comparison results

Six FINISHED runs used the same split, seed, architecture and maximum training budget. Each variant changed one reference setting.

| Run | Validation loss | ROC-AUC | Accuracy | F1 |
|---|---:|---:|---:|---:|
| unfreeze_8 | 0.4473 | 0.8756 | 82.88% | 0.8428 |
| unfreeze_4 | 0.4844 | 0.8526 | 78.77% | 0.8166 |
| no_augmentation | 0.5379 | 0.8049 | 71.23% | 0.7375 |
| reference | 0.5391 | 0.7962 | 70.55% | 0.7456 |
| higher_dropout | 0.5437 | 0.7903 | 70.55% | 0.7394 |
| lower_lr | 0.5445 | 0.7896 | 70.55% | 0.7456 |

## Short justification

I selected unfreeze_8 because it had the lowest validation loss (0.4473), supported by the highest validation ROC-AUC (0.8756). Selection used validation data; the test set was used only afterward.

Final test accuracy: 79.59%; F1: 0.8148; ROC-AUC: 0.8648. The decision threshold stayed at 0.5.

## Evidence

Use comparison.png and capture the DagsHub comparison view of all six FINISHED 3.5 runs. Show the changed settings and selection_val metrics. One interrupted attempt remains KILLED and is excluded.

- [unfreeze_8](https://dagshub.com/ammarbinmudassar/MLOPS-Assignment2.mlflow/#/experiments/0/runs/36cdf49361924598b55143251cc26845)
- [unfreeze_4](https://dagshub.com/ammarbinmudassar/MLOPS-Assignment2.mlflow/#/experiments/0/runs/93f7b0553b60447085968c9822b038b5)
- [no_augmentation](https://dagshub.com/ammarbinmudassar/MLOPS-Assignment2.mlflow/#/experiments/0/runs/7951c99fecf04e08969ec480f70dcbad)
- [reference](https://dagshub.com/ammarbinmudassar/MLOPS-Assignment2.mlflow/#/experiments/0/runs/86129a92bc2844e58d3d9ced0830d433)
- [higher_dropout](https://dagshub.com/ammarbinmudassar/MLOPS-Assignment2.mlflow/#/experiments/0/runs/decb7dbf0da74c08b528d0cf37a195b9)
- [lower_lr](https://dagshub.com/ammarbinmudassar/MLOPS-Assignment2.mlflow/#/experiments/0/runs/ef767d2d4d754505962a043640232c72)

All six models and comparison files are saved remotely. The selected model was downloaded and its predictions verified. Common head-training histories match across reference, lower_lr, unfreeze_4 and unfreeze_8.
