# Task 3.4 — run artifacts

The attachment script is `task3/attach_artifacts.py`. It uses the saved Task 3.3
checkpoints and test predictions; it does not retrain or select another model.

Open either Task 3.3 run on DagsHub and select **Artifacts**:

- `evaluation/confusion_matrix.png`
- `evaluation/roc_curve.png`
- `splits/train.csv`, `validation.csv`, `test.csv`
- `environment/packages.txt`
- `model/MLmodel` and `model/data/model.keras`

Both the manual and autologging runs receive these artifacts. Additional split
metadata, test predictions and `dataset.dvc` are included for traceability.

**Evidence:** take a screenshot of the Artifacts tree with `evaluation`, `splits`,
`environment` and `model` expanded. Use a second screenshot if needed to show
the Keras file inside `model/data`.

**Optional caption:** I attached evaluation plots, split lists, the package list
and the trained model to each run.

Models are stored as MLflow Keras run artifacts, compatible with this hosted
server. The script downloads the required files, checks their hashes and reloads
each remote model to verify its predictions. Completion evidence is saved in
`outputs/task3/3_4/verification.json`.
