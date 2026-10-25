# Task 3.2 — DagsHub setup

Repository supplied by the user:
https://dagshub.com/ammarbinmudassar/MLOPS-Assignment2

Tracking endpoint:
https://dagshub.com/ammarbinmudassar/MLOPS-Assignment2.mlflow

`.env` contains the tracking URL, username and token. It is ignored by Git;
`.env.example` is a safe template with no token. The Python code reads environment
variables instead of storing credentials in source code.

## Private step: create and enter your token

1. Open https://dagshub.com/user/settings/tokens while signed in.
2. Create an access token for this assignment. Do not share it in chat or screenshots.
3. From the assignment folder, run:

```bash
.venv/bin/python -m task3.set_token
```

Paste the token at the hidden prompt. The helper saves it only in the ignored
`.env` file with owner-only file permissions. It refuses to save if `.env` is tracked.

## Check and repeat the toy lab remotely

```bash
.venv/bin/python -m task3.check_dagshub
bash run_gpu.sh python -u -m task3.local_lab --dagshub
```

The lab creates two fresh runs, tagged `task=3.2`, under `23L-0630_A2_Ops` on DagsHub.
It trains the same toy model, logging parameters, per-epoch metrics, plots and a model.
Models are stored as run artifacts for compatibility with hosted MLflow servers.
The saved model is loaded back and its predictions verified.

The authenticated connection and both remote runs completed successfully.
Learning rate `0.001` reached 84.25% validation accuracy; `0.0001` reached 78.60%.
Both saved models were downloaded, reloaded and checked against the original predictions.
Results are saved under `outputs/task3/3_2/`.

## Evidence for the report

- Screenshot of the DagsHub project showing its connected GitHub repository.
- Screenshot of a completed `3.2_fashion_lr_...` run with parameters and metrics.
- Screenshot showing its plot and model artifacts.
- Terminal screenshot of `git check-ignore -v .env`; never display `.env` contents.

**Short explanation:** I connected the project to DagsHub and used its hosted
MLflow server. The tracking URL and credentials are loaded from environment
variables through an ignored `.env` file, keeping the token out of source control.

Open the repository's **Experiments** page, select `23L-0630_A2_Ops`, and select
the two `3.2_fashion_lr_...` runs to compare them. The browser pane requires
its own DagsHub sign-in; screenshots must be taken while signed in.
