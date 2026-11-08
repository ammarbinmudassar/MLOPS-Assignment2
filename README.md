# 23L-0630_A2_Ops — VisionGuard

[![CI](https://github.com/ammarbinmudassar/MLOPS-Assignment2/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ammarbinmudassar/MLOPS-Assignment2/actions/workflows/ci.yml)

TensorFlow/Keras AI-versus-real image classification coursework.

- [GitHub](https://github.com/ammarbinmudassar/MLOPS-Assignment2)
- [DagsHub](https://dagshub.com/ammarbinmudassar/MLOPS-Assignment2)
- [MLflow](https://dagshub.com/ammarbinmudassar/MLOPS-Assignment2.mlflow)
- [Hosted API](https://visionguard-a2-api.vercel.app)

## Source and evidence

| Path | Purpose |
|---|---|
| `dataset_audit.py`, `visual_eda.py` | Dataset audit and plots |
| `task2/` | Input pipeline, CNN, transfer learning, tuning and evaluation |
| `task3/` | MLflow lab, DagsHub tracking, artifacts, comparisons and registry |
| `task4/`, `visionguard/` | Flask UI, validation, inference and history storage |
| `task7/`, `deploy/vercel_api/` | Model conversion, validation and hosted LiteRT API |
| `tests/`, `.github/workflows/` | Tests and CI |
| `outputs/` | Saved results, figures, screenshots and run evidence |
| `postman/` | API request collection |

## Training and MLflow

Use Python 3.11 in the local training environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python dataset_audit.py
python visual_eda.py
bash run_gpu.sh python -m task2.train
bash run_gpu.sh python -m task2.evaluate
python -m task2.report
bash scripts/start_mlflow.sh
```

Open http://127.0.0.1:5000 for local MLflow. Run `python -m task3.local_lab`
in another terminal, or add `--dagshub` for hosted tracking. Copy `.env.example`
to the ignored `.env` and enter DagsHub credentials privately.
`run_gpu.sh` exposes NVIDIA libraries from the local environment; CPU training
can use Python directly. Test evaluation refuses to overwrite completed results.

## Web application

For local model inference, prepare the champion once and start the UI:

```bash
python -m task4.prepare_model
python -m task4.run --server gunicorn
```

Copy `.env.web.example` to `.env.web` first. The default UI is
http://127.0.0.1:8000. The local model and dataset are excluded from Git.

For the existing hosted API, the ignored `.env.remote` is already configured:

```bash
VISIONGUARD_WEB_ENV=.env.remote python -m task4.run
```

Open http://127.0.0.1:8001. A fresh remote configuration needs
`INFERENCE_API_URL`, `INFERENCE_API_KEY`, `MAX_UPLOAD_BYTES=3145728` and
`APP_PORT=8001`. Results are stored in Neon PostgreSQL. Images are processed
in memory; confidence is an uncalibrated model score.

## Docker and deployment

```bash
docker compose up -d --build --wait
npx vercel deploy --prod --cwd deploy/vercel_api
```

Compose uses the private PostgreSQL settings in `.env` and the prepared champion
model; its UI is http://127.0.0.1:8083. `compose.failure.yaml` is the intentional
failure exercise. Vercel uses Python 3.12 and the bundled 4.52 MB float16 model.
Its production settings require private `DATABASE_URL` and `VISIONGUARD_API_KEY`
variables, configured with `vercel env add` before deployment.

Import `postman/VisionGuard.postman_collection.json` for API checks. `/health`
is public. `/predict` accepts a multipart `image` file up to 3 MiB;
`/history` and `/results/<id>` return saved metadata. These three routes require
`X-API-Key`. Never commit keys or database passwords.

## Checks

```bash
ruff check .
python -m unittest discover -s tests -v
```

AI assistance was used for implementation, testing and documentation of Tasks 2–8.
