# 23L-0630_A2_Ops — VisionGuard

[![VisionGuard CI](https://github.com/ammarbinmudassar/MLOPS-Assignment2/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ammarbinmudassar/MLOPS-Assignment2/actions/workflows/ci.yml)

TensorFlow/Keras coursework for distinguishing the dataset's AI and Real image
classes. The original dataset is kept locally and excluded from Git.

Task 1 audit and EDA are in `dataset_audit.py`, `visual_eda.py` and `outputs/`.
Task 2 code is in `task2/`; its explanations, commands and screenshot checklist are
in [docs/TASK2_GUIDE.md](docs/TASK2_GUIDE.md). Generated results and figures live in
`outputs/task2/`. Large models, downloads and tuner state are Git-ignored.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash run_gpu.sh python -u -m task2.train
bash run_gpu.sh python -u -m task2.evaluate
python -m task2.show_results
```

The GPU wrapper uses Bash and exposes the NVIDIA libraries installed in this
environment. A compatible working NVIDIA driver is still required for GPU training.
The model code also supports CPU execution with `.venv/bin/python` directly.
Initial transfer training downloads official ImageNet weights; the toy lab uses
Fashion-MNIST. Internet access is needed unless these downloads are already cached.

Test data is reserved until tuning is complete. The evaluation command refuses to
overwrite completed test results. Use saved predictions for later analysis.

Task 3 MLflow/DagsHub tracking, comparisons and the model registry are implemented
in `task3/`. Subsection guides and evidence instructions are in `docs/TASK3_1.md`
through `docs/TASK3_6.md`; results are under `outputs/task3/`.
The registered model is `23L-0630_VisionGuard`, with `champion` pointing to version 2.
Run `bash run_gpu.sh python -m task3.predict_champion` to load it by alias and
classify the first saved test image. Credentials come from the Git-ignored `.env`.

Task 4.1 local inference is in `task4/`; see [docs/TASK4_1.md](docs/TASK4_1.md).
The local model package is ignored by Git. After preparation, inference works
offline and reuses one loaded model for multiple predictions.

Tasks 4.2–4.4 provide validated uploads, a simple frontend, persistent SQLite
history and environment-based settings. Run `.venv/bin/python -m task4.run
--server gunicorn` and open http://127.0.0.1:8000. Use `.env.web.example` as the
template for the ignored `.env.web` settings. Guides and screenshot mappings are
in `docs/TASK4_2.md` through `docs/TASK4_4.md`.

Task 5 verifies the persistent database trail; see [docs/TASK5.md](docs/TASK5.md)
for the short explanation, query output and history screenshot.

Task 6.1 Docker warm-up commands, definitions and evidence mappings are in
[docs/TASK6_1.md](docs/TASK6_1.md). Actual command outputs are in `outputs/task6/6_1/`.

Task 6.2 adds a slim, non-root CPU inference image with a health check and a
runtime-only build context. Build with `docker build -t visionguard:6.2 .`;
model mounting, startup and terminal evidence commands are in
[docs/TASK6_2.md](docs/TASK6_2.md).

Task 6.3 adds `compose.yaml` with PostgreSQL, health-based startup ordering,
a persistent database volume and a read-only model mount. Private database
credentials belong in the ignored `.env`; see `.env.example`. Run
`docker compose up -d --build --wait`, then open http://127.0.0.1:8083.
See [docs/TASK6_3.md](docs/TASK6_3.md) for terminal evidence commands.

Task 6.4 measures a 5.59% size reduction from disabling pip bytecode compilation.
Compose uses the verified optimized image. See [docs/TASK6_4.md](docs/TASK6_4.md)
for before/after sizes, largest layers and terminal screenshot commands.

Task 6.5 demonstrates operations on both services and a controlled missing-model
failure, diagnosed from logs and recovered without losing history. Commands and
evidence are in [docs/TASK6_5.md](docs/TASK6_5.md). `compose.failure.yaml` is an
explicit troubleshooting override; normal startup uses only `compose.yaml`.

Task 6.6 defines GitHub Actions CI in `.github/workflows/ci.yml`: install
dependencies, lint, run the full tests and build Docker. See
[docs/TASK6_6.md](docs/TASK6_6.md) for hosted-run evidence status and secrets guidance.

Public deployment is not implemented yet.
AI assistance was used to implement, test and document Tasks 2, 3, 4, 5 and 6.1–6.6;
the final submitted report must disclose that assistance.
