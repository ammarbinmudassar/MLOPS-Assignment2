# 23L-0630_A2_Ops — VisionGuard

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

Docker, CI and public deployment are not implemented yet.
AI assistance was used to implement, test and document Tasks 2, 3 and 4.1–4.4;
the final submitted report must disclose that assistance.
