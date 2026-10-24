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

MLflow/DagsHub, serving, database, Docker, CI and deployment tasks are not implemented
by this Task 2 work. AI assistance was used to implement, test and document Task 2;
the final submitted report must disclose that assistance.
