#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$project_dir/.mlflow/artifacts"

exec "$project_dir/.venv/bin/mlflow" server \
  --backend-store-uri "sqlite:///$project_dir/.mlflow/mlflow.db" \
  --artifacts-destination "$project_dir/.mlflow/artifacts" \
  --host 127.0.0.1 \
  --port "${MLFLOW_PORT:-5000}" \
  --workers 1

