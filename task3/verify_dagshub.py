"""Verify the two saved Task 3.2 runs directly on the hosted server."""

import json

from mlflow import MlflowClient
from task2.common import ROOT
from task3.dagshub_config import configure_dagshub


def main():
    client = MlflowClient(tracking_uri=configure_dagshub())
    destination = ROOT / "outputs/task3/3_2"
    runs = json.loads((destination / "runs.json").read_text())
    checks = []
    for saved in runs:
        run_id = saved["run_id"]
        run = client.get_run(run_id)
        assert run.info.status == "FINISHED", run_id
        histories = {}
        for key in ("loss", "accuracy", "val_loss", "val_accuracy"):
            history = client.get_metric_history(run_id, key)
            assert sorted(point.step for point in history) == [1, 2, 3, 4, 5]
            histories[key] = len(history)
        required = {
            "plots": "plots/training_curves.png",
            "tables": "tables/history.csv",
            "model": "model/MLmodel",
            "model/data": "model/data/model.keras",
        }
        for folder, path in required.items():
            assert path in {item.path for item in client.list_artifacts(run_id, folder)}, path
        assert run.data.tags.get("model_reload_verified") == "true"
        assert run.data.params["learning_rate"] == str(saved["learning_rate"])
        checks.append({
            "run_id": run_id, "status": run.info.status,
            "metric_points": histories, "required_artifacts_verified": True,
            "model_reload_verified": True,
        })
    (destination / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    print("Verified two FINISHED remote runs, five epochs, plots, tables and reloadable models.")


if __name__ == "__main__":
    main()
