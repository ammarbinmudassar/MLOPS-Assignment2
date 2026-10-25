"""Task 3.6: register two trained versions and promote the validation winner."""

import json

import mlflow
from mlflow import MlflowClient

from task2.common import ROOT, git_commit, write_json
from task3.dagshub_config import configure_dagshub

MODEL_NAME = "23L-0630_VisionGuard"
OUT = ROOT / "outputs/task3/3_6"


def main():
    configure_dagshub()
    client = MlflowClient()
    selection = json.loads((ROOT / "outputs/task3/3_5/selection.json").read_text())
    runs = json.loads((ROOT / "outputs/task3/3_5/runs.json").read_text())
    winner = next(row for row in runs if row["run_id"] == selection["run_id"])
    previous = next(row for row in runs if row["name"] == "unfreeze_4")
    registered = []
    for row in (previous, winner):
        # Re-running the setup must reuse a version from this run, not duplicate it.
        versions = client.search_model_versions(f"name = '{MODEL_NAME}'")
        existing = next((item for item in versions if item.run_id == row["run_id"]), None)
        if existing is None:
            existing = mlflow.register_model(
                row["model_uri"], MODEL_NAME, await_registration_for=60,
                tags={"task": "3.6", "configuration": row["name"],
                      "validation_loss": str(row["selection_val_loss"]),
                      "validation_roc_auc": str(row["selection_val_roc_auc"]),
                      "decision_threshold": str(selection["decision_threshold"]),
                      "positive_class": "AI", "registry_code_commit": git_commit()},
            )
        version = client.get_model_version(MODEL_NAME, existing.version)
        assert version.status == "READY", version.status
        client.update_model_version(
            MODEL_NAME, version.version,
            description=f"Task 3.5 {row['name']}; validation loss {row['selection_val_loss']:.4f}; "
                        f"validation ROC-AUC {row['selection_val_roc_auc']:.4f}. "
                        "RGB letterbox 160x160, raw pixels [0,255]; preprocessing is inside the model.",
        )
        registered.append({"version": version.version, "configuration": row["name"],
                           "run_id": version.run_id, "status": version.status,
                           "source": version.source})
    champion = next(item for item in registered if item["run_id"] == winner["run_id"])
    client.set_registered_model_alias(MODEL_NAME, "champion", champion["version"])
    resolved = client.get_model_version_by_alias(MODEL_NAME, "champion")
    assert resolved.version == champion["version"] and resolved.run_id == winner["run_id"]
    model = client.get_registered_model(MODEL_NAME)
    assert model.aliases["champion"] == champion["version"]
    write_json(OUT / "registry.json", {
        "model_name": MODEL_NAME, "versions": registered,
        "alias": "champion", "champion_version": resolved.version,
        "champion_run_id": resolved.run_id, "alias_verified": True,
        "model_uri": f"models:/{MODEL_NAME}@champion",
    })
    for item in registered:
        print(f"Version {item['version']}: {item['configuration']} ({item['status']})")
    print(f"Alias champion -> version {resolved.version}")
    print(f"Model URI: models:/{MODEL_NAME}@champion")


if __name__ == "__main__":
    main()
