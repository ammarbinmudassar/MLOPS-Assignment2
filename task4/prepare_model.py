"""Download an immutable champion version once, outside request handling."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import mlflow

from task2.common import ROOT, write_json
from task3.dagshub_config import configure_dagshub
from task3.register_models import MODEL_NAME


def main():
    configure_dagshub()
    client = mlflow.MlflowClient()
    champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    run = client.get_run(champion.run_id)
    # Resolve the alias once and download that fixed version, even if the alias moves.
    version_uri = f"models:/{MODEL_NAME}/{champion.version}"
    destination = ROOT / "models/champion"
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        downloaded = Path(mlflow.artifacts.download_artifacts(artifact_uri=version_uri, dst_path=temporary))
        source = downloaded / "data/model.keras"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        metadata = {
            "model_name": MODEL_NAME, "model_version": champion.version,
            "source_run_id": champion.run_id, "resolved_alias": "champion",
            "source_uri": version_uri, "sha256": digest,
            "threshold": float(champion.tags["decision_threshold"]),
            "positive_class": "AI", "image_size": int(run.data.params["image_size"]),
            "preprocessing": "rgb_letterbox_bilinear_antialias_uint8_v1",
            "model_internal_scaling": "MobileNetV2 [-1,1]; input pixels remain [0,255]",
        }
        staged = destination / "model.keras.partial"
        shutil.copyfile(source, staged)
        staged.replace(destination / "model.keras")
        write_json(destination / "metadata.json", metadata)
    write_json(ROOT / "outputs/task4/4_1/model_package.json", metadata)
    print(f"Prepared local {MODEL_NAME} version {champion.version}")
    print(f"Model: {destination / 'model.keras'}")
    print(f"SHA-256: {digest}")
    print("Inference can now run without DagsHub access.")


if __name__ == "__main__":
    main()
