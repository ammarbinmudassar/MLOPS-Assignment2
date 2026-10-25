"""Check remote access; print no credentials or detailed request headers."""

import json

from task2.common import ROOT
from task3.dagshub_config import configure_dagshub


def main():
    try:
        uri = configure_dagshub()
    except ValueError as error:
        raise SystemExit(str(error)) from None
    from mlflow import MlflowClient
    try:
        MlflowClient(tracking_uri=uri).search_experiments(max_results=1)
    except Exception as error:
        raise SystemExit(
            f"DagsHub check failed ({type(error).__name__}). "
            "Check your connection, repository access, username and token. No token was printed."
        ) from None
    destination = ROOT / "outputs/task3/3_2"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "connection.json").write_text(json.dumps({
        "tracking_uri": uri, "authenticated_check": "passed", "token_recorded": False,
    }, indent=2) + "\n")
    print("DagsHub connection OK:", uri)
    print("Credentials were loaded from .env; the token was not displayed.")


if __name__ == "__main__":
    main()

