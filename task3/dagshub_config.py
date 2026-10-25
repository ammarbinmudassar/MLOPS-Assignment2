"""Read DagsHub settings without displaying the access token."""

import os
from urllib.parse import urlsplit

from dotenv import load_dotenv

from task2.common import ROOT


def configure_dagshub():
    load_dotenv(ROOT / ".env", override=True)
    required = ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]
    missing = [name for name in required if not os.environ.get(name, "").strip()]
    if missing:
        raise ValueError("Fill these variables in .env: " + ", ".join(missing))
    uri = os.environ["MLFLOW_TRACKING_URI"].rstrip("/")
    try:
        parsed = urlsplit(uri)
        port = parsed.port
    except ValueError:
        raise ValueError("Invalid tracking URL. Use the HTTPS DagsHub tracking endpoint.") from None
    parts = parsed.path.strip("/").split("/")
    if (parsed.scheme != "https" or parsed.hostname != "dagshub.com"
            or port not in (None, 443) or parsed.username or parsed.password
            or parsed.query or parsed.fragment or len(parts) != 2
            or not all(parts) or not parts[-1].endswith(".mlflow")):
        raise ValueError("Use the HTTPS DagsHub tracking URL without embedded credentials.")
    import mlflow
    mlflow.set_tracking_uri(uri)
    return uri

