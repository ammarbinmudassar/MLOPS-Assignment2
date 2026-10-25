"""Web settings come from environment variables, with local coursework defaults."""

import os
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def settings_from_env():
    model_dir = Path(os.environ.get("MODEL_DIR", "models/champion")).expanduser()
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    threshold = float(os.environ.get("DECISION_THRESHOLD", "0.5"))
    maximum = int(os.environ.get("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))
    pixels = int(os.environ.get("MAX_IMAGE_PIXELS", "16000000"))
    port = int(os.environ.get("APP_PORT", "8000"))
    display_timezone = os.environ.get("APP_TIMEZONE", "Asia/Karachi")
    ZoneInfo(display_timezone)
    if not 0 <= threshold <= 1 or maximum < 1 or pixels < 1 or not 1 <= port <= 65535:
        raise ValueError("Invalid threshold, upload limit, image dimensions or port in web settings.")
    return {"MODEL_DIR": model_dir, "DECISION_THRESHOLD": threshold,
            "DATABASE_URL": os.environ.get("DATABASE_URL", "sqlite:///var/visionguard.sqlite3"),
            "MAX_FILE_BYTES": maximum, "MAX_CONTENT_LENGTH": maximum + 64 * 1024,
            "MAX_FORM_MEMORY_SIZE": 512 * 1024, "MAX_FORM_PARTS": 4,
            "MAX_IMAGE_PIXELS": pixels, "APP_HOST": os.environ.get("APP_HOST", "127.0.0.1"),
            "APP_PORT": port, "APP_TIMEZONE": display_timezone, "DEBUG": False}
