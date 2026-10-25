"""Task 4.2: Flask routes, memory-only uploads and friendly validation errors."""

from collections import deque
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from threading import Lock
from uuid import uuid4
import warnings

from flask import Flask, Request, jsonify, redirect, render_template, request, url_for
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 5 * 1024 * 1024
FORMATS = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG"}


class MemoryUploadRequest(Request):
    def _get_file_stream(self, total_content_length, content_type, filename=None, content_length=None):
        # Werkzeug normally spools larger uploads to a temporary disk file.
        return BytesIO()


class RecentResults:
    """Bounded, thread-safe metadata history; Task 5 will provide persistence."""

    def __init__(self):
        self.records = deque(maxlen=100)
        self.lock = Lock()

    def add(self, record):
        with self.lock:
            self.records.appendleft(record)

    def list(self):
        with self.lock:
            return list(self.records)

    def get(self, identifier):
        return next((row for row in self.list() if row["id"] == identifier), None)


def create_app(test_config=None, predictor=None):
    app = Flask(__name__)
    app.request_class = MemoryUploadRequest
    app.config.update(
        MAX_FILE_BYTES=MAX_FILE_BYTES,
        MAX_CONTENT_LENGTH=MAX_FILE_BYTES + 64 * 1024,
        MAX_FORM_MEMORY_SIZE=512 * 1024, MAX_FORM_PARTS=4,
        MAX_IMAGE_PIXELS=16_000_000,
    )
    if test_config:
        app.config.update(test_config)
    if predictor is None:
        from task4.inference import VisionGuardPredictor
        predictor = VisionGuardPredictor(ROOT / "models/champion")
    app.extensions["predictor"] = predictor
    app.extensions["recent_results"] = RecentResults()

    def error_page(title, message, status):
        return render_template("error.html", title=title, message=message), status

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(error):
        return error_page("Image too large", "Choose an image smaller than 5 MB and try again.", 413)

    @app.errorhandler(404)
    def not_found(error):
        return error_page("Page or result not found", "Check recent history or upload an image to start again.", 404)

    @app.errorhandler(405)
    def wrong_method(error):
        return error_page("Upload an image first", "Use the upload form to check an image.", 405)

    @app.errorhandler(500)
    def unavailable(error):
        return error_page("Something went wrong", "Please try again. Your image has not been saved.", 500)

    @app.get("/")
    def upload():
        return render_template("upload.html")

    @app.post("/predict")
    def predict():
        incoming = request.files.get("image")
        if incoming is None or not incoming.filename:
            return error_page("No image selected", "Choose a PNG or JPEG image before submitting.", 400)
        # Keep only a safe display name; no upload path is ever created.
        filename = secure_filename(incoming.filename)
        extension = Path(filename).suffix.lower()
        if extension not in FORMATS:
            return error_page("Unsupported file type", "Upload a PNG or JPEG image (.png, .jpg or .jpeg).", 415)
        filename = filename[:100] + extension if len(filename) > 120 else filename
        contents = incoming.stream.read(app.config["MAX_FILE_BYTES"] + 1)
        if len(contents) > app.config["MAX_FILE_BYTES"]:
            raise RequestEntityTooLarge()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(contents)) as image:
                    if image.format != FORMATS[extension]:
                        return error_page("Image format mismatch", "Save the image as PNG or JPEG with the matching file extension.", 422)
                    if image.width * image.height > app.config["MAX_IMAGE_PIXELS"]:
                        return error_page("Image dimensions too large", "Resize the image to 16 million pixels or fewer and try again.", 422)
                    image.verify()
                # verify() alone does not fully decode every JPEG.
                with Image.open(BytesIO(contents)) as image:
                    image.load()
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning):
            return error_page("Image could not be opened", "This file is not a readable image. Try a different PNG or JPEG.", 422)
        try:
            result = app.extensions["predictor"].predict_bytes(contents)
        except Exception:
            app.logger.exception("Image prediction failed")
            return error_page("Image check unavailable", "We could not check this image. Please try again with another image.", 503)
        record = {"id": uuid4().hex, "filename": filename,
                  "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **result}
        app.extensions["recent_results"].add(record)
        return redirect(url_for("result", identifier=record["id"]), code=303)

    @app.get("/result/<identifier>")
    def result(identifier):
        record = app.extensions["recent_results"].get(identifier)
        if record is None:
            return not_found(None)
        return render_template("result.html", result=record)

    @app.get("/history")
    def history():
        return render_template("history.html", records=app.extensions["recent_results"].list())

    @app.get("/health")
    def health():
        metadata = app.extensions["predictor"].metadata
        return jsonify(status="ok", model_name=metadata["model_name"], model_version=metadata["model_version"])

    return app
