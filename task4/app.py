"""Task 4.2: Flask routes, memory-only uploads and friendly validation errors."""

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo
import warnings

from flask import Flask, Request, jsonify, redirect, render_template, request, url_for
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
from task4.settings import settings_from_env
from task4.storage import ResultStore

ROOT = Path(__file__).resolve().parents[1]
FORMATS = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG"}


class MemoryUploadRequest(Request):
    def _get_file_stream(self, total_content_length, content_type, filename=None, content_length=None):
        # Werkzeug normally spools larger uploads to a temporary disk file.
        return BytesIO()


def create_app(test_config=None, predictor=None):
    app = Flask(__name__)
    app.request_class = MemoryUploadRequest
    app.config.update(settings_from_env())
    if test_config:
        app.config.update(test_config)
    if predictor is None:
        from task4.inference import VisionGuardPredictor
        predictor = VisionGuardPredictor(app.config["MODEL_DIR"], threshold=app.config["DECISION_THRESHOLD"])
    app.extensions["predictor"] = predictor
    app.extensions["recent_results"] = ResultStore(app.config["DATABASE_URL"])

    @app.context_processor
    def shared_view_data():
        return {"model_version": predictor.metadata["model_version"],
                "upload_limit_mb": f"{app.config['MAX_FILE_BYTES'] / (1024 * 1024):g}"}

    @app.template_filter("local_time")
    def local_time(timestamp):
        return datetime.fromisoformat(timestamp).astimezone(ZoneInfo(app.config["APP_TIMEZONE"])).strftime("%d %b %Y, %I:%M %p %Z")

    def error_page(title, message, status):
        return render_template("error.html", title=title, message=message), status

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(error):
        limit = app.config["MAX_FILE_BYTES"] / (1024 * 1024)
        return error_page("Image too large", f"Choose an image smaller than {limit:g} MB and try again.", 413)

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
                        limit = app.config["MAX_IMAGE_PIXELS"]
                        return error_page("Image dimensions too large", f"Resize the image to {limit:,} pixels or fewer and try again.", 422)
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
        try:
            app.extensions["recent_results"].add(record)
        except Exception:
            app.logger.exception("Prediction history write failed")
            return error_page("Result could not be saved", "Please try again in a moment.", 503)
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
        try:
            ready = app.extensions["recent_results"].ping()
        except Exception:
            ready = False
        return jsonify(status="ok" if ready else "unavailable", model_name=metadata["model_name"],
                       model_version=metadata["model_version"], threshold=predictor.threshold,
                       history_backend="sqlite", database_ready=ready, debug=app.debug), (200 if ready else 503)

    return app
