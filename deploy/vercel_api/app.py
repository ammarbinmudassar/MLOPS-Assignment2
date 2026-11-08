"""Small protected prediction/history API for Vercel; no TensorFlow dependency."""
from datetime import datetime, timezone
from functools import wraps
from io import BytesIO
from threading import Lock
from uuid import uuid4
import hmac
import os
import warnings

from flask import Flask, Request, jsonify, request
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

try:
    from .db import PostgresStore
except ImportError:
    from db import PostgresStore


class MemoryRequest(Request):
    def _get_file_stream(self, *args, **kwargs):
        return BytesIO()


def create_app(predictor=None, store=None, test_config=None):
    app = Flask(__name__)
    app.request_class = MemoryRequest
    app.config.update(API_KEY=os.environ.get('VISIONGUARD_API_KEY', ''),
                      DATABASE_URL=os.environ.get('DATABASE_URL', ''),
                      MAX_CONTENT_LENGTH=3 * 1024 * 1024 + 65536,
                      MAX_FILE_BYTES=3 * 1024 * 1024,
                      MAX_FORM_MEMORY_SIZE=512 * 1024,
                      MAX_FORM_PARTS=4, DEBUG=False)
    if test_config:
        app.config.update(test_config)
    state = {'predictor': predictor, 'store': store}
    init_lock = Lock()

    def resources():
        with init_lock:
            if state['predictor'] is None:
                try:
                    from .lite_model import LitePredictor
                except ImportError:
                    from lite_model import LitePredictor
                state['predictor'] = LitePredictor()
            if state['store'] is None:
                state['store'] = PostgresStore(app.config['DATABASE_URL'])
        return state['predictor'], state['store']

    def protected(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            expected = app.config['API_KEY']
            if not expected:
                return jsonify(error='API is not configured'), 503
            if not hmac.compare_digest(request.headers.get('X-API-Key', ''), expected):
                return jsonify(error='Unauthorized'), 401
            return function(*args, **kwargs)
        return wrapped

    @app.errorhandler(RequestEntityTooLarge)
    def oversized(error):
        return jsonify(error='Choose an image no larger than 3 MiB'), 413

    @app.errorhandler(404)
    def not_found(error):
        return jsonify(error='Not found'), 404

    @app.errorhandler(500)
    def internal(error):
        return jsonify(error='Service temporarily unavailable'), 503

    @app.get('/')
    def index():
        return jsonify(service='VisionGuard inference API', health='/health',
                       prediction='POST /predict; multipart image; X-API-Key required')

    @app.get('/health')
    def health():
        try:
            model, database = resources()
            ready = bool(app.config['API_KEY']) and database.ping()
            return jsonify(status='ok' if ready else 'unavailable', database_ready=ready,
                           model_name=model.metadata['model_name'], model_version=model.metadata['model_version'],
                           threshold=model.threshold, runtime='LiteRT float16', history_backend='postgresql', debug=False), (200 if ready else 503)
        except Exception:
            app.logger.exception('API startup or database health failed')
            return jsonify(status='unavailable'), 503

    @app.post('/predict')
    @protected
    def predict():
        image = request.files.get('image')
        if image is None or not image.filename:
            return jsonify(error='Choose a PNG or JPEG image'), 400
        filename = secure_filename(image.filename)[:120]
        extension = filename.rsplit('.', 1)[-1].lower()
        formats = {'png': 'PNG', 'jpg': 'JPEG', 'jpeg': 'JPEG'}
        if extension not in formats:
            return jsonify(error='Use PNG or JPEG'), 415
        contents = image.stream.read(app.config['MAX_FILE_BYTES'] + 1)
        if len(contents) > app.config['MAX_FILE_BYTES']:
            raise RequestEntityTooLarge()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', Image.DecompressionBombWarning)
                with Image.open(BytesIO(contents)) as decoded:
                    if decoded.format != formats[extension] or decoded.width * decoded.height > 16000000:
                        return jsonify(error='Image format or dimensions are invalid'), 422
                    decoded.verify()
                with Image.open(BytesIO(contents)) as decoded:
                    decoded.load()
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError,
                Image.DecompressionBombWarning, Image.DecompressionBombError):
            return jsonify(error='Image could not be opened'), 422
        try:
            model, database = resources()
            record = {'id': uuid4().hex, 'filename': filename,
                      'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                      **model.predict_bytes(contents)}
            database.add(record)
            return jsonify(record), 201
        except Exception:
            app.logger.exception('Prediction or history save failed')
            return jsonify(error='Image check temporarily unavailable'), 503

    @app.get('/history')
    @protected
    def history():
        try:
            _, database = resources()
            return jsonify(results=database.list())
        except Exception:
            app.logger.exception('History read failed')
            return jsonify(error='History temporarily unavailable'), 503

    @app.get('/results/<identifier>')
    @protected
    def result(identifier):
        try:
            _, database = resources()
            record = database.get(identifier)
            return (jsonify(record), 200) if record else (jsonify(error='Result not found'), 404)
        except Exception:
            app.logger.exception('Result read failed')
            return jsonify(error='Result temporarily unavailable'), 503

    return app


app = create_app()
