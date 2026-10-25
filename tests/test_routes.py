from io import BytesIO
import random
import unittest
from unittest.mock import Mock

from flask import request
from PIL import Image

from task4.app import create_app


def png():
    stream = BytesIO()
    Image.new("RGB", (20, 10), "white").save(stream, format="PNG")
    return stream.getvalue()


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.predictor = Mock(metadata={"model_name": "test", "model_version": "2"})
        self.predictor.predict_bytes.return_value = {
            "label": "AI", "confidence": .8, "p_ai": .8, "threshold": .5,
            "model_name": "test", "model_version": "2", "confidence_is_calibrated": False,
        }
        self.app = create_app({"TESTING": True}, predictor=self.predictor)
        self.client = self.app.test_client()

    def post(self, contents, filename):
        return self.client.post("/predict", data={"image": (BytesIO(contents), filename)})

    def test_pages_health_and_empty_history(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertIn(b"No images checked yet", self.client.get("/history").data)
        self.assertEqual(self.client.get("/health").json["model_version"], "2")
        self.predictor.predict_bytes.assert_not_called()

    def test_success_sanitizes_name_and_stores_only_metadata(self):
        response = self.post(png(), "../../unsafe name.png")
        self.assertEqual(response.status_code, 303)
        result = self.client.get(response.location)
        self.assertEqual(result.status_code, 200)
        self.assertIn(b"unsafe_name.png", result.data)
        self.assertIn(b"80.00%", result.data)
        records = self.app.extensions["recent_results"].list()
        self.assertEqual(len(records), 1)
        self.assertFalse(any(isinstance(value, bytes) for value in records[0].values()))
        self.assertIn(b"unsafe_name.png", self.client.get("/history").data)

    def test_same_predictor_handles_two_requests(self):
        self.post(png(), "first.png")
        self.post(png(), "second.png")
        self.assertIs(self.app.extensions["predictor"], self.predictor)
        self.assertEqual(self.predictor.predict_bytes.call_count, 2)

    def test_valid_image_larger_than_parser_chunk_is_accepted(self):
        image = Image.frombytes("RGB", (400, 400), random.Random(42).randbytes(480000))
        stream = BytesIO()
        image.save(stream, format="PNG")
        self.assertGreater(len(stream.getvalue()), 400000)
        self.assertEqual(self.post(stream.getvalue(), "valid_large.png").status_code, 303)

    def test_wrong_extension_is_rejected_before_prediction(self):
        response = self.post(b"plain text", "notes.txt")
        self.assertEqual(response.status_code, 415)
        self.assertIn(b"Unsupported file type", response.data)
        self.predictor.predict_bytes.assert_not_called()

    def test_file_and_total_request_size_are_bounded(self):
        self.app.config.update(MAX_FILE_BYTES=100, MAX_CONTENT_LENGTH=1000)
        self.assertEqual(self.post(b"x" * 101, "large.png").status_code, 413)
        self.assertEqual(self.post(b"x" * 2000, "large.png").status_code, 413)
        self.predictor.predict_bytes.assert_not_called()

    def test_fake_image_is_rejected_before_prediction(self):
        response = self.post(b"this is not a PNG", "fake.png")
        self.assertEqual(response.status_code, 422)
        self.assertIn(b"Image could not be opened", response.data)
        self.predictor.predict_bytes.assert_not_called()

    def test_format_mismatch_and_missing_upload(self):
        self.assertEqual(self.post(png(), "actually_png.jpg").status_code, 422)
        self.assertEqual(self.client.post("/predict").status_code, 400)
        self.predictor.predict_bytes.assert_not_called()

    def test_large_dimensions_are_rejected(self):
        self.app.config["MAX_IMAGE_PIXELS"] = 100
        self.assertEqual(self.post(png(), "wide.png").status_code, 422)
        self.predictor.predict_bytes.assert_not_called()

    def test_upload_stream_remains_in_memory(self):
        with self.app.test_request_context("/predict", method="POST", data={"image": (BytesIO(b"x" * 600000), "large.png")}):
            self.assertIsInstance(request.files["image"].stream, BytesIO)
            request.environ["wsgi.input"].close()

    def test_unknown_result_has_friendly_error(self):
        response = self.client.get("/result/missing")
        self.assertEqual(response.status_code, 404)
        self.assertIn(b"Page or result not found", response.data)


if __name__ == "__main__":
    unittest.main()
