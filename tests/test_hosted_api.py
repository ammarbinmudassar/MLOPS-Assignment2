from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from deploy.vercel_api.app import create_app
from task4.storage import ResultStore
from task4.remote import RemoteClient


def picture():
    buffer = BytesIO()
    Image.new('RGB', (20, 10), 'white').save(buffer, format='PNG')
    return buffer.getvalue()


class HostedAPITests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.store = ResultStore('sqlite:///' + str(Path(folder.name) / 'results.sqlite3'))
        self.model = Mock(metadata={'model_name': 'test', 'model_version': '2-fp16'}, threshold=.5)
        self.model.predict_bytes.return_value = dict(label='AI', p_ai=.8, confidence=.8, threshold=.5,
                                                    model_name='test', model_version='2-fp16', confidence_is_calibrated=False)
        self.app = create_app(self.model, self.store, {'TESTING': True, 'API_KEY': 'test-secret'})
        self.client = self.app.test_client()
        self.headers = {'X-API-Key': 'test-secret'}

    def test_private_routes_reject_missing_or_wrong_keys(self):
        self.assertEqual(self.client.post('/predict').status_code, 401)
        self.assertEqual(self.client.get('/history', headers={'X-API-Key': 'wrong'}).status_code, 401)
        self.assertEqual(self.client.get('/results/unknown').status_code, 401)
        self.model.predict_bytes.assert_not_called()

    def test_prediction_is_saved_once_and_readable_from_api_history(self):
        response = self.client.post('/predict', headers=self.headers,
                                    data={'image': (BytesIO(picture()), '../../example.png')})
        self.assertEqual(response.status_code, 201)
        record = response.json
        self.assertEqual(record['filename'], 'example.png')
        self.assertEqual(self.client.get('/results/' + record['id'], headers=self.headers).json['p_ai'], .8)
        self.assertEqual(len(self.client.get('/history', headers=self.headers).json['results']), 1)
        self.assertNotIn('test-secret', response.get_data(as_text=True))

    def test_bad_image_never_calls_model_or_creates_history(self):
        response = self.client.post('/predict', headers=self.headers,
                                    data={'image': (BytesIO(b'fake png'), 'fake.png')})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.store.list(), [])
        self.model.predict_bytes.assert_not_called()

    def test_health_is_public_but_does_not_expose_credentials(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json['database_ready'])
        self.assertNotIn('test-secret', response.get_data(as_text=True))
        self.app.config['API_KEY'] = ''
        self.assertEqual(self.client.get('/health').status_code, 503)

    def test_file_limit_is_enforced(self):
        self.app.config['MAX_FILE_BYTES'] = 10
        response = self.client.post('/predict', headers=self.headers,
                                    data={'image': (BytesIO(picture()), 'large.png')})
        self.assertEqual(response.status_code, 413)
        self.model.predict_bytes.assert_not_called()


class RemoteClientTests(unittest.TestCase):
    @patch('task4.remote.requests.request')
    def test_redirects_do_not_forward_the_key_and_errors_do_not_echo_it(self, request):
        request.return_value = Mock(status_code=200)
        request.return_value.json.return_value = {'model_name': 'test', 'model_version': '2-fp16', 'threshold': .5}
        remote = RemoteClient('https://example.vercel.app', 'private-key')
        self.assertFalse(request.call_args.kwargs['allow_redirects'])
        request.return_value.status_code = 302
        with self.assertRaises(RuntimeError) as context:
            remote.list()
        self.assertNotIn('private-key', str(context.exception))

    def test_plain_http_and_embedded_credentials_are_rejected(self):
        for url in ('http://example.com', 'https://user:password@example.com'):
            with self.assertRaises(ValueError):
                RemoteClient(url, 'key')

    @patch('task4.app.ResultStore')
    @patch('task4.remote.RemoteClient')
    def test_remote_ui_does_not_open_a_local_database(self, client, store):
        client.return_value.metadata = {'model_name': 'test', 'model_version': '2-fp16'}
        from task4.app import create_app as create_ui
        app = create_ui({'INFERENCE_API_URL': 'https://example.vercel.app', 'INFERENCE_API_KEY': 'test-key'})
        self.assertIs(app.extensions['recent_results'], client.return_value)
        store.assert_not_called()
