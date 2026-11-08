"""Local UI client for the protected hosted model/history API."""
from urllib.parse import urlsplit
import requests


class RemoteClient:
    backend = 'remote-postgresql'

    def __init__(self, url, key, allow_http=False):
        parts = urlsplit(url)
        local_http = allow_http and parts.scheme == 'http' and parts.hostname in ('127.0.0.1', 'localhost')
        if not parts.hostname or (parts.scheme != 'https' and not local_http) or parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError('Use an HTTPS inference API URL without credentials')
        if not key:
            raise ValueError('INFERENCE_API_KEY is required')
        self.url = url.rstrip('/')
        self.key = key
        health = self._request('GET', '/health')
        self.metadata = {key: health[key] for key in ('model_name', 'model_version')}
        self.threshold = float(health.get('threshold', .5))

    def _request(self, method, path, **kwargs):
        response = requests.request(method, self.url + path,
                                    headers={'X-API-Key': self.key}, timeout=(10, 60),
                                    allow_redirects=False, **kwargs)
        if response.status_code == 404:
            return None
        if response.status_code not in (200, 201):
            raise RuntimeError(f'Inference API returned HTTP {response.status_code}')
        return response.json()

    def predict_upload(self, contents, filename):
        return self._request('POST', '/predict', files={'image': (filename, contents)})

    def get(self, identifier):
        return self._request('GET', '/results/' + identifier)

    def list(self):
        return self._request('GET', '/history')['results']

    def ping(self):
        return self._request('GET', '/health')['status'] == 'ok'
