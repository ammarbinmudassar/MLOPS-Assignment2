import os
import unittest
from unittest.mock import patch

from task3.dagshub_config import configure_dagshub


class DagsHubConfigurationTests(unittest.TestCase):
    def configure(self, uri, password="dummy-test-value"):
        values = {"MLFLOW_TRACKING_URI": uri, "MLFLOW_TRACKING_USERNAME": "test-user",
                  "MLFLOW_TRACKING_PASSWORD": password}
        with patch.dict(os.environ, values, clear=True), \
                patch("task3.dagshub_config.load_dotenv"), \
                patch("mlflow.set_tracking_uri") as set_uri:
            result = configure_dagshub()
            set_uri.assert_called_once_with(result)
            return result

    def test_missing_token_identifies_variable_without_echoing_credentials(self):
        with self.assertRaisesRegex(ValueError, "MLFLOW_TRACKING_PASSWORD"):
            self.configure("https://dagshub.com/user/repo.mlflow", "")

    def test_https_dagshub_endpoint_is_accepted(self):
        uri = "https://dagshub.com/user/repo.mlflow"
        self.assertEqual(self.configure(uri + "/"), uri)

    def test_credentials_cannot_be_sent_to_another_host(self):
        with self.assertRaises(ValueError):
            self.configure("https://example.com/user/repo.mlflow")

    def test_credentials_cannot_be_embedded_in_url(self):
        with self.assertRaises(ValueError):
            self.configure("https://user:dummy@dagshub.com/user/repo.mlflow")

    def test_invalid_port_is_not_echoed(self):
        with self.assertRaises(ValueError) as context:
            self.configure("https://dagshub.com:private-value/user/repo.mlflow")
        self.assertNotIn("private-value", str(context.exception))


if __name__ == "__main__":
    unittest.main()
