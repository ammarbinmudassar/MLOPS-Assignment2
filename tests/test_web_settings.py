import os
import unittest
from unittest.mock import patch

from task4.settings import ROOT, settings_from_env


class WebSettingsTests(unittest.TestCase):
    def test_settings_are_read_from_environment(self):
        with patch.dict(os.environ, {"MODEL_DIR": "custom/model", "DECISION_THRESHOLD": "0.7",
                                    "DATABASE_URL": "sqlite:///custom/history.sqlite3",
                                    "MAX_UPLOAD_BYTES": "1000000", "APP_PORT": "8123"}, clear=True):
            settings = settings_from_env()
        self.assertEqual(settings["MODEL_DIR"], ROOT / "custom/model")
        self.assertEqual(settings["DECISION_THRESHOLD"], .7)
        self.assertEqual(settings["DATABASE_URL"], "sqlite:///custom/history.sqlite3")
        self.assertEqual(settings["MAX_FILE_BYTES"], 1000000)
        self.assertEqual(settings["APP_PORT"], 8123)
        self.assertFalse(settings["DEBUG"])

    def test_bad_threshold_fails_at_startup(self):
        with patch.dict(os.environ, {"DECISION_THRESHOLD": "1.5"}):
            with self.assertRaisesRegex(ValueError, "Invalid"):
                settings_from_env()
