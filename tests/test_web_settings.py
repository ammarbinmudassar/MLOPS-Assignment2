import os
import unittest
from unittest.mock import patch

from task4.settings import ROOT, settings_from_env
from task4.run import main


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

    def test_compose_credentials_use_service_name_and_escape_reserved_characters(self):
        with patch.dict(os.environ, {
            "DATABASE_HOST": "db", "DATABASE_USER": "course@user",
            "DATABASE_PASSWORD": "test:/?#@", "DATABASE_NAME": "visionguard",
            "DATABASE_URL": "sqlite:////data/visionguard.sqlite3",
        }, clear=True):
            settings = settings_from_env()
        self.assertEqual(settings["DATABASE_URL"],
                         "postgresql://course%40user:test%3A%2F%3F%23%40@db:5432/visionguard")

    def test_flask_launcher_disables_debug_reload_and_implicit_dotenv(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch("sys.argv", ["run", "--server", "flask"]), \
                patch("task4.run.load_dotenv"), \
                patch("task4.app.create_app") as create:
            main()
            arguments = create.return_value.run.call_args.kwargs
        self.assertFalse(arguments["debug"])
        self.assertFalse(arguments["use_reloader"])
        self.assertFalse(arguments["load_dotenv"])
