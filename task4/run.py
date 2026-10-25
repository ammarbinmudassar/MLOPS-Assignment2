"""Run Flask for coursework checks or Gunicorn as the production WSGI server."""

import argparse
import os
import sys

from dotenv import load_dotenv

from task4.settings import ROOT, settings_from_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", choices=("flask", "gunicorn"), default="flask")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env.web", override=False)
    os.environ["FLASK_DEBUG"] = "0"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    config = settings_from_env()
    if args.server == "flask":
        from task4.app import create_app
        create_app().run(host=config["APP_HOST"], port=config["APP_PORT"], debug=False, use_reloader=False)
    else:
        workers = int(os.environ.get("WEB_WORKERS", "1"))
        threads = int(os.environ.get("WEB_THREADS", "2"))
        timeout = int(os.environ.get("WEB_TIMEOUT", "60"))
        if min(workers, threads, timeout) < 1:
            raise ValueError("Workers, threads and timeout must be positive.")
        os.execv(sys.executable, [sys.executable, "-m", "gunicorn", "--bind",
                 f"{config['APP_HOST']}:{config['APP_PORT']}", "--workers", str(workers),
                 "--threads", str(threads), "--timeout", str(timeout),
                 "--access-logfile", "-", "--error-logfile", "-", "task4.app:create_app()"])


if __name__ == "__main__":
    main()
