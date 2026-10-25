# Task 4.4 — settings and server

Model path, decision threshold and database location come from environment settings.
Debug is disabled, and Gunicorn serves the app as a production WSGI server.

## Run

The ignored `.env.web` contains local web settings; `.env.web.example` is the
shareable template. The launcher loads `.env.web` explicitly.

From the assignment folder, run one server at a time:

```bash
.venv/bin/python -m task4.run --server flask
```

Or use Gunicorn:

```bash
.venv/bin/python -m task4.run --server gunicorn
```

Open http://127.0.0.1:8000. Stop the current server with Ctrl+C before switching.
The model package must already exist; see `TASK4_1.md` for preparation.

## Evidence

- `outputs/task4/4_4/screenshots/health.jpg`: healthy model/database and `debug: false`.
- `flask_check.json`: actual Flask health and prediction checks.
- `gunicorn_check.json`: actual Gunicorn checks, persistence across restart and
  upload rejection statuses 415, 413 and 422.
- `configuration.json`: public settings used for verification.

The server runs locally on port 8000. Hosting and public deployment are separate work.
