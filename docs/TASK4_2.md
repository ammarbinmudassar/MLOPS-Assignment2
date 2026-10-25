# Task 4.2 — routes and validation

Run from the assignment folder:

```bash
.venv/bin/python -m flask --app task4.app:create_app run --host 127.0.0.1 --port 8000 --no-debugger --no-reload
```

Open http://127.0.0.1:8000. Routes: `/`, POST `/predict`, `/result/<id>`,
`/history` and `/health`. The local model loads once when the app starts.

PNG/JPEG uploads are limited to 5 MiB. The app checks extensions, actual image
format and full decoding, sanitizes names and stores only result metadata.
Multipart upload streams use memory instead of temporary disk files.

## Screenshots to submit

Files under `outputs/task4/4_2/screenshots/`:

- `wrong_type.jpg`: unsupported text file rejected (415).
- `too_large.jpg`: oversized upload rejected (413).
- `unreadable.jpg`: fake PNG rejected (422).

Additional screenshots show a successful result and recent history.
To repeat the cases, use files in `outputs/task4/4_2/fixtures/`.
Regenerate the ignored oversized image with:

```bash
.venv/bin/python -m task4.validation_fixtures
```

Actual HTTP checks are recorded in `http_checks.json`. All 24 tests passed,
including a valid large-image regression test. At this checkpoint, history is
bounded in-memory metadata; database persistence belongs to the next stage.
