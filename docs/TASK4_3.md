# Task 4.3 — frontend

The upload page accepts an image and shows its predicted class and model confidence.
The history page reads saved results from SQLite; uploaded images are not saved.

## Screenshots to submit

Use these files from `outputs/task4/4_3/screenshots/`:

- `upload.jpg`: upload page.
- `result.jpg`: prediction, confidence and uncertainty note.
- `history.jpg`: results read from the database.

Optional mobile evidence: `upload_mobile.jpg`, `result_mobile.jpg` and
`history_mobile.jpg`. `error.jpg` shows the styled invalid-image response.

History survives an app restart, verified between the Flask and Gunicorn runs.
Saved timestamps use the configured display timezone (Asia/Karachi by default).
