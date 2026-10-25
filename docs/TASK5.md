# Task 5 — Database Trail

Each successful prediction is saved in SQLite with its timestamp, class,
confidence and model name/version. The History page reads these records from
the database, so results remain available after restarting the app.

## Evidence

- `outputs/task5/history.jpg`: history page with two saved predictions.
- `outputs/task5/database_query.txt`: direct database query showing all required fields.
- `outputs/task5/database_check.json`: schema, records and successful integrity check.
- `outputs/task4/4_4/gunicorn_check.json`: earlier Flask result remained available
  after switching to Gunicorn.

Implementation: `task4/storage.py` and the `/predict`, `/history` and `/result/<id>`
routes in `task4/app.py`. Database address comes from `DATABASE_URL` in `.env.web`;
the default file is `var/visionguard.sqlite3` and is excluded from Git.

Timestamps are stored in UTC and displayed in Asia/Karachi. Image bytes are not
stored. History displays the latest 100 results; older records remain queryable.
Rejected uploads do not create prediction records because no prediction is made.
The Real sample shown was misclassified as AI; history records the model decision,
not the ground-truth class.
