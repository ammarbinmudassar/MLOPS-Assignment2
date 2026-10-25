"""SQLite result metadata; uploaded image bytes are never stored."""

from contextlib import contextmanager
from pathlib import Path
import sqlite3

from task4.settings import ROOT

FIELDS = ("id", "filename", "created_at", "label", "confidence", "p_ai", "threshold", "model_name", "model_version", "confidence_is_calibrated")


class ResultStore:
    def __init__(self, database_url):
        if not database_url.startswith("sqlite:///"):
            raise ValueError("Use a sqlite:/// database address for this assignment.")
        path = Path(database_url.removeprefix("sqlite:///")).expanduser()
        self.path = path if path.is_absolute() else ROOT / path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("""CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY, filename TEXT NOT NULL, created_at TEXT NOT NULL,
                label TEXT NOT NULL CHECK(label IN ('AI','Real')),
                confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
                p_ai REAL NOT NULL CHECK(p_ai BETWEEN 0 AND 1),
                threshold REAL NOT NULL CHECK(threshold BETWEEN 0 AND 1),
                model_name TEXT NOT NULL, model_version TEXT NOT NULL,
                confidence_is_calibrated INTEGER NOT NULL DEFAULT 0
            )""")

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def add(self, record):
        with self.connection() as connection:
            connection.execute(
                f"INSERT INTO predictions ({','.join(FIELDS)}) VALUES ({','.join('?' for _ in FIELDS)})",
                tuple(record[key] for key in FIELDS),
            )

    def list(self, limit=100):
        with self.connection() as connection:
            return [dict(row) for row in connection.execute(
                "SELECT * FROM predictions ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,),
            )]

    def get(self, identifier):
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM predictions WHERE id = ?", (identifier,)).fetchone()
            return dict(row) if row else None

    def ping(self):
        with self.connection() as connection:
            return connection.execute("SELECT 1").fetchone()[0] == 1
