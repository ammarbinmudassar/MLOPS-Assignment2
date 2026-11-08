"""Neon/PostgreSQL prediction metadata, with no uploaded image storage."""
from contextlib import contextmanager

FIELDS = ('id', 'filename', 'created_at', 'label', 'confidence', 'p_ai', 'threshold',
          'model_name', 'model_version', 'confidence_is_calibrated')


class PostgresStore:
    def __init__(self, url):
        if not url or not url.startswith(('postgresql://', 'postgres://')):
            raise ValueError('A PostgreSQL DATABASE_URL is required')
        self.url = url
        with self.connection() as connection:
            connection.execute('SELECT pg_advisory_xact_lock(6302026)')
            connection.execute('''CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY, filename TEXT NOT NULL, created_at TEXT NOT NULL,
                label TEXT NOT NULL CHECK(label IN ('AI','Real')),
                confidence DOUBLE PRECISION NOT NULL CHECK(confidence BETWEEN 0 AND 1),
                p_ai DOUBLE PRECISION NOT NULL CHECK(p_ai BETWEEN 0 AND 1),
                threshold DOUBLE PRECISION NOT NULL CHECK(threshold BETWEEN 0 AND 1),
                model_name TEXT NOT NULL, model_version TEXT NOT NULL,
                confidence_is_calibrated INTEGER NOT NULL DEFAULT 0)''')

    @contextmanager
    def connection(self):
        import psycopg
        from psycopg.rows import dict_row
        with psycopg.connect(self.url, row_factory=dict_row, connect_timeout=15) as connection:
            yield connection

    def add(self, record):
        with self.connection() as connection:
            connection.execute(f"INSERT INTO predictions ({','.join(FIELDS)}) VALUES ({','.join(['%s'] * len(FIELDS))})",
                               tuple(int(record[key]) if key == 'confidence_is_calibrated' else record[key] for key in FIELDS))

    def get(self, identifier):
        with self.connection() as connection:
            return connection.execute('SELECT * FROM predictions WHERE id=%s', (identifier,)).fetchone()

    def list(self):
        with self.connection() as connection:
            return connection.execute('SELECT * FROM predictions ORDER BY created_at DESC, id DESC LIMIT 100').fetchall()

    def ping(self):
        with self.connection() as connection:
            return connection.execute('SELECT 1 AS ready').fetchone()['ready'] == 1
