"""DATABASE_URL vs the split DB_* vars. No database connection."""
from __future__ import annotations

import os
import unittest

from govuk_corpus.db_env import conninfo, postgres_selected

_KEYS = ("DATABASE_URL", "DB_BACKEND", "DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD")


class TestDbEnv(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in _KEYS}
        for k in _KEYS:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_split_vars(self):
        os.environ["DB_HOST"] = "db.internal"
        os.environ["DB_PORT"] = "5433"
        os.environ["DB_NAME"] = "corpus"
        os.environ["DB_USER"] = "app"
        os.environ["DB_PASSWORD"] = "s3cret"
        self.assertEqual(
            conninfo(),
            "host=db.internal port=5433 dbname=corpus user=app password=s3cret",
        )
        self.assertTrue(postgres_selected())

    def test_defaults_when_backend_forced(self):
        os.environ["DB_BACKEND"] = "postgres"
        self.assertEqual(
            conninfo(),
            "host=localhost port=5432 dbname=gov_uk_corpus user=corpus password=",
        )
        self.assertTrue(postgres_selected())

    def test_database_url_used_whole_and_wins(self):
        url = "postgresql://app:p%40ss@db.example:5432/gov?sslmode=require"
        os.environ["DATABASE_URL"] = f"  {url}  "
        os.environ["DB_HOST"] = "localhost"
        os.environ["DB_USER"] = "other"
        self.assertEqual(conninfo(), url)
        self.assertTrue(postgres_selected())

    def test_blank_database_url_falls_back(self):
        os.environ["DATABASE_URL"] = "   "
        os.environ["DB_HOST"] = "db.internal"
        self.assertTrue(conninfo().startswith("host=db.internal "))
        self.assertTrue(postgres_selected())

    def test_database_url_alone_selects_postgres(self):
        os.environ["DATABASE_URL"] = "postgres://u:p@h:5432/db"
        self.assertTrue(postgres_selected())
        self.assertEqual(conninfo(), "postgres://u:p@h:5432/db")

    def test_sqlite_when_unset(self):
        self.assertFalse(postgres_selected())
