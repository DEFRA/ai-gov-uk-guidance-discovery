"""Postgres selection and conninfo, with no driver import.

`DATABASE_URL`, when non-blank, is the whole conninfo (URI or libpq keyword
form) and selects Postgres. Otherwise the split `DB_HOST` / `DB_PORT` /
`DB_NAME` / `DB_USER` / `DB_PASSWORD` vars are used, and `DB_HOST` or
`DB_BACKEND=postgres` selects Postgres.
"""
from __future__ import annotations

import os


def postgres_selected() -> bool:
    if os.getenv("DB_BACKEND", "").strip().lower() == "postgres":
        return True
    if os.getenv("DB_HOST", "").strip():
        return True
    if os.getenv("DATABASE_URL", "").strip():
        return True
    return False


def conninfo() -> str:
    """Conninfo for psycopg. A set `DATABASE_URL` wins over the split vars."""
    url = os.getenv("DATABASE_URL", "").strip()
    if url:
        return url
    return (
        f"host={os.getenv('DB_HOST', 'localhost')} "
        f"port={os.getenv('DB_PORT', '5432')} "
        f"dbname={os.getenv('DB_NAME', 'gov_uk_corpus')} "
        f"user={os.getenv('DB_USER', 'corpus')} "
        f"password={os.getenv('DB_PASSWORD', '')}"
    )
