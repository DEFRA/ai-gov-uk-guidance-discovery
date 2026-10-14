"""Postgres selection and conninfo, with no driver import.

`DATABASE_URL`, when non-blank, is the whole conninfo (URI or libpq keyword
form) and selects Postgres. Otherwise the split `DB_HOST` / `DB_PORT` /
`DB_NAME` / `DB_USER` / `DB_PASSWORD` vars are used, and `DB_HOST` or
`DB_BACKEND=postgres` selects Postgres.
"""
from __future__ import annotations

import os
from urllib.parse import parse_qs, urlparse

# Grep container logs for this. Bump it when the db-env startup line changes,
# so a replica that prints it is running this file and not an older image.
CODE_MARK = "db-env-log-1"


def postgres_selected() -> bool:
    if os.getenv("DB_BACKEND", "").strip().lower() == "postgres":
        return True
    if os.getenv("DB_HOST", "").strip():
        return True
    if os.getenv("DATABASE_URL", "").strip():
        return True
    return False


def _database_url() -> str:
    """Read DATABASE_URL. Blank and unset both come back as ''."""
    return os.getenv("DATABASE_URL", "").strip()


def conninfo() -> str:
    """Conninfo for psycopg. A set `DATABASE_URL` wins over the split vars."""
    url = _database_url()
    if url:
        return url
    return (
        f"host={os.getenv('DB_HOST', 'localhost')} "
        f"port={os.getenv('DB_PORT', '5432')} "
        f"dbname={os.getenv('DB_NAME', 'gov_uk_corpus')} "
        f"user={os.getenv('DB_USER', 'corpus')} "
        f"password={os.getenv('DB_PASSWORD', '')}"
    )


def _redacted_target(conn: str) -> str:
    """Host/db/user/sslmode only. Never the password or the raw string."""
    if "://" in conn.split()[0]:
        parsed = urlparse(conn)
        sslmode = (parse_qs(parsed.query).get("sslmode") or [""])[0]
        return (
            f"scheme={parsed.scheme} host={parsed.hostname or ''} "
            f"port={parsed.port or ''} dbname={parsed.path.lstrip('/')} "
            f"user={parsed.username or ''} sslmode={sslmode}"
        )
    parts = {}
    for token in conn.split():
        key, sep, value = token.partition("=")
        if sep and key.lower() != "password":
            parts[key.lower()] = value
    return " ".join(
        f"{key}={parts[key]}"
        for key in ("host", "port", "dbname", "user", "sslmode")
        if key in parts
    )


def describe_db_env() -> str:
    """One log line: which file this is, and what DATABASE_URL resolved to.

    `key_present` is whether the variable exists at all (a Container App secret
    ref that failed to inject is absent; a blank value is present but empty).
    The target is redacted — password and the raw URL are not included.
    """
    key_present = "DATABASE_URL" in os.environ
    url = _database_url()
    if url:
        source = "DATABASE_URL"
        target = _redacted_target(url)
    else:
        source = "split-vars"
        target = _redacted_target(conninfo())
    return (
        f"code={CODE_MARK} file={__file__} "
        f"postgres={postgres_selected()} source={source} "
        f"DATABASE_URL_key_present={key_present} DATABASE_URL_len={len(url)} "
        f"target=({target}) "
        f"DB_BACKEND={os.getenv('DB_BACKEND', '').strip()!r} "
        f"DB_HOST={os.getenv('DB_HOST', '').strip()!r}"
    )
