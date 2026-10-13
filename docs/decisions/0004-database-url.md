# 0004 — Accept a single Postgres connection string

- **Status:** accepted (2026-10-13)

## Context

Deployments (Azure Container Apps) inject one Postgres connection string. The app
only read `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, and `DB_PASSWORD`, and it
selected Postgres when `DB_HOST` or `DB_BACKEND=postgres` was set. A URI also
carries `sslmode`, which the split vars dropped.

## Decision

`DATABASE_URL`, when non-blank, is the conninfo passed to `psycopg.connect` and
it selects the Postgres backend on its own. URI (`postgresql://…`, `postgres://…`)
and libpq keyword form both pass through unchanged, including query parameters
such as `sslmode`. When it is unset, the split `DB_*` vars behave as before.
`DB_STATEMENT_TIMEOUT_MS` still applies on top of either form.

## Consequences

- A process with `DATABASE_URL` set will not fall back to SQLite, same as `DB_HOST`.
- The split vars are ignored for the connection while `DATABASE_URL` is set. Host
  overrides have to be done on the URL itself.
