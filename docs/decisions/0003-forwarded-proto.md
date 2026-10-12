# 0003 — Trust X-Forwarded-Proto behind the container ingress

- **Status:** accepted (2026-10-12)

## Context

The app is served over HTTPS by Azure Container Apps. The ingress terminates TLS
and forwards plain HTTP to uvicorn, so `request.url.scheme` is `http`. Starlette's
`url_for()` then emits absolute `http://…` URLs. On an HTTPS page the browser
blocks those stylesheets and scripts as mixed content, and form actions / fetches
point at the wrong scheme.

uvicorn 0.24 (pinned) only honours forwarded headers from `127.0.0.1`, and its
`trusted_hosts="*"` wildcard does not exist in that version. The ingress is not
localhost.

## Decision

1. Static assets (`govuk.css`, `notgovuk.svg`, `ai_pipeline.js`) are root-relative
   (`/static/…`), so they inherit the page's scheme and host.
2. An outermost ASGI middleware copies `X-Forwarded-Proto` onto `scope["scheme"]`
   when the value is `http` or `https`. Every other `url_for()` link, redirect, and
   form action then matches the browser's scheme. Local requests with no header
   stay `http`.

## Consequences

- A client that can reach the container port directly can spoof the scheme. The
  container is only reached through the ingress; do not publish port 8600.
- Password-reset and other absolute links generated behind the proxy are `https`.
