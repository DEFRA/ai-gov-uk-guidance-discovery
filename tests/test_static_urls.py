"""Static assets stay root-relative, and X-Forwarded-Proto fixes absolute links.

Azure Container Apps terminates TLS, so the app sees HTTP. url_for() then emits
http:// URLs that the browser blocks as mixed content. Run:
python3 -m unittest -v tests.test_static_urls
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import fastapi  # noqa: F401
    _HAS_WEBAPP = True
except Exception:
    _HAS_WEBAPP = False


@unittest.skipUnless(_HAS_WEBAPP, "web app deps (fastapi) not installed")
class TestStaticUrls(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".db"); os.close(fd)
        os.environ["CORPUS_DB"] = self.path
        os.environ.pop("DB_HOST", None)
        os.environ.pop("DATABASE_URL", None)
        os.environ.pop("DASHBOARD_PASSWORD", None)
        import importlib
        import webapp.app as app
        importlib.reload(app)
        self.app = app
        conn = app.connect(); app.db.init_db(conn); conn.close()

    def tearDown(self):
        if os.path.exists(self.path):
            os.unlink(self.path)

    def _client(self):
        from fastapi.testclient import TestClient
        return TestClient(self.app.app)

    def test_stylesheet_and_logo_are_root_relative(self):
        r = self._client().get("/profile")
        self.assertEqual(r.status_code, 200)
        self.assertIn('href="/static/govuk.css"', r.text)
        self.assertIn('src="/static/notgovuk.svg"', r.text)
        self.assertNotIn("http://testserver/static/", r.text)
        self.assertNotIn("https://testserver/static/", r.text)

    def test_forwarded_proto_https_rewrites_absolute_links(self):
        r = self._client().get("/profile", headers={"x-forwarded-proto": "https"})
        self.assertIn('href="https://testserver/sustainability"', r.text)
        self.assertIn('href="/static/govuk.css"', r.text)

    def test_forwarded_proto_takes_first_value(self):
        r = self._client().get("/profile", headers={"x-forwarded-proto": "https,http"})
        self.assertIn('href="https://testserver/sustainability"', r.text)

    def test_no_forwarded_proto_stays_http(self):
        r = self._client().get("/profile")
        self.assertIn('href="http://testserver/sustainability"', r.text)

    def test_bogus_forwarded_proto_is_ignored(self):
        r = self._client().get("/profile", headers={"x-forwarded-proto": "javascript"})
        self.assertIn('href="http://testserver/sustainability"', r.text)


if __name__ == "__main__":
    unittest.main()
