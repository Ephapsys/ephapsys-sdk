import importlib.util
from pathlib import Path

import requests


_HTTP_PATH = Path(__file__).parent / "ephapsys" / "http.py"
_SPEC = importlib.util.spec_from_file_location("ephapsys_http_module", _HTTP_PATH)
assert _SPEC and _SPEC.loader
http = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(http)


class _Resp:
    status_code = 403
    text = '{"detail":"anchor=none is disabled"}'
    content = b'{"detail":"anchor=none is disabled"}'

    def raise_for_status(self):
        raise requests.HTTPError("403 Client Error", response=self)


def test_request_includes_response_body_on_requests_http_error(monkeypatch):
    monkeypatch.setattr(http.requests, "request", lambda *args, **kwargs: _Resp())

    try:
        http.request("GET", "https://api.example.com", "/x")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        msg = str(exc)
        assert "HTTP error" in msg
        assert "Response body" in msg
        assert "anchor=none is disabled" in msg
