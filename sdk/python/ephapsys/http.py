# SPDX-License-Identifier: Apache-2.0
import json, ssl, os
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

try:
    import requests  # type: ignore
except Exception:
    requests = None

def _join(base: str, path: str) -> str:
    if base.endswith("/") and path.startswith("/"):
        return base[:-1] + path
    if not base.endswith("/") and not path.startswith("/"):
        return base + "/" + path
    return base + path

def request(method: str, base_url: str, path: str, headers=None, params=None, json_body=None, data=None, timeout=15, verify_ssl=True):
    url = _join(base_url, path)
    # Enforce outbound allowlist if configured (lazy import to avoid circular dependency)
    raw_allow = os.getenv("AOC_NETWORK_ALLOWLIST", "")
    allowed = [t.strip() for t in raw_allow.split(",") if t.strip()]
    if allowed:
        try:
            from .agent import _enforce_network_allowlist  # type: ignore
            _enforce_network_allowlist(url, allowed)
        except Exception as e:
            raise

    if params:
        q = urlencode(params)
        url = url + ("&" if "?" in url else "?") + q
    headers = headers or {}
    if requests:
        try:
            r = requests.request(method.upper(), url, headers=headers, json=json_body, data=data, timeout=timeout, verify=verify_ssl)
            r.raise_for_status()
            if r.content:
                try:
                    return r.json()
                except Exception:
                    return r.text
            return None
        except Exception as e:
            raise RuntimeError(f"HTTP error: {e}")
    # urllib fallback
    data_bytes = None
    if json_body is not None:
        headers.setdefault("Content-Type", "application/json")
        data_bytes = json.dumps(json_body).encode("utf-8")
    elif data is not None:
        data_bytes = data if isinstance(data, (bytes, bytearray)) else str(data).encode("utf-8")
    req = Request(url, data=data_bytes, method=method.upper(), headers=headers)
    ctx = None
    if not verify_ssl:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    try:
        with urlopen(req, context=ctx, timeout=timeout) as resp:
            content = resp.read()
            if not content:
                return None
            try:
                return json.loads(content.decode("utf-8"))
            except Exception:
                return content.decode("utf-8", "ignore")
    except HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.reason}")
    except URLError as e:
        raise RuntimeError(f"HTTP error: {e.reason}")


    def get_bytes(self, url: str, headers: dict | None = None, timeout: int = 15) -> bytes:
        import requests
        h = self._auth_headers()
        if headers: h.update(headers)
        r = requests.get(url, headers=h, timeout=timeout)
        r.raise_for_status()
        return r.content
    
