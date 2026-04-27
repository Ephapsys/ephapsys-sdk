from conftest import a2a


class _Resp:
    def __init__(self, *, ok=True, status_code=200, payload=None, text=""):
        self.ok = ok
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class _Recorder:
    """Captures HTTP calls and returns canned responses keyed by (method, path)."""

    def __init__(self):
        self.calls = []
        self.responses = {}

    def respond(self, method, path_suffix, response):
        self.responses[(method.upper(), path_suffix)] = response

    def _resolve(self, method, url):
        for (m, suffix), resp in self.responses.items():
            if m == method.upper() and url.endswith(suffix):
                return resp
        return _Resp(ok=False, status_code=500, text=f"unexpected {method} {url}")

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append(("GET", url, params, None))
        return self._resolve("GET", url)

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append(("POST", url, None, json))
        return self._resolve("POST", url)

    def patch(self, url, headers=None, json=None, timeout=None):
        self.calls.append(("PATCH", url, None, json))
        return self._resolve("PATCH", url)

    def delete(self, url, headers=None, timeout=None):
        self.calls.append(("DELETE", url, None, None))
        return self._resolve("DELETE", url)


def _client_with(monkeypatch, recorder):
    monkeypatch.setattr(a2a.requests, "get", recorder.get)
    monkeypatch.setattr(a2a.requests, "post", recorder.post)
    monkeypatch.setattr(a2a.requests, "patch", recorder.patch)
    monkeypatch.setattr(a2a.requests, "delete", recorder.delete)
    return a2a.A2AClient(base_url="http://localhost:7001", token="tok")


def test_create_cluster_posts_payload(monkeypatch):
    rec = _Recorder()
    rec.respond("POST", "/clusters", _Resp(payload={"ok": True, "cluster": {"cluster_id": "c1"}}))
    client = _client_with(monkeypatch, rec)

    resp = client.create_cluster(
        label="Fleet",
        agent_ids=["alice", "bob"],
        policy={"propagate_status": True, "isolate_on_revoke": True, "require_signatures": False},
        cluster_id="c1",
    )
    assert resp["ok"] is True
    method, url, _, body = rec.calls[0]
    assert method == "POST"
    assert url.endswith("/clusters")
    assert body == {
        "label": "Fleet",
        "agent_ids": ["alice", "bob"],
        "cluster_id": "c1",
        "policy": {"propagate_status": True, "isolate_on_revoke": True, "require_signatures": False},
    }


def test_create_cluster_omits_optional_fields_when_none(monkeypatch):
    rec = _Recorder()
    rec.respond("POST", "/clusters", _Resp(payload={"ok": True, "cluster": {}}))
    client = _client_with(monkeypatch, rec)

    client.create_cluster(label="Fleet")
    _, _, _, body = rec.calls[0]
    assert body == {"label": "Fleet", "agent_ids": []}


def test_list_clusters_passes_agent_filter(monkeypatch):
    rec = _Recorder()
    rec.respond("GET", "/clusters", _Resp(payload={"items": []}))
    client = _client_with(monkeypatch, rec)

    client.list_clusters(agent_id="alice")
    _, _, params, _ = rec.calls[0]
    assert params == {"agent_id": "alice"}

    client.list_clusters()
    _, _, params2, _ = rec.calls[1]
    assert params2 is None


def test_get_cluster(monkeypatch):
    rec = _Recorder()
    rec.respond("GET", "/clusters/c1", _Resp(payload={"cluster": {"cluster_id": "c1"}}))
    client = _client_with(monkeypatch, rec)

    resp = client.get_cluster(cluster_id="c1")
    assert resp["cluster"]["cluster_id"] == "c1"
    assert rec.calls[0][1].endswith("/clusters/c1")


def test_update_cluster_only_sends_provided_fields(monkeypatch):
    rec = _Recorder()
    rec.respond("PATCH", "/clusters/c1", _Resp(payload={"ok": True, "cluster": {}}))
    client = _client_with(monkeypatch, rec)

    client.update_cluster(cluster_id="c1", label="renamed")
    _, _, _, body = rec.calls[0]
    assert body == {"label": "renamed"}

    client.update_cluster(
        cluster_id="c1",
        agent_ids=["alice"],
        policy={"propagate_status": False, "isolate_on_revoke": True, "require_signatures": False},
    )
    _, _, _, body2 = rec.calls[1]
    assert "label" not in body2
    assert body2["agent_ids"] == ["alice"]
    assert body2["policy"]["propagate_status"] is False


def test_delete_cluster(monkeypatch):
    rec = _Recorder()
    rec.respond("DELETE", "/clusters/c1", _Resp(payload={"ok": True}))
    client = _client_with(monkeypatch, rec)

    resp = client.delete_cluster(cluster_id="c1")
    assert resp["ok"] is True


def test_add_and_remove_member(monkeypatch):
    rec = _Recorder()
    rec.respond("POST", "/clusters/c1/members", _Resp(payload={"ok": True, "cluster": {}}))
    rec.respond(
        "DELETE", "/clusters/c1/members/alice", _Resp(payload={"ok": True, "cluster": {}})
    )
    client = _client_with(monkeypatch, rec)

    client.add_member(cluster_id="c1", agent_id="alice")
    _, url, _, body = rec.calls[0]
    assert url.endswith("/clusters/c1/members")
    assert body == {"agent_id": "alice"}

    client.remove_member(cluster_id="c1", agent_ref="alice")
    _, url2, _, _ = rec.calls[1]
    assert url2.endswith("/clusters/c1/members/alice")


def test_join_and_leave_are_aliases(monkeypatch):
    rec = _Recorder()
    rec.respond("POST", "/clusters/c1/members", _Resp(payload={"ok": True}))
    rec.respond("DELETE", "/clusters/c1/members/alice", _Resp(payload={"ok": True}))
    client = _client_with(monkeypatch, rec)

    client.join_cluster(cluster_id="c1", agent_id="alice")
    client.leave_cluster(cluster_id="c1", agent_id="alice")
    methods = [c[0] for c in rec.calls]
    assert methods == ["POST", "DELETE"]


def test_create_cluster_raises_on_http_error(monkeypatch):
    rec = _Recorder()
    rec.respond("POST", "/clusters", _Resp(ok=False, status_code=409, text="duplicate"))
    client = _client_with(monkeypatch, rec)

    try:
        client.create_cluster(label="dup", cluster_id="x")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "409" in str(exc)
        assert "duplicate" in str(exc)
