from conftest import a2a


class _Resp:
    def __init__(self, *, ok=True, status_code=200, payload=None, text=""):
        self.ok = ok
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def _build_get(cluster_members, sender_statuses):
    """sender_statuses maps agent_id -> {"status": ..., "state": {...}} or None for 503."""

    def _fake_get(url, headers=None, params=None, timeout=None):
        if "/clusters/" in url and not url.endswith("/members"):
            return _Resp(payload={"cluster": {"agent_refs": list(cluster_members)}})
        if "/agents/" in url and url.endswith("/status"):
            agent_id = url.split("/agents/")[1].split("/")[0]
            status = sender_statuses.get(agent_id)
            if status is None:
                return _Resp(ok=False, status_code=503, text="status unavailable")
            return _Resp(payload=status)
        return _Resp(ok=False, status_code=500, text=f"unexpected GET {url}")

    return _fake_get


def test_cluster_info_aggregates_member_health(monkeypatch):
    fake_get = _build_get(
        cluster_members=["alice", "bob", "carol", "dave", "eve"],
        sender_statuses={
            "alice": {"status": "ENABLED", "state": {}},
            "bob": {"status": "DISABLED", "state": {"enabled": False}},
            "carol": {"status": "REVOKED", "state": {"revoked": True}},
            "dave": {"status": "ENABLED", "state": {}},
            # eve omitted -> 503 -> unknown
        },
    )
    monkeypatch.setattr(a2a.requests, "get", fake_get)

    client = a2a.A2AClient(base_url="http://localhost:7001", token="tok")
    info = client.cluster_info(cluster_id="c1")

    assert info["health"] == {"healthy": 2, "degraded": 1, "revoked": 1, "unknown": 1}
    assert info["cluster"]["agent_refs"] == ["alice", "bob", "carol", "dave", "eve"]

    by_id = {m["agent_id"]: m for m in info["members"]}
    assert by_id["alice"]["status"] == "ENABLED"
    assert by_id["bob"]["status"] == "DISABLED"
    assert by_id["carol"]["status"] == "REVOKED"
    assert by_id["eve"]["status"] == "UNKNOWN"


def test_cluster_info_revoked_state_overrides_status(monkeypatch):
    # Even if status field is stale ENABLED, state.revoked=True still counts as revoked.
    fake_get = _build_get(
        cluster_members=["alice"],
        sender_statuses={"alice": {"status": "ENABLED", "state": {"revoked": True}}},
    )
    monkeypatch.setattr(a2a.requests, "get", fake_get)

    client = a2a.A2AClient(base_url="http://localhost:7001", token="tok")
    info = client.cluster_info(cluster_id="c1")
    assert info["health"]["revoked"] == 1
    assert info["health"]["healthy"] == 0


def test_cluster_info_empty_cluster(monkeypatch):
    fake_get = _build_get(cluster_members=[], sender_statuses={})
    monkeypatch.setattr(a2a.requests, "get", fake_get)

    client = a2a.A2AClient(base_url="http://localhost:7001", token="tok")
    info = client.cluster_info(cluster_id="c1")
    assert info["health"] == {"healthy": 0, "degraded": 0, "revoked": 0, "unknown": 0}
    assert info["members"] == []
