from conftest import a2a, agent_mod


class _Resp:
    def __init__(self, *, ok=True, status_code=200, payload=None, text=""):
        self.ok = ok
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


# -- _resolve_cluster_join_targets ----------------------------------------


def test_resolve_targets_explicit_arg_wins(monkeypatch):
    monkeypatch.setenv("EPHAPSYS_AGENT_CLUSTER_IDS", "from-env")
    out = agent_mod._resolve_cluster_join_targets(["fleet-alpha", "fleet-bravo"])
    assert out == ["fleet-alpha", "fleet-bravo"]


def test_resolve_targets_explicit_empty_list_skips_env(monkeypatch):
    # An explicit empty list is still "the user said no clusters".
    monkeypatch.setenv("EPHAPSYS_AGENT_CLUSTER_IDS", "from-env")
    out = agent_mod._resolve_cluster_join_targets([])
    assert out == []


def test_resolve_targets_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("EPHAPSYS_AGENT_CLUSTER_IDS", "fleet-alpha, fleet-bravo,, fleet-alpha")
    out = agent_mod._resolve_cluster_join_targets(None)
    # Trimmed, empties skipped, deduped.
    assert out == ["fleet-alpha", "fleet-bravo"]


def test_resolve_targets_no_arg_no_env_returns_empty(monkeypatch):
    monkeypatch.delenv("EPHAPSYS_AGENT_CLUSTER_IDS", raising=False)
    assert agent_mod._resolve_cluster_join_targets(None) == []


def test_resolve_targets_dedupes_explicit_arg(monkeypatch):
    out = agent_mod._resolve_cluster_join_targets(["c1", "c1", " c2 ", "", "c2"])
    assert out == ["c1", "c2"]


# -- _auto_join_clusters --------------------------------------------------


def test_auto_join_calls_join_for_each_cluster(monkeypatch):
    join_calls = []

    def _fake_post(url, headers=None, json=None, timeout=None):
        if url.endswith("/members"):
            cluster_id = url.split("/clusters/")[1].split("/")[0]
            join_calls.append({"cluster_id": cluster_id, "agent_id": json["agent_id"]})
            return _Resp(payload={"ok": True, "cluster": {}})
        return _Resp(ok=False, status_code=500, text="unexpected")

    monkeypatch.setattr(a2a.requests, "post", _fake_post)

    results = agent_mod._auto_join_clusters(
        api_base="http://localhost:7001",
        api_key="tok",
        agent_id="my-agent",
        cluster_ids=["fleet-alpha", "fleet-bravo"],
    )
    assert results == {"fleet-alpha": True, "fleet-bravo": True}
    assert join_calls == [
        {"cluster_id": "fleet-alpha", "agent_id": "my-agent"},
        {"cluster_id": "fleet-bravo", "agent_id": "my-agent"},
    ]


def test_auto_join_continues_after_individual_failure(monkeypatch):
    def _fake_post(url, headers=None, json=None, timeout=None):
        if url.endswith("/members"):
            cluster_id = url.split("/clusters/")[1].split("/")[0]
            if cluster_id == "missing-cluster":
                return _Resp(ok=False, status_code=404, text="cluster not found")
            return _Resp(payload={"ok": True})
        return _Resp(ok=False, status_code=500, text="unexpected")

    monkeypatch.setattr(a2a.requests, "post", _fake_post)

    results = agent_mod._auto_join_clusters(
        api_base="http://localhost:7001",
        api_key="tok",
        agent_id="my-agent",
        cluster_ids=["fleet-alpha", "missing-cluster", "fleet-bravo"],
    )
    assert results == {
        "fleet-alpha": True,
        "missing-cluster": False,
        "fleet-bravo": True,
    }


def test_auto_join_noop_when_cluster_ids_empty(monkeypatch):
    def _explode(*args, **kwargs):
        raise AssertionError("no HTTP calls expected for empty cluster_ids")
    monkeypatch.setattr(a2a.requests, "post", _explode)

    results = agent_mod._auto_join_clusters(
        api_base="http://localhost:7001",
        api_key="tok",
        agent_id="my-agent",
        cluster_ids=[],
    )
    assert results == {}


def test_auto_join_skips_when_required_args_missing(monkeypatch):
    def _explode(*args, **kwargs):
        raise AssertionError("no HTTP calls expected when args incomplete")
    monkeypatch.setattr(a2a.requests, "post", _explode)

    results = agent_mod._auto_join_clusters(
        api_base="",  # missing
        api_key="tok",
        agent_id="my-agent",
        cluster_ids=["c1"],
    )
    assert results == {"c1": False}


def test_auto_join_swallows_unexpected_exceptions(monkeypatch):
    def _boom(url, headers=None, json=None, timeout=None):
        raise RuntimeError("network exploded")
    monkeypatch.setattr(a2a.requests, "post", _boom)

    results = agent_mod._auto_join_clusters(
        api_base="http://localhost:7001",
        api_key="tok",
        agent_id="my-agent",
        cluster_ids=["c1", "c2"],
    )
    # Failures are recorded as False; no exception escapes.
    assert results == {"c1": False, "c2": False}
