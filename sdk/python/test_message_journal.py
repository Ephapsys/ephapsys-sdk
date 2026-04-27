import json

from conftest import journal_mod


def test_journal_append_and_read_round_trip(tmp_path):
    j = journal_mod.MessageJournal(path=str(tmp_path / "j.jsonl"))

    a = j.record(agent_id="me", message_id="m1", from_agent_id="alice", decision="verified")
    b = j.record(
        agent_id="me",
        message_id="m2",
        from_agent_id="bob",
        decision="rejected",
        reason="sender_revoked",
        ts=1700000000,
    )

    entries = list(j.read())
    assert len(entries) == 2
    assert entries[0]["message_id"] == "m1"
    assert entries[0]["decision"] == "verified"
    assert entries[0]["reason"] is None
    assert entries[1]["message_id"] == "m2"
    assert entries[1]["reason"] == "sender_revoked"
    assert entries[1]["ts"] == 1700000000
    assert a["agent_id"] == "me"
    assert b["from_agent_id"] == "bob"


def test_journal_read_filters_by_since_ts(tmp_path):
    j = journal_mod.MessageJournal(path=str(tmp_path / "j.jsonl"))
    j.record(agent_id="me", message_id="m1", decision="verified", ts=1000)
    j.record(agent_id="me", message_id="m2", decision="verified", ts=2000)
    j.record(agent_id="me", message_id="m3", decision="verified", ts=3000)

    after = [e["message_id"] for e in j.read(since_ts=2000)]
    assert after == ["m2", "m3"]


def test_journal_count_by_decision(tmp_path):
    j = journal_mod.MessageJournal(path=str(tmp_path / "j.jsonl"))
    for i in range(3):
        j.record(agent_id="me", message_id=f"v{i}", decision="verified")
    j.record(agent_id="me", message_id="r1", decision="rejected", reason="sender_revoked")
    j.record(agent_id="me", message_id="g1", decision="guardrail_blocked", reason="prompt_injection")
    j.record(agent_id="me", message_id="q1", decision="quarantine_alert")

    counts = j.count_by_decision()
    assert counts["verified"] == 3
    assert counts["rejected"] == 1
    assert counts["guardrail_blocked"] == 1
    assert counts["quarantine_alert"] == 1
    assert counts["system_event"] == 0


def test_journal_rejects_unknown_decision(tmp_path):
    j = journal_mod.MessageJournal(path=str(tmp_path / "j.jsonl"))
    try:
        j.record(agent_id="me", message_id="m1", decision="not_a_decision")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "unknown decision" in str(exc)


def test_journal_skips_malformed_lines(tmp_path):
    path = tmp_path / "j.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": 1, "agent_id": "me", "message_id": "m1", "decision": "verified"}) + "\n")
        fh.write("not-json\n")
        fh.write("\n")
        fh.write(json.dumps({"ts": 2, "agent_id": "me", "message_id": "m2", "decision": "verified"}) + "\n")

    j = journal_mod.MessageJournal(path=str(path))
    ids = [e["message_id"] for e in j.read()]
    assert ids == ["m1", "m2"]


def test_journal_from_env_uses_path_var(tmp_path, monkeypatch):
    target = tmp_path / "env-journal.jsonl"
    monkeypatch.setenv("EPHAPSYS_A2A_JOURNAL_PATH", str(target))
    j = journal_mod.MessageJournal.from_env()
    j.record(agent_id="me", message_id="m1", decision="verified")
    assert target.exists()
    assert "verified" in target.read_text()


def test_journal_creates_parent_directory(tmp_path):
    nested = tmp_path / "a" / "b" / "c" / "j.jsonl"
    j = journal_mod.MessageJournal(path=str(nested))
    j.record(agent_id="me", message_id="m1", decision="verified")
    assert nested.exists()
