#!/usr/bin/env python3
"""Tests for MCP tool exposure (issue #125): ToolRegistry + MCPToolServer.

Fully offline — uses a duck-typed FakeAgent, so no torch/AOC needed.

Usage:
    cd ephapsys-sdk
    PYTHONPATH=sdk/python python3 tests/test_mcp_toolserver.py
    # or: pytest tests/test_mcp_toolserver.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdk", "python"))

from ephapsys.mcp import MCPToolServer, ToolRegistry  # noqa: E402


class FakeAgent:
    """Duck-typed stand-in for TrustedAgent (run + models)."""

    def __init__(self, kinds, revoked=False):
        self._kinds = kinds
        self._revoked = revoked
        self.calls = []

    def models(self):
        return [{"model_kind": k} for k in self._kinds]

    def run(self, input_data, model_kind, **kwargs):
        if self._revoked:
            raise RuntimeError("Agent revoked; inference blocked")
        self.calls.append((model_kind, input_data))
        return f"ran:{model_kind}:{input_data}"


def test_registry_basic():
    r = ToolRegistry()
    r.register_tool("echo", "Echo upper", lambda input: input.upper(), {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]})
    tools = r.list_tools()
    assert tools == [{"name": "echo", "description": "Echo upper", "inputSchema": {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}}], tools
    assert r.run_tool("echo", {"input": "hi"}) == "HI"
    try:
        r.run_tool("missing", {})
    except ValueError as e:
        assert "not registered" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown tool")


def test_derives_tools_from_agent_model_kinds():
    srv = MCPToolServer(FakeAgent(["language", "stt"]))
    names = {t["name"] for t in srv.list_tools()}
    assert names == {"language", "stt"}, names
    for t in srv.list_tools():
        assert t["inputSchema"]["required"] == ["input"], t


def test_run_tool_dispatches_through_agent_run():
    agent = FakeAgent(["language"])
    srv = MCPToolServer(agent)
    out = srv.run_tool("language", {"input": "hello"})
    assert out == "ran:language:hello", out
    assert agent.calls == [("language", "hello")]


def test_jsonrpc_tools_list_and_call():
    srv = MCPToolServer(FakeAgent(["language", "stt"]))
    listed = srv.handle_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert len(listed["result"]["tools"]) == 2, listed
    called = srv.handle_jsonrpc(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "stt", "arguments": {"input": "a.wav"}}}
    )
    assert called["result"]["content"][0]["text"] == "ran:stt:a.wav", called


def test_jsonrpc_unknown_tool_is_error_result():
    srv = MCPToolServer(FakeAgent(["language"]))
    resp = srv.handle_jsonrpc(
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "nope", "arguments": {}}}
    )
    assert resp["result"]["isError"] is True, resp
    assert "not registered" in resp["result"]["content"][0]["text"]


def test_jsonrpc_unknown_method_is_protocol_error():
    srv = MCPToolServer(FakeAgent(["language"]))
    resp = srv.handle_jsonrpc({"jsonrpc": "2.0", "id": 4, "method": "resources/list"})
    assert resp["error"]["code"] == -32601, resp


def test_governance_failure_surfaces_as_iserror():
    # A revoked agent's run() raises -> tools/call returns isError (fail-closed),
    # i.e. tool calls route THROUGH the governance gate.
    srv = MCPToolServer(FakeAgent(["language"], revoked=True))
    resp = srv.handle_jsonrpc(
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "language", "arguments": {"input": "x"}}}
    )
    assert resp["result"]["isError"] is True, resp
    assert "revoked" in resp["result"]["content"][0]["text"].lower()


def test_explicit_tool_specs():
    srv = MCPToolServer(
        FakeAgent(["language"]),
        tools=[{"name": "shout", "handler": lambda input: input.upper(), "description": "Uppercase", "input_schema": {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}}],
    )
    assert {t["name"] for t in srv.list_tools()} == {"shout"}
    assert srv.run_tool("shout", {"input": "hi"}) == "HI"


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
