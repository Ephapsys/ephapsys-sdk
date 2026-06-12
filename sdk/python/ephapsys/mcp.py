# SPDX-License-Identifier: Apache-2.0
"""MCP tool exposure for TrustedAgent (issue #125 / docs/A2A_MCP.md).

Exposes a governed agent's capabilities as MCP tools so external MCP-aware
clients (Claude, Autogen, LangChain, ...) can discover and call them.

Two layers, matching A2A_MCP.md's "unified tool abstraction":

  * ``ToolRegistry`` — the capability layer: ``register_tool`` /
    ``list_tools`` (MCP ``tools/list`` shape) / ``run_tool`` (``tools/call``).
  * ``MCPToolServer`` — wraps a ``TrustedAgent``, auto-registers its
    ``model_kind``s as tools, and serves them over MCP HTTP/JSON-RPC.

Governance: ``run_tool`` dispatches to ``agent.run(...)``, which runs the
SDK's fail-closed preflight (status / certs / attestation / kill-switch).
Tool calls therefore route *through* the governance gate, not around it —
a revoked agent's tools fail closed.

Transport scope: this ships a minimal HTTP/JSON-RPC ``serve_mcp`` using only
the standard library (the SDK keeps a minimal dependency set). It implements
the ``tools/list`` and ``tools/call`` methods. Dynamic discovery
(``notifications/tools/list_changed``) is intentionally out of scope here —
the tool list is static per server instance (see #125 open decision).
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional


def _default_input_schema(kind: str) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": f"Input passed to the agent's '{kind}' model.",
            }
        },
        "required": ["input"],
        "additionalProperties": False,
    }


class ToolRegistry:
    """Capability layer: register tools and discover/execute them.

    Descriptors follow the MCP ``tools/list`` shape
    (``{name, description, inputSchema}``). ``run_tool`` invokes the handler
    with the call's ``arguments`` as keyword arguments (the MCP ``tools/call``
    contract).
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register_tool(
        self,
        name: str,
        description: str,
        handler: Callable[..., Any],
        input_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not name:
            raise ValueError("tool name is required")
        self._tools[name] = {
            "description": description or "",
            "handler": handler,
            "input_schema": input_schema or {"type": "object", "properties": {}},
        }

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": name,
                "description": spec["description"],
                "inputSchema": spec["input_schema"],
            }
            for name, spec in self._tools.items()
        ]

    def run_tool(self, name: str, args: Optional[Dict[str, Any]] = None) -> Any:
        if name not in self._tools:
            raise ValueError(f"Tool {name!r} not registered")
        return self._tools[name]["handler"](**(args or {}))


class MCPToolServer(ToolRegistry):
    """Expose a ``TrustedAgent``'s capabilities as MCP tools.

    Parameters
    ----------
    agent:
        Object exposing ``run(input_data, model_kind=...)`` and (optionally)
        ``models()`` returning a list of model dicts carrying ``model_kind``.
    tools:
        Optional explicit tool specs. Each is a dict with ``name`` plus either
        ``model_kind`` (dispatch to ``agent.run(input, model_kind=...)``) or a
        ``handler`` callable, and optional ``description`` / ``input_schema``.
        When omitted, tools are derived from the agent's distinct
        ``model_kind``s.
    """

    def __init__(self, agent: Any, tools: Optional[List[Dict[str, Any]]] = None) -> None:
        super().__init__()
        self.agent = agent
        if tools is not None:
            for spec in tools:
                self._register_spec(spec)
        else:
            self._register_from_agent()

    def _register_spec(self, spec: Dict[str, Any]) -> None:
        name = spec.get("name")
        description = spec.get("description") or ""
        input_schema = spec.get("input_schema")
        handler = spec.get("handler")
        if handler is None:
            kind = (spec.get("model_kind") or name or "").strip().lower()
            description = description or f"Run the agent's '{kind}' model."
            input_schema = input_schema or _default_input_schema(kind)

            def handler(input, _kind=kind):  # noqa: A002 - matches inputSchema field
                return self.agent.run(input, model_kind=_kind)

        self.register_tool(name, description, handler, input_schema)

    def _register_from_agent(self) -> None:
        kinds: List[str] = []
        try:
            models = self.agent.models() or []
        except Exception:
            models = []
        for m in models:
            if not isinstance(m, dict):
                continue
            kind = (m.get("model_kind") or m.get("kind") or "").strip().lower()
            if kind and kind not in kinds:
                kinds.append(kind)
        for kind in kinds:
            self._register_spec({"name": kind, "model_kind": kind})

    # ---- MCP JSON-RPC ----

    def handle_jsonrpc(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a single MCP JSON-RPC request (``tools/list`` / ``tools/call``)."""
        rid = request.get("id")
        method = request.get("method")
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": rid, "result": {"tools": self.list_tools()}}
        if method == "tools/call":
            params = request.get("params") or {}
            name = params.get("name")
            args = params.get("arguments") or {}
            try:
                result = self.run_tool(name, args)
            except Exception as exc:  # noqa: BLE001 - tool errors are data, not protocol errors
                # MCP convention: execution errors are reported in the result
                # (isError), not as JSON-RPC protocol errors. A governance
                # failure (revoked/disabled) surfaces here as isError.
                return {
                    "jsonrpc": "2.0",
                    "id": rid,
                    "result": {"isError": True, "content": [{"type": "text", "text": str(exc)}]},
                }
            text = result if isinstance(result, str) else json.dumps(result, default=str)
            return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": text}]}}
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "error": {"code": -32601, "message": f"method not found: {method!r}"},
        }

    def make_http_server(self, host: str = "0.0.0.0", port: int = 8081):
        """Build (but don't start) an HTTPServer serving MCP JSON-RPC over POST.

        Returns the ``http.server.HTTPServer``; call ``.serve_forever()`` to run
        or ``.shutdown()`` to stop. Useful for tests and for callers that manage
        the lifecycle themselves.
        """
        from http.server import BaseHTTPRequestHandler, HTTPServer

        server = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - http.server API
                length = int(self.headers.get("Content-Length", "0") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    req = json.loads(raw or b"{}")
                except Exception:
                    self._send(400, {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}})
                    return
                self._send(200, server.handle_jsonrpc(req))

            def _send(self, code: int, obj: Dict[str, Any]) -> None:
                body = json.dumps(obj).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):  # silence default stderr logging
                return

        return HTTPServer((host, port), _Handler)

    def serve_mcp(self, host: str = "0.0.0.0", port: int = 8081) -> None:
        """Start a blocking MCP HTTP/JSON-RPC server (Ctrl-C / shutdown to stop)."""
        self.make_http_server(host, port).serve_forever()
