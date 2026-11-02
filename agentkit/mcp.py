"""A minimal MCP client: start a tool server, list its tools, call them.

MCP (Model Context Protocol) is how one tool server can serve many harnesses: the same server
plugs into agentkit (this file), LM Studio's chat app (mcp.json) and Claude Code (--mcp-config).
The wire format is JSON-RPC 2.0, one JSON object per line on the server's stdin/stdout.
"""
import json
import os
import subprocess
import threading


class MCPServer:
    """One stdio MCP server process. Thread-safe: worker threads share it, one request at a time."""

    def __init__(self, command: list[str], env: dict | None = None, cwd=None, log=None):
        """`env` adds to (not replaces) this process's environment; `log` is a file path for the server's stderr."""
        self._log = open(log, "a") if log else subprocess.DEVNULL
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                                     text=True, bufsize=1, env={**os.environ, **(env or {})}, cwd=cwd)
        self._lock, self._next_id = threading.Lock(), 0
        self.info = self._request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                                 "clientInfo": {"name": "agentkit", "version": "1.0"}})
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.tools = self._request("tools/list", {}).get("tools", [])

    def _send(self, msg):
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def _request(self, method, params):
        with self._lock:
            self._next_id += 1
            self._send({"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params})
            while True:  # skip notifications and log messages until our answer arrives
                line = self.proc.stdout.readline()
                if not line:
                    raise RuntimeError(f"MCP server exited during {method}")
                msg = json.loads(line)
                if msg.get("id") == self._next_id:
                    if "error" in msg:
                        raise RuntimeError(f"MCP {method}: {msg['error'].get('message')}")
                    return msg["result"]

    def call(self, name, arguments):
        """Call a tool. Returns its structured result if it has one, else its text."""
        res = self._request("tools/call", {"name": name, "arguments": arguments})
        if res.get("isError"):
            return {"error": " ".join(c.get("text", "") for c in res.get("content", []))}
        if res.get("structuredContent") is not None:
            out = res["structuredContent"]
            return out["result"] if set(out) == {"result"} else out   # unwrap non-dict returns
        text = " ".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc.stdout.close()
        if self._log is not subprocess.DEVNULL:
            self._log.close()
