"""Message your laptop's local assistant from your phone: a small, locked-down HTTP server (standard library only).

The phone gets a chat page. Every message goes through one router (Gemma 4 12B, constrained to six labels) to
one of five fixed handlers, or to a polite refusal. Calendar adds come back as a proposal with Confirm/Cancel;
only a tap on Confirm writes, and only to a sandbox copy of the calendar. Nothing is ever sent or paid.

Run (from the repo root, LM Studio running with google/gemma-4-12b loaded):
  .venv/bin/python assistant/server.py                    http://127.0.0.1:8787, this Mac only (the default)
  .venv/bin/python assistant/server.py --demo-samples     also offer the sample receipts in the photo picker
  .venv/bin/python assistant/server.py --host 100.x.y.z   your Tailscale (or home Wi-Fi) address: opt in, read the warning
  .venv/bin/python assistant/server.py --show-token       print the access token again

Security:
  - binds 127.0.0.1 unless you pass --host; refuses public addresses, and 0.0.0.0 unless --all-interfaces
  - a shared secret token, made on first run, stored 0600 in the state folder (default ~/.config/speed-of-thought-
    assistant, outside the repo). The phone sends it once on the sign-in screen (never in a URL); the server sets
    an HttpOnly, SameSite=Strict cookie. Scripts can send it as the X-Assistant-Token header. Compared in constant time.
  - Host header allow-list (stops DNS rebinding), Origin check and JSON-only POSTs (stops cross-site requests)
  - rate limits per client: 240 requests/min, 20 messages/min, 5 bad tokens per 10 min (then 429), 50 bad tokens
    per 10 min across all clients
  - uploads: 9 MB request body, 6 MB photo, JPEG/PNG/WebP only, re-encoded to a PNG with no EXIF (no GPS)
  - audit log of every request and action: <state folder>/audit.jsonl (never the token)
  - the models get no tools; code holds list_events and create_event, and only a Confirm tap calls create_event

State folder (override with ASSISTANT_STATE_DIR or --state-dir): token, audit.jsonl, calendar.json (the sandbox
calendar, copied from sample_data/inbox/calendar_before.json on first run), gcal_server.log.
Phone access, the Telegram adapter, and keeping it running: see assistant/README.md.
"""
import argparse
import base64
import binascii
import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import signal
import sys
import threading
from collections import deque
from datetime import datetime
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from assistant import core  # noqa: E402

STATIC = HERE / "static"
STATIC_FILES = {"/": ("index.html", "text/html; charset=utf-8"),
                "/index.html": ("index.html", "text/html; charset=utf-8"),
                "/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/static/style.css": ("style.css", "text/css; charset=utf-8"),
                "/static/icon.svg": ("icon.svg", "image/svg+xml"),
                "/manifest.webmanifest": ("manifest.webmanifest", "application/manifest+json")}
SAMPLES = core.ROOT / "sample_data" / "receipts"
MAX_BODY = 9 * 1024 * 1024
COOKIE = "assistant_session"
TOKEN_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"   # no 0/o, 1/l/i: easy to type on a phone
CGNAT = ipaddress.ip_network("100.64.0.0/10")         # Tailscale addresses live here
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; "
                               "connect-src 'self'; manifest-src 'self'; base-uri 'none'; form-action 'self'; "
                               "frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "geolocation=(), microphone=(), payment=()", "Cross-Origin-Opener-Policy": "same-origin"}


def default_state_dir():
    return Path(os.environ.get("ASSISTANT_STATE_DIR") or Path.home() / ".config" / "speed-of-thought-assistant")


def write_private(path: Path, text: str):
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)
    os.chmod(path, 0o600)


def load_or_create_token(state_dir: Path, rotate=False):
    """(token, created). Six groups of four from a 31-letter alphabet: about 119 bits."""
    path = state_dir / "token"
    if path.exists() and not rotate:
        token = path.read_text().strip()
        if len(token) >= 20:
            os.chmod(path, 0o600)
            return token, False
    token = "-".join("".join(secrets.choice(TOKEN_ALPHABET) for _ in range(4)) for _ in range(6))
    write_private(path, token + "\n")
    return token, True


def normalize_token(text):
    return "".join(str(text).split()).lower()


def bind_kind(host, all_interfaces=False):
    """'loopback' or 'network'; raises SystemExit for addresses a student should not expose by accident."""
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        raise SystemExit(f"--host must be an IP address (e.g. your Tailscale 100.x.y.z or Wi-Fi 192.168.x.y), not {host!r}")
    if ip.is_loopback:
        return "loopback"
    if ip.is_unspecified and not all_interfaces:
        raise SystemExit(f"{host} listens on every network this Mac joins, café Wi-Fi included. Pass the one address "
                         "you mean (Tailscale or home Wi-Fi), or add --all-interfaces if you really want this.")
    if not (ip.is_unspecified or ip.is_private or ip.is_link_local or (ip.version == 4 and ip in CGNAT)):
        raise SystemExit(f"{host} is a public address; this server is for a private network only.")
    return "network"


class App:
    """Everything the request handler shares: the assistant, the token, limits, history and the audit log."""

    def __init__(self, assistant, token, state_dir, allowed_hosts, demo_samples=False, secure_cookie=False,
                 clock_label=""):
        self.assistant, self._token = assistant, normalize_token(token)
        self.state_dir = Path(state_dir)
        self.allowed_hosts = {h.lower() for h in allowed_hosts}
        self.demo_samples, self.secure_cookie, self.clock_label = demo_samples, secure_cookie, clock_label
        self.audit = core.AuditLog(self.state_dir / "audit.jsonl")
        self.limit_all = core.RateLimiter(240, 60)
        self.limit_msg = core.RateLimiter(20, 60)
        self.bad_auth = core.RateLimiter(5, 600)
        self.bad_auth_all = core.RateLimiter(50, 600)
        self.busy = threading.BoundedSemaphore(2)       # at most two messages at the models at once
        self.history = deque(maxlen=80)
        self.lock = threading.Lock()
        self.samples = sorted(p.stem for p in SAMPLES.glob("r*.png")) if demo_samples else []

    def token_ok(self, given):
        return hmac.compare_digest(normalize_token(given).encode(), self._token.encode())

    def locked_out(self, ip):
        return self.bad_auth.count(ip) >= self.bad_auth.limit or self.bad_auth_all.count("*") >= self.bad_auth_all.limit

    def record_bad(self, ip):
        self.bad_auth.allow(ip)
        self.bad_auth_all.allow("*")

    def remember(self, entry):
        with self.lock:
            self.history.append(entry)

    def mark_proposal(self, pid, status):
        with self.lock:
            for h in self.history:
                if (h.get("proposal") or {}).get("id") == pid:
                    h["proposal"]["status"] = status


class Handler(BaseHTTPRequestHandler):
    server_version = "local-assistant"
    sys_version = ""
    timeout = 30                    # a slow client cannot hold a thread forever

    def log_message(self, fmt, *args):   # the audit log replaces the default stderr log
        pass

    @property
    def app(self) -> App:
        return self.server.app

    # ------------------------------------------------------------ plumbing
    def _send(self, status, body=b"", ctype="application/json", headers=None):
        self.send_response(status)
        for k, v in {**SECURITY_HEADERS, **(headers or {})}.items():
            self.send_header(k, v)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
        self._status = status

    def _json(self, status, obj, headers=None):
        self._send(status, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8", headers)

    def _audit(self, status=None, **extra):
        self.app.audit.write(ip=self.client_address[0], method=self.command, path=urlsplit(self.path).path[:80],
                             status=status or getattr(self, "_status", None), **extra)

    def _host_ok(self):
        host = (self.headers.get("Host") or "").lower()
        name = host.rsplit("]", 1)[0] + "]" if host.startswith("[") else host.split(":", 1)[0]
        return name in self.app.allowed_hosts

    def _origin_ok(self):
        origin = self.headers.get("Origin")
        if not origin:
            return True                  # same-origin fetches from some browsers, curl, the Telegram-free tests
        return (urlsplit(origin).hostname or "").lower() in self.app.allowed_hosts

    def _auth(self):
        given = self.headers.get("X-Assistant-Token")
        if given is None and self.headers.get("Cookie"):
            jar = cookies.SimpleCookie()
            try:
                jar.load(self.headers["Cookie"])
            except cookies.CookieError:
                pass
            if COOKIE in jar:
                given = jar[COOKIE].value
        if given is None:
            return "missing"
        return "ok" if self.app.token_ok(given) else "bad"

    def _read_json(self):
        if not (self.headers.get("Content-Type") or "").startswith("application/json"):
            self._json(415, {"error": "send JSON"})
            return None
        length = self.headers.get("Content-Length")
        if length is None or not length.isdigit():
            self._json(411, {"error": "Content-Length required"})
            return None
        if int(length) > MAX_BODY:
            self.close_connection = True          # do not read the body
            self._json(413, {"error": f"request larger than {MAX_BODY // (1024 * 1024)} MB"})
            return None
        try:
            body = json.loads(self.rfile.read(int(length)) or b"{}")
        except (ValueError, UnicodeDecodeError):
            self._json(400, {"error": "malformed JSON"})
            return None
        if not isinstance(body, dict):
            self._json(400, {"error": "send a JSON object"})
            return None
        return body

    def _gate(self, need_auth=True):
        """Host, rate limit and (optionally) token checks. Returns the auth state, or None if already answered."""
        ip = self.client_address[0]
        if not self._host_ok():
            self._json(421, {"error": "unknown host name; start the server with --allow-hostname for it"})
            return None
        if not self.app.limit_all.allow(ip):
            self._json(429, {"error": "slow down"}, {"Retry-After": "60"})
            return None
        if not need_auth:
            return "n/a"
        if self.app.locked_out(ip):
            self._json(429, {"error": "too many bad tokens; wait 10 minutes"}, {"Retry-After": "600"})
            return None
        state = self._auth()
        if state == "bad":
            self.app.record_bad(ip)
        if state != "ok":
            self._json(401, {"error": "sign in with your token"})
            return None
        return state

    # ------------------------------------------------------------ GET
    def do_GET(self):
        path = urlsplit(self.path).path
        if path in STATIC_FILES:
            if self._gate(need_auth=False) is None:
                return self._audit(auth="n/a")
            name, ctype = STATIC_FILES[path]
            self._send(200, (STATIC / name).read_bytes(), ctype)
            return self._audit(auth="n/a")
        if path == "/api/state":
            auth = self._gate()
            if auth is not None:
                with self.app.lock:
                    history = list(self.app.history)
                self._json(200, {"clock": self.app.clock_label, "capabilities": [c for _, c in core.CAPABILITIES],
                                 "samples": self.app.samples, "history": history})
            return self._audit(auth=auth or "rejected")
        if path.startswith("/api/sample/"):
            auth = self._gate()
            if auth is not None:
                name = path.rsplit("/", 1)[1].removesuffix(".png")
                if name in self.app.samples:                     # allow-list: never a path from the request
                    self._send(200, (SAMPLES / f"{name}.png").read_bytes(), "image/png")
                else:
                    self._json(404, {"error": "no such sample"})
            return self._audit(auth=auth or "rejected", event="sample")
        if self._gate(need_auth=False) is not None:
            self._json(404, {"error": "not found"})
        self._audit(auth="n/a")

    do_HEAD = do_GET

    # ------------------------------------------------------------ POST
    def do_POST(self):
        path = urlsplit(self.path).path
        if path not in ("/api/login", "/api/logout", "/api/message", "/api/decide"):
            if self._gate(need_auth=False) is not None:
                self._json(404, {"error": "not found"})
            return self._audit(auth="n/a")
        if not self._origin_ok():
            self._json(403, {"error": "cross-site request refused"})
            return self._audit(auth="n/a", event="bad origin", origin=self.headers.get("Origin", "")[:80])
        if path == "/api/login":
            return self._login()
        auth = self._gate()
        if auth is None:
            return self._audit(auth="rejected")
        if path == "/api/logout":
            self._json(200, {"ok": True}, {"Set-Cookie": f"{COOKIE}=; Max-Age=0; Path=/; HttpOnly; SameSite=Strict"})
            return self._audit(auth=auth, event="logout")
        if not self.app.limit_msg.allow(self.client_address[0]):
            self._json(429, {"error": "too many messages; wait a minute"}, {"Retry-After": "60"})
            return self._audit(auth=auth, event="rate limited")
        body = self._read_json()
        if body is None:
            return self._audit(auth=auth, event="bad body")
        return self._message(body) if path == "/api/message" else self._decide(body)

    def _login(self):
        ip = self.client_address[0]
        if self._gate(need_auth=False) is None:
            return self._audit(auth="n/a", event="login")
        if self.app.locked_out(ip):
            self._json(429, {"error": "too many bad tokens; wait 10 minutes"}, {"Retry-After": "600"})
            return self._audit(auth="locked out", event="login")
        body = self._read_json()
        if body is None:
            return self._audit(auth="n/a", event="login")
        if not self.app.token_ok(str(body.get("token", ""))):
            self.app.record_bad(ip)
            self._json(401, {"error": "that token is not right"})
            return self._audit(auth="bad", event="login")
        flags = "; Secure" if self.app.secure_cookie else ""
        cookie = f"{COOKIE}={self.app._token}; Max-Age=2592000; Path=/; HttpOnly; SameSite=Strict{flags}"
        self._json(200, {"ok": True}, {"Set-Cookie": cookie})
        self._audit(auth="ok", event="login")

    def _message(self, body):
        text = body.get("text", "")
        if not isinstance(text, str) or len(text) > 4 * core.MAX_TEXT:
            self._json(413, {"error": "message too long"})
            return self._audit(auth="ok", event="message", outcome="too long")
        photo, photo_info = None, {}
        raw = body.get("photo")
        if raw:
            try:
                if not isinstance(raw, str):
                    raise core.UploadError("photo must be base64 text")
                data = base64.b64decode(raw.split(",", 1)[1] if raw.startswith("data:") else raw, validate=True)
                photo = core.sanitize_image(data)
                photo_info = {"photo_bytes": len(data), "photo_sha256": hashlib.sha256(data).hexdigest()[:16]}
            except (binascii.Error, ValueError) as e:
                status = e.status if isinstance(e, core.UploadError) else 400
                self._json(status, {"error": str(e) if isinstance(e, core.UploadError) else "photo is not valid base64"})
                return self._audit(auth="ok", event="message", outcome="bad photo", text=text)
        if not self.app.busy.acquire(timeout=120):
            self._json(503, {"error": "the laptop is busy; try again in a minute"})
            return self._audit(auth="ok", event="message", outcome="busy", text=text)
        try:
            result = self.app.assistant.handle(text, photo, owner="web")
        finally:
            self.app.busy.release()
        now = datetime.now(core.TZ).isoformat(timespec="seconds")
        self.app.remember({"role": "user", "text": core.clean_text(text)[:core.MAX_TEXT], "photo": photo is not None,
                           "t": now})
        reply = {"role": "assistant", "t": now, **{k: result.get(k) for k in
                 ("text", "intent", "outcome", "proposal", "cards", "suggestions", "latency_s")}}
        if reply["proposal"]:
            reply["proposal"] = {**reply["proposal"], "status": "open"}
        self.app.remember(reply)
        # An action is in the audit log before its result reaches the phone.
        self._audit(200, auth="ok", event="message", text=text, intent=result["intent"], routed_by=result["routed_by"],
                    outcome=result["outcome"], proposal_id=(result.get("proposal") or {}).get("id"),
                    latency_s=result["latency_s"], model_calls=result["model_calls"], **photo_info)
        self._json(200, reply)

    def _decide(self, body):
        pid, decision = str(body.get("proposal_id", ""))[:64], str(body.get("decision", ""))[:16]
        try:
            result = self.app.assistant.decide(pid, decision, owner="web")
        except core.CalendarError as e:
            result = {"outcome": "error", "text": f"The calendar refused the change ({e}). Nothing was added."}
        if result["outcome"] in ("confirmed", "cancelled", "error"):
            self.app.mark_proposal(pid, result["outcome"])
        reply = {"role": "assistant", "t": datetime.now(core.TZ).isoformat(timespec="seconds"),
                 "text": result["text"], "outcome": result["outcome"], "proposal_id": pid}
        self.app.remember(reply)
        self._audit(200, auth="ok", event="decide", decision=decision, proposal_id=pid, outcome=result["outcome"],
                    event_id=result.get("event_id"))
        self._json(200, reply)

    def do_PUT(self):
        self._json(405, {"error": "method not allowed"})
        self._audit(auth="n/a")

    do_DELETE = do_PATCH = do_OPTIONS = do_PUT


def make_server(assistant, token, state_dir, host="127.0.0.1", port=8787, allow_hostnames=(), demo_samples=False,
                secure_cookie=False, clock_label=""):
    allowed = {"127.0.0.1", "localhost", "[::1]", host.lower() if not host.startswith("[") else host,
               *(h.lower() for h in allow_hostnames)}
    if ":" in host and not host.startswith("["):
        allowed.add(f"[{host.lower()}]")
    httpd = ThreadingHTTPServer((host.strip("[]"), port), Handler)
    httpd.daemon_threads = True
    httpd.app = App(assistant, token, state_dir, allowed, demo_samples, secure_cookie, clock_label)
    return httpd


def main():
    ap = argparse.ArgumentParser(description="Local phone assistant (see the docstring at the top of this file).")
    ap.add_argument("--host", default="127.0.0.1", help="address to listen on (default 127.0.0.1, this Mac only)")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--allow-hostname", action="append", default=[],
                    help="extra Host name the phone may use, e.g. my-mac.tailnet-name.ts.net (repeatable)")
    ap.add_argument("--all-interfaces", action="store_true", help="allow --host 0.0.0.0 (not recommended)")
    ap.add_argument("--https-proxy", action="store_true", help="behind an HTTPS proxy such as `tailscale serve`: "
                                                               "mark the cookie Secure")
    ap.add_argument("--state-dir", type=Path, default=None)
    ap.add_argument("--now", default="demo", help="'demo' (Mon Nov 2 2026, 8 am: matches the sample data), 'live', "
                                                  "or an ISO time")
    ap.add_argument("--demo-samples", action="store_true", help="offer the sample receipts in the photo picker")
    ap.add_argument("--reset-calendar", action="store_true", help="start again from the sample calendar")
    ap.add_argument("--show-token", action="store_true", help="print the access token")
    ap.add_argument("--rotate-token", action="store_true", help="make a new token (signs every phone out)")
    ap.add_argument("--notes-dir", type=Path, default=core.ROOT / "outputs" / "class_notes")
    ap.add_argument("--flashcards", type=Path, default=core.ROOT / "outputs" / "study_pack" / "flashcards.csv")
    args = ap.parse_args()

    kind = bind_kind(args.host, args.all_interfaces)
    state = args.state_dir or default_state_dir()
    state.mkdir(parents=True, exist_ok=True)
    os.chmod(state, 0o700)
    token, created = load_or_create_token(state, args.rotate_token)
    cal = state / "calendar.json"
    if args.reset_calendar:
        cal.unlink(missing_ok=True)
    now = {"demo": core.DEMO_NOW, "live": None}.get(args.now, args.now)
    if now:
        datetime.fromisoformat(now)      # fail early on a bad --now
    clock = f"Demo clock: {core.fmt_when(now)}" if now else "Live clock"
    cfg = core.Config(calendar_file=cal, notes_dir=args.notes_dir, flashcards=args.flashcards, now=now,
                      mcp_log=state / "gcal_server.log")
    assistant = core.Assistant(cfg)
    httpd = make_server(assistant, token, state, args.host, args.port, args.allow_hostname, args.demo_samples,
                        args.https_proxy, clock)
    port = httpd.server_address[1]
    shown = f"[{args.host}]" if ":" in args.host else args.host

    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)

    print(f"\nLocal assistant on http://{shown}:{port}   ({'this Mac only' if kind == 'loopback' else 'NETWORK'})")
    print(f"  {clock}. Calendar: sandbox copy {cal}")
    print(f"  Audit log: {state / 'audit.jsonl'}")
    if created or args.rotate_token or args.show_token:
        print(f"  Access token: {token}\n  (type it into the sign-in screen once; it is also in {state / 'token'}, "
              "mode 600. Never put it in a URL.)")
    else:
        print(f"  Access token: in {state / 'token'} (or run with --show-token)")
    if kind == "network":
        bar = "!" * 78
        print(f"\n{bar}\n  WARNING: listening on {shown}:{port}. Any device that can reach this address can try to\n"
              "  sign in, so the token is all that stands between it and your calendar, notes and receipts.\n"
              "  - Prefer a Tailscale address (100.x.y.z): only your own devices, and the traffic is encrypted.\n"
              "  - On home Wi-Fi, traffic is plain HTTP that others on the network could read. Never use café Wi-Fi.\n"
              "  - Stop with Ctrl-C. If the token leaks, restart with --rotate-token.\n"
              f"{bar}\n")
    print("  Ctrl-C to stop.\n", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        httpd.server_close()
        assistant.close()


if __name__ == "__main__":
    main()
