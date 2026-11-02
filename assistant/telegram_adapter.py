"""Optional: reach the same local assistant from Telegram, with no inbound port on your laptop.

The adapter long-polls the Telegram Bot API (getUpdates) over outbound HTTPS, answers only your own chat id,
and sends every message through the same Assistant.handle() and Assistant.decide() as the web page: same router,
same five handlers, same Confirm/Cancel (as inline buttons) before any calendar write.

STATUS: tested only against a local stub of the Bot API (tests/test_assistant.py). The live Telegram path has
NOT been verified: no bot was created and no real token was used while building this demo.

Set up your own bot (about five minutes):
  1. In Telegram, open @BotFather, send /newbot, pick a name. BotFather replies with a bot token.
  2. Save the token where only you can read it, outside the repo:
        mkdir -p ~/.config/speed-of-thought-assistant && chmod 700 ~/.config/speed-of-thought-assistant
        (umask 077; pbpaste > ~/.config/speed-of-thought-assistant/telegram_token)    # after copying the token
     or export TELEGRAM_BOT_TOKEN for one shell session. Never commit it, never paste it into a chat.
  3. Send your bot any message from your phone, then find your chat id:
        .venv/bin/python assistant/telegram_adapter.py --print-chat-ids
  4. Run it with your chat id on the allow-list (everyone else is ignored, silently):
        .venv/bin/python assistant/telegram_adapter.py --allow-chat-id 123456789
  Telegram's servers see your messages and photos (bot chats are not end-to-end encrypted). The web page over
  Tailscale keeps everything on your own devices; pick the trade-off you prefer.

Like the web server, the adapter keeps its audit log in the state folder (audit.jsonl, source "telegram") and
rate-limits per chat. Only private chats are answered; group chats are ignored even if allow-listed by mistake.
"""
import argparse
import html
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from assistant import core  # noqa: E402

API_BASE = os.environ.get("TELEGRAM_API_BASE", "https://api.telegram.org")
MAX_AGE = 6 * 3600              # ignore messages that waited longer than this (laptop asleep or off)
MAX_MESSAGE = 4000              # Telegram's limit is 4096 characters


class TelegramError(RuntimeError):
    pass


class BotAPI:
    """The four Bot API methods the adapter needs, plus file download. The token is never logged or printed."""

    def __init__(self, token, base=API_BASE, timeout=70):
        self._token, self.base, self.timeout = token, base.rstrip("/"), timeout

    def call(self, method, **params):
        req = urllib.request.Request(f"{self.base}/bot{self._token}/{method}", data=json.dumps(params).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read())
        except urllib.error.HTTPError as e:      # str(e) has the status only, never the URL with the token
            try:
                body = json.loads(e.read())
            except ValueError:
                body = {"ok": False, "description": f"HTTP {e.code}"}
            e.close()
        except urllib.error.URLError as e:
            raise TelegramError(f"{method}: network error ({e.reason})") from None
        if not body.get("ok"):
            raise TelegramError(f"{method}: {body.get('description', 'failed')} ({body.get('error_code', '?')})")
        return body["result"]

    def download(self, file_path, limit):
        req = urllib.request.Request(f"{self.base}/file/bot{self._token}/{file_path}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = resp.read(limit + 1)
        except urllib.error.URLError as e:
            raise TelegramError(f"file download failed ({getattr(e, 'code', '') or e.reason})") from None
        if len(data) > limit:
            raise core.UploadError("photo is too large", 413)
        return data


def render(result):
    """A reply as Telegram HTML (every piece of text escaped) plus inline Confirm/Cancel buttons for a proposal."""
    parts = [html.escape(result.get("text", ""))]
    for i, c in enumerate(result.get("cards") or [], 1):
        parts.append(f"\n<b>{i}. {html.escape(c['q'])}</b>\n<tg-spoiler>{html.escape(c['a'])}</tg-spoiler>")
    text = "\n".join(parts)[:MAX_MESSAGE]
    markup = None
    if result.get("proposal"):
        pid = result["proposal"]["id"]
        markup = {"inline_keyboard": [[{"text": "Confirm", "callback_data": f"ok:{pid}"},
                                       {"text": "Cancel", "callback_data": f"no:{pid}"}]]}
    return text, markup


class TelegramAdapter:
    def __init__(self, api: BotAPI, assistant, allowed_chat_ids, audit, max_age=MAX_AGE):
        self.api, self.assistant = api, assistant
        self.allowed = {int(c) for c in allowed_chat_ids}
        self.audit, self.max_age = audit, max_age
        self.limit = core.RateLimiter(20, 60)
        self.offset = None

    def _allowed(self, chat):
        return chat.get("type") == "private" and chat.get("id") in self.allowed

    def send(self, chat_id, result):
        text, markup = render(result)
        params = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
        if markup:
            params["reply_markup"] = markup
        return self.api.call("sendMessage", **params)

    def poll_once(self, timeout=50):
        params = {"timeout": timeout, "allowed_updates": ["message", "callback_query"]}
        if self.offset is not None:
            params["offset"] = self.offset
        updates = self.api.call("getUpdates", **params)
        for upd in updates:
            self.offset = upd["update_id"] + 1          # confirm it, even if handling fails below
            try:
                if "callback_query" in upd:
                    self.on_callback(upd["callback_query"])
                elif "message" in upd:
                    self.on_message(upd["message"])
            except TelegramError as e:
                self.audit.write(source="telegram", event="error", error=str(e)[:200])
        return len(updates)

    def on_message(self, msg):
        chat = msg.get("chat", {})
        if not self._allowed(chat):
            self.audit.write(source="telegram", event="ignored", chat_id=chat.get("id"), chat_type=chat.get("type"))
            return
        chat_id = chat["id"]
        if time.time() - msg.get("date", time.time()) > self.max_age:
            self.audit.write(source="telegram", event="too old", chat_id=chat_id)
            return
        if not self.limit.allow(str(chat_id)):
            self.send(chat_id, {"text": "Too many messages; wait a minute."})
            return
        text = msg.get("text") or msg.get("caption") or ""
        photo, info = None, {}
        try:
            file_id = self._photo_file_id(msg)
            if file_id:
                meta = self.api.call("getFile", file_id=file_id)
                if meta.get("file_size", 0) > core.MAX_PHOTO_BYTES:
                    raise core.UploadError("photo is too large", 413)
                raw = self.api.download(meta["file_path"], core.MAX_PHOTO_BYTES)
                photo = core.sanitize_image(raw)
                info = {"photo_bytes": len(raw)}
        except core.UploadError as e:
            self.send(chat_id, {"text": f"I couldn't use that photo: {e}."})
            self.audit.write(source="telegram", event="message", chat_id=chat_id, outcome="bad photo", text=text)
            return
        if not text and photo is None and "text" not in msg:
            return          # stickers, voice notes, locations ...: nothing to do
        result = self.assistant.handle(text, photo, owner=f"telegram:{chat_id}")
        self.send(chat_id, result)
        self.audit.write(source="telegram", event="message", chat_id=chat_id, text=text, intent=result["intent"],
                         outcome=result["outcome"], proposal_id=(result.get("proposal") or {}).get("id"),
                         latency_s=result["latency_s"], model_calls=result["model_calls"], **info)

    @staticmethod
    def _photo_file_id(msg):
        if msg.get("photo"):
            sizes = [p for p in msg["photo"] if p.get("file_size", 0) <= core.MAX_PHOTO_BYTES] or msg["photo"][:1]
            return max(sizes, key=lambda p: p.get("width", 0) * p.get("height", 0))["file_id"]
        doc = msg.get("document") or {}
        if str(doc.get("mime_type", "")).startswith("image/"):
            return doc["file_id"]
        return None

    def on_callback(self, cq):
        chat = (cq.get("message") or {}).get("chat", {})
        user = cq.get("from", {})
        if not self._allowed(chat) or user.get("id") != chat.get("id"):
            self.audit.write(source="telegram", event="ignored callback", chat_id=chat.get("id"))
            self.api.call("answerCallbackQuery", callback_query_id=cq["id"])
            return
        chat_id = chat["id"]
        kind, _, pid = str(cq.get("data", "")).partition(":")
        decision = {"ok": "confirm", "no": "cancel"}.get(kind, "")
        try:
            result = self.assistant.decide(pid, decision, owner=f"telegram:{chat_id}")
        except core.CalendarError as e:
            result = {"outcome": "error", "text": f"The calendar refused the change ({e}). Nothing was added."}
        self.api.call("answerCallbackQuery", callback_query_id=cq["id"], text=result["outcome"].capitalize())
        message_id = (cq.get("message") or {}).get("message_id")
        if message_id is not None:          # remove the buttons so the proposal cannot be tapped again
            self.api.call("editMessageReplyMarkup", chat_id=chat_id, message_id=message_id,
                          reply_markup={"inline_keyboard": []})
        self.send(chat_id, result)
        self.audit.write(source="telegram", event="decide", chat_id=chat_id, decision=decision, proposal_id=pid,
                         outcome=result["outcome"], event_id=result.get("event_id"))

    def run_forever(self):
        backoff = 1
        while True:
            try:
                self.poll_once()
                backoff = 1
            except TelegramError as e:
                if "(409)" in str(e):
                    print("Telegram says another getUpdates poller or a webhook is active for this bot. Stop the other "
                          "copy (or deleteWebhook) and start again.", file=sys.stderr)
                    raise SystemExit(1)
                if "(401)" in str(e) or "(404)" in str(e):
                    print("Telegram rejected the bot token. Check the token file.", file=sys.stderr)
                    raise SystemExit(1)
                print(f"{datetime.now():%H:%M:%S} {e}; retrying in {backoff}s", file=sys.stderr)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)


def read_bot_token(state_dir):
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    path = state_dir / "telegram_token"
    if not token and path.exists():
        if path.stat().st_mode & 0o077:
            os.chmod(path, 0o600)
        token = path.read_text().strip()
    if not token:
        raise SystemExit(f"No bot token: set TELEGRAM_BOT_TOKEN or put it in {path} (see the top of this file).")
    return token


def main():
    sys.path.insert(0, str(HERE))
    from server import default_state_dir   # same state folder as the web server

    ap = argparse.ArgumentParser(description="Telegram long-polling adapter for the local assistant.")
    ap.add_argument("--allow-chat-id", action="append", default=[], help="your own chat id (repeatable)")
    ap.add_argument("--print-chat-ids", action="store_true", help="show who has messaged the bot, then exit")
    ap.add_argument("--state-dir", type=Path, default=None)
    ap.add_argument("--now", default="demo", help="'demo', 'live' or an ISO time (as in server.py)")
    ap.add_argument("--once", action="store_true", help="handle one batch of updates and exit")
    args = ap.parse_args()
    state = args.state_dir or default_state_dir()
    state.mkdir(parents=True, exist_ok=True)
    os.chmod(state, 0o700)
    api = BotAPI(read_bot_token(state))
    if args.print_chat_ids:
        seen = {}
        for upd in api.call("getUpdates", timeout=0):
            chat = (upd.get("message") or {}).get("chat") or {}
            if chat:
                seen[chat["id"]] = (chat.get("type"), chat.get("username") or chat.get("first_name", ""))
        for cid, (kind, name) in seen.items():
            print(f"chat id {cid}  ({kind}, {name})")
        print("" if seen else "No messages yet: send your bot a message from your phone first.")
        return
    allowed = args.allow_chat_id or [c for c in os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS", "").split(",") if c.strip()]
    if not allowed:
        raise SystemExit("Pass --allow-chat-id <your chat id> (find it with --print-chat-ids).")
    now = {"demo": core.DEMO_NOW, "live": None}.get(args.now, args.now)
    cfg = core.Config(calendar_file=state / "calendar.json", now=now, mcp_log=state / "gcal_server.log")
    assistant = core.Assistant(cfg)
    adapter = TelegramAdapter(api, assistant, allowed, core.AuditLog(state / "audit.jsonl"))
    print(f"Telegram adapter polling; answering chat ids {sorted(adapter.allowed)} only. Ctrl-C to stop.", flush=True)
    try:
        adapter.poll_once(timeout=0) if args.once else adapter.run_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        assistant.close()


if __name__ == "__main__":
    main()
