"""Google Calendar as five MCP tools: list_events, get_event, create_event, update_event, delete_event.

One tool server, three harnesses:
  agentkit      workflows/inbox_calendar.py starts it with agentkit.mcp.MCPServer (stdio)
  LM Studio     the chat app starts it from mcp.json (stdio), so you can ask a model to fix your calendar by hand
  Claude Code   claude --mcp-config claude_compare/mcp/gcal.json (stdio)

Tools take flat ISO 8601 strings (easy for a 12B model) and return events in Google Calendar API v3
format (id, status, summary, description, location, start/end {dateTime, timeZone}, extendedProperties).

Backends, chosen by the GCAL_BACKEND environment variable:
  sandbox (default)  a local JSON file shaped like Google's events.list response. Every measured run and test
                     uses it. GCAL_SANDBOX_FILE = the calendar file (default calendar.json in the current folder);
                     GCAL_SANDBOX_SEED = a file to copy into GCAL_SANDBOX_FILE the first time, if it is missing.
  google             the real Calendar API v3 (https://www.googleapis.com/calendar/v3) for your own account.
                     1. In Google Cloud console: enable the Google Calendar API, create an OAuth client of type
                        "Desktop app", download it as credentials.json.
                     2. pip install google-auth-oauthlib   (only needed for this one-time step)
                        python mcp_servers/gcal_auth.py credentials.json token.json
                        (Google's installed-app flow opens a browser; scope calendar.events; writes token.json)
                     3. GCAL_BACKEND=google GCAL_TOKEN_FILE=/abs/path/token.json   (optional: GCAL_CALENDAR_ID,
                        default "primary"). The server refreshes the access token itself when it expires.
                     Or set GCAL_ACCESS_TOKEN to a short-lived access token instead of a token file.
                     The google backend is checked against a local stub of the documented endpoints
                     (tests/test_gcal.py), not against the live API.

Safety rules the server enforces, whatever the model asks:
  - recurring events (class meetings, weekly office hours) are never changed or deleted;
  - deleted events are marked status "cancelled" (as Google does), so a sandbox delete can be inspected.

Run:  .venv/bin/python mcp_servers/gcal_server.py               stdio (what the harnesses use)
      .venv/bin/python mcp_servers/gcal_server.py --http 8765   streamable HTTP at http://127.0.0.1:8765/mcp

By hand in LM Studio: Program tab > Install > Edit mcp.json, add this inside "mcpServers" (absolute paths),
then turn the gcal tools on in a chat and ask "Put my CS 152 deadlines for this week on my calendar":
    "gcal": {"command": "/ABS/repo/.venv/bin/python", "args": ["/ABS/repo/mcp_servers/gcal_server.py"],
             "env": {"GCAL_BACKEND": "sandbox", "GCAL_SANDBOX_FILE": "/ABS/repo/outputs/inbox/calendar_lmstudio.json",
                     "GCAL_SANDBOX_SEED": "/ABS/repo/sample_data/inbox/calendar_before.json"}}
"""
import json
import os
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

TZ_NAME = "America/New_York"
TZ = ZoneInfo(TZ_NAME)
DEADLINE_MINUTES = 30          # an event with no end lasts 30 minutes (a deadline marker)
LINK = "source_message_ids"    # extendedProperties.private key that links an event to inbox messages
KEEP = ("id", "status", "summary", "description", "location", "start", "end", "recurrence",
        "recurringEventId", "extendedProperties", "updated")   # fields the tools return (keeps replies short)

app = MCPServer("gcal", instructions="The student's Google Calendar. Times are ISO 8601; the calendar's time zone "
                                     "is America/New_York. Recurring events (classes) cannot be changed.",
                log_level="WARNING")


# ---------------------------------------------------------------- small helpers
def when(text: str) -> str:
    """ISO 8601 in, ISO 8601 in America/New_York out. A time with no offset is taken as New York time."""
    try:
        t = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        raise ToolError(f"{text!r} is not an ISO 8601 time like 2026-11-06T17:00:00-05:00")
    return (t.replace(tzinfo=TZ) if t.tzinfo is None else t).astimezone(TZ).isoformat()


def slot(iso: str) -> dict:
    return {"dateTime": iso, "timeZone": TZ_NAME}


def at(x) -> datetime:
    """An event's start/end slot (or an ISO string) as a datetime, for comparing times."""
    if isinstance(x, dict) and "dateTime" not in x:
        raise ToolError("this is an all-day event (it has a date, not a time); this server only edits timed events")
    return datetime.fromisoformat(x["dateTime"] if isinstance(x, dict) else x)


def links(event: dict) -> list[str]:
    raw = event.get("extendedProperties", {}).get("private", {}).get(LINK, "")
    return [x.strip() for x in raw.split(",") if x.strip()]


def merge_links(old: list[str], new: str) -> str:
    out = list(old)
    out += [x.strip() for x in new.split(",") if x.strip() and x.strip() not in out]
    return ",".join(out)


def short(event: dict) -> dict:
    return {k: event[k] for k in KEEP if k in event}


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def refuse_recurring(event: dict):
    if event.get("recurrence") or event.get("recurringEventId"):
        raise ToolError(f"{event['id']} ({event.get('summary')}) is a recurring event, such as a class meeting. "
                        "This server never changes recurring events; leave it alone.")


# ---------------------------------------------------------------- the five tools
@app.tool()
def list_events(time_min: str, time_max: str, q: str = "") -> dict:
    """List the calendar's events that overlap a time window, earliest first.

    time_min, time_max: ISO 8601 times with offset, e.g. "2026-11-02T00:00:00-05:00".
    q: optional words to search for in the title, description or location (leave empty for all events).
    Weekly events such as class meetings come back as single occurrences (with a recurringEventId).
    """
    return {"timeZone": TZ_NAME, "items": [short(e) for e in backend().list(when(time_min), when(time_max), q)]}


@app.tool()
def get_event(event_id: str) -> dict:
    """Get one event by its id (the "id" field from list_events)."""
    return short(backend().get(event_id))


@app.tool()
def create_event(summary: str, start: str, end: str = "", description: str = "", location: str = "",
                 source_message_ids: str = "") -> dict:
    """Add an event to the calendar. Returns the new event with its id.

    summary: the title, e.g. "CHEM 263: Lab report 4 due".
    start, end: ISO 8601 with offset. Leave end empty for a deadline: it then lasts 30 minutes from start.
    source_message_ids: comma-separated inbox message ids this event came from, e.g. "m99". They are saved
    in extendedProperties.private so a later run can find the event again instead of adding a duplicate.
    """
    start = when(start)
    end = when(end) if end else (datetime.fromisoformat(start) + timedelta(minutes=DEADLINE_MINUTES)).isoformat()
    if at(end) < at(start):
        raise ToolError(f"end {end} is before start {start}")
    body = {"summary": summary, "start": slot(start), "end": slot(end)}
    if description:
        body["description"] = description
    if location:
        body["location"] = location
    if source_message_ids:
        body["extendedProperties"] = {"private": {LINK: merge_links([], source_message_ids)}}
    return short(backend().insert(body))


@app.tool()
def update_event(event_id: str, summary: str = "", start: str = "", end: str = "", description: str = "",
                 location: str = "", source_message_ids: str = "") -> dict:
    """Change some fields of an existing event (patch: fields left empty stay as they are).

    Moving start without giving end keeps the event's length. source_message_ids are ADDED to the event's
    existing links (e.g. an update message "m99" next to the original "m98"). Recurring events are refused.
    """
    b = backend()
    event = b.get(event_id)
    refuse_recurring(event)
    if event.get("status") == "cancelled":
        raise ToolError(f"{event_id} was deleted")
    patch = {}
    if summary:
        patch["summary"] = summary
    if description:
        patch["description"] = description
    if location:
        patch["location"] = location
    if start:
        new_start = when(start)
        patch["start"] = slot(new_start)
        patch["end"] = slot(when(end) if end else (at(new_start) + (at(event["end"]) - at(event["start"]))).isoformat())
    elif end:
        patch["end"] = slot(when(end))
    if "end" in patch and at(patch["end"]) < at(patch.get("start", event["start"])):
        raise ToolError("end is before start")
    if source_message_ids:
        merged = merge_links(links(event), source_message_ids)
        # Google merges nested objects on patch, but sending the whole private map is safe either way.
        patch["extendedProperties"] = {"private": {**event.get("extendedProperties", {}).get("private", {}), LINK: merged}}
    # Drop fields that already have these values; nothing left means no write, so re-running is harmless.
    patch = {k: v for k, v in patch.items()
             if not (at(v) == at(event[k]) if k in ("start", "end") else v == event.get(k))}
    if not patch:
        return short(event)
    return short(b.patch(event_id, patch))


@app.tool()
def delete_event(event_id: str) -> dict:
    """Delete one event by id. Recurring events (class meetings) are refused."""
    b = backend()
    event = b.get(event_id)
    refuse_recurring(event)
    if event.get("status") != "cancelled":
        b.delete(event_id)
    return {"deleted": event_id, "summary": event.get("summary", "")}


# ---------------------------------------------------------------- backend 1: a local JSON file
class Sandbox:
    """Google's data model in a JSON file: {"kind": "calendar#events", "timeZone", "items": [event, ...]}."""

    def __init__(self, path, seed=None):
        self.path = Path(path)
        if not self.path.exists():
            if not seed:
                raise ToolError(f"sandbox calendar {self.path} does not exist (set GCAL_SANDBOX_SEED to start one)")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(seed, self.path)

    def _load(self):
        return json.loads(self.path.read_text())

    def _save(self, cal):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cal, indent=2) + "\n")
        tmp.replace(self.path)    # atomic: a crash never leaves half a calendar

    def list(self, time_min, time_max, q=""):
        out = []
        for e in self._load()["items"]:
            for occ in (expand(e, time_min, time_max) if e.get("recurrence") else [e]):
                if occ.get("status") == "cancelled":
                    continue
                # Google's rule: timeMin bounds an event's END, timeMax bounds its START (both exclusive).
                if not (at(occ["end"]) > at(time_min) and at(occ["start"]) < at(time_max)):
                    continue
                text = " ".join(occ.get(k, "") for k in ("summary", "description", "location")).lower()
                if all(w in text for w in q.lower().split()):
                    out.append(occ)
        return sorted(out, key=lambda e: at(e["start"]))

    def get(self, event_id):
        master_id, _, instant = event_id.partition("_")
        for e in self._load()["items"]:
            if e["id"] == event_id:
                return e
            if instant and e["id"] == master_id and e.get("recurrence"):
                try:   # an occurrence id is <series id>_<start in UTC>, e.g. cs152lec2026fa_20261103T153000Z
                    t = datetime.strptime(instant, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
                except ValueError:
                    break
                day = timedelta(days=1)
                for occ in expand(e, (t - day).isoformat(), (t + day).isoformat()):
                    if occ["id"] == event_id:
                        return occ
        raise ToolError(f"no event with id {event_id!r}; use an id from list_events")

    def insert(self, body):
        cal = self._load()
        now = stamp()
        event = {"kind": "calendar#event", "id": uuid.uuid4().hex, "status": "confirmed", "created": now,
                 "updated": now, **body}
        cal["items"].append(event)
        self._save(cal)
        return event

    def patch(self, event_id, body):
        cal = self._load()
        event = next(e for e in cal["items"] if e["id"] == event_id)
        merge(event, body)
        event["updated"] = stamp()
        self._save(cal)
        return event

    def delete(self, event_id):
        self.patch(event_id, {"status": "cancelled"})   # Google keeps deleted events as status "cancelled"


def merge(target, patch):
    """Google's patch semantics: nested objects merge, arrays and plain values replace, null deletes."""
    for k, v in patch.items():
        if v is None:
            target.pop(k, None)
        elif isinstance(v, dict) and isinstance(target.get(k), dict):
            merge(target[k], v)
        else:
            target[k] = v


DAYS = ["MO", "TU", "WE", "TH", "FR", "SA", "SU"]


def expand(event, time_min, time_max):
    """Occurrences of a weekly event (RRULE:FREQ=WEEKLY;BYDAY=..;UNTIL=..|COUNT=..), like singleEvents=true."""
    rule = dict(p.split("=") for p in event["recurrence"][0].removeprefix("RRULE:").split(";"))
    if rule.get("FREQ") != "WEEKLY":
        return []
    first = at(event["start"]).astimezone(TZ)
    length = at(event["end"]) - at(event["start"])
    until = (datetime.strptime(rule["UNTIL"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
             if "UNTIL" in rule else datetime.max.replace(tzinfo=timezone.utc))
    days = [DAYS.index(d) for d in rule.get("BYDAY", DAYS[first.weekday()]).split(",")]
    lo, hi = at(time_min), at(time_max)
    out, n, day = [], 0, first.date()
    while n < int(rule.get("COUNT", 10**6)):
        if day.weekday() in days:
            # Same wall-clock time every week, so a 10:30 class stays at 10:30 after the DST change.
            start = datetime(day.year, day.month, day.day, first.hour, first.minute, tzinfo=TZ)
            if start > until or start > hi:
                break
            n += 1
            if start + length > lo:
                utc = start.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                occ = {k: v for k, v in event.items() if k not in ("recurrence", "id")}
                out.append({"id": f"{event['id']}_{utc}", **occ, "recurringEventId": event["id"],
                            "originalStartTime": slot(start.isoformat()),
                            "start": slot(start.isoformat()), "end": slot((start + length).isoformat())})
        day += timedelta(days=1)
    return out


# ---------------------------------------------------------------- backend 2: the real Google Calendar API v3
class Google:
    """REST calls as documented at developers.google.com/workspace/calendar/api/v3/reference/events."""

    def __init__(self):
        self.base = os.environ.get("GCAL_API_BASE", "https://www.googleapis.com/calendar/v3").rstrip("/")
        self.calendar = urllib.parse.quote(os.environ.get("GCAL_CALENDAR_ID", "primary"), safe="")
        self.token = os.environ.get("GCAL_ACCESS_TOKEN") or self._token_from_file(os.environ.get("GCAL_TOKEN_FILE", ""))

    def _token_from_file(self, path):
        """Read token.json from Google's installed-app flow; refresh the access token if it has expired."""
        if not path:
            raise ToolError("google backend needs GCAL_TOKEN_FILE (token.json) or GCAL_ACCESS_TOKEN")
        info = json.loads(Path(path).read_text())
        expiry = info.get("expiry", "").removesuffix("Z").split(".")[0]
        fresh = expiry and datetime.fromisoformat(expiry) > datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=2)
        if info.get("token") and fresh:
            return info["token"]
        # OAuth 2.0 refresh (developers.google.com/identity/protocols/oauth2/native-app#offline)
        form = urllib.parse.urlencode({"client_id": info["client_id"], "client_secret": info.get("client_secret", ""),
                                       "refresh_token": info["refresh_token"], "grant_type": "refresh_token"}).encode()
        uri = os.environ.get("GCAL_TOKEN_URI", "https://oauth2.googleapis.com/token")
        req = urllib.request.Request(uri, data=form, method="POST",
                                     headers={"Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            got = json.loads(resp.read())
        info["token"] = got["access_token"]
        info["expiry"] = (datetime.now(timezone.utc) + timedelta(seconds=int(got.get("expires_in", 3600)))
                          ).strftime("%Y-%m-%dT%H:%M:%S.000000Z")
        Path(path).write_text(json.dumps(info))     # like Google's quickstart, keep the refreshed token
        return info["token"]

    def _call(self, method, path, params=None, body=None):
        url = f"{self.base}/calendars/{self.calendar}/events{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Authorization": f"Bearer {self.token}", "Accept": "application/json"}
        if data:
            headers["Content-Type"] = "application/json"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=data, method=method, headers=headers),
                                        timeout=30) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            e.close()
            try:
                detail = json.loads(detail)["error"]["message"]
            except (ValueError, KeyError, TypeError):
                pass
            raise ToolError(f"Google Calendar API {e.code} on {method} {path or '/'}: {detail}")
        return json.loads(raw) if raw.strip() else {}

    def list(self, time_min, time_max, q=""):
        params = {"timeMin": time_min, "timeMax": time_max, "singleEvents": "true", "orderBy": "startTime",
                  "maxResults": 250, "timeZone": TZ_NAME}
        if q:
            params["q"] = q
        items, page = [], None
        while True:
            res = self._call("GET", "", {**params, **({"pageToken": page} if page else {})})
            items += res.get("items", [])
            page = res.get("nextPageToken")
            if not page:
                return items

    def get(self, event_id):
        return self._call("GET", "/" + urllib.parse.quote(event_id, safe=""), {"timeZone": TZ_NAME})

    def insert(self, body):
        return self._call("POST", "", body=body)

    def patch(self, event_id, body):
        return self._call("PATCH", "/" + urllib.parse.quote(event_id, safe=""), body=body)

    def delete(self, event_id):
        try:
            self._call("DELETE", "/" + urllib.parse.quote(event_id, safe=""))
        except ToolError as e:
            if " 410 " not in str(e):     # 410 Gone = already deleted, which is fine
                raise


def backend():
    """A fresh backend per call: the sandbox file is re-read each time, so outside edits are seen."""
    kind = os.environ.get("GCAL_BACKEND", "sandbox")
    if kind == "google":
        global _google
        _google = _google or Google()       # one token refresh per process
        return _google
    if kind == "sandbox":
        return Sandbox(os.environ.get("GCAL_SANDBOX_FILE", "calendar.json"), os.environ.get("GCAL_SANDBOX_SEED"))
    raise ToolError(f"unknown GCAL_BACKEND {kind!r}; use sandbox or google")


_google = None

if __name__ == "__main__":
    if "--http" in sys.argv:
        port = int(sys.argv[sys.argv.index("--http") + 1])
        app.run(transport="streamable-http", port=port)
    else:
        app.run()
