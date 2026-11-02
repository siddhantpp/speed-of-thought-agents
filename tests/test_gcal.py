"""Tests for the calendar MCP server (both backends) and the matching code in mcp_servers/gcal_reconcile.py.

No model, no LM Studio and no Google account are needed. The google backend runs against a local stub HTTP
server that checks the documented Calendar API v3 paths, methods, query parameters and bodies
(developers.google.com/workspace/calendar/api/v3/reference/events). The live Google API is never contacted.

    .venv/bin/python -m unittest tests.test_gcal        # from the repo root
    .venv/bin/python tests/test_gcal.py                 # or run it directly
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.parse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Repo layout (tests/ next to agentkit/) or this source tree (tools/repo_extras/tests -> demos/).
ROOT = next(p for p in (HERE.parent, HERE.parents[2] / "demos") if (p / "agentkit").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "mcp_servers")]
import gcal_reconcile as rec  # noqa: E402
import gcal_server as gcal  # noqa: E402
from agentkit.mcp import MCPServer  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

DATA = ROOT / "sample_data" / "inbox"
KEY = json.loads((DATA / "answer_key.json").read_text())
CODES = ["CS 152", "ECON 140", "HIST 97", "STAT 121"]
NOW, LATER = "2026-11-02T08:00:00-05:00", "2026-12-02T08:00:00-05:00"


class SandboxCase(unittest.TestCase):
    """Each test gets its own copy of calendar_before.json as the sandbox calendar."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.file = self.dir / "calendar_after.json"
        self.env = {"GCAL_BACKEND": "sandbox", "GCAL_SANDBOX_FILE": str(self.file),
                    "GCAL_SANDBOX_SEED": str(DATA / "calendar_before.json")}
        self.old = {k: os.environ.get(k) for k in self.env}
        os.environ.update(self.env)

    def tearDown(self):
        for k, v in self.old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        shutil.rmtree(self.dir)

    def saved(self, event_id):
        gcal.backend()   # seeds the sandbox file on first use
        return next(e for e in json.loads(self.file.read_text())["items"] if e["id"] == event_id)


class SandboxBackendTest(SandboxCase):
    def test_seeded_from_the_sample_and_listed_in_time_order(self):
        items = gcal.list_events(NOW, "2026-11-04T00:00:00-05:00")["items"]
        self.assertTrue(self.file.exists())
        starts = [datetime.fromisoformat(e["start"]["dateTime"]) for e in items]
        self.assertEqual(starts, sorted(starts))
        self.assertIn("climbkiko1103", [e["id"] for e in items])
        self.assertNotIn("cs152ps4regrade", [e["id"] for e in items])   # ended before time_min

    def test_weekly_classes_expand_to_occurrences_across_the_dst_change(self):
        items = gcal.list_events("2026-11-03T00:00:00-05:00", "2026-11-04T00:00:00-05:00", q="CS 152 lecture")["items"]
        self.assertEqual([e["id"] for e in items], ["cs152lec2026fa_20261103T153000Z"])
        self.assertEqual(items[0]["start"]["dateTime"], "2026-11-03T10:30:00-05:00")   # 10:30 in EST, as in EDT
        self.assertEqual(items[0]["recurringEventId"], "cs152lec2026fa")
        self.assertEqual(gcal.get_event(items[0]["id"])["end"]["dateTime"], "2026-11-03T11:45:00-05:00")

    def test_class_meetings_match_the_class_notes_schedule(self):
        # classes/schedule.json (the class-notes workflow's day) and this calendar describe the same timetable:
        # the day's class meetings are exactly those classes, at the same start and end.
        sched = json.loads((ROOT / "sample_data" / "classes" / "schedule.json").read_text())
        day = sched["date"]
        items = gcal.list_events(f"{day}T00:00:00-05:00", f"{day}T23:59:59-05:00")["items"]
        got = [(e["summary"], e["start"]["dateTime"], e["end"]["dateTime"]) for e in items if e.get("recurringEventId")]
        want = [(f"{c['course'].split(':')[0]} {c['session'].split()[0].lower()}", c["start"], c["end"])
                for c in sched["classes"]]
        self.assertEqual(got, want)

    def test_create_defaults_to_a_30_minute_deadline_and_links_the_message(self):
        e = gcal.create_event("CS 152: Problem Set 5 due", "2026-11-06T22:00:00Z", source_message_ids="m05")
        self.assertRegex(e["id"], r"^[a-v0-9]{5,1024}$")       # Google's id alphabet (base32hex)
        self.assertEqual(e["start"], {"dateTime": "2026-11-06T17:00:00-05:00", "timeZone": "America/New_York"})
        self.assertEqual(e["end"]["dateTime"], "2026-11-06T17:30:00-05:00")
        self.assertEqual(e["extendedProperties"], {"private": {"source_message_ids": "m05"}})
        self.assertEqual(gcal.get_event(e["id"])["summary"], "CS 152: Problem Set 5 due")

    def test_update_is_a_patch_that_keeps_length_and_adds_links(self):
        before = self.saved("econ140ess2m01")
        e = gcal.update_event("econ140ess2m01", start="2026-11-09T23:59:00-05:00", source_message_ids="m14")
        self.assertEqual(e["end"]["dateTime"], "2026-11-10T00:29:00-05:00")
        self.assertEqual(e["extendedProperties"]["private"]["source_message_ids"], "m01,m14")
        after = self.saved("econ140ess2m01")
        self.assertEqual((after["summary"], after["description"]), (before["summary"], before["description"]))
        self.assertNotEqual(after["updated"], before["updated"])
        # Asking again for the same thing writes nothing.
        gcal.update_event("econ140ess2m01", source_message_ids="m14")
        self.assertEqual(self.saved("econ140ess2m01")["updated"], after["updated"])

    def test_delete_marks_cancelled_and_hides_it(self):
        self.assertEqual(gcal.delete_event("hist97fieldrpt")["deleted"], "hist97fieldrpt")
        self.assertEqual(self.saved("hist97fieldrpt")["status"], "cancelled")
        ids = [e["id"] for e in gcal.list_events(NOW, LATER)["items"]]
        self.assertNotIn("hist97fieldrpt", ids)
        with self.assertRaises(ToolError):
            gcal.update_event("hist97fieldrpt", start="2026-11-05T09:00:00-05:00")

    def test_recurring_events_are_refused(self):
        for eid in ("stat121tfoh2026fa", "stat121tfoh2026fa_20261103T200000Z"):
            with self.subTest(eid), self.assertRaises(ToolError):
                gcal.update_event(eid, start="2026-11-04T14:00:00-05:00")
            with self.subTest(eid), self.assertRaises(ToolError):
                gcal.delete_event(eid)
        self.assertEqual(self.saved("stat121tfoh2026fa")["status"], "confirmed")

    def test_bad_input_is_a_tool_error_not_a_crash(self):
        with self.assertRaises(ToolError):
            gcal.create_event("x", "next Friday")
        with self.assertRaises(ToolError):
            gcal.create_event("x", "2026-11-06T17:00:00-05:00", end="2026-11-06T16:00:00-05:00")
        with self.assertRaises(ToolError):
            gcal.get_event("nosuchevent")

    def test_over_stdio_with_the_agentkit_client(self):
        server = MCPServer([sys.executable, str(ROOT / "mcp_servers" / "gcal_server.py")], env={**os.environ, **self.env})
        try:
            self.assertEqual(sorted(t["name"] for t in server.tools),
                             ["create_event", "delete_event", "get_event", "list_events", "update_event"])
            got = server.call("list_events", {"time_min": NOW, "time_max": LATER, "q": "essay"})
            self.assertEqual([e["id"] for e in got["items"]], ["econ140ess2m01"])
            refused = server.call("delete_event", {"event_id": "cs152lec2026fa"})
            self.assertIn("recurring", refused["error"])
        finally:
            server.close()
            server.proc.stdout.close()


# ---------------------------------------------------------------- google backend against a stub
class Stub(BaseHTTPRequestHandler):
    """Records each request; answers from `routes[(method, path)]` = (status, json body or None)."""
    routes, log = {}, []

    def _serve(self):
        url = urllib.parse.urlsplit(self.path)
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        Stub.log.append({"method": self.command, "path": url.path, "query": urllib.parse.parse_qs(url.query),
                         "headers": dict(self.headers), "body": raw.decode()})
        status, body = Stub.routes.get((self.command, url.path), (404, {"error": {"code": 404, "message": "Not Found"}}))
        if callable(body):
            body = body(Stub.log[-1])
        data = b"" if body is None else json.dumps(body).encode()
        self.send_response(status)
        if data:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_POST = do_PATCH = do_DELETE = _serve

    def log_message(self, *args):
        pass


EVENTS = "/calendar/v3/calendars/primary/events"
ONE = {"kind": "calendar#event", "id": "abc123", "status": "confirmed", "summary": "ECON 140: Essay 2 draft due",
       "start": {"dateTime": "2026-11-05T23:59:00-05:00", "timeZone": "America/New_York"},
       "end": {"dateTime": "2026-11-06T00:29:00-05:00", "timeZone": "America/New_York"},
       "extendedProperties": {"private": {"source_message_ids": "m01"}}}


class GoogleBackendContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.http = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
        threading.Thread(target=cls.http.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.http.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown()

    def setUp(self):
        self.env = {"GCAL_BACKEND": "google", "GCAL_API_BASE": self.base + "/calendar/v3", "GCAL_ACCESS_TOKEN": "test-token",
                    "GCAL_CALENDAR_ID": "primary", "GCAL_TOKEN_FILE": "", "GCAL_TOKEN_URI": self.base + "/token"}
        self.old = {k: os.environ.get(k) for k in self.env}
        os.environ.update(self.env)
        gcal._google = None
        Stub.routes, Stub.log = {}, []

    def tearDown(self):
        for k, v in self.old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        gcal._google = None

    def test_list_uses_documented_query_parameters_and_follows_pages(self):
        pages = {None: {"kind": "calendar#events", "items": [ONE], "nextPageToken": "p2"},
                 "p2": {"kind": "calendar#events", "items": [{**ONE, "id": "def456"}]}}
        Stub.routes[("GET", EVENTS)] = (200, lambda req: pages[req["query"].get("pageToken", [None])[0]])
        got = gcal.list_events(NOW, LATER, q="essay")
        self.assertEqual([e["id"] for e in got["items"]], ["abc123", "def456"])
        first, second = Stub.log
        self.assertEqual(first["method"], "GET")
        self.assertEqual(first["headers"]["Authorization"], "Bearer test-token")
        q = {k: v[0] for k, v in first["query"].items()}
        self.assertEqual(q, {"timeMin": NOW, "timeMax": LATER, "singleEvents": "true", "orderBy": "startTime",
                             "maxResults": "250", "timeZone": "America/New_York", "q": "essay"})
        self.assertEqual(first["body"], "")
        self.assertEqual(second["query"]["pageToken"], ["p2"])

    def test_create_posts_an_event_resource(self):
        Stub.routes[("POST", EVENTS)] = (200, lambda req: {**json.loads(req["body"]), "id": "new123", "status": "confirmed"})
        e = gcal.create_event("CS 152: Problem Set 5 due", "2026-11-06T17:00:00-05:00", description="From m05",
                              source_message_ids="m05")
        req = Stub.log[-1]
        self.assertEqual((req["method"], req["path"]), ("POST", EVENTS))
        self.assertEqual(req["headers"]["Content-Type"], "application/json")
        self.assertEqual(json.loads(req["body"]), {
            "summary": "CS 152: Problem Set 5 due", "description": "From m05",
            "start": {"dateTime": "2026-11-06T17:00:00-05:00", "timeZone": "America/New_York"},
            "end": {"dateTime": "2026-11-06T17:30:00-05:00", "timeZone": "America/New_York"},
            "extendedProperties": {"private": {"source_message_ids": "m05"}}})
        self.assertEqual(e["id"], "new123")

    def test_get_and_patch_send_only_the_changed_fields(self):
        Stub.routes[("GET", EVENTS + "/abc123")] = (200, ONE)
        Stub.routes[("PATCH", EVENTS + "/abc123")] = (200, lambda req: {**ONE, **json.loads(req["body"])})
        gcal.update_event("abc123", start="2026-11-09T23:59:00-05:00", source_message_ids="m14")
        get, patch = Stub.log
        self.assertEqual((get["method"], get["path"]), ("GET", EVENTS + "/abc123"))
        self.assertEqual((patch["method"], patch["path"]), ("PATCH", EVENTS + "/abc123"))
        self.assertEqual(json.loads(patch["body"]), {
            "start": {"dateTime": "2026-11-09T23:59:00-05:00", "timeZone": "America/New_York"},
            "end": {"dateTime": "2026-11-10T00:29:00-05:00", "timeZone": "America/New_York"},
            "extendedProperties": {"private": {"source_message_ids": "m01,m14"}}})

    def test_delete_checks_then_deletes_and_accepts_an_empty_204(self):
        Stub.routes[("GET", EVENTS + "/abc123")] = (200, ONE)
        Stub.routes[("DELETE", EVENTS + "/abc123")] = (204, None)
        self.assertEqual(gcal.delete_event("abc123")["deleted"], "abc123")
        self.assertEqual([(r["method"], r["path"]) for r in Stub.log],
                         [("GET", EVENTS + "/abc123"), ("DELETE", EVENTS + "/abc123")])
        Stub.routes[("DELETE", EVENTS + "/abc123")] = (410, {"error": {"code": 410, "message": "Resource has been deleted"}})
        self.assertEqual(gcal.delete_event("abc123")["deleted"], "abc123")   # already gone is fine

    def test_recurring_occurrence_is_refused_before_any_write(self):
        occ = {**ONE, "id": "cls_20261103T153000Z", "recurringEventId": "cls"}
        Stub.routes[("GET", EVENTS + "/cls_20261103T153000Z")] = (200, occ)
        with self.assertRaises(ToolError):
            gcal.delete_event("cls_20261103T153000Z")
        self.assertEqual([r["method"] for r in Stub.log], ["GET"])

    def test_calendar_id_is_url_encoded(self):
        os.environ["GCAL_CALENDAR_ID"] = "riley.park@college.example.edu"
        Stub.routes[("GET", "/calendar/v3/calendars/riley.park%40college.example.edu/events")] = (200, {"items": []})
        self.assertEqual(gcal.list_events(NOW, LATER)["items"], [])

    def test_api_errors_come_back_as_tool_errors(self):
        with self.assertRaisesRegex(ToolError, "404.*Not Found"):
            gcal.get_event("missing")

    def test_expired_token_file_is_refreshed_with_the_oauth_refresh_grant(self):
        token = Path(tempfile.mkdtemp()) / "token.json"
        token.write_text(json.dumps({"token": "old", "refresh_token": "r-1", "client_id": "cid", "client_secret": "cs",
                                     "token_uri": "https://oauth2.googleapis.com/token",
                                     "scopes": ["https://www.googleapis.com/auth/calendar.events"],
                                     "expiry": "2020-01-01T00:00:00.000000Z"}))
        os.environ.pop("GCAL_ACCESS_TOKEN")
        os.environ["GCAL_TOKEN_FILE"] = str(token)
        Stub.routes[("POST", "/token")] = (200, {"access_token": "fresh", "expires_in": 3599, "token_type": "Bearer"})
        Stub.routes[("GET", EVENTS)] = (200, {"items": []})
        gcal.list_events(NOW, LATER)
        refresh, call = Stub.log
        self.assertEqual(refresh["headers"]["Content-Type"], "application/x-www-form-urlencoded")
        self.assertEqual(urllib.parse.parse_qs(refresh["body"]), {"client_id": ["cid"], "client_secret": ["cs"],
                                                                 "refresh_token": ["r-1"], "grant_type": ["refresh_token"]})
        self.assertEqual(call["headers"]["Authorization"], "Bearer fresh")
        self.assertEqual(json.loads(token.read_text())["token"], "fresh")
        shutil.rmtree(token.parent)


# ---------------------------------------------------------------- code-side matching and checks
def item(task, course, start, sources, kind="deadline", status="new"):
    ci = {"item_id": sources[0] + "-0", "kind": kind, "course": course, "task": task, "status": status,
          "start": start, "source_message_ids": sources}
    ci["end"] = rec.end_time({"kind": kind, "due": start})
    return ci


class MatchingTest(SandboxCase):
    def setUp(self):
        super().setUp()
        self.events = gcal.list_events(NOW, LATER)["items"]

    def decide(self, ci, action, event_id=""):
        job = {"item": ci, "cands": rec.candidates(ci, self.events, CODES)}
        return rec.check_decision(job, {"action": action, "event_id": event_id, "reason": "test"})

    def test_words_and_course_codes(self):
        self.assertEqual(rec.words("CS 152: PS5 (Hash Tables) due", CODES), {"problem", "set", "5", "hash", "table"})
        self.assertEqual(rec.course_in(["#cs152-study-group"], CODES), "CS 152")
        self.assertEqual(rec.course_in(["Pemberton Hall, Room 210", "Nov 13"], CODES), "")

    def test_end_times_come_from_copied_words(self):
        ev = {"kind": "event", "due": "2026-11-03T19:00:00-05:00", "phrase": "tomorrow (Tuesday) at 7pm", "ends": "until 9:30"}
        self.assertEqual(rec.end_time(ev), "2026-11-03T21:30:00-05:00")
        ev = {"kind": "event", "due": "2026-11-05T16:00:00-05:00", "phrase": "Thursday, November 5, 4:00 to 5:15 pm", "ends": ""}
        self.assertEqual(rec.end_time(ev), "2026-11-05T17:15:00-05:00")
        self.assertEqual(rec.end_time({"kind": "deadline", "due": "2026-11-08T23:59:00-05:00"}), "2026-11-09T00:29:00-05:00")
        # A model that writes the end as ISO instead of words is accepted; an ISO date is never read as a range.
        ev = {"kind": "event", "due": "2026-11-03T19:00:00-05:00", "phrase": "tomorrow at 7pm", "ends": "2026-11-03T21:30:00-05:00"}
        self.assertEqual(rec.end_time(ev), "2026-11-03T21:30:00-05:00")
        ev = {"kind": "event", "due": "2026-11-05T16:00:00-05:00", "phrase": "2026-11-05 at 4pm", "ends": ""}
        self.assertEqual(rec.end_time(ev), "2026-11-05T17:00:00-05:00")

    def test_same_title_in_another_course_is_not_a_candidate(self):
        ps5 = item("Problem Set 5 (Hash Tables)", "CS 152", "2026-11-06T17:00:00-05:00", ["m05"])
        self.assertEqual(rec.candidates(ps5, self.events, CODES), [])
        self.assertIsNone(self.decide(ps5, "create"))
        self.assertIsNotNone(self.decide(ps5, "update", "econ140ps5"))

    def test_already_there_means_no_change(self):
        hw7 = item("Homework 7", "STAT 121", "2026-11-05T23:59:00-05:00", ["m02"])
        [c] = rec.candidates(hw7, self.events, CODES)
        self.assertEqual((c["event_id"], c["same_time"], c["linked"]), ("stat121hmk7", True, False))
        self.assertIsNotNone(self.decide(hw7, "create"))
        self.assertIsNone(self.decide(hw7, "no_change", "stat121hmk7"))
        proposal = item("Final project proposal", "STAT 121", "2026-11-16T09:00:00-05:00", ["m02"])
        self.assertIsNone(self.decide(proposal, "create"))

    def test_moved_deadline_updates_the_linked_event(self):
        essay = item("Essay 2 draft", "ECON 140", "2026-11-09T23:59:00-05:00", ["m01", "m14"], status="update")
        [c] = rec.candidates(essay, self.events, CODES)
        self.assertEqual((c["event_id"], c["linked"], c["same_time"]), ("econ140ess2m01", True, False))
        self.assertIsNotNone(self.decide(essay, "create"))
        self.assertIsNotNone(self.decide(essay, "no_change", "econ140ess2m01"))
        self.assertIsNone(self.decide(essay, "update", "econ140ess2m01"))

    def test_cancelled_item_deletes_and_nothing_else_does(self):
        report = item("Walking Tour Field Report", "HIST 97", "2026-11-04T09:00:00-05:00", ["m12"], status="cancelled")
        self.assertIsNone(self.decide(report, "delete", "hist97fieldrpt"))
        self.assertIsNotNone(self.decide(report, "no_change", "hist97fieldrpt"))
        self.assertIsNotNone(self.decide(report, "create"))
        hw7 = item("Homework 7", "STAT 121", "2026-11-05T23:59:00-05:00", ["m02"])
        self.assertIsNotNone(self.decide(hw7, "delete", "stat121hmk7"))

    def test_a_cancellation_never_removes_the_event_that_replaced_it(self):
        moved = gcal.create_event("STAT 121: TF office hours", "2026-11-04T14:00:00-05:00", "2026-11-04T16:00:00-05:00",
                                  source_message_ids="m15")
        self.events = gcal.list_events(NOW, LATER)["items"]
        tuesday = item("TF office hours", "STAT 121", "2026-11-03T15:00:00-05:00", ["m15"], kind="event", status="cancelled")
        self.assertEqual(rec.candidates(tuesday, self.events, CODES), [])
        self.assertIsNotNone(self.decide(tuesday, "delete", moved["id"]))
        self.assertIsNone(self.decide(tuesday, "no_change"))

    def test_recurring_office_hours_are_never_candidates(self):
        oh = item("STAT 121 TF office hours (moved)", "STAT 121", "2026-11-04T14:00:00-05:00", ["m15"], kind="event")
        self.assertEqual(rec.candidates(oh, self.events, CODES), [])
        tuesday = item("TF office hours", "STAT 121", "2026-11-03T15:00:00-05:00", ["m15"], kind="event", status="cancelled")
        self.assertIsNone(self.decide(tuesday, "no_change"))

    def test_overlaps_are_found_by_code(self):
        changes = [{**item("PS5 study group", "CS 152", "2026-11-03T19:00:00-05:00", ["m18"], kind="event"),
                    "action": "create", "summary": "CS 152: PS5 study group", "end": "2026-11-03T21:30:00-05:00"},
                   {**item("TF office hours", "STAT 121", "2026-11-04T14:00:00-05:00", ["m15"], kind="event"),
                    "action": "create", "summary": "STAT 121: TF office hours", "end": "2026-11-04T16:00:00-05:00"}]
        found = rec.overlaps(changes, self.events)
        self.assertTrue(any("Climbing gym with Kiko" in x for x in found))
        self.assertTrue(any("HIST 97 seminar" in x for x in found))      # Wed 1:30-2:45 pm
        self.assertFalse(any("STAT 121 lecture" in x for x in found))    # Wed 12-1:15 pm, over before the TF's hours
        self.assertEqual(len(found), 2)

    def test_answer_key_changes_score_full_and_a_second_pass_changes_nothing(self):
        import score
        cal = KEY["calendar"]["expected_changes"]
        changes = []
        for c in cal["create"]:
            e = gcal.create_event(c["title"], c["start"], c["end"], source_message_ids=",".join(c["source_message_ids"]))
            changes.append({"action": "create", "event_id": e["id"], "start": c["start"], "end": c["end"]})
        for u in cal["update"]:
            gcal.update_event(u["event_id"], start=u["start"], source_message_ids=",".join(u["source_message_ids"]))
            changes.append({"action": "update", "event_id": u["event_id"], "start": u["start"], "end": u["end"]})
        for d in cal["delete"]:
            gcal.delete_event(d["event_id"])
            changes.append({"action": "delete", "event_id": d["event_id"]})
        (self.dir / "calendar_plan.json").write_text(json.dumps(
            {"applied": True, "changes": changes, "deletions": [{"event_id": d["event_id"]} for d in cal["delete"]]}))
        got = score.score_calendar(self.dir, KEY["calendar"])
        self.assertTrue(got["ok"], got)
        # Idempotency: against the new calendar, every item finds its own event and only no_change passes the check.
        self.events = gcal.list_events(NOW, LATER)["items"]
        for c in cal["create"] + cal["update"]:
            ci = item(c.get("title", "Essay 2 draft"), rec.course_in([c.get("title", "ECON 140")], CODES),
                      c["start"], c["source_message_ids"], kind=c.get("kind", "deadline"))
            cands = rec.candidates(ci, self.events, CODES)
            self.assertTrue(cands and cands[0]["linked"] and cands[0]["same_time"], (ci["task"], cands))
            self.assertIsNotNone(self.decide(ci, "create"), ci["task"])
            self.assertIsNone(self.decide(ci, "no_change", cands[0]["event_id"]), ci["task"])

    def test_a_sloppy_run_is_caught_by_the_scorer(self):
        import score
        gcal.create_event("ECON 140: Essay 2 draft due", "2026-11-09T23:59:00-05:00", source_message_ids="m14")  # duplicate
        gcal.update_event("econ140ps5", start="2026-11-06T17:00:00-05:00")                                     # wrong course
        (self.dir / "calendar_plan.json").write_text(json.dumps({"applied": True, "changes": [], "deletions": []}))
        got = score.score_calendar(self.dir, KEY["calendar"])
        self.assertFalse(got["ok"])
        self.assertTrue(any("econ140ps5" in x for x in got["collateral_changes"]))
        self.assertTrue(got["duplicates"])
        self.assertEqual(got["deletes"], "0/1")


if __name__ == "__main__":
    unittest.main()
