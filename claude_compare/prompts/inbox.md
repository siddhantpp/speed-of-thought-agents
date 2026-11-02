You are helping a student, Riley Park, with their inbox and their calendar. The current time is Monday 2026-11-02 08:00 America/New_York (EST, UTC-05:00).

Inputs:
- `messages.jsonl` in this folder: 18 messages from email, the course site and chat; one JSON object per line with id, source, from, from_name, received, subject, body.
- Riley's Google Calendar, through the `gcal` MCP tools only: `list_events`, `get_event`, `create_event`, `update_event`, `delete_event`. The calendar's time zone is America/New_York. (It is a sandbox: `calendar_before.json` is its starting state and the server keeps the live calendar in `calendar_after.json`. Never create, edit or delete those two files yourself; change the calendar only by calling the tools.)

Do all of this, then write the files below into this folder:
1. Classify every message into exactly one category: deadline (something to submit/complete/register by a date, including messages that move or cancel a deadline), event (something to attend), admin (institutional info, nothing due), social, or promo (marketing, spam, phishing). Note whether the student must act.
2. Extract every upcoming deadline and event with an exact time (events also with their end time). Resolve relative dates ("the Tuesday after next") from the message's received time, convert other time zones to America/New_York, apply updates (keep only the newest time), drop cancelled items and anything already past, and merge duplicates.
3. Reconcile the calendar with the inbox for 2026-11-02T08:00:00-05:00 to 2026-12-02T08:00:00-05:00. Call `list_events` first. Then, for each item:
   - an upcoming deadline or event that is not on the calendar: create it;
   - one that is already on the calendar at an old time: update that event (never add a second one);
   - one that is already on the calendar at the right time: leave it alone;
   - a cancelled item that is on the calendar: delete that event. Delete nothing else.
   Never change recurring events (class meetings, weekly office hours; the server refuses), personal events, events in the past, or events not about these messages. The same title in a different course is a different thing.
   Deadlines become an event titled "COURSE: task due" (just "task due" when there is no course), starting at the due time and lasting 30 minutes (leave `end` empty and `create_event` does this). Events get their real start and end. Pass `source_message_ids` (comma-separated message ids, e.g. "m98,m99") to every `create_event` and `update_event` call, so each event stays linked to the messages it came from.
4. Before changing anything, write the change plan: `calendar_plan.json` and a readable `calendar_plan.md` that lists any deletions first. Then apply exactly that plan with the MCP tools (the student approved applying it in this run) and set `"applied": true` in `calendar_plan.json`.
5. Write a short daily digest (Markdown, at most 350 words): Today, This week (deadlines in time order with weekday and time), Events, Calendar (what was added, moved from which time to which, and removed, plus any overlaps between new events and existing ones), FYI, Ignored count.

Output files (exact formats; they are machine-graded):
- `labels.json`: `[{"id": "m99", "category": "social", "action_required": false, "reason": "..."}, ...]` for all 18 messages.
- `schedule.json`: `{"deadlines": [{"course": "...", "task": "...", "due": "2026-12-04T09:20:00-05:00", "source_message_ids": ["m99"]}], "events": [{"title": "...", "start": "ISO 8601 with offset", "end": "ISO 8601 with offset", "source_message_ids": ["..."]}], "conflicts": []}`
- `calendar_plan.json`: `{"applied": true, "deletions": [{"event_id": "...", "summary": "...", "start": "..."}], "changes": [{"action": "create", "event_id": null, "summary": "...", "start": "ISO 8601 with offset", "end": "ISO 8601 with offset", "source_message_ids": ["m99"], "reason": "..."}]}`. One entry per item. `action` is one of `create`, `update`, `delete`, `no_change`; `event_id` is the existing event's id (null for create); `start`/`end` are the new times for create and update and the event's current times otherwise. Every delete appears in both `changes` and `deletions`.
- `calendar_plan.md`
- `digest.md`
(`calendar_after.json` is written by the MCP server as you call its tools.)
Do not ask questions; make reasonable decisions and finish.
