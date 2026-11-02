# A local assistant you can message from your phone

Leave the laptop plugged in, and message it from your phone at any time. It does five things and nothing else:

| You send | Who does each step | What can change |
|---|---|---|
| "What's due this week?" / "What's on today?" | **router** (12B) labels it; **code** works out the window and filters the calendar (MCP `list_events`); **12B** writes one opening line, which code checks | nothing (read only) |
| "Add to my calendar: study group Thursday 7pm" | **12B** copies the title and the date words; **code** resolves the date (`resolve_date`), rejects past or unparseable dates and duplicates; **you** tap Confirm or Cancel | one event, only after your tap, on a sandbox copy |
| "Notes from ECON 140" | **code** finds the class and returns the summary from `outputs/class_notes/` | nothing |
| "Quiz me on load factor" | **code** picks matching cards from `outputs/study_pack/flashcards.csv` | nothing |
| "Split this receipt" + a photo | **12B vision** reads it; **code** checks it with `sum_items` (one retry with the finding); **12B** assigns items per `house_rules.md`; **code** computes shares (`compute_balances`) | nothing: you get draft messages, nothing is sent or paid |
| anything else | **router** says `refuse`; **code** replies with the list above | nothing |

The models get no tools at all. Code holds the calendar's `list_events` and `create_event` (its `update_event` and `delete_event` are not even registered), and `create_event` runs only from `Assistant.decide()`, after a Confirm tap. Even a router fooled by a clever message can at worst read the calendar or make a proposal you then cancel. The 27B is not needed: there is nothing to plan or judge here.

## Run it

```bash
.venv/bin/python assistant/server.py --demo-samples
```

The first run prints an access token and saves it to `~/.config/speed-of-thought-assistant/token` (mode 600, outside the repo). Open http://127.0.0.1:8787 and type the token into the sign-in screen once. The token is never put in a URL. `--demo-samples` adds the sample receipts to the photo picker, which helps on stage. `--now live` uses the real clock instead of the sample data's Monday, Nov 2, 2026, 8 am.

## Reach it from your phone

The server listens on 127.0.0.1 (this Mac only) unless you opt in. Pick one way:

1. **Same Wi-Fi (home only).** Find the Mac's address (System Settings > Wi-Fi > Details), then run `server.py --host 192.168.x.y` and open `http://192.168.x.y:8787` on the phone. The server prints a warning: this is plain HTTP that others on the network could read. Never do this on café or campus Wi-Fi.
2. **Tailscale (anywhere, recommended).** Tailscale is a private network made of your own devices; the traffic is WireGuard-encrypted. Install it on the Mac and the phone and sign in to both with the same account (not done on this machine; nothing here installs it). Then either:
   - `server.py --host 100.x.y.z` (the Mac's Tailscale address, from `tailscale ip -4`), and open `http://100.x.y.z:8787`; or
   - keep the default 127.0.0.1 and let Tailscale add HTTPS in front: `tailscale serve --bg 8787`, then `server.py --allow-hostname <mac-name>.<tailnet>.ts.net --https-proxy`, and open `https://<mac-name>.<tailnet>.ts.net`. Turn it off with `tailscale serve reset`. Check `tailscale serve --help` for your version. Never use `tailscale funnel`, which publishes to the whole internet.
3. **Telegram (no inbound port at all).** `assistant/telegram_adapter.py` long-polls the Bot API over outbound HTTPS and answers only your own chat id. Setup is at the top of that file. The trade-off is that Telegram's servers see your messages. **Only verified against a local stub of the Bot API, not live Telegram.**

None of these paths was tested from a real phone while building this demo. The phone screenshots are the built-in browser at a phone-sized viewport on 127.0.0.1.

## Keep it running

- **Stay awake on power:** `caffeinate -s .venv/bin/python assistant/server.py` keeps the Mac from sleeping while the server runs and the Mac is on AC power. On battery it sleeps as usual. A closed lid still sleeps unless the Mac is in clamshell mode with an external display, so leave the lid open. Nothing here changes `pmset` or other system settings.
- **Keep the model loaded in LM Studio:** a model you load yourself (in the app, or with `lms load gemma-4-12b-it-mlx --identifier google/gemma-4-12b -c 32768 --parallel 4 -y`; use whatever key `lms ls` shows for your download, and keep the `--identifier`) has no idle timeout by default and stays loaded until you unload it. Models loaded "just in time" (JIT) by the first request unload after 60 idle minutes by default, and with Auto-Evict on, a new JIT load unloads the previous one (Developer tab > Server Settings). So load the 12B yourself, or give JIT loads a long TTL (`lms load <model> --ttl <seconds>`). To have the server come back after a reboot, turn on the app setting that runs the LLM server on login, or use `lms server start`. See lmstudio.ai/docs, "Idle TTL and Auto-Evict" and "Run LM Studio as a service".
- **Start at login:** `assistant/launchd/edu.example.speed-of-thought.assistant.plist` is a sample LaunchAgent that runs the server under `caffeinate -s` at login and restarts it after a crash. It is not installed; the install and removal commands are in the file. Run the server once by hand first so it creates your token.

## Security, in one list

- Binds 127.0.0.1 by default; `--host` must be a private address (Tailscale 100.64/10, 10/8, 172.16/12, 192.168/16); `0.0.0.0` needs `--all-interfaces`; public addresses are refused. A non-loopback bind prints a warning.
- A shared-secret token (about 119 bits), compared with `hmac.compare_digest`; sent once in a POST body and then held in an `HttpOnly; SameSite=Strict` cookie (`Secure` behind `--https-proxy`), or as an `X-Assistant-Token` header for scripts. A token in a query string is ignored.
- A Host header allow-list (DNS rebinding), an Origin check and JSON-only POSTs (cross-site requests), a strict Content-Security-Policy, no third-party scripts or fonts.
- Rate limits: 240 requests and 20 messages per minute per client; 5 bad tokens per client (or 50 across all clients) in 10 minutes locks sign-in for 10 minutes.
- Uploads: 9 MB per request, 6 MB per photo, JPEG/PNG/WebP only, decompression-bomb guard, re-encoded as a PNG of at most 2048 px with no EXIF (no GPS location).
- Audit log: every request and action (message, intent, outcome, proposal id, decision, latency, photo hash) in `~/.config/speed-of-thought-assistant/audit.jsonl`, mode 600; never the token. The Telegram adapter writes to the same log.
- Proposals are one use only, bound to the channel that made them (web or your Telegram chat), and expire after 10 minutes. A text "yes, confirm" never confirms anything.

## Measure it

```bash
.venv/bin/python workflows/assistant_eval.py      # 45 phone messages through the same router and handlers
.venv/bin/python score.py assistant               # router accuracy, refusals, no write without confirm, handlers
.venv/bin/python -m unittest tests.test_assistant # no model needed
```

The test set is `sample_data/assistant/messages.jsonl`: in-scope messages with typos, ambiguous ones, out-of-scope requests, 12 injection and privilege-escalation attempts, and two receipt photos. The key is `answer_key.json` in the same folder.
