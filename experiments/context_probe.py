"""Does one small model with everything in one prompt get worse as the prompt grows?

One Gemma 4 12B call gets the whole inbox (the 18 real messages) and must return every upcoming
deadline and event, exactly as the multi-agent pipeline does. Then we pad the inbox with ordinary
campus mail (newsletters, recaps, tips: nothing due, nothing to attend) and ask again. The real
messages and the question never change; only the amount of text around them grows.
Graded with score.py's matcher against the inbox answer key. Also records time to first token,
which is the prefill cost of a long prompt on the laptop.

Run:  .venv/bin/python experiments/context_probe.py [--tools] [target_tokens ...]   ->  results/context-probe.json
      --tools gives the one agent the same resolve_date tool the pipeline's workers use, so date
      arithmetic is off the table and what remains is the effect of the long prompt.
      (the model must be loaded with a context window larger than the biggest target)
"""
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from score import match  # noqa: E402  (the same exact-minute matcher the pipeline is graded with)

OUT = ROOT / "results" / "context-probe.json"
MODEL = os.environ.get("PROBE_MODEL", "google/gemma-4-12b")
client = OpenAI(base_url=os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"), api_key="lmstudio")

INSTRUCTIONS = """You are helping a student, Riley Park, with their inbox. The current time is Monday 2026-11-02 08:00 America/New_York (EST, UTC-05:00).
Below is the inbox, one JSON message per line. Return every UPCOMING deadline (something to submit, complete or register by a time) and every UPCOMING event (something to attend), each with an exact ISO 8601 time with offset in America/New_York.
Rules: resolve relative dates ("next Friday") from the message's received time; convert other time zones to America/New_York; if a later message moves a deadline, keep only the new time; drop cancelled items and anything already past; merge duplicates; ignore marketing, spam and phishing."""

SCHEMA = {"type": "object", "properties": {
    "deadlines": {"type": "array", "items": {"type": "object", "properties": {
        "task": {"type": "string"}, "due": {"type": "string"}}, "required": ["task", "due"]}},
    "events": {"type": "array", "items": {"type": "object", "properties": {
        "title": {"type": "string"}, "start": {"type": "string"}}, "required": ["title", "start"]}}},
    "required": ["deadlines", "events"]}

# Filler: ordinary campus mail with nothing due and nothing to attend (only past events or no dates).
SENDERS = [("news@college.example.edu", "Campus Daily"), ("dining@college.example.edu", "Dining Services"),
           ("library@college.example.edu", "College Library"), ("reslife@college.example.edu", "Residence Life"),
           ("athletics@college.example.edu", "Athletics"), ("alumni@college.example.edu", "Alumni Office"),
           ("sustainability@college.example.edu", "Green Campus"), ("it@college.example.edu", "IT Services")]
PARAGRAPHS = [
    "The north quad renovation crews finished laying the new brick paths last week, and the grass seed went down on Saturday. Facilities asks everyone to stay on the marked walkways while the lawn takes root.",
    "Last Thursday's a cappella showcase drew more than four hundred people to the chapel. Thanks to everyone who came out, and congratulations to the Treble Makers on their first full-length set.",
    "This week's featured dish at the main dining hall is a roasted squash curry with jasmine rice. The vegan station has added a rotating soup, and the late-night grill now stocks gluten-free buns.",
    "Researchers in the biology department published a study on how city pigeons adjust their flight paths around tall buildings. The team tracked birds with lightweight sensors over two summers.",
    "Did you know you can borrow laptop chargers, calculators and umbrellas from the library's front desk? Items are lent for four hours at a time and can be renewed once.",
    "The women's soccer team closed its regular season with a two-to-one win over the visiting Hawks. Senior captain Dana Ruiz scored the winning goal in the final ten minutes.",
    "A reminder that the fitness center's climbing wall is open to all students with a valid ID. Belay certification is free and takes about forty minutes with a staff member.",
    "Our alumni spotlight this month features a graduate who now designs wind turbine blades. She credits a first-year physics seminar with sparking her interest in fluid dynamics.",
    "The campus composting program diverted nearly two tons of food scraps from landfill last month. Green bins are now available in every residence hall kitchen.",
    "IT Services completed the wireless network upgrade in the science center. Students should notice faster connections in the basement labs and the third-floor lounge.",
    "The photography club's autumn exhibition, hung in the student center gallery, features thirty prints of local architecture. Several members experimented with long exposures at night.",
    "Residence Life thanks everyone who took part in last weekend's hall clean-up. Volunteers filled forty bags of litter along the river path and planted a row of tulip bulbs.",
    "The college orchestra recorded its fall concert, and the audio is now available on the music department's website. The program included works by Dvorak and a new piece by a student composer.",
    "Parking services reminds drivers that the lot behind the gym is reserved for staff during weekdays. Student permits remain valid in the east and west lots at all hours.",
    "A new study room booking system went live in the library this term. You can see which rooms are free in real time, and each booking includes a whiteboard and a wall screen.",
    "The debate team returned from the regional tournament with two speaker awards. Coaches say the team's research on renewable energy policy impressed the judges.",
    "The campus bookstore has expanded its section of used textbooks. Prices are typically forty to sixty percent lower than new copies, and stock changes daily.",
    "Our weather station recorded the first frost of the season early on Sunday morning. Grounds staff have moved the potted plants from the main entrance into the greenhouse.",
    "Student government passed a resolution supporting longer library hours during reading period. The proposal now goes to the dean of students for review.",
    "The chemistry department welcomed a visiting scholar who studies battery materials. She will be working with two lab groups for the rest of the academic year.",
    "Dining Services is piloting reusable takeout containers at the grab-and-go counter. Return them to any collection bin and the container fee is refunded to your card.",
    "The men's cross-country team placed third at the conference championship. First-year runner Theo Banks finished in the top ten in his first collegiate championship race.",
    "The writing center has added evening appointments staffed by graduate tutors. Sessions cover everything from thesis statements to citation formats and lab reports.",
    "A family of red-tailed hawks has nested on the clock tower again this year. The ornithology club has set up a camera, and the feed is linked from the club's page.",
    "The art history department acquired a collection of nineteenth-century travel posters. A selection is on display in the second-floor hallway of the humanities building.",
    "IT Services reminds everyone to keep their devices updated. The latest operating system updates fix several security issues and improve battery life on many laptops.",
    "The outing club's fall hike to the ridge trail drew a record turnout. Participants spotted a pair of bald eagles and enjoyed clear views across three counties.",
    "Our sustainability office calculated that the new solar panels on the athletics center now supply about a fifth of that building's electricity on sunny days.",
    "The career center's resume review service is available year-round. Upload a draft through the student portal and an advisor will return written comments.",
    "The theater department's set shop salvaged and reused most of the lumber from last spring's production. Designers say the reclaimed wood gave the new set a weathered look.",
]
SUBJECTS = ["This week on campus", "News you can use", "Campus roundup", "Updates from around the college",
            "In case you missed it", "Notes and highlights", "Around the quad", "Good to know"]


def model_memory_gb():
    """Sum of macOS `footprint` over LM Studio's model processes (as experiments/mem_sampler.py does)."""
    ps = subprocess.run(["ps", "-axo", "pid,command"], capture_output=True, text=True).stdout
    total = 0.0
    for line in ps.splitlines():
        if ".lmstudio/.internal/utils/node" in line:
            out = subprocess.run(["footprint", "-p", line.split()[0]], capture_output=True, text=True).stdout
            m = re.search(r"Footprint:\s*([\d.]+)\s*(MB|GB)", out)
            total += float(m.group(1)) / (1024 if m.group(2) == "MB" else 1) if m else 0.0
    return round(total, 2)


def filler(n, rng):
    start = datetime.fromisoformat("2026-10-29T07:00:00-04:00")
    out = []
    for i in range(n):
        email, name = rng.choice(SENDERS)
        when = start + timedelta(minutes=rng.randrange(0, 4 * 24 * 60))
        out.append({"id": f"f{i + 1:03d}", "source": "email", "from": email, "from_name": name,
                    "received": when.isoformat(), "subject": rng.choice(SUBJECTS),
                    "body": "\n\n".join(rng.sample(PARAGRAPHS, 3))})
    return out


def run(messages):
    prompt = "\n".join(json.dumps(m) for m in sorted(messages, key=lambda m: m["received"]))
    start, first, chunks, usage = time.time(), None, [], None
    stream = client.chat.completions.create(
        model=MODEL, temperature=0.2, max_tokens=3000, stream=True, stream_options={"include_usage": True},
        response_format={"type": "json_schema", "json_schema": {"name": "items", "schema": SCHEMA}},
        messages=[{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": prompt}])
    for ev in stream:
        if ev.usage:
            usage = ev.usage
        if ev.choices and ev.choices[0].delta.content:
            first = first or time.time()
            chunks.append(ev.choices[0].delta.content)
    end = time.time()
    return "".join(chunks), usage, (first or end) - start, end - start


def run_with_tools(messages):
    """The same single call, but as an agentkit agent that may call resolve_date as often as it likes."""
    from agentkit.agent import Agent, AgentCard
    from agentkit.tools import default_toolbox
    prompt = "\n".join(json.dumps(m) for m in sorted(messages, key=lambda m: m["received"]))
    card = AgentCard(name="one_agent", tier="worker", job="find every upcoming deadline and event in the inbox",
                     instructions=INSTRUCTIONS + "\nAlways call resolve_date for every date phrase instead of working dates out yourself.",
                     tools=["resolve_date"], output_schema=SCHEMA, max_tool_turns=16, max_tokens=3000, model=MODEL)
    calls = []
    class Count:  # minimal tracer: keeps each model call's prompt size
        def llm(self, reply, **kw): calls.append(reply.prompt_tokens)
        def tool(self, *a, **kw): pass
    start = time.time()
    got = Agent(card, default_toolbox(ROOT / "outputs"), tracer=Count()).run(prompt, label="one-agent")
    return got, max(calls) if calls else None, len(calls), time.time() - start


def main(targets, tools=False):
    key = json.loads((ROOT / "sample_data" / "inbox" / "answer_key.json").read_text())
    real = [json.loads(line) for line in (ROOT / "sample_data" / "inbox" / "messages.jsonl").read_text().splitlines()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(OUT.read_text())["runs"] if OUT.exists() else []
    for target in targets:
        rng = random.Random(target)
        n = 0 if target <= 6000 else max(0, (target - 5200) // 290)   # ~290 tokens per filler message
        if tools:
            got, prompt_tokens, n_calls, total = run_with_tools(real + filler(n, rng))
            ttft = None
        else:
            text, usage, ttft, total = run(real + filler(n, rng))
            prompt_tokens, n_calls = (usage.prompt_tokens if usage else None), 1
            try:
                got = json.loads(text)
            except json.JSONDecodeError:
                got = {"deadlines": [], "events": []}
        tp, fp, missed = match(got.get("deadlines", []), key["deadlines"], "due")
        etp, efp, emissed = match(got.get("events", []), key["events"], "start")
        row = {"mode": "tools" if tools else "plain", "target_tokens": target, "filler_messages": n,
               "prompt_tokens": prompt_tokens, "model_calls": n_calls,
               "ttft_s": None if ttft is None else round(ttft, 1), "total_s": round(total, 1),
               "model_memory_gb_after": model_memory_gb(),
               "deadlines_correct": tp, "deadlines_expected": len(key["deadlines"]), "deadlines_extra": len(fp),
               "events_correct": etp, "events_expected": len(key["events"]), "events_extra": len(efp),
               "missed": [d["task"] for d in missed] + [e["title"] for e in emissed],
               "extra": [f"{d.get('task')} @ {d.get('due')}" for d in fp] + [f"{e.get('title')} @ {e.get('start')}" for e in efp]}
        results.append(row)
        print(f"{row['mode']:<6} {row['prompt_tokens']:>7} tokens  {row['total_s']:>6}s  deadlines {tp}/{len(key['deadlines'])} "
              f"(+{len(fp)} extra)  events {etp}/{len(key['events'])} (+{len(efp)} extra)")
        OUT.write_text(json.dumps({"model": MODEL, "what": __doc__.strip().splitlines()[0], "runs": results}, indent=2))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--tools"]
    main([int(a) for a in args] or [5000, 16000, 28000], tools="--tools" in sys.argv)
