"""Tests for the cloud backends in agentkit/providers.py, with fake SDK clients: no key, no network, no model.

They check how agentkit's chat() inputs map onto each API (system prompt, tools and tool results, JSON-schema
output, images, reasoning, temperature), how usage is accounted (uncached input, cache reads and writes, output,
reasoning), that a whole Agent.run tool loop works on each backend, that keys are read only from the SOT_*
variables or the key file and never reach a trace, and that the LM Studio path still sends exactly what it did.

    .venv/bin/python -m unittest tests.test_providers      # from the repo root
    .venv/bin/python tests/test_providers.py               # or run it directly
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
# Works in the exported repo (tests/ next to agentkit/) and in the source tree (tools/repo_extras/tests).
ROOT = next(p for p in (HERE.parent, HERE.parents[2] / "demos") if (p / "agentkit").is_dir())
sys.path.insert(0, str(ROOT))
import anthropic  # noqa: E402
import openai  # noqa: E402
from anthropic.types import Message  # noqa: E402
from openai.types.chat import ChatCompletion  # noqa: E402

from agentkit import Agent, AgentCard, Toolbox, Tracer, llm, providers  # noqa: E402

FAKE_KEY = "test-key-not-a-real-one-0123456789"
PNG = b"\x89PNG\r\n\x1a\n-fake-image-bytes"
TOOL = {"type": "function", "function": {"name": "add", "description": "Add two numbers.", "parameters": {
    "type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}}}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["answer", "items"], "properties": {
    "answer": {"type": "string", "maxLength": 40},
    "note": {"type": "string"},                                   # optional
    "items": {"type": "array", "maxItems": 3, "items": {"type": "object", "properties": {
        "name": {"type": "string"}, "qty": {"type": "integer", "minimum": 1}}, "required": ["name"]}}}}


# ---------------------------------------------------------------- fake SDK clients
def oa_completion(content=None, tool_calls=None, finish="stop", prompt=1000, cached=0, written=0, completion=50,
                  reasoning=0, refusal=None):
    msg = {"role": "assistant", "content": content, "refusal": refusal}
    if tool_calls:
        msg["tool_calls"] = [{"id": f"call_{i}", "type": "function",
                              "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(tool_calls)]
    return ChatCompletion.model_validate({
        "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "small-model-2026-01-01",
        "service_tier": "default",
        "choices": [{"index": 0, "finish_reason": finish, "message": msg, "logprobs": None}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion,
                  "prompt_tokens_details": {"cached_tokens": cached, "cache_write_tokens": written},
                  "completion_tokens_details": {"reasoning_tokens": reasoning}}})


def an_message(blocks, stop="end_turn", inp=100, out=20, read=0, write5=0, write1h=0, thinking=0):
    return Message.model_validate({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "small-model", "content": blocks,
        "stop_reason": stop, "stop_sequence": None,
        "usage": {"input_tokens": inp, "output_tokens": out, "cache_read_input_tokens": read,
                  "cache_creation_input_tokens": write5 + write1h,
                  "cache_creation": {"ephemeral_5m_input_tokens": write5, "ephemeral_1h_input_tokens": write1h},
                  "output_tokens_details": {"thinking_tokens": thinking}}})


class FakeBadRequest(openai.BadRequestError):
    """A 400 from the API, without building an HTTP response."""
    def __init__(self, message):
        Exception.__init__(self, message)


class FakeOpenAI:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.embeddings = SimpleNamespace(create=self._embed)

    def _create(self, **kw):
        self.calls.append(kw)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def _embed(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.0, 1.0]) for _ in kw["input"]])


class FakeAnthropic:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(json.loads(json.dumps(kw)))   # a copy, as sent
        return self.responses.pop(0)


def user_with_image(text):
    return {"role": "user", "content": [{"type": "text", "text": text},
                                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}]}


class ProviderTest(unittest.TestCase):
    def setUp(self):
        providers._KNOWN_TOOLS.clear()
        providers._NO_TEMPERATURE.clear()


# ---------------------------------------------------------------- OpenAI
class OpenAIMapping(ProviderTest):
    def test_no_reasoning_sends_effort_none_and_temperature(self):
        fake = FakeOpenAI(oa_completion("ok"))
        providers.openai_chat("small-model", [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
                              reasoning="none", max_tokens=300, temperature=0.2, client=fake)
        kw = fake.calls[0]
        self.assertEqual(kw["reasoning_effort"], "none")
        self.assertEqual(kw["temperature"], 0.2)
        self.assertEqual(kw["max_completion_tokens"], 300)
        self.assertNotIn("max_tokens", kw)
        self.assertEqual(kw["messages"][0], {"role": "system", "content": "sys"})

    def test_reasoning_level_maps_to_effort_and_drops_temperature(self):
        for level in ("low", "medium", "high"):
            fake = FakeOpenAI(oa_completion("ok", reasoning=40))
            r = providers.openai_chat("small-model", [{"role": "user", "content": "hi"}], reasoning=level,
                                      temperature=0.4, client=fake)
            self.assertEqual(fake.calls[0]["reasoning_effort"], level)
            self.assertNotIn("temperature", fake.calls[0])
            self.assertEqual(r.usage["reasoning"], 40)

    def test_temperature_rejected_once_then_never_sent(self):
        err = FakeBadRequest("Unsupported parameter: 'temperature' is not supported with this model.")
        fake = FakeOpenAI(err, oa_completion("ok"), oa_completion("ok"))
        r = providers.openai_chat("small-model", [{"role": "user", "content": "hi"}], client=fake)
        self.assertIn("temperature", fake.calls[0])
        self.assertNotIn("temperature", fake.calls[1])
        self.assertIn("rejected", r.meta["sent"]["temperature"])
        providers.openai_chat("small-model", [{"role": "user", "content": "again"}], client=fake)
        self.assertNotIn("temperature", fake.calls[2])

    def test_schema_is_made_strict_and_added_nulls_are_removed(self):
        answer = {"answer": "x", "note": None, "items": [{"name": "a", "qty": None}]}
        fake = FakeOpenAI(oa_completion(json.dumps(answer)))
        r = providers.openai_chat("small-model", [{"role": "user", "content": "hi"}], schema=SCHEMA, client=fake)
        rf = fake.calls[0]["response_format"]
        self.assertEqual(rf["type"], "json_schema")
        self.assertTrue(rf["json_schema"]["strict"])
        s = rf["json_schema"]["schema"]
        self.assertEqual(sorted(s["required"]), ["answer", "items", "note"])
        self.assertEqual(s["properties"]["note"]["type"], ["string", "null"])
        self.assertNotIn("maxLength", s["properties"]["answer"])                    # moved into the description
        self.assertIn("maxLength: 40", s["properties"]["answer"]["description"])
        self.assertEqual(s["properties"]["items"]["maxItems"], 3)                  # supported: kept
        item = s["properties"]["items"]["items"]
        self.assertFalse(item["additionalProperties"])
        self.assertEqual(item["properties"]["qty"]["type"], ["integer", "null"])
        self.assertEqual(json.loads(r.content), {"answer": "x", "items": [{"name": "a"}]})
        self.assertEqual(SCHEMA["properties"]["answer"], {"type": "string", "maxLength": 40})   # input untouched

    def test_tools_images_and_history_are_passed_in_chat_completions_shape(self):
        fake = FakeOpenAI(oa_completion(None, tool_calls=[("add", {"a": 1, "b": 2})], finish="tool_calls"),
                          oa_completion('{"answer": "3", "items": []}'))
        msgs = [{"role": "system", "content": "sys"}, user_with_image("read it")]
        r = providers.openai_chat("small-model", msgs, tools=[TOOL], client=fake)
        self.assertEqual(fake.calls[0]["tools"], [TOOL])
        self.assertEqual(fake.calls[0]["messages"][1]["content"][1]["image_url"]["url"], "data:image/png;base64,QUJD")
        self.assertEqual(r.finish_reason, "tool_calls")
        self.assertEqual((r.tool_calls[0].function.name, json.loads(r.tool_calls[0].function.arguments)),
                         ("add", {"a": 1, "b": 2}))
        # Agent.run appends the SDK message dump (with extra fields) and the tool result, then asks for JSON.
        msgs += [r.raw_message.model_dump(exclude_none=True),
                 {"role": "tool", "tool_call_id": r.tool_calls[0].id, "content": "3"},
                 {"role": "assistant", "content": "(tools finished)"}, {"role": "user", "content": "JSON now"}]
        providers.openai_chat("small-model", msgs, schema=SCHEMA, client=fake)
        sent = fake.calls[1]
        self.assertEqual(sent["messages"][2], {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_0", "type": "function", "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'}}]})
        self.assertEqual(sent["messages"][3], {"role": "tool", "tool_call_id": "call_0", "content": "3"})
        self.assertEqual(sent["tools"], [TOOL])           # the earlier tool, so the history stays valid
        self.assertEqual(sent["tool_choice"], "none")

    def test_gpt6_tools_only_without_reasoning(self):
        msgs = [{"role": "user", "content": "add"}]
        kw, _ = providers.openai_request("gpt-6-luna", msgs, tools=[TOOL], reasoning="none")
        self.assertEqual(kw["reasoning_effort"], "none")
        with self.assertRaises(ValueError):          # Chat Completions: function calls only with effort none
            providers.openai_request("gpt-6-luna", msgs, tools=[TOOL], reasoning="low")
        kw, _ = providers.openai_request("gpt-6-luna", msgs, reasoning="low")   # reasoning without tools is fine
        self.assertEqual(kw["reasoning_effort"], "low")

    def test_usage_accounting(self):
        fake = FakeOpenAI(oa_completion("ok", prompt=5000, cached=3000, written=1500, completion=400, reasoning=120))
        r = providers.openai_chat("small-model", [{"role": "user", "content": "hi"}], client=fake)
        self.assertEqual(r.usage, {"prompt_total": 5000, "input_uncached": 500, "cache_read": 3000, "cache_write": 1500,
                                   "cache_write_1h": 0, "output": 400, "reasoning": 120, "images": 0})
        self.assertEqual((r.prompt_tokens, r.completion_tokens), (5000, 400))
        self.assertEqual((r.provider, r.meta["response_model"], r.meta["service_tier"]),
                         ("openai", "small-model-2026-01-01", "default"))

    def test_refusal_is_reported(self):
        fake = FakeOpenAI(oa_completion(None, refusal="I can't help with that."))
        r = providers.openai_chat("small-model", [{"role": "user", "content": "hi"}], schema=SCHEMA, client=fake)
        self.assertEqual((r.content, r.finish_reason), ("", "refusal"))


# ---------------------------------------------------------------- Anthropic
class AnthropicMapping(ProviderTest):
    def test_system_cache_no_temperature_and_thinking_off(self):
        fake = FakeAnthropic(an_message([{"type": "text", "text": "ok"}]))
        providers.anthropic_chat("small-model", [{"role": "system", "content": "You are X."},
                                                 {"role": "user", "content": "hi"}],
                                 reasoning="none", max_tokens=200, temperature=0.0, client=fake)
        kw = fake.calls[0]
        self.assertEqual(kw["system"], [{"type": "text", "text": "You are X.", "cache_control": {"type": "ephemeral"}}])
        self.assertEqual(kw["messages"], [{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
        self.assertEqual(kw["thinking"], {"type": "disabled"})
        self.assertEqual(kw["max_tokens"], 200)
        for k in ("temperature", "top_p", "top_k", "output_config"):
            self.assertNotIn(k, kw)

    def test_prompt_cache_can_be_turned_off(self):
        fake = FakeAnthropic(an_message([{"type": "text", "text": "ok"}]))
        with mock.patch.dict(os.environ, {"AGENTKIT_PROMPT_CACHE": "off"}):
            providers.anthropic_chat("small-model", [{"role": "system", "content": "S"},
                                                     {"role": "user", "content": "hi"}], client=fake)
        self.assertEqual(fake.calls[0]["system"], [{"type": "text", "text": "S"}])

    def test_reasoning_level_is_adaptive_thinking_with_effort(self):
        for level in ("low", "medium", "high"):
            fake = FakeAnthropic(an_message([{"type": "thinking", "thinking": "", "signature": "s"},
                                             {"type": "text", "text": "ok"}], thinking=30))
            r = providers.anthropic_chat("small-model", [{"role": "user", "content": "hi"}], reasoning=level,
                                         client=fake)
            self.assertEqual(fake.calls[0]["thinking"], {"type": "adaptive"})
            self.assertEqual(fake.calls[0]["output_config"], {"effort": level})
            self.assertEqual((r.content, r.usage["reasoning"]), ("ok", 30))

    def test_schema_uses_output_config_format(self):
        fake = FakeAnthropic(an_message([{"type": "text", "text": '{"answer": "x", "items": []}'}]))
        r = providers.anthropic_chat("small-model", [{"role": "user", "content": "hi"}], schema=SCHEMA, client=fake)
        fmt = fake.calls[0]["output_config"]["format"]
        self.assertEqual(fmt["type"], "json_schema")
        self.assertEqual(fmt["schema"], json.loads(json.dumps(anthropic.transform_schema(SCHEMA))))
        self.assertFalse(fmt["schema"]["properties"]["items"]["items"]["additionalProperties"])
        self.assertEqual(json.loads(r.content), {"answer": "x", "items": []})

    def test_images_become_base64_blocks_in_order(self):
        fake = FakeAnthropic(an_message([{"type": "text", "text": "ok"}]))
        providers.anthropic_chat("small-model", [user_with_image("read it")], client=fake)
        self.assertEqual(fake.calls[0]["messages"][0]["content"], [
            {"type": "text", "text": "read it"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}}])

    def test_tool_loop_and_format_turn(self):
        fake = FakeAnthropic(
            an_message([{"type": "thinking", "thinking": "", "signature": "sig1"},
                        {"type": "tool_use", "id": "toolu_1", "name": "add", "input": {"a": 1, "b": 2}},
                        {"type": "tool_use", "id": "toolu_2", "name": "add", "input": {"a": 3, "b": 4}}],
                       stop="tool_use"),
            an_message([{"type": "text", "text": "3 and 7"}]),
            an_message([{"type": "text", "text": '{"answer": "3, 7", "items": []}'}]))
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "add"}]
        r = providers.anthropic_chat("small-model", msgs, tools=[TOOL], reasoning="low", client=fake)
        self.assertEqual(fake.calls[0]["tools"], [{"name": "add", "description": "Add two numbers.",
                                                   "input_schema": TOOL["function"]["parameters"]}])
        self.assertEqual(r.finish_reason, "tool_calls")
        self.assertEqual([(t.id, t.function.name, json.loads(t.function.arguments)) for t in r.tool_calls],
                         [("toolu_1", "add", {"a": 1, "b": 2}), ("toolu_2", "add", {"a": 3, "b": 4})])
        msgs.append(r.raw_message.model_dump(exclude_none=True))
        msgs += [{"role": "tool", "tool_call_id": "toolu_1", "content": "3"},
                 {"role": "tool", "tool_call_id": "toolu_2", "content": "7"}]
        r2 = providers.anthropic_chat("small-model", msgs, tools=[TOOL], reasoning="low", client=fake)
        sent = fake.calls[1]["messages"]
        self.assertEqual(sent[1]["content"][0], {"type": "thinking", "thinking": "", "signature": "sig1"})  # unchanged
        self.assertEqual(sent[2], {"role": "user", "content": [   # both results in one user message
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "3"},
            {"type": "tool_result", "tool_use_id": "toolu_2", "content": "7"}]})
        # Agent.run's format turn: no tools, thinking off, a JSON schema.
        msgs += [{"role": "assistant", "content": r2.content}, {"role": "user", "content": "Now JSON."}]
        providers.anthropic_chat("small-model", msgs, schema=SCHEMA, reasoning="none", client=fake)
        kw = fake.calls[2]
        self.assertEqual(kw["tool_choice"], {"type": "none"})
        self.assertEqual([t["name"] for t in kw["tools"]], ["add"])
        self.assertNotIn("thinking", [b["type"] for m in kw["messages"] for b in m["content"]])   # stripped: thinking off
        self.assertEqual(kw["messages"][-1], {"role": "user", "content": [
            {"type": "text", "text": "Now JSON."}]})
        self.assertEqual([m["role"] for m in kw["messages"]], ["user", "assistant", "user", "assistant", "user"])

    def test_usage_accounting_and_stop_reasons(self):
        fake = FakeAnthropic(an_message([{"type": "text", "text": "cut"}], stop="max_tokens", inp=40, out=500,
                                        read=900, write5=600, write1h=100, thinking=200))
        r = providers.anthropic_chat("small-model", [{"role": "user", "content": "hi"}], client=fake)
        self.assertEqual(r.usage, {"input_uncached": 40, "cache_read": 900, "cache_write": 600, "cache_write_1h": 100,
                                   "output": 500, "reasoning": 200, "prompt_total": 1640})
        self.assertEqual((r.prompt_tokens, r.completion_tokens, r.finish_reason), (1640, 500, "length"))
        fake = FakeAnthropic(an_message([], stop="refusal"))
        r = providers.anthropic_chat("small-model", [{"role": "user", "content": "hi"}], client=fake)
        self.assertEqual((r.content, r.finish_reason), ("", "refusal"))

    def test_empty_assistant_turn_is_not_sent_empty(self):
        fake = FakeAnthropic(an_message([{"type": "text", "text": "{}"}]))
        providers.anthropic_chat("small-model", [{"role": "user", "content": "a"}, {"role": "assistant", "content": ""},
                                                 {"role": "user", "content": "b"}], client=fake)
        self.assertEqual(fake.calls[0]["messages"][1]["content"], [{"type": "text", "text": "(empty answer)"}])


# ---------------------------------------------------------------- the whole Agent.run loop on each backend
class AgentOnProviders(ProviderTest):
    def run_agent(self, provider, fake):
        tb = Toolbox()
        def add(a, b):
            return {"sum": a + b}
        tb.register(add, "Add two numbers.", TOOL["function"]["parameters"]["properties"], ["a", "b"])
        card = AgentCard(name="adder", tier="worker", job="add numbers", instructions="Use the tool.", tools=["add"],
                         output_schema={"type": "object", "additionalProperties": False, "required": ["sum"],
                                        "properties": {"sum": {"type": "number"}}}, model="small-model")
        tracer = Tracer("unit", quiet=True)
        with mock.patch.object(llm, "PROVIDER", provider), mock.patch.object(providers, "get_client", lambda p: fake):
            out = Agent(card, tb, tracer).run("add 2 and 3", label="t")
        return out, tracer

    def test_openai(self):
        fake = FakeOpenAI(oa_completion(None, tool_calls=[("add", {"a": 2, "b": 3})], finish="tool_calls", cached=0),
                          oa_completion("The sum is 5."), oa_completion('{"sum": 5}', prompt=1200, cached=1024))
        out, tracer = self.run_agent("openai", fake)
        self.assertEqual(out, {"sum": 5})
        self.assertEqual(fake.calls[1]["messages"][-1]["role"], "tool")
        llm_events = [e for e in tracer.events if e["kind"] == "llm"]
        self.assertEqual([e["provider"] for e in llm_events], ["openai"] * 3)
        self.assertEqual(llm_events[-1]["usage"]["cache_read"], 1024)
        self.assertEqual(tracer.summary()["usage"]["cache_read"], 1024)

    def test_anthropic(self):
        fake = FakeAnthropic(
            an_message([{"type": "tool_use", "id": "toolu_9", "name": "add", "input": {"a": 2, "b": 3}}], stop="tool_use"),
            an_message([{"type": "text", "text": "The sum is 5."}]),
            an_message([{"type": "text", "text": '{"sum": 5}'}], read=512))
        out, tracer = self.run_agent("anthropic", fake)
        self.assertEqual(out, {"sum": 5})
        self.assertEqual(fake.calls[1]["messages"][-1]["content"][0]["type"], "tool_result")
        self.assertEqual(json.loads(fake.calls[1]["messages"][-1]["content"][0]["content"]), {"sum": 5})
        self.assertEqual(fake.calls[2]["tool_choice"], {"type": "none"})
        self.assertEqual([e["usage"]["cache_read"] for e in tracer.events if e["kind"] == "llm"], [0, 0, 512])


# ---------------------------------------------------------------- keys, traces, the LM Studio path
class KeysAndTraces(ProviderTest):
    def setUp(self):
        super().setUp()
        providers._keys.clear()
        self.addCleanup(providers._keys.clear)

    def test_key_from_sot_variable_only(self):
        with mock.patch.dict(os.environ, {"SOT_ANTHROPIC_API_KEY": FAKE_KEY, "ANTHROPIC_API_KEY": "generic"}), \
                mock.patch.object(providers, "KEY_FILE", Path("/nonexistent/keys.env")):
            self.assertEqual(providers.api_key("anthropic"), FAKE_KEY)

    def test_key_from_file_and_missing_key_message(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "keys.env"
            f.write_text(f"# keys\nexport SOT_OPENAI_API_KEY=\"{FAKE_KEY}\"\nOTHER=1\n")
            env = {k: v for k, v in os.environ.items() if not k.startswith("SOT_")}
            with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(providers, "KEY_FILE", f):
                self.assertEqual(providers.api_key("openai"), FAKE_KEY)
                with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "generic", "ANTHROPIC_API_KEY": "generic"}):
                    with self.assertRaises(RuntimeError) as err:
                        providers.api_key("anthropic")       # the generic variable is never used
                self.assertNotIn(FAKE_KEY, str(err.exception))

    def test_trace_is_named_by_label_records_setup_and_never_holds_a_key(self):
        providers._keys["anthropic"] = FAKE_KEY
        tracer = Tracer("unit", quiet=True)
        reply = llm.Reply(content="x", prompt_tokens=10, completion_tokens=2, provider="anthropic",
                          usage={"input_uncached": 10, "output": 2}, meta={"response_model": "m", "sent": {}})
        tracer.llm("a", "worker", "small-model", "l", reply)
        tracer.note(f"something echoed {FAKE_KEY}")
        with tempfile.TemporaryDirectory() as d, mock.patch.object(llm, "PROVIDER", "anthropic"), \
                mock.patch.dict(os.environ, {"AGENTKIT_RUN_LABEL": "haiku"}):
            path = tracer.save(Path(d))
            text = path.read_text()
        self.assertRegex(path.name, r"^unit-haiku-\d{8}-\d{6}\.json$")
        self.assertNotIn(FAKE_KEY, text)
        d = json.loads(text)
        self.assertEqual(d["setup"]["provider"], "anthropic")
        self.assertIn("local", d["setup"]["embeddings"]["where"])
        self.assertEqual(d["events"][0]["usage"], {"input_uncached": 10, "output": 2})

    def test_local_trace_name_unchanged(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AGENTKIT_RUN_LABEL", None)
            os.environ.pop("RUN_TAG", None)
            path = Tracer("unit", quiet=True).save(Path(d))
        self.assertRegex(path.name, r"^unit-local-\d{8}-\d{6}\.json$")

    def test_usage_log_has_tokens_and_no_key(self):
        providers._keys["openai"] = FAKE_KEY
        fake = FakeOpenAI(oa_completion("ok", prompt=1200, cached=1024))
        with tempfile.TemporaryDirectory() as d, mock.patch.object(providers, "get_client", lambda p: fake):
            log = Path(d) / "usage.jsonl"
            with mock.patch.dict(os.environ, {"AGENTKIT_USAGE_LOG": str(log)}):
                providers.chat("openai", "small-model", [{"role": "user", "content": "hi"}])
            text = log.read_text()
        self.assertNotIn(FAKE_KEY, text)
        line = json.loads(text)
        self.assertEqual((line["provider"], line["model"], line["usage"]["cache_read"]), ("openai", "small-model", 1024))

    def test_lmstudio_request_is_unchanged(self):
        fake = FakeOpenAI(oa_completion("ok"))
        with mock.patch.object(llm, "PROVIDER", "lmstudio"), mock.patch.object(llm, "client", fake):
            llm.chat("google/gemma-4-12b", [{"role": "user", "content": "hi"}], tools=[TOOL], schema=SCHEMA,
                     reasoning="none", max_tokens=99, temperature=0.3)
            llm.embed(["a", "b"])
        self.assertEqual(fake.calls[0], {
            "model": "google/gemma-4-12b", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 99,
            "temperature": 0.3, "reasoning_effort": "none", "tools": [TOOL],
            "response_format": {"type": "json_schema", "json_schema": {"name": "output", "strict": True,
                                                                       "schema": SCHEMA}}})
        self.assertEqual(fake.calls[1], {"model": llm.EMBED_MODEL, "input": ["a", "b"]})


if __name__ == "__main__":
    unittest.main()
