"""govuk_corpus.llm: the request builder (pure) and the sampling-capability guard. No network.
Run: python3 -m unittest -v tests.test_llm"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from govuk_corpus import llm

CFG = {"provider": "anthropic", "model": "claude-haiku-4-5-20251001", "key": "k", "base_url": "",
       "label": "Anthropic", "price_in": 1.0, "price_out": 5.0}
MSGS = [{"role": "user", "content": "hi"}]


class TestRequestKwargs(unittest.TestCase):
    def test_defaults_send_no_sampling_controls(self):
        kw = llm.request_kwargs(CFG, "", MSGS, 4096)
        self.assertEqual(set(kw), {"model", "max_tokens", "messages"})
        self.assertEqual(kw["max_tokens"], 4096)

    def test_temperature_only_when_given(self):
        kw = llm.request_kwargs(CFG, "sys", MSGS, 100, temperature=0)
        self.assertEqual(kw["temperature"], 0.0)
        self.assertEqual(kw["system"], "sys")
        self.assertNotIn("thinking", kw)

    def test_system_cache_control_on_anthropic_and_foundry(self):
        kw = llm.request_kwargs(CFG, "sys", MSGS, 100, cache_system=True)
        self.assertEqual(kw["system"][0]["cache_control"], {"type": "ephemeral"})
        fo = dict(CFG, provider="foundry", model="claude-haiku-4-5")
        self.assertEqual(llm.request_kwargs(fo, "sys", MSGS, 100, cache_system=True)["system"][0]["cache_control"],
                         {"type": "ephemeral"})
        ds = dict(CFG, provider="deepseek", model="deepseek-chat")
        self.assertEqual(llm.request_kwargs(ds, "sys", MSGS, 100, cache_system=True)["system"], "sys")

    def test_thinking_and_effort_only_for_claude_providers(self):
        kw = llm.request_kwargs(CFG, "", MSGS, 100, thinking={"type": "adaptive"}, effort="low")
        self.assertEqual(kw["thinking"], {"type": "adaptive"})
        self.assertEqual(kw["output_config"], {"effort": "low"})
        ds = dict(CFG, provider="deepseek", model="deepseek-chat")
        kw = llm.request_kwargs(ds, "", MSGS, 100, thinking={"type": "adaptive"}, effort="low")
        self.assertNotIn("thinking", kw)
        self.assertNotIn("output_config", kw)
        br = dict(CFG, provider="bedrock", model="eu.anthropic.claude-haiku-4-5-20251001-v1:0")
        self.assertIn("thinking", llm.request_kwargs(br, "", MSGS, 100, thinking={"type": "adaptive"}))
        fo = dict(CFG, provider="foundry", model="claude-haiku-4-5")
        kw = llm.request_kwargs(fo, "", MSGS, 100, thinking={"type": "adaptive"}, effort="low")
        self.assertEqual(kw["thinking"], {"type": "adaptive"})
        self.assertEqual(kw["output_config"], {"effort": "low"})

    def test_temperature_refused_on_models_that_reject_it(self):
        for m in ("claude-sonnet-5", "claude-opus-5", "claude-opus-4-7", "claude-fable-5-1",
                  "eu.anthropic.claude-sonnet-5-v1:0"):
            self.assertFalse(llm.supports_sampling(m), m)
            with self.assertRaises(ValueError):
                llm.request_kwargs(dict(CFG, model=m), "", MSGS, 100, temperature=0)
            # …but without a temperature the request is fine.
            llm.request_kwargs(dict(CFG, model=m), "", MSGS, 100)
        for m in ("claude-haiku-4-5-20251001", "claude-sonnet-4-6", "deepseek-chat"):
            self.assertTrue(llm.supports_sampling(m), m)

    def test_chat_reports_refused_temperature_as_fatal(self):
        res = llm.chat(dict(CFG, model="claude-sonnet-5"), "", MSGS, 10, temperature=0)
        self.assertTrue(res.get("fatal"))
        self.assertIn("temperature", res["error"])

    def test_chat_without_key_is_fatal(self):
        res = llm.chat(dict(CFG, key=None), "", MSGS, 10)
        self.assertTrue(res.get("fatal"))


class TestReply(unittest.TestCase):
    def test_reply_passes_sampling_through(self):
        seen = {}

        def fake_chat(config, system, messages, max_tokens=1024, cache_system=False, **kw):
            seen.update(kw)
            seen["max_tokens"] = max_tokens
            return {"reply": "{}"}
        orig = llm.chat
        llm.chat = fake_chat
        try:
            llm.reply(CFG, "", "p", sampling={"temperature": 0.0, "thinking": None, "effort": None})
        finally:
            llm.chat = orig
        self.assertEqual(seen["temperature"], 0.0)
        self.assertIsNone(seen["thinking"])
        self.assertEqual(seen["max_tokens"], llm.eval_max_tokens())


_FOUNDRY_ENVS = ("ANTHROPIC_FOUNDRY_API_KEY", "ANTHROPIC_FOUNDRY_RESOURCE",
                 "ANTHROPIC_FOUNDRY_BASE_URL")


def _has_azure_identity() -> bool:
    try:
        return __import__("importlib").util.find_spec("azure.identity") is not None
    except ImportError:   # parent `azure` namespace itself missing
        return False


class TestFoundryCreds(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in _FOUNDRY_ENVS}
        for k in _FOUNDRY_ENVS:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_unconfigured_without_endpoint(self):
        self.assertIsNone(llm.foundry_creds())
        self.assertFalse(llm.provider_configured("foundry"))
        os.environ["ANTHROPIC_FOUNDRY_API_KEY"] = "k"   # key alone is not enough
        self.assertIsNone(llm.foundry_creds())

    def test_key_plus_resource(self):
        os.environ["ANTHROPIC_FOUNDRY_API_KEY"] = "k"
        os.environ["ANTHROPIC_FOUNDRY_RESOURCE"] = "res"
        creds = llm.foundry_creds()
        self.assertEqual((creds["api_key"], creds["resource"]), ("k", "res"))
        self.assertTrue(llm.provider_configured("foundry"))

    def test_base_url_alternative_to_resource(self):
        os.environ["ANTHROPIC_FOUNDRY_API_KEY"] = "k"
        os.environ["ANTHROPIC_FOUNDRY_BASE_URL"] = "https://res.services.ai.azure.com/anthropic/"
        creds = llm.foundry_creds()
        self.assertEqual(creds["base_url"], "https://res.services.ai.azure.com/anthropic")

    def test_chat_without_foundry_config_is_fatal(self):
        fo = dict(CFG, provider="foundry", model="claude-haiku-4-5", key=None)
        res = llm.chat(fo, "", MSGS, 10)
        self.assertTrue(res.get("fatal"))

    @unittest.skipUnless(_has_azure_identity(), "no azure-identity")
    def test_entra_when_keyless_with_endpoint(self):
        os.environ["ANTHROPIC_FOUNDRY_RESOURCE"] = "res"
        creds = llm.foundry_creds()
        self.assertTrue(creds.get("use_entra"))
        self.assertTrue(llm.provider_configured("foundry"))


_OPENAI_ENVS = ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_BASE_URL", "AZURE_OPENAI_RESOURCE",
                "FOUNDRY_OPENAI_API_KEY", "FOUNDRY_OPENAI_BASE_URL", "FOUNDRY_OPENAI_RESOURCE",
                "ANTHROPIC_FOUNDRY_API_KEY", "ANTHROPIC_FOUNDRY_RESOURCE",
                "ANTHROPIC_FOUNDRY_BASE_URL")


class TestFoundryOpenAI(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in _OPENAI_ENVS}
        for k in _OPENAI_ENVS:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_unconfigured_without_endpoint(self):
        self.assertIsNone(llm.foundry_openai_creds())
        self.assertFalse(llm.provider_configured("foundry_openai"))
        os.environ["AZURE_OPENAI_API_KEY"] = "k"
        self.assertIsNone(llm.foundry_openai_creds())

    def test_explicit_url_and_key(self):
        os.environ["AZURE_OPENAI_API_KEY"] = "ok"
        os.environ["AZURE_OPENAI_BASE_URL"] = "https://custom.openai.azure.com/openai/v1/"
        os.environ["ANTHROPIC_FOUNDRY_RESOURCE"] = "other"
        creds = llm.foundry_openai_creds()
        self.assertEqual(creds["api_key"], "ok")
        self.assertEqual(creds["base_url"], "https://custom.openai.azure.com/openai/v1")
        self.assertTrue(llm.provider_configured("foundry_openai"))

    def test_reuses_claude_foundry_resource(self):
        os.environ["ANTHROPIC_FOUNDRY_API_KEY"] = "k"
        os.environ["ANTHROPIC_FOUNDRY_RESOURCE"] = "res"
        creds = llm.foundry_openai_creds()
        self.assertEqual(creds["api_key"], "k")
        self.assertEqual(creds["base_url"], "https://res.services.ai.azure.com/openai/v1")

    def test_rewrites_claude_base_url(self):
        os.environ["ANTHROPIC_FOUNDRY_API_KEY"] = "k"
        os.environ["ANTHROPIC_FOUNDRY_BASE_URL"] = "https://res.services.ai.azure.com/anthropic/"
        self.assertEqual(llm.foundry_openai_creds()["base_url"],
                         "https://res.services.ai.azure.com/openai/v1")

    def test_request_is_chat_completions(self):
        cfg = dict(CFG, provider="foundry_openai", model="gpt-5.4")
        kw = llm.openai_request_kwargs(cfg, "sys", MSGS, 4096, temperature=0)
        self.assertEqual(kw["model"], "gpt-5.4")
        self.assertEqual(kw["max_completion_tokens"], 4096)
        self.assertNotIn("max_tokens", kw)
        self.assertEqual(kw["temperature"], 0.0)
        self.assertEqual(kw["messages"][0], {"role": "system", "content": "sys"})
        self.assertNotIn("reasoning_effort", kw)
        kw = llm.openai_request_kwargs(cfg, "", MSGS, 100, effort="low")
        self.assertEqual(kw["reasoning_effort"], "low")
        self.assertNotIn("temperature", kw)
        with self.assertRaises(ValueError):
            llm.openai_request_kwargs(cfg, "", MSGS, 100, temperature=0, effort="low")

    def test_chat_without_config_is_fatal(self):
        fo = dict(CFG, provider="foundry_openai", model="gpt-5.4", key=None, base_url="")
        res = llm.chat(fo, "", MSGS, 10)
        self.assertTrue(res.get("fatal"))

    def test_chat_prices_cached_prompt_tokens(self):
        seen = {}

        class _Msg:
            content = '{"keep": true}'
            refusal = None

        class _Choice:
            message = _Msg()
            finish_reason = "stop"

        class _Completion:
            model = "gpt-5.4-2026-03-05"
            choices = [_Choice()]
            usage = {"prompt_tokens": 1000, "completion_tokens": 20,
                     "prompt_tokens_details": {"cached_tokens": 800}}

        class _Completions:
            @staticmethod
            def create(**kwargs):
                seen.update(kwargs)
                return _Completion()

        class _Chat:
            completions = _Completions()

        class _Client:
            chat = _Chat()

        orig = llm._openai_client
        llm._openai_client = lambda config, timeout, max_retries: (_Client(), None)
        try:
            cfg = dict(CFG, provider="foundry_openai", model="gpt-5.4", key="k",
                       base_url="https://res.openai.azure.com/openai/v1",
                       price_in=2.5, price_out=15.0,
                       grid={"in_hit_off": 0.25, "in_hit_peak": 0.25,
                             "in_miss_off": 2.5, "in_miss_peak": 2.5,
                             "out_off": 15.0, "out_peak": 15.0})
            res = llm.chat(cfg, "sys", MSGS, 128, temperature=0)
        finally:
            llm._openai_client = orig
        self.assertEqual(res["reply"], '{"keep": true}')
        self.assertEqual(res["actual_model"], "gpt-5.4-2026-03-05")
        self.assertEqual(res["stop_reason"], "stop")
        self.assertEqual((res["input_tokens"], res["cache_hit_tokens"], res["output_tokens"]),
                         (1000, 800, 20))
        self.assertAlmostEqual(res["cost_usd"],
                               (200 / 1e6) * 2.5 + (800 / 1e6) * 0.25 + (20 / 1e6) * 15.0, places=6)
        self.assertEqual(seen["max_completion_tokens"], 128)
        self.assertEqual(seen["temperature"], 0.0)

    @unittest.skipUnless(_has_azure_identity(), "no azure-identity")
    def test_entra_when_keyless_with_endpoint(self):
        os.environ["AZURE_OPENAI_RESOURCE"] = "res"
        creds = llm.foundry_openai_creds()
        self.assertTrue(creds.get("use_entra"))
        self.assertTrue(llm.provider_configured("foundry_openai"))


if __name__ == "__main__":
    unittest.main()
