"""DeepSeek classification falls back to OpenRouter on hard failures."""

from __future__ import annotations

import io
import json
import os
import unittest
import urllib.error
from email.message import Message
from pathlib import Path
from unittest import mock

import deepseek_classifier as dc
from deepseek_classifier import (
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_OPENROUTER_MODEL,
    ProviderError,
    classify_messages,
    classify_report,
)

ROOT = Path(__file__).resolve().parents[1]
OK_JSON = json.dumps(
    {
        "report_type": "公司研究",
        "institution": "示例证券",
        "title": "示例公司点评",
        "companies": ["示例公司"],
        "tickers": ["000001"],
        "industries": ["电子"],
        "sentiment": "正面",
        "summary": "需求回升。",
        "tags": ["点评"],
    },
    ensure_ascii=False,
)


def _chat_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


class FallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = {
            "DEEPSEEK_API_KEY": "test-deepseek-key",
            "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
            "DEEPSEEK_MODEL": "deepseek-chat",
            "OPENROUTER_API_KEY": "test-openrouter-key",
            "OPENROUTER_BASE_URL": DEFAULT_OPENROUTER_BASE_URL,
            "OPENROUTER_MODEL": DEFAULT_OPENROUTER_MODEL,
        }
        self.env_patch = mock.patch.dict(os.environ, self.env, clear=False)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def test_deepseek_success_does_not_call_openrouter(self) -> None:
        calls: list[str] = []

        def fake(url, api_key, payload, timeout, *, provider):
            calls.append(provider)
            self.assertEqual(payload["messages"][0]["role"], "system")
            return _chat_response(OK_JSON)

        with mock.patch.object(dc, "_post_chat", side_effect=fake):
            result = classify_report("营收同比增长。", title="示例点评")
        self.assertEqual(calls, ["deepseek"])
        self.assertEqual(result.provider, "deepseek")
        self.assertEqual(result.model, "deepseek-chat")
        self.assertEqual(result.data["report_type"], "公司研究")

    def test_http_402_falls_back_and_logs_the_switch(self) -> None:
        def fake(url, api_key, payload, timeout, *, provider):
            if provider == "deepseek":
                raise ProviderError(
                    "deepseek",
                    402,
                    '{"error":{"message":"Insufficient Balance"}}',
                    "Insufficient Balance",
                )
            self.assertEqual(payload["model"], DEFAULT_OPENROUTER_MODEL)
            self.assertNotIn(":free", payload["model"])
            return _chat_response(OK_JSON)

        with mock.patch.object(dc, "_post_chat", side_effect=fake):
            with self.assertLogs("deepseek_classifier", level="INFO") as logs:
                result = classify_report("正文", title="标题")
        self.assertEqual(result.provider, "openrouter")
        self.assertEqual(result.model, DEFAULT_OPENROUTER_MODEL)
        joined = "\n".join(logs.output)
        self.assertIn("falling back to OpenRouter", joined)
        self.assertIn("HTTP 402", joined)
        self.assertIn(DEFAULT_OPENROUTER_MODEL, joined)
        self.assertIn("OpenRouter fallback completed", joined)

    def test_balance_language_on_other_status_falls_back(self) -> None:
        def fake(url, api_key, payload, timeout, *, provider):
            if provider == "deepseek":
                raise ProviderError("deepseek", 400, "余额不足，请充值", "余额不足")
            return _chat_response(OK_JSON)

        with mock.patch.object(dc, "_post_chat", side_effect=fake):
            result = classify_messages([{"role": "user", "content": "hi"}])
        self.assertEqual(result.provider, "openrouter")

    def test_plain_400_does_not_fall_back(self) -> None:
        def fake(url, api_key, payload, timeout, *, provider):
            raise ProviderError("deepseek", 400, "invalid request", "invalid request")

        with mock.patch.object(dc, "_post_chat", side_effect=fake):
            with self.assertLogs("deepseek_classifier", level="ERROR") as logs:
                with self.assertRaises(ProviderError):
                    classify_messages([{"role": "user", "content": "hi"}])
        self.assertIn("not falling back", "\n".join(logs.output))

    def test_server_error_and_network_error_fall_back(self) -> None:
        for status, message in ((503, "upstream unavailable"), (None, "timed out")):
            with self.subTest(status=status):
                def fake(url, api_key, payload, timeout, *, provider, status=status, message=message):
                    if provider == "deepseek":
                        raise ProviderError("deepseek", status, message, message)
                    return _chat_response('{"ok": true}')

                with mock.patch.object(dc, "_post_chat", side_effect=fake):
                    result = classify_messages([{"role": "user", "content": "hi"}])
                self.assertEqual(result.provider, "openrouter")
                self.assertEqual(json.loads(result.content), {"ok": True})

    def test_missing_deepseek_key_uses_openrouter(self) -> None:
        os.environ["DEEPSEEK_API_KEY"] = ""

        def fake(url, api_key, payload, timeout, *, provider):
            self.assertEqual(provider, "openrouter")
            return _chat_response('{"ok": true}')

        with mock.patch.object(dc, "_post_chat", side_effect=fake):
            with self.assertLogs("deepseek_classifier", level="WARNING") as logs:
                result = classify_messages([{"role": "user", "content": "hi"}])
        self.assertEqual(result.provider, "openrouter")
        self.assertIn("using OpenRouter model", "\n".join(logs.output))

    def test_neither_provider_configured(self) -> None:
        os.environ["DEEPSEEK_API_KEY"] = ""
        os.environ["OPENROUTER_API_KEY"] = ""
        with mock.patch.object(dc, "_post_chat") as post:
            with self.assertRaises(ProviderError):
                classify_messages([{"role": "user", "content": "hi"}])
        post.assert_not_called()

    def test_openrouter_failure_after_402_is_logged(self) -> None:
        def fake(url, api_key, payload, timeout, *, provider):
            if provider == "deepseek":
                raise ProviderError("deepseek", 402, "Insufficient Balance", "Insufficient Balance")
            raise ProviderError("openrouter", 401, "unauthorized", "unauthorized")

        with mock.patch.object(dc, "_post_chat", side_effect=fake):
            with self.assertLogs("deepseek_classifier", level="WARNING") as logs:
                with self.assertRaises(ProviderError) as caught:
                    classify_messages([{"role": "user", "content": "hi"}])
        self.assertEqual(caught.exception.provider, "openrouter")
        self.assertIn("after DeepSeek failure", str(caught.exception))
        self.assertIn("OpenRouter fallback failed", "\n".join(logs.output))

    def test_retired_free_model_404_is_called_out(self) -> None:
        os.environ["OPENROUTER_MODEL"] = "minimax/minimax-m3.1-flash:free"

        def fake(url, api_key, payload, timeout, *, provider):
            if provider == "deepseek":
                raise ProviderError("deepseek", 402, "Insufficient Balance", "Insufficient Balance")
            self.assertEqual(payload["model"], "minimax/minimax-m3.1-flash:free")
            raise ProviderError("openrouter", 404, "No such model", "No such model")

        with mock.patch.object(dc, "_post_chat", side_effect=fake):
            with self.assertLogs("deepseek_classifier", level="WARNING") as logs:
                with self.assertRaises(ProviderError):
                    classify_messages([{"role": "user", "content": "hi"}])
        joined = "\n".join(logs.output)
        self.assertIn("HTTP 404", joined)
        self.assertIn(":free", joined)
        self.assertIn(DEFAULT_OPENROUTER_MODEL, joined)

    def test_empty_model_env_uses_documented_default(self) -> None:
        os.environ["OPENROUTER_MODEL"] = ""
        seen: list[str] = []

        def fake(url, api_key, payload, timeout, *, provider):
            if provider == "deepseek":
                raise ProviderError("deepseek", 402, "Insufficient Balance", "Insufficient Balance")
            seen.append(payload["model"])
            seen.append(url)
            return _chat_response('{"ok": true}')

        with mock.patch.object(dc, "_post_chat", side_effect=fake):
            result = classify_messages([{"role": "user", "content": "hi"}])
        self.assertEqual(seen[0], DEFAULT_OPENROUTER_MODEL)
        self.assertEqual(seen[1], DEFAULT_OPENROUTER_BASE_URL + "/chat/completions")
        self.assertEqual(result.model, DEFAULT_OPENROUTER_MODEL)

    def test_same_messages_are_sent_to_both_providers(self) -> None:
        payloads: list[list[dict]] = []

        def fake(url, api_key, payload, timeout, *, provider):
            payloads.append(payload["messages"])
            if provider == "deepseek":
                raise ProviderError("deepseek", 402, "Insufficient Balance", "Insufficient Balance")
            return _chat_response('{"ok": true}')

        messages = [{"role": "user", "content": "same body"}]
        with mock.patch.object(dc, "_post_chat", side_effect=fake):
            classify_messages(messages)
        self.assertEqual(payloads[0], payloads[1])

    def test_json_fence_is_parsed(self) -> None:
        fenced = "```json\n" + OK_JSON + "\n```"

        def fake(url, api_key, payload, timeout, *, provider):
            return _chat_response(fenced)

        with mock.patch.object(dc, "_post_chat", side_effect=fake):
            result = classify_report("正文", title="标题")
        self.assertEqual(result.data["sentiment"], "正面")

    def test_secrets_in_provider_errors_are_redacted(self) -> None:
        leaked = "Bearer test-openrouter-key"

        def fake(url, api_key, payload, timeout, *, provider):
            if provider == "deepseek":
                raise ProviderError("deepseek", 402, f"bad {leaked}", f"bad {leaked}")
            return _chat_response('{"ok": true}')

        with mock.patch.object(dc, "_post_chat", side_effect=fake):
            with self.assertLogs("deepseek_classifier", level="WARNING") as logs:
                classify_messages([{"role": "user", "content": "hi"}])
        joined = "\n".join(logs.output)
        self.assertNotIn("test-openrouter-key", joined)
        self.assertIn("[redacted]", joined)

    def test_post_chat_turns_http_402_into_provider_error(self) -> None:
        body = b'{"error":{"message":"Insufficient Balance","type":"insufficient_quota"}}'
        error = urllib.error.HTTPError(
            url="https://api.deepseek.com/chat/completions",
            code=402,
            msg="Payment Required",
            hdrs=Message(),
            fp=io.BytesIO(body),
        )
        captured = {}

        def urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["auth"] = request.get_header("Authorization")
            raise error

        with mock.patch("urllib.request.urlopen", side_effect=urlopen):
            with self.assertRaises(ProviderError) as caught:
                dc._post_chat(
                    "https://api.deepseek.com/chat/completions",
                    "test-deepseek-key",
                    {"model": "deepseek-chat", "messages": []},
                    5,
                    provider="deepseek",
                )
        self.assertEqual(caught.exception.status, 402)
        self.assertIn("Insufficient Balance", caught.exception.message)
        self.assertEqual(captured["auth"], "Bearer test-deepseek-key")
        self.assertTrue(captured["url"].endswith("/chat/completions"))


class SummaryPromptTests(unittest.TestCase):
    def test_summary_prompt_encodes_style_without_a_fixed_report(self) -> None:
        text = dc.SUMMARY_PROMPT
        self.assertIn(text.strip(), dc.SYSTEM_PROMPT)
        for phrase in (
            "结论先行",
            "现在怎么了",
            "关键风险",
            "EPS",
            "避开",
            "日历",
            "触发点",
        ):
            self.assertIn(phrase, text)
        self.assertNotIn("120", dc.SYSTEM_PROMPT)
        for banned in ("贵州茅台", "宁德时代", "600519", "000001.SZ", "示例公司"):
            self.assertNotIn(banned, text)
            self.assertNotIn(banned, dc.SYSTEM_PROMPT)


class EnvExampleTests(unittest.TestCase):
    def test_example_documents_openrouter_without_secrets(self) -> None:
        text = (ROOT / ".env.example").read_text(encoding="utf-8")
        for name in ("OPENROUTER_API_KEY", "OPENROUTER_MODEL", "OPENROUTER_BASE_URL"):
            self.assertIn(name, text)
        self.assertIn(DEFAULT_OPENROUTER_MODEL, text)
        self.assertIn(DEFAULT_OPENROUTER_BASE_URL, text)
        self.assertIn(":free", text)
        self.assertIn("404", text)
        lines = [line.strip() for line in text.splitlines()]
        self.assertIn("OPENROUTER_API_KEY=", lines)
        self.assertIn("DEEPSEEK_API_KEY=", lines)
        self.assertNotIn("sk-", text)
        self.assertNotRegex(text, r"Bearer\s+\S+")


if __name__ == "__main__":
    unittest.main()
