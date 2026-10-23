"""Classify research-report text with DeepSeek, then OpenRouter on hard failure.

Production calls DeepSeek's OpenAI-compatible chat completions API. When that
call cannot be served — HTTP 402 / insufficient balance, auth failures, and
similar hard errors — the same messages are retried on OpenRouter.

OpenRouter model in use: ``minimax/minimax-m3.1-flash``.
``minimax/minimax-m3.1-flash:free`` was requested and returns HTTP 404
(retired / unavailable). Do not use the ``:free`` slug.

Environment:
    DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_BASE_URL
    OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_BASE_URL
        (base URL defaults to https://openrouter.ai/api/v1)
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_OPENROUTER_MODEL = "minimax/minimax-m3.1-flash"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# The :free variant is documented here because it 404s. It is not the default.
RETIRED_OPENROUTER_FREE_MODEL = "minimax/minimax-m3.1-flash:free"

MAX_INPUT_CHARS = 100_000
DEFAULT_TIMEOUT_SECONDS = 180

# Status codes that mean DeepSeek will not complete this classification.
# 402 is the balance failure this fallback exists for. 401/403 are auth,
# 404 is a missing model or route, 408/429 and 5xx are hard for this call.
_HARD_STATUSES = frozenset({401, 402, 403, 404, 408, 409, 429})

_BALANCE_HINTS = (
    "insufficient balance",
    "insufficient_balance",
    "insufficient quota",
    "insufficient_quota",
    "payment required",
    "exceeded your current quota",
    "credit balance",
    "account balance",
    "余额不足",
    "余额不够",
    "欠费",
    "账户余额",
    "请充值",
)

_SECRET_RE = re.compile(r"(Bearer\s+)\S+|sk-[A-Za-z0-9_\-]{8,}", re.IGNORECASE)

# Style for the summary string. Describes how to write, not any one report.
SUMMARY_PROMPT = """\
用白话中文写，结论先行，写短，少行话、少翻译腔。只根据给定的这一篇来写，不要套用别的研报里的公司、数字或结论。
先用一两句话写出整体判断。后面按这篇材料往下写，材料里没有的部分就省略：
- 现在怎么了
- 关键风险，保留具体数字和阈值
- 分地区或 EPS 的含义（正文谈到才写）
- 可以买什么、避开什么
- 近期日历，以及会改变判断的触发点
数字和门槛用正文里的具体值，不要改成“大幅”“较高”这类含糊说法。读起来像在跟人讲。这段里可以用短标题或几条列表。
"""

SYSTEM_PROMPT = f"""你是卖方研报分类器。根据给定研报标题和正文，只输出一个 JSON 对象。不要用 Markdown 代码块包裹整个结果，不要在 JSON 外面加说明。
JSON 字段：
- report_type: 只能是 公司研究、行业研究、策略、宏观、固收、金工、其他 之一
- institution: 出具机构，无法判断则为空字符串
- title: 报告标题
- companies: 涉及公司名称数组，没有则为 []
- tickers: 证券代码数组，没有则为 []
- industries: 行业名称数组，没有则为 []
- sentiment: 只能是 正面、中性、负面、不确定 之一
- summary: 字符串，写法见下方「摘要」
- tags: 主题标签数组，最多 8 个

摘要：
{SUMMARY_PROMPT.strip()}
"""


class ProviderError(Exception):
    """A classification provider failed before returning usable text."""

    def __init__(self, provider: str, status: int | None, body: str, message: str):
        self.provider = provider
        self.status = status
        self.body = body
        self.message = message
        label = f"HTTP {status}" if status is not None else "network error"
        detail = _redact((message or body).strip().replace("\n", " "))[:300]
        text = f"{provider} {label}: {detail}" if detail else f"{provider} {label}"
        super().__init__(text)


@dataclass
class ClassifyResult:
    provider: str
    model: str
    content: str
    data: dict | None = None


def load_dotenv(path: str | Path | None = None) -> None:
    """Load KEY=VALUE pairs into the environment without overriding existing vars.

    Existing process environment wins, so a real key exported by the shell is
    never replaced by a file. Empty values are ignored.
    """
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    else:
        candidates.append(Path.cwd() / ".env")
        candidates.append(Path(__file__).resolve().parent / ".env")
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        for raw_line in resolved.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and value and key not in os.environ:
                os.environ[key] = value


def classify_report(text: str, *, title: str = "") -> ClassifyResult:
    """Classify one research report. DeepSeek first, OpenRouter on hard failure."""
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("report text is empty")
    if len(cleaned) > MAX_INPUT_CHARS:
        logger.warning(
            "Truncating report text from %s to %s characters",
            len(cleaned),
            MAX_INPUT_CHARS,
        )
        cleaned = cleaned[:MAX_INPUT_CHARS]
    heading = title.strip() or "（无）"
    result = classify_messages(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"标题: {heading}\n\n正文:\n{cleaned}"},
        ]
    )
    result.data = _parse_json_content(result.content)
    return result


def classify_messages(messages: list[dict]) -> ClassifyResult:
    """Send chat messages to DeepSeek, falling back to OpenRouter when needed."""
    load_dotenv()
    timeout = _timeout()
    deepseek_key = _env("DEEPSEEK_API_KEY")
    deepseek_model = _env("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
    deepseek_base = _env("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)
    openrouter_key = _env("OPENROUTER_API_KEY")
    openrouter_model = _env("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
    openrouter_base = _env("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL)

    deepseek_error: ProviderError | None = None
    if deepseek_key:
        try:
            data = _post_chat(
                _chat_url(deepseek_base),
                deepseek_key,
                _payload(deepseek_model, messages),
                timeout,
                provider="deepseek",
            )
            content = _message_text(data, provider="deepseek")
            logger.debug("DeepSeek classification completed model=%s", deepseek_model)
            return ClassifyResult("deepseek", deepseek_model, content)
        except ProviderError as exc:
            deepseek_error = exc
            if not _is_hard_failure(exc.status, f"{exc.message}\n{exc.body}"):
                logger.error(
                    "DeepSeek classification failed (%s); not falling back to OpenRouter",
                    exc,
                )
                raise
    else:
        deepseek_error = ProviderError(
            "deepseek", None, "", "DEEPSEEK_API_KEY is not set"
        )

    if not openrouter_key:
        logger.error(
            "DeepSeek classification failed (%s); OpenRouter fallback is not configured "
            "(set OPENROUTER_API_KEY)",
            deepseek_error,
        )
        raise deepseek_error

    _warn_if_retired_free_model(openrouter_model)
    if deepseek_key:
        logger.warning(
            "DeepSeek classification failed (%s); falling back to OpenRouter model %s",
            deepseek_error,
            openrouter_model,
        )
    else:
        logger.warning(
            "DEEPSEEK_API_KEY is not set; using OpenRouter model %s",
            openrouter_model,
        )

    try:
        data = _post_chat(
            _chat_url(openrouter_base),
            openrouter_key,
            _payload(openrouter_model, messages),
            timeout,
            provider="openrouter",
        )
    except ProviderError as exc:
        if exc.status == 404 and ":free" in openrouter_model:
            logger.error(
                "OpenRouter model %s returned HTTP 404. The :free slug may be retired "
                "or unavailable; use %s.",
                openrouter_model,
                DEFAULT_OPENROUTER_MODEL,
            )
        logger.error(
            "OpenRouter fallback failed (%s) after DeepSeek failure (%s)",
            exc,
            deepseek_error,
        )
        raise ProviderError(
            "openrouter",
            exc.status,
            exc.body,
            f"{exc.message} (after DeepSeek failure: {deepseek_error})",
        ) from exc

    content = _message_text(data, provider="openrouter")
    logger.info("OpenRouter fallback completed model=%s", openrouter_model)
    return ClassifyResult("openrouter", openrouter_model, content)


def _warn_if_retired_free_model(model: str) -> None:
    if model.endswith(":free"):
        logger.warning(
            "OPENROUTER_MODEL=%s uses the :free suffix. %s returned HTTP 404 "
            "(retired / unavailable). Prefer %s.",
            model,
            RETIRED_OPENROUTER_FREE_MODEL,
            DEFAULT_OPENROUTER_MODEL,
        )


def _is_hard_failure(status: int | None, text: str) -> bool:
    if status is None or status in _HARD_STATUSES or status >= 500:
        return True
    blob = f"{text}\n{text.lower()}"
    return any(hint in blob for hint in _BALANCE_HINTS)


def _post_chat(
    url: str,
    api_key: str,
    payload: dict,
    timeout: float,
    *,
    provider: str,
) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raw_error = exc.read() if exc.fp is not None else b""
        error_body = raw_error.decode("utf-8", errors="replace")
        message = _error_message(error_body) or str(getattr(exc, "reason", "") or "")
        raise ProviderError(provider, int(exc.code), error_body, message) from exc
    except Exception as exc:
        raise ProviderError(provider, None, "", str(exc)) from exc

    text = raw.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProviderError(provider, 200, text[:500], "response was not JSON") from exc
    if isinstance(parsed, dict) and parsed.get("error") and "choices" not in parsed:
        err = parsed["error"]
        message = err.get("message") if isinstance(err, dict) else str(err)
        raise ProviderError(provider, 200, text[:500], str(message or "provider error"))
    return parsed


def _message_text(data: dict, *, provider: str) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(
            provider,
            200,
            json.dumps(data, ensure_ascii=False)[:500],
            "response did not include message content",
        ) from exc
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
        text = "".join(parts).strip()
    else:
        text = str(content).strip()
    if not text:
        raise ProviderError(provider, 200, "", "model returned empty content")
    return text


def _parse_json_content(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        snippet = _redact(content.strip().replace("\n", " "))[:300]
        raise ValueError(f"classification output was not JSON: {snippet}") from exc
    if not isinstance(data, dict):
        raise ValueError("classification JSON must be an object")
    return data


def _payload(model: str, messages: list[dict]) -> dict:
    return {"model": model, "messages": messages, "temperature": 0}


def _chat_url(base: str) -> str:
    root = base.rstrip("/")
    if root.endswith("/chat/completions"):
        return root
    return root + "/chat/completions"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or default


def _timeout() -> float:
    raw = _env("CLASSIFY_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS))
    try:
        value = float(raw)
    except ValueError:
        return float(DEFAULT_TIMEOUT_SECONDS)
    return value if value > 0 else float(DEFAULT_TIMEOUT_SECONDS)


def _error_message(body: str) -> str:
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return body.strip()
    if not isinstance(parsed, dict):
        return body.strip()
    err = parsed.get("error", parsed)
    if isinstance(err, dict):
        message = err.get("message") or err.get("code") or ""
        return str(message)
    return str(err)


def _redact(text: str) -> str:
    return _SECRET_RE.sub(
        lambda match: (match.group(1) + "[redacted]") if match.group(1) else "[redacted]",
        text,
    )
