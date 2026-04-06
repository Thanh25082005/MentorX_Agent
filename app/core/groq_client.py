"""
app/core/groq_client.py
────────────────────────
Wrapper cho Groq API.
Cung cấp 2 hàm chính:
  • chat()           — gọi Groq với messages thông thường
  • chat_json()      — gọi Groq với JSON mode (structured output)
  • health_check()   — kiểm tra kết nối Groq

Tính năng:
  • Retry with exponential backoff (configurable)
  • Request timeout
  • Structured error handling
  • Dễ swap sang OpenAI / Anthropic / local LLM
"""

from __future__ import annotations

import json
import time
from typing import Any

from groq import Groq
from loguru import logger

from app.core.config import settings


class GroqClient:
    """Thin wrapper quanh Groq SDK với retry + timeout."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        self._api_key = api_key or settings.groq_api_key
        self._model = model or settings.groq_model
        self._timeout = timeout or settings.groq_timeout
        self._max_retries = max_retries or settings.groq_max_retries
        self._client = Groq(api_key=self._api_key, timeout=self._timeout)
        logger.info(
            f"GroqClient initialized: model={self._model}, "
            f"timeout={self._timeout}s, max_retries={self._max_retries}"
        )

    # ── Retry helper ─────────────────────────────────────────

    def _call_with_retry(self, fn, *args, **kwargs) -> Any:
        """
        Gọi fn() với exponential backoff retry.
        Retry khi gặp rate-limit (429) hoặc server error (5xx).
        """
        last_exception = None
        for attempt in range(1, self._max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                # Retry cho rate-limit và server errors
                is_retryable = (
                    "rate_limit" in error_str
                    or "429" in error_str
                    or "500" in error_str
                    or "502" in error_str
                    or "503" in error_str
                    or "timeout" in error_str
                )
                if is_retryable and attempt < self._max_retries:
                    wait = 2 ** attempt  # 2s, 4s, 8s...
                    logger.warning(
                        f"[Groq] Attempt {attempt}/{self._max_retries} failed: {e}. "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    # Non-retryable hoặc hết retry
                    break
        raise last_exception  # type: ignore[misc]

    # ── Chat (text response) ────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """Gọi Groq chat completions, trả về text response."""
        def _do_call():
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""

        try:
            content = self._call_with_retry(_do_call)
            logger.debug(f"[Groq] response length={len(content)}")
            return content
        except Exception as e:
            logger.error(f"[Groq] chat error after {self._max_retries} attempts: {e}")
            raise

    # ── Chat JSON (structured output) ───────────────────────

    def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """
        Gọi Groq với JSON mode → parse response thành dict.
        System prompt PHẢI chứa từ 'JSON' để Groq bật json_object mode.
        """
        raw = ""

        def _do_call():
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content or "{}"

        try:
            raw = self._call_with_retry(_do_call)
            logger.debug(f"[Groq JSON] raw={raw[:200]}")
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"[Groq] JSON parse failed: {e}, raw={raw[:300]}")
            return {"error": "JSON parse failed", "raw": raw}
        except Exception as e:
            logger.error(
                f"[Groq] chat_json error after {self._max_retries} attempts: {e}"
            )
            raise

    # ── Health Check ────────────────────────────────────────

    def health_check(self) -> dict[str, Any]:
        """Kiểm tra kết nối Groq bằng một request nhỏ."""
        try:
            start = time.time()
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            latency = time.time() - start
            return {
                "status": "ok",
                "model": self._model,
                "latency_ms": round(latency * 1000),
            }
        except Exception as e:
            return {
                "status": "error",
                "model": self._model,
                "error": str(e),
            }


# Singleton instance — import từ nơi khác
groq_client = GroqClient()
