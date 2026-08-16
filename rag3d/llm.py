"""Adaptador de LLM agnóstico de provedor — "qualquer IA".

Sem dependências: fala HTTP puro (urllib) com:
  - Anthropic (ANTHROPIC_API_KEY)
  - OpenAI e qualquer endpoint OpenAI-compatível (OPENAI_API_KEY / OPENAI_BASE_URL)
  - Ollama local (endpoint OpenAI-compatível em http://localhost:11434/v1)
  - callable Python arbitrário (para plugar qualquer outra IA)

"auto" escolhe na ordem: Anthropic > OpenAI > Ollama > none (só retrieval).
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Callable, List, Optional

Message = dict  # {"role": "user"|"assistant", "content": str}
MAX_LLM_RESPONSE_BYTES = 1024 * 1024


def validate_llm_text(
    value: str, maximum: int = MAX_LLM_RESPONSE_BYTES
) -> str:
    """Validate an untrusted provider/callable response before further work."""
    if not isinstance(value, str):
        raise TypeError("LLM text must be a string")
    # Character preflight prevents an unbounded UTF-8 allocation. A surviving
    # value allocates at most four times ``maximum`` for the byte check.
    if len(value) > maximum or len(value.encode("utf-8")) > maximum:
        raise ValueError(
            f"LLM text exceeds maximum of {maximum} UTF-8 bytes"
        )
    return value


def _read_bounded_response(response, maximum: int = MAX_LLM_RESPONSE_BYTES) -> bytes:
    payload = response.read(maximum + 1)
    if len(payload) > maximum:
        raise RuntimeError("LLM response exceeds size limit")
    return payload


class LLM:
    """Interface mínima: complete(system, messages) -> texto."""

    provider = "none"
    model = ""

    def complete(self, system: str, messages: List[Message], max_tokens: int = 1500) -> str:
        raise NotImplementedError

    def available(self) -> bool:
        return True


class NoLLM(LLM):
    provider = "none"

    def complete(self, system: str, messages: List[Message], max_tokens: int = 1500) -> str:
        raise RuntimeError("Nenhum LLM configurado (modo só-retrieval).")

    def available(self) -> bool:
        return False


def _post_json(url: str, headers: dict, payload: dict, timeout: int = 120) -> dict:
    body = json.dumps(payload).encode("utf-8")
    last_err: Optional[Exception] = None
    for attempt in range(3):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(_read_bounded_response(resp).decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503, 529) and attempt < 2:
                time.sleep(2.0 * (attempt + 1))
                continue
            detail = e.read(501).decode("utf-8", "replace")[:500]
            raise RuntimeError(f"LLM HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            last_err = e
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
                continue
            raise RuntimeError(f"LLM inacessível: {e}") from e
    raise RuntimeError(f"LLM falhou: {last_err}")


class AnthropicLLM(LLM):
    provider = "anthropic"

    def __init__(self, model: str = ""):
        self.key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model or "claude-sonnet-5"

    def available(self) -> bool:
        return bool(self.key)

    def complete(self, system: str, messages: List[Message], max_tokens: int = 1500) -> str:
        data = _post_json(
            "https://api.anthropic.com/v1/messages",
            {
                "x-api-key": self.key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            {
                "model": self.model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": messages,
            },
        )
        return "".join(b.get("text", "") for b in data.get("content", []))


class OpenAICompatLLM(LLM):
    """OpenAI, Ollama, vLLM, LM Studio, Groq... qualquer /v1/chat/completions."""

    provider = "openai"

    def __init__(self, model: str = "", base_url: str = "", api_key: str = ""):
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("TRIRAG_LLM_MODEL", "gpt-4o-mini")

    def available(self) -> bool:
        return bool(self.key) or "localhost" in self.base_url or "127.0.0.1" in self.base_url

    def complete(self, system: str, messages: List[Message], max_tokens: int = 1500) -> str:
        data = _post_json(
            f"{self.base_url}/chat/completions",
            {"Authorization": f"Bearer {self.key or 'none'}", "Content-Type": "application/json"},
            {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}] + messages,
            },
        )
        return data["choices"][0]["message"]["content"] or ""


class CallableLLM(LLM):
    """Embrulha qualquer função Python (system, messages, max_tokens) -> str."""

    provider = "callable"

    def __init__(self, fn: Callable[[str, List[Message], int], str], name: str = "callable"):
        self.fn = fn
        self.model = name

    def complete(self, system: str, messages: List[Message], max_tokens: int = 1500) -> str:
        return self.fn(system, messages, max_tokens)


def _ollama_up(base: str = "http://localhost:11434") -> bool:
    try:
        with urllib.request.urlopen(base + "/api/tags", timeout=1) as r:
            return r.status == 200
    except Exception:
        return False


def make_llm(provider: str = "auto", model: str = "") -> LLM:
    if provider == "anthropic":
        return AnthropicLLM(model)
    if provider == "openai":
        return OpenAICompatLLM(model)
    if provider == "ollama":
        return OpenAICompatLLM(model or "llama3.1", base_url="http://localhost:11434/v1", api_key="ollama")
    if provider == "none":
        return NoLLM()
    # auto
    a = AnthropicLLM(model)
    if a.available():
        return a
    o = OpenAICompatLLM(model)
    if os.environ.get("OPENAI_API_KEY"):
        return o
    if _ollama_up():
        first = ""
        try:
            with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
                tags = json.loads(_read_bounded_response(r).decode())
                if tags.get("models"):
                    first = tags["models"][0]["name"]
        except Exception:
            pass
        return OpenAICompatLLM(model or first or "llama3.1", base_url="http://localhost:11434/v1", api_key="ollama")
    return NoLLM()
