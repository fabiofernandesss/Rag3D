from __future__ import annotations

import json

import pytest

from rag3d.config import TriRagConfig
from rag3d.llm import _post_json
from rag3d.reader import Reader


class _OversizedResponse:
    def __init__(self):
        self.requested = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size=-1):
        self.requested.append(size)
        return b"x" * size


def test_http_transport_caps_response_before_json_decoding(monkeypatch):
    response = _OversizedResponse()
    monkeypatch.setattr("rag3d.llm.urllib.request.urlopen", lambda *_a, **_k: response)
    monkeypatch.setattr(
        "rag3d.llm.json.loads",
        lambda _raw: (_ for _ in ()).throw(
            AssertionError("oversized body must not reach JSON decoding")
        ),
    )

    with pytest.raises(RuntimeError, match="response exceeds size limit"):
        _post_json("https://example.invalid/v1", {}, {})

    assert response.requested == [1024 * 1024 + 1]


def test_reader_axis_failure_does_not_echo_provider_secret():
    secret = "postgresql://admin:super-secret@example.invalid/prod"

    class FinalLLM:
        def available(self):
            return True

        def complete(self, *_args, **_kwargs):
            return "resposta final"

    class BrokenAxisLLM(FinalLLM):
        def complete(self, *_args, **_kwargs):
            raise RuntimeError(secret)

    reader = Reader(
        TriRagConfig(read_mode="tri"),
        FinalLLM(),
        axis_llms={"semantico": BrokenAxisLLM()},
    )
    context = {
        "mode": "retrieval",
        "summary": None,
        "recent": [],
        "blocks": [{"chosen": "evidência"}],
        "views": {"semantico": [{"text": "evidência"}]},
    }

    result = reader.read_tri("consulta", context)

    assert secret not in json.dumps(result, ensure_ascii=False)
    assert result["sub_answers"]["semantico"] == "(leitor do eixo falhou)"
