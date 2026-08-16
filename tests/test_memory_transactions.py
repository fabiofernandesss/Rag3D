from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy

import numpy as np
import pytest

from rag3d.config import TriRagConfig
from rag3d.encoders import HashEncoder, TriVec
from rag3d.memory import ChatMemory
from rag3d.store import TriStore


class _LLM:
    def available(self):
        return True

    def complete(self, *_args, **_kwargs):
        return "novo resumo"


class _Encoder:
    def __init__(self, store):
        self.store = store
        self.called_inside_transaction = None

    def encode(self, _texts, is_query=False):
        self.called_inside_transaction = self.store.in_transaction
        return [
            TriVec(
                dense=np.array([1.0, 0.0], dtype=np.float32),
                sparse={1: 1.0},
                tokens=np.array([[1.0, 0.0]], dtype=np.float32),
            )
        ]


class _RollbackStore:
    def __init__(self):
        self.in_transaction = False
        self.meta = {
            "rolling_summary": "resumo antigo",
            "rolling_summary_turn": "10",
            "rolling_summary_chunk": "5",
        }
        self.chunks = {5: "resumo antigo"}

    def get_meta(self, key):
        return self.meta.get(key)

    def set_meta(self, key, value):
        self.meta[key] = value

    def all_texts(self, kinds=()):
        return [{"turn_no": 11, "text": "novo turno"}]

    def delete_chunk(self, chunk_id):
        self.chunks.pop(chunk_id, None)

    def add_chunk(self, *_args, **_kwargs):
        raise RuntimeError("injected summary write failure")

    @contextmanager
    def transaction(self):
        snapshot = (deepcopy(self.meta), deepcopy(self.chunks))
        self.in_transaction = True
        try:
            yield
        except BaseException:
            self.meta, self.chunks = snapshot
            raise
        finally:
            self.in_transaction = False


def test_chat_summary_prepares_vectors_outside_and_rolls_back_all_state() -> None:
    store = _RollbackStore()
    encoder = _Encoder(store)
    memory = ChatMemory(
        store,
        encoder,
        TriRagConfig(summary_every_turns=12),
        llm=_LLM(),
    )

    with pytest.raises(RuntimeError, match="summary write failure"):
        memory._consolidate(12)

    assert encoder.called_inside_transaction is False
    assert store.meta == {
        "rolling_summary": "resumo antigo",
        "rolling_summary_turn": "10",
        "rolling_summary_chunk": "5",
    }
    assert store.chunks == {5: "resumo antigo"}


def test_oversized_summary_response_preserves_existing_memory_state() -> None:
    store = _RollbackStore()
    encoder = _Encoder(store)

    class OversizedLLM(_LLM):
        def complete(self, *_args, **_kwargs):
            return "x" * (1024 * 1024 + 1)

    memory = ChatMemory(
        store,
        encoder,
        TriRagConfig(summary_every_turns=12),
        llm=OversizedLLM(),
    )

    memory._consolidate(12)

    assert encoder.called_inside_transaction is None
    assert store.meta["rolling_summary"] == "resumo antigo"
    assert store.chunks == {5: "resumo antigo"}


def test_record_turn_rejects_oversized_messages_before_encoding_or_write(
    tmp_path,
) -> None:
    store = TriStore(tmp_path / "bounded-memory.db")
    memory = ChatMemory(store, HashEncoder(8, 4, 8), TriRagConfig())

    with pytest.raises(ValueError, match="LLM text.*maximum"):
        memory.record_turn("pergunta", "x" * (1024 * 1024 + 1))

    assert store.last_turn_no() == 0
    store.close()
