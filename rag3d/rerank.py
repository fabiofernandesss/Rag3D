"""Optional, bounded rerankers used after candidate fusion.

Rerankers are ordinal stages: a valid result may improve or worsen the prior
ranking.  Availability, model errors and invalid outputs therefore fail back
to the exact input order without discarding candidates.
"""
from __future__ import annotations

import importlib.util
import json
import math
from collections.abc import Sized
from itertools import islice
from numbers import Integral, Real
from typing import Any, List, Optional, Protocol, Sequence, runtime_checkable

from .backend import DEFAULT_RETRIEVAL_LIMITS
from .llm import LLM

_SYS = (
    "Você é um reordenador de relevância. Recebe uma pergunta e trechos "
    "numerados. Responda APENAS um array JSON com os números dos trechos, do "
    "mais relevante para o menos relevante. Inclua os números úteis primeiro; "
    "os itens omitidos serão mantidos no fim. Exemplo: [3, 1, 5]. Sem texto "
    "fora do array."
)

MAX_RERANKER_SNIPPET_CHARS = 10_000
MAX_RERANKER_RESPONSE_BYTES = DEFAULT_RETRIEVAL_LIMITS.max_query_bytes
# Backward-compatible internal name retained for callers that imported it.
MAX_CROSS_ENCODER_SNIPPET_CHARS = MAX_RERANKER_SNIPPET_CHARS


def _validated_top_k(top_k: Optional[int]) -> Optional[int]:
    if top_k is None:
        return None
    if isinstance(top_k, bool) or not isinstance(top_k, Integral):
        raise TypeError("top_k must be an integer, not bool")
    if top_k < 0 or top_k > DEFAULT_RETRIEVAL_LIMITS.max_pool:
        raise ValueError(
            f"top_k must be between 0 and {DEFAULT_RETRIEVAL_LIMITS.max_pool}"
        )
    return int(top_k)


def _copy_hits(hits: Sequence[dict], top_k: Optional[int]) -> List[dict]:
    maximum = DEFAULT_RETRIEVAL_LIMITS.max_pool
    if isinstance(hits, (str, bytes)):
        raise TypeError("hits must be a sequence of mappings")
    if isinstance(hits, Sized) and len(hits) > maximum:
        raise ValueError(f"hits exceed maximum of {maximum}")
    copied: List[dict] = []
    for item_count, hit in enumerate(hits, start=1):
        if item_count > maximum:
            raise ValueError(f"hits exceed maximum of {maximum}")
        copied.append(dict(hit))
    return copied if top_k is None else copied[:top_k]


@runtime_checkable
class RerankerProtocol(Protocol):
    """Minimal contract accepted by :class:`RetrievalV2`."""

    name: str

    def available(self) -> bool:
        ...

    def rerank(
        self, query: str, hits: Sequence[dict], top_k: Optional[int] = None
    ) -> List[dict]:
        ...


class NoOpReranker:
    """Identity reranker used as the explicit rollback path."""

    name = "none"

    def available(self) -> bool:
        return True

    def rerank(
        self, query: str, hits: Sequence[dict], top_k: Optional[int] = None
    ) -> List[dict]:
        del query
        return _copy_hits(hits, _validated_top_k(top_k))


class LLMListwiseReranker:
    """Listwise LLM reranker that preserves omitted candidates at the tail."""

    name = "llm"

    def __init__(self, llm: LLM, snippet_chars: int = 500):
        if isinstance(snippet_chars, bool) or not isinstance(snippet_chars, Integral):
            raise TypeError("snippet_chars must be an integer, not bool")
        if not 1 <= snippet_chars <= MAX_RERANKER_SNIPPET_CHARS:
            raise ValueError(
                "snippet_chars must be between 1 and "
                f"{MAX_RERANKER_SNIPPET_CHARS}"
            )
        self.llm = llm
        self.snippet_chars = int(snippet_chars)
        self.last_status = "skipped"
        self.last_reason = "not-run"

    def available(self) -> bool:
        try:
            return self.llm is not None and bool(self.llm.available())
        except Exception:
            return False

    def rerank(
        self, query: str, hits: Sequence[dict], top_k: Optional[int] = None
    ) -> List[dict]:
        self.last_status, self.last_reason = "skipped", "not-run"
        limit = _validated_top_k(top_k)
        bounded_hits = _copy_hits(hits, None)
        fallback = bounded_hits if limit is None else bounded_hits[:limit]
        if limit == 0:
            self.last_reason = "empty-pool"
            return []
        if not self.available() or len(bounded_hits) < 2:
            self.last_reason = (
                "unavailable" if not self.available() else "empty-pool"
            )
            return fallback

        listing = "\n".join(
            f"[{index}] {str(hit.get('text', ''))[:self.snippet_chars]}"
            for index, hit in enumerate(bounded_hits)
        )
        prompt = (
            f"PERGUNTA: {query}\n\nTRECHOS:\n{listing}\n\n"
            "Ordem de relevância (array JSON):"
        )
        try:
            raw = self.llm.complete(
                _SYS, [{"role": "user", "content": prompt}], max_tokens=200
            )
            order = self._parse(raw, len(bounded_hits))
        except Exception:
            self.last_status, self.last_reason = "fallback", "stage-error"
            return fallback
        if order is None:
            self.last_status, self.last_reason = "fallback", "invalid-output"
            return fallback

        seen = set(order)
        complete_order = order + [
            index for index in range(len(bounded_hits)) if index not in seen
        ]
        reordered: List[dict] = []
        for rank, index in enumerate(complete_order):
            hit = dict(bounded_hits[index])
            hit["rerank"] = rank
            reordered.append(hit)
        self.last_status, self.last_reason = "applied", "none"
        return reordered if limit is None else reordered[:limit]

    @staticmethod
    def _parse(raw: str, n: int) -> Optional[List[int]]:
        """Accept only one canonical JSON array of unique zero-based indices."""
        if not isinstance(raw, str):
            return None
        # A provider-supplied ``max_tokens`` is not a trust boundary.  Bound
        # both characters and UTF-8 bytes before handing data to the JSON
        # decoder; the first check also bounds the encoding allocation.
        if (
            len(raw) > MAX_RERANKER_RESPONSE_BYTES
            or len(raw.encode("utf-8")) > MAX_RERANKER_RESPONSE_BYTES
        ):
            return None
        try:
            values: Any = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(values, list):
            return None
        out: List[int] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, Integral):
                return None
            index = int(value)
            if not 0 <= index < n or index in out:
                return None
            out.append(index)
        return out


class CrossEncoderReranker:
    """Optional cross-encoder reranker with lazy dependency/model loading."""

    name = "cross-encoder"

    def __init__(
        self,
        model: Optional[Any] = None,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        snippet_chars: int = 500,
    ):
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if isinstance(snippet_chars, bool) or not isinstance(snippet_chars, Integral):
            raise TypeError("snippet_chars must be an integer, not bool")
        if not 1 <= snippet_chars <= MAX_RERANKER_SNIPPET_CHARS:
            raise ValueError(
                "snippet_chars must be between 1 and "
                f"{MAX_RERANKER_SNIPPET_CHARS}"
            )
        self.model = model
        self.model_name = model_name
        self.snippet_chars = int(snippet_chars)
        self._load_attempted = model is not None
        self.last_status = "skipped"
        self.last_reason = "not-run"

    def _load_model(self) -> Optional[Any]:
        if self.model is not None or self._load_attempted:
            return self.model
        self._load_attempted = True
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name)
        except Exception:
            self.model = None
        return self.model

    def available(self) -> bool:
        if self.model is not None:
            return True
        try:
            return importlib.util.find_spec("sentence_transformers") is not None
        except Exception:
            return False

    def rerank(
        self, query: str, hits: Sequence[dict], top_k: Optional[int] = None
    ) -> List[dict]:
        self.last_status, self.last_reason = "skipped", "not-run"
        limit = _validated_top_k(top_k)
        bounded_hits = _copy_hits(hits, None)
        fallback = bounded_hits if limit is None else bounded_hits[:limit]
        if limit == 0:
            self.last_reason = "empty-pool"
            return []
        if len(bounded_hits) < 2:
            self.last_reason = "empty-pool"
            return fallback
        model = self._load_model()
        if model is None:
            self.last_reason = "unavailable"
            return fallback

        pairs = [
            (query, str(hit.get("text", ""))[: self.snippet_chars])
            for hit in bounded_hits
        ]
        try:
            raw_scores = model.predict(pairs) if hasattr(model, "predict") else model(pairs)
            expected = len(bounded_hits)
            if isinstance(raw_scores, Sized) and len(raw_scores) != expected:
                self.last_status, self.last_reason = "fallback", "invalid-output"
                return fallback
            scores = list(islice(iter(raw_scores), expected + 1))
            if len(scores) != len(bounded_hits):
                self.last_status, self.last_reason = "fallback", "invalid-output"
                return fallback
            normalized: List[float] = []
            for score in scores:
                if isinstance(score, bool) or not isinstance(score, Real):
                    self.last_status, self.last_reason = "fallback", "invalid-output"
                    return fallback
                value = float(score)
                if not math.isfinite(value):
                    self.last_status, self.last_reason = "fallback", "invalid-output"
                    return fallback
                normalized.append(value)
        except Exception:
            self.last_status, self.last_reason = "fallback", "stage-error"
            return fallback

        order = sorted(
            range(len(bounded_hits)),
            key=lambda index: (
                -normalized[index],
                int(bounded_hits[index].get("id", index)),
                index,
            ),
        )
        reordered: List[dict] = []
        for rank, index in enumerate(order):
            hit = dict(bounded_hits[index])
            hit["rerank"] = rank
            hit["rerank_score"] = normalized[index]
            reordered.append(hit)
        self.last_status, self.last_reason = "applied", "none"
        return reordered if limit is None else reordered[:limit]


class Reranker(LLMListwiseReranker):
    """Backward-compatible name for the original listwise reranker."""
