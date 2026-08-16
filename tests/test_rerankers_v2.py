import math

import pytest

from rag3d.backend import DEFAULT_RETRIEVAL_LIMITS
from rag3d.rerank import (
    CrossEncoderReranker,
    LLMListwiseReranker,
    NoOpReranker,
    Reranker,
)

MAX_TEST_RERANKER_RESPONSE_BYTES = 64 * 1024


class FakeLLM:
    def __init__(self, response="[]", error=None):
        self.response = response
        self.error = error

    def available(self):
        return True

    def complete(self, *_args, **_kwargs):
        if self.error is not None:
            raise self.error
        return self.response


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores

    def predict(self, _pairs):
        return self.scores


def _hits():
    return [
        {"id": 1, "text": "primeiro", "score": 0.9},
        {"id": 2, "text": "segundo", "score": 0.8},
        {"id": 3, "text": "terceiro", "score": 0.7},
    ]


def test_noop_preserves_order_without_mutating_callers_hits():
    hits = _hits()
    before = [dict(hit) for hit in hits]

    result = NoOpReranker().rerank("consulta", hits, top_k=2)

    assert [hit["id"] for hit in result] == [1, 2]
    assert hits == before
    assert result[0] is not hits[0]


def test_rerankers_reject_an_honestly_oversized_pool_before_iteration():
    class ExplodingHits:
        def __len__(self):
            return DEFAULT_RETRIEVAL_LIMITS.max_pool + 1

        def __iter__(self):
            raise AssertionError("oversized hit pools must fail before iteration")

    with pytest.raises(ValueError, match="hits.*maximum"):
        NoOpReranker().rerank("consulta", ExplodingHits(), top_k=1)


def test_rerankers_stop_a_lying_pool_at_the_first_item_above_the_bound():
    class LyingHits:
        def __init__(self):
            self.consumed = 0

        def __len__(self):
            return 0

        def __iter__(self):
            for chunk_id in range(DEFAULT_RETRIEVAL_LIMITS.max_pool + 1):
                self.consumed += 1
                yield {"id": chunk_id, "text": "bounded"}

    hits = LyingHits()
    with pytest.raises(ValueError, match="hits.*maximum"):
        NoOpReranker().rerank("consulta", hits, top_k=1)

    assert hits.consumed == DEFAULT_RETRIEVAL_LIMITS.max_pool + 1


def test_listwise_valid_order_never_discards_unmentioned_candidates():
    hits = _hits()
    reranker = LLMListwiseReranker(FakeLLM("[2, 0]"))

    result = reranker.rerank("consulta", hits)

    assert [hit["id"] for hit in result] == [3, 1, 2]
    assert [hit["rerank"] for hit in result] == [0, 1, 2]
    assert all("rerank" not in hit for hit in hits)


def test_listwise_failure_or_parse_without_ids_preserves_previous_order():
    hits = _hits()

    failed = LLMListwiseReranker(FakeLLM(error=RuntimeError("offline"))).rerank(
        "consulta", hits
    )
    unparsable = LLMListwiseReranker(FakeLLM("nenhum identificador")).rerank(
        "consulta", hits
    )

    assert [hit["id"] for hit in failed] == [1, 2, 3]
    assert [hit["id"] for hit in unparsable] == [1, 2, 3]
    assert failed is not hits


def test_listwise_oversized_response_fails_closed_before_json_parsing(monkeypatch):
    response = "[" + ("0," * MAX_TEST_RERANKER_RESPONSE_BYTES) + "0]"
    reranker = LLMListwiseReranker(FakeLLM(response))
    monkeypatch.setattr(
        "rag3d.rerank.json.loads",
        lambda _raw: (_ for _ in ()).throw(
            AssertionError("oversized output must fail before JSON parsing")
        ),
    )

    result = reranker.rerank("consulta", _hits())

    assert [hit["id"] for hit in result] == [1, 2, 3]
    assert reranker.last_status == "fallback"
    assert reranker.last_reason == "invalid-output"


@pytest.mark.parametrize(
    "response",
    [
        "Não consegui reordenar; erro 2.",
        "ordem: [2, 0]",
        "```json\n[2, 0]\n```",
        '["2", 0]',
        "[2, 2]",
        "[2, 99]",
        "[true, 0]",
        '{"order": [2, 0]}',
    ],
)
def test_listwise_noncanonical_or_invalid_json_preserves_previous_order(response):
    result = LLMListwiseReranker(FakeLLM(response)).rerank("consulta", _hits())

    assert [hit["id"] for hit in result] == [1, 2, 3]
    assert all("rerank" not in hit for hit in result)


def test_a_valid_listwise_response_is_allowed_to_worsen_the_prior_order():
    result = LLMListwiseReranker(FakeLLM("[1, 0, 2]")).rerank("consulta", _hits())

    assert [hit["id"] for hit in result] == [2, 1, 3]


def test_cross_encoder_uses_injected_model_and_returns_finite_scores():
    hits = _hits()
    reranker = CrossEncoderReranker(model=FakeCrossEncoder([0.1, 3.0, 1.0]))

    result = reranker.rerank("consulta", hits)

    assert [hit["id"] for hit in result] == [2, 3, 1]
    assert [hit["rerank_score"] for hit in result] == [3.0, 1.0, 0.1]
    assert all(math.isfinite(hit["rerank_score"]) for hit in result)
    assert all("rerank_score" not in hit for hit in hits)


def test_cross_encoder_bounds_candidate_text_before_calling_the_model():
    class CapturingCrossEncoder:
        def __init__(self):
            self.pairs = []

        def predict(self, pairs):
            self.pairs = list(pairs)
            return [0.1, 0.2]

    model = CapturingCrossEncoder()
    reranker = CrossEncoderReranker(model=model, snippet_chars=64)
    hits = [
        {"id": 1, "text": "a" * 100_000},
        {"id": 2, "text": "b" * 100_000},
    ]

    reranker.rerank("consulta", hits)

    assert [len(text) for _query, text in model.pairs] == [64, 64]


@pytest.mark.parametrize("snippet_chars", [True, 0, 10_001])
def test_cross_encoder_rejects_unbounded_or_invalid_snippet_limits(snippet_chars):
    with pytest.raises((TypeError, ValueError), match="snippet_chars"):
        CrossEncoderReranker(
            model=FakeCrossEncoder([0.1, 0.2]),
            snippet_chars=snippet_chars,
        )


@pytest.mark.parametrize("snippet_chars", [True, 0, 10_001])
def test_llm_reranker_rejects_unbounded_or_invalid_snippet_limits(snippet_chars):
    with pytest.raises((TypeError, ValueError), match="snippet_chars"):
        LLMListwiseReranker(FakeLLM("[]"), snippet_chars=snippet_chars)


def test_cross_encoder_failure_or_nonfinite_score_preserves_previous_order():
    hits = _hits()
    broken = CrossEncoderReranker(model=FakeCrossEncoder([0.1, math.nan, 0.2]))

    result = broken.rerank("consulta", hits)

    assert [hit["id"] for hit in result] == [1, 2, 3]
    assert all("rerank_score" not in hit for hit in result)


def test_cross_encoder_stops_unbounded_scores_at_expected_cardinality_plus_one():
    class UnboundedScores:
        def __init__(self):
            self.consumed = 0

        def __iter__(self):
            for _ in range(100_000):
                self.consumed += 1
                yield 1.0

    scores = UnboundedScores()
    result = CrossEncoderReranker(model=FakeCrossEncoder(scores)).rerank(
        "consulta", _hits()
    )

    assert [hit["id"] for hit in result] == [1, 2, 3]
    assert scores.consumed == len(_hits()) + 1


def test_cross_encoder_score_ties_are_broken_by_id():
    hits = [
        {"id": 3, "text": "três"},
        {"id": 1, "text": "um"},
        {"id": 2, "text": "dois"},
    ]

    result = CrossEncoderReranker(
        model=FakeCrossEncoder([0.5, 0.5, 0.5])
    ).rerank("consulta", hits)

    assert [hit["id"] for hit in result] == [1, 2, 3]


def test_cross_encoder_availability_probe_fails_closed(monkeypatch):
    def broken_probe(_name):
        raise RuntimeError("broken import metadata")

    monkeypatch.setattr("rag3d.rerank.importlib.util.find_spec", broken_probe)

    assert CrossEncoderReranker().available() is False


@pytest.mark.parametrize(
    "reranker",
    [
        NoOpReranker(),
        LLMListwiseReranker(FakeLLM("[2, 1, 0]")),
        CrossEncoderReranker(model=FakeCrossEncoder([1.0, 2.0, 3.0])),
    ],
)
def test_all_rerankers_treat_top_k_zero_as_empty(reranker):
    assert reranker.rerank("consulta", _hits(), top_k=0) == []


def test_legacy_reranker_name_remains_compatible():
    reranker = Reranker(FakeLLM("[0]"))

    assert isinstance(reranker, LLMListwiseReranker)
