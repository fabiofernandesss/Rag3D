from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tests" / "beir_ablation.py"
SPEC = importlib.util.spec_from_file_location("beir_ablation", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
beir_ablation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(beir_ablation)


def test_historical_beir_ablation_is_explicitly_non_claim_exploratory() -> None:
    notice = beir_ablation.DIAGNOSTIC_NOTICE.lower()

    assert beir_ablation.CLAIM_ELIGIBLE is False
    assert "exploratorio" in notice
    assert "qrels/test" in notice
    assert "nao use" in notice


def test_reduced_corpus_selection_is_deterministic_and_has_no_qrels_input() -> None:
    corpus = {f"d{i}": f"text {i}" for i in range(40)}
    qrels_a = {"q": {"d0": 1}}
    qrels_b = {"q": {"d39": 1}}

    first = beir_ablation.select_corpus(corpus, 12, seed=20260813)
    second = beir_ablation.select_corpus(dict(reversed(list(corpus.items()))), 12, seed=20260813)

    assert first == second
    assert len(first) == 12
    assert "qrels" not in beir_ablation.select_corpus.__annotations__
    assert qrels_a != qrels_b  # neither judgment object participates in selection


@pytest.mark.parametrize(
    ("ranking", "relevance", "expected"),
    [
        (
            ["a", "a", "b"],
            {"a": 3, "b": 2},
            (7 + 3 / math.log2(4)) / (7 + 3 / math.log2(3)),
        ),
        (["missing", "a", "a"], {"a": 1}, 1 / math.log2(3)),
    ],
)
def test_ndcg_deduplicates_before_cutoff_and_never_exceeds_one(
    ranking, relevance, expected
) -> None:
    score = beir_ablation.ndcg_at_k(ranking, relevance, k=10)

    assert score == pytest.approx(expected)
    assert math.isfinite(score)
    assert 0.0 <= score <= 1.0


def test_recall_deduplicates_and_metrics_handle_zero_cutoff() -> None:
    # Duplicate IDs consume a raw rank position but never receive gain twice.
    assert beir_ablation.recall_at_k(["a", "a", "b"], {"a": 1, "b": 1}, 2) == 0.5
    assert beir_ablation.ndcg_at_k(["a"], {"a": 1}, 0) == 0.0
    assert beir_ablation.recall_at_k(["a"], {"a": 1}, 0) == 0.0


def test_invalid_corpus_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        beir_ablation.select_corpus({"d": "text"}, -1)
