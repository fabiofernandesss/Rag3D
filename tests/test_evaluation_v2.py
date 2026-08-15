from __future__ import annotations

import inspect
import importlib.util
import json
import math
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag3d.evaluation import (
    BOOTSTRAP_SEED,
    MAX_BOOTSTRAP_SAMPLES,
    MAX_BOOTSTRAP_WORK,
    RETRIEVAL_STAGES,
    StageRecorder,
    aggregate_query_metrics,
    canonical_sha256,
    citation_precision,
    clustered_percentile_bootstrap,
    coverage_at_k,
    deduplicate_ranking,
    duplicate_rate_at_k,
    evaluate_query,
    latency_percentiles,
    mrr_at_k,
    ndcg_at_k,
    no_answer_correct,
    paired_bootstrap,
    recall_at_k,
    select_corpus_by_hash,
    validate_stage_lineage,
    validate_split_protocol,
    write_json_report,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "benchmarks" / "run_retrieval_v2.py"


def _load_runner_module():
    specification = importlib.util.spec_from_file_location(
        "retrieval_v2_benchmark_for_tests", RUNNER
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_runner_hydrates_backend_chunks_in_bounded_batches() -> None:
    runner = _load_runner_module()
    limit = runner.DEFAULT_RETRIEVAL_LIMITS.max_pool

    class BoundedStore:
        def __init__(self):
            self.batch_sizes = []

        def get_chunks(self, chunk_ids):
            ids = list(chunk_ids)
            assert len(ids) <= limit
            self.batch_sizes.append(len(ids))
            return [{"id": chunk_id, "doc_id": chunk_id} for chunk_id in ids]

    store = BoundedStore()
    rows = runner._get_chunks_batched(store, range(limit + 1))

    assert store.batch_sizes == [limit, 1]
    assert [row["id"] for row in rows] == list(range(limit + 1))


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--dense-dim", "4097", "dense-dim"),
        ("--structural-dim", "1025", "structural-dim"),
        ("--max-structural-tokens", "1025", "max-structural-tokens"),
    ],
)
def test_runner_rejects_embedding_dimensions_that_exceed_memory_bounds(
    flag, value, message, capsys
) -> None:
    runner = _load_runner_module()

    with pytest.raises(SystemExit):
        runner._parse_arguments([flag, value])

    assert message in capsys.readouterr().err


def test_runner_rejects_an_unbounded_structural_tensor_product(capsys) -> None:
    runner = _load_runner_module()

    with pytest.raises(SystemExit):
        runner._parse_arguments(
            ["--structural-dim", "1024", "--max-structural-tokens", "1024"]
        )

    assert "structural tensor" in capsys.readouterr().err


def test_runner_embedding_memory_estimate_counts_structural_bytes(capsys) -> None:
    runner = _load_runner_module()

    with pytest.raises(SystemExit):
        runner._parse_arguments(
            [
                "--scale",
                "100000",
                "--dense-dim",
                "1",
                "--structural-dim",
                "80",
                "--max-structural-tokens",
                "64",
            ]
        )

    assert "embedding payload" in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments",
    [
        ["--hnsw-m", "1"],
        ["--hnsw-m", "101"],
        ["--hnsw-ef-construction", "3"],
        ["--hnsw-ef-construction", "1001"],
        ["--hnsw-m", "40", "--hnsw-ef-construction", "64"],
    ],
)
def test_runner_rejects_invalid_hnsw_build_options_before_ingestion(arguments) -> None:
    runner = _load_runner_module()

    with pytest.raises(SystemExit):
        runner._parse_arguments(arguments)


def test_ir_metrics_apply_the_raw_cutoff_before_deduplication() -> None:
    ranking = ["a", "a", "c", "unknown"]
    relevance = {"a": 2, "c": 1}

    assert deduplicate_ranking(ranking) == (["a", "c", "unknown"], 1)
    assert recall_at_k(ranking, relevance, 2) == 0.5
    assert mrr_at_k(ranking, relevance, 20) == 1.0
    assert ndcg_at_k(ranking, relevance, 2) == pytest.approx(
        3.0 / (3.0 + 1.0 / math.log2(3))
    )
    assert 0.0 <= ndcg_at_k(ranking, relevance, 10) <= 1.0


def test_duplicates_occupy_rank_and_never_backfill_beyond_the_raw_cutoff() -> None:
    assert recall_at_k(["x", "x", "a"], {"a": 1}, 2) == 0.0
    assert mrr_at_k(["x", "x", "a"], {"a": 1}, 2) == 0.0
    assert ndcg_at_k(["x", "x", "a"], {"a": 1}, 2) == 0.0
    assert recall_at_k(["a", "a", "b"], {"a": 1, "b": 1}, 2) == 0.5


def test_graded_ndcg_matches_a_manual_example() -> None:
    score = ndcg_at_k(["a", "b", "c"], {"a": 2, "c": 1}, 3)

    expected = (3.0 + 1.0 / math.log2(4)) / (3.0 + 1.0 / math.log2(3))
    assert score == pytest.approx(expected)


def test_ndcg_stays_finite_for_extreme_finite_relevance_grades() -> None:
    score = ndcg_at_k(["a", "b"], {"a": 100_000.0, "b": 99_999.0}, 2)

    assert score is not None
    assert math.isfinite(score)
    assert 0.0 <= score <= 1.0


@pytest.mark.parametrize("grade", [1e-300, 1e-20, 1.0, 100_000.0])
def test_ndcg_preserves_tiny_and_huge_positive_relevance(grade: float) -> None:
    score = ndcg_at_k(["a"], {"a": grade}, 10)

    assert score == pytest.approx(1.0)
    assert math.isfinite(score)


@pytest.mark.parametrize("metric", [recall_at_k, mrr_at_k, ndcg_at_k])
def test_quality_metrics_are_not_applicable_without_relevance_labels(metric) -> None:
    assert metric(["a"], {}, 10) is None


@pytest.mark.parametrize("k", [0, 1, 5, 20])
def test_quality_metrics_are_finite_and_bounded_for_edge_cutoffs(k: int) -> None:
    relevance = {"a": 3, "c": 1}

    for metric in (recall_at_k, mrr_at_k, ndcg_at_k):
        value = metric(["missing", "a", "a", "c"], relevance, k)
        assert value is not None
        assert math.isfinite(value)
        assert 0.0 <= value <= 1.0


def test_coverage_uses_distinct_required_facts_and_is_null_without_labels() -> None:
    document_facts = {
        "a": ["f1", "f1"],
        "b": ["f2"],
        "c": ["unneeded"],
    }

    assert coverage_at_k(["a", "a", "b"], document_facts, ["f1", "f2"], 2) == 0.5
    assert coverage_at_k(["a"], document_facts, None, 2) is None
    assert coverage_at_k(["a"], document_facts, [], 2) is None


def test_duplicate_rate_counts_exact_content_pairs_in_the_raw_prefix() -> None:
    fingerprints = {"a": "same", "b": "same", "c": "different"}

    assert duplicate_rate_at_k(["a", "b", "c"], fingerprints, 3) == pytest.approx(1 / 3)
    assert duplicate_rate_at_k(["a"], fingerprints, 10) == 0.0
    assert duplicate_rate_at_k([], fingerprints, 10) == 0.0


def test_no_answer_and_citation_metrics_remain_null_without_required_labels() -> None:
    metrics = evaluate_query(["a"], {}, top_k=20)

    assert metrics["no_answer_correct"] is None
    assert metrics["citation_precision"] is None
    assert no_answer_correct(None, True) is None
    assert no_answer_correct(True, None) is None
    assert no_answer_correct(True, True) == 1.0
    assert citation_precision(None) is None
    assert citation_precision([]) is None
    assert citation_precision([True, False, True]) == pytest.approx(2 / 3)


def test_evaluate_query_publishes_required_cutoffs_and_duplicate_count() -> None:
    metrics = evaluate_query(
        ["a", "a", "b", "c"],
        {"a": 2, "c": 1},
        top_k=20,
        document_fingerprints={"a": "x", "b": "x", "c": "z"},
    )

    assert set(metrics) >= {
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "mrr_at_20",
        "ndcg_at_10",
        "coverage_at_20",
        "duplicate_rate_at_20",
        "duplicate_ids_removed",
        "no_answer_correct",
        "citation_precision",
    }
    assert metrics["duplicate_ids_removed"] == 1
    assert metrics["ndcg_at_10"] <= 1.0


def test_evaluate_query_duplicate_rate_uses_exactly_the_at_20_cutoff() -> None:
    ranking = [f"d{i}" for i in range(22)]
    fingerprints = {document_id: document_id for document_id in ranking}
    fingerprints["d20"] = fingerprints["d21"] = "duplicate-after-cutoff"

    metrics = evaluate_query(
        ranking,
        {"d0": 1},
        top_k=22,
        document_fingerprints=fingerprints,
    )

    assert metrics["duplicate_rate_at_20"] == 0.0


def test_evaluate_query_never_scores_or_counts_past_top_k() -> None:
    metrics = evaluate_query(
        ["x", "a", "a"],
        {"a": 1},
        top_k=1,
        document_fingerprints={"a": "same"},
    )

    assert metrics["recall_at_20"] == 0.0
    assert metrics["duplicate_ids_removed"] == 0


def test_evaluate_query_rejects_a_string_ranking() -> None:
    with pytest.raises(TypeError, match="ranking"):
        evaluate_query("document-id", {"document-id": 1})


def test_stage_recorder_reports_every_stage_with_explicit_limits_and_recall() -> None:
    recorder = StageRecorder()
    recorder("dense", ["a", "x", "a"])
    recorder("sparse", ["b"])
    recorder("union", ["a", "b", "x"])
    recorder("fusion", ["a", "x", "b"])
    recorder("structural", ["b", "a", "x"])
    recorder("reranker", ["b", "a"])
    recorder("final", ["b"])

    report = recorder.report(
        {"a": 1, "b": 1},
        limits={"dense": 3, "sparse": 1, "union": 3, "final": 1},
    )

    assert tuple(report) == RETRIEVAL_STAGES
    assert report["dense"] == {
        "status": "observed",
        "limit": 3,
        "candidate_count": 2,
        "duplicate_ids_removed": 1,
        "recall": 0.5,
    }
    assert report["union"]["recall"] == 1.0
    assert report["final"]["recall"] == 0.5


def test_stage_recorder_is_resettable_and_returns_null_recall_without_qrels() -> None:
    recorder = StageRecorder()
    recorder("dense", [1, 2])
    recorder.reset()

    report = recorder.report({})

    assert all(item["status"] == "unobserved" for item in report.values())
    assert all(item["candidate_count"] is None for item in report.values())
    assert all(item["recall"] is None for item in report.values())


def test_stage_lineage_rejects_invented_ids_duplicates_and_limit_overflow() -> None:
    rankings = {
        "dense": ["a", "b", "b"],
        "sparse": ["c"],
        "union": ["a", "b", "c"],
        "fusion": ["a", "invented"],
        "structural": ["a", "invented"],
        "reranker": ["a"],
        "final": ["a", "extra"],
    }

    result = validate_stage_lineage(
        rankings,
        limits={"dense": 2, "final": 1},
        top_k=1,
    )

    assert result["status"] == "failed"
    codes = {item["code"] for item in result["violations"]}
    assert {"duplicate_id", "limit_exceeded", "invented_id"} <= codes


def test_native_stage_lineage_fails_closed_when_any_stage_is_missing() -> None:
    empty = validate_stage_lineage({}, top_k=20)
    missing = {stage: [] for stage in RETRIEVAL_STAGES if stage != "reranker"}
    incomplete = validate_stage_lineage(missing, top_k=20)

    assert empty["status"] == "failed"
    assert {item["stage"] for item in empty["violations"] if item["code"] == "missing_stage"} == set(
        RETRIEVAL_STAGES
    )
    assert incomplete["status"] == "failed"
    assert {item["stage"] for item in incomplete["violations"] if item["code"] == "missing_stage"} == {
        "reranker"
    }


def test_stage_limits_match_the_full_post_structural_candidate_pool() -> None:
    runner = _load_runner_module()

    limits = runner._stage_limits(
        SimpleNamespace(
            top_k=20,
            channel_k=20,
            structural_depth=5,
        )
    )

    # RetrievalV2 observes the complete candidate ranking after structural
    # reranking; structural_depth limits only how many candidates are scored.
    assert limits["fusion"] == 40
    assert limits["structural"] == 40
    assert limits["reranker"] == 40
    assert limits["final"] == 20


def test_paired_bootstrap_is_deterministic_paired_and_zero_for_identical_systems() -> None:
    baseline = {"q1": 0.25, "q2": 0.5, "q3": 1.0}
    candidate = {"q1": 0.5, "q2": 0.75, "q3": 1.0}

    first = paired_bootstrap(baseline, candidate, samples=2_000)
    second = paired_bootstrap(baseline, candidate, samples=2_000)
    identical = paired_bootstrap(baseline, baseline, samples=500)

    assert first == second
    assert first["seed"] == BOOTSTRAP_SEED == 20260813
    assert first["samples"] == 2_000
    assert first["n_queries"] == 3
    assert first["absolute_delta"] == pytest.approx(1 / 6)
    assert first["absolute_ci"][0] <= first["absolute_delta"] <= first["absolute_ci"][1]
    assert first["direction"] == "candidate_minus_baseline"
    assert identical["absolute_delta"] == 0.0
    assert identical["absolute_ci"] == [0.0, 0.0]


def test_paired_bootstrap_aligns_by_query_id_and_nulls_relative_when_baseline_zero() -> None:
    baseline = {"q2": 0.0, "q1": 0.0}
    candidate = {"q1": 1.0, "q2": 0.0}

    result = paired_bootstrap(baseline, candidate, samples=500)

    assert result["absolute_delta"] == pytest.approx(0.5)
    assert result["relative_gain"] is None
    assert result["relative_ci"] is None


def test_paired_bootstrap_uses_stable_direct_deltas_for_extreme_values() -> None:
    values = {"q1": 1e308, "q2": 1e308}

    result = paired_bootstrap(values, values, samples=250)

    assert result["absolute_delta"] == 0.0
    assert result["absolute_ci"] == [0.0, 0.0]


def test_paired_bootstrap_does_not_publish_a_conditional_relative_interval() -> None:
    result = paired_bootstrap(
        {"q1": 0.0, "q2": 1.0},
        {"q1": 1.0, "q2": 1.0},
        samples=500,
    )

    assert result["relative_gain"] == pytest.approx(1.0)
    assert result["relative_ci"] is None
    assert result["relative_defined_samples"] > 0
    assert result["relative_undefined_samples"] > 0


def test_paired_bootstrap_names_the_requested_confidence_level() -> None:
    result = paired_bootstrap({"q": 1.0}, {"q": 2.0}, samples=50, confidence=0.8)

    assert result["confidence"] == 0.8
    assert "absolute_ci" in result
    assert "absolute_ci95" not in result


def test_bootstrap_helpers_reject_unbounded_sample_counts() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        paired_bootstrap(
            {"q": 1.0},
            {"q": 1.0},
            samples=MAX_BOOTSTRAP_SAMPLES + 1,
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        clustered_percentile_bootstrap(
            {"q": [1.0]},
            samples=MAX_BOOTSTRAP_SAMPLES + 1,
        )


def test_bootstrap_helpers_reject_unbounded_sample_query_products() -> None:
    query_count = MAX_BOOTSTRAP_WORK // MAX_BOOTSTRAP_SAMPLES + 1
    baseline = {f"q{index}": 1.0 for index in range(query_count)}
    candidate = dict(baseline)
    latency = {query_id: [1.0] for query_id in baseline}

    with pytest.raises(ValueError, match="work"):
        paired_bootstrap(
            baseline,
            candidate,
            samples=MAX_BOOTSTRAP_SAMPLES,
        )
    with pytest.raises(ValueError, match="work"):
        clustered_percentile_bootstrap(
            latency,
            samples=MAX_BOOTSTRAP_SAMPLES,
        )


@pytest.mark.parametrize(
    ("baseline", "candidate"),
    [
        ({1: 0.1}, {1: 0.2}),
        ({True: 0.1}, {True: 0.2}),
        ({"q": 0.1, 1: 0.2}, {"q": 0.2, 1: 0.3}),
    ],
)
def test_paired_bootstrap_rejects_non_string_and_mixed_query_ids(
    baseline, candidate
) -> None:
    with pytest.raises(TypeError, match="query IDs.*strings"):
        paired_bootstrap(baseline, candidate, samples=100)


def test_clustered_percentile_bootstrap_resamples_queries_with_all_repetitions() -> None:
    grouped = {"q1": [1.0, 1.0], "q2": [100.0, 100.0]}

    first = clustered_percentile_bootstrap(grouped, percentile=0.95, samples=200)
    second = clustered_percentile_bootstrap(grouped, percentile=0.95, samples=200)

    assert first == second
    assert first["unit"] == "query_cluster"
    assert first["queries"] == 2
    assert first["observations"] == 4
    assert first["ci"][0] <= first["estimate"] <= first["ci"][1]


def test_latency_percentiles_use_linear_interpolation_and_validate_samples() -> None:
    assert latency_percentiles([1.0, 2.0, 3.0, 4.0]) == pytest.approx(
        {"p50": 2.5, "p95": 3.85, "p99": 3.97}
    )
    assert latency_percentiles([]) == {"p50": None, "p95": None, "p99": None}
    with pytest.raises(ValueError, match="finite|non-negative"):
        latency_percentiles([1.0, math.nan])


def test_aggregate_query_metrics_ignores_null_labels_without_turning_them_into_zero() -> None:
    aggregate = aggregate_query_metrics(
        [
            {"recall_at_20": 1.0, "citation_precision": None},
            {"recall_at_20": 0.0, "citation_precision": None},
        ]
    )

    assert aggregate["recall_at_20"] == 0.5
    assert aggregate["citation_precision"] is None
    assert aggregate["support"] == {"citation_precision": 0, "recall_at_20": 2}


def test_aggregate_query_metrics_uses_a_stable_mean_for_extreme_values() -> None:
    aggregate = aggregate_query_metrics([{"score": 1e308}, {"score": 1e308}])

    assert aggregate["score"] == 1e308


def test_hash_corpus_selection_is_stable_and_has_no_qrels_input() -> None:
    corpus = {f"d{i}": {"text": str(i)} for i in range(30)}
    qrels_a = {"q": {"d0": 1}}
    qrels_b = {"q": {"d29": 1}}

    first = select_corpus_by_hash(corpus, 8, seed=20260813)
    second = select_corpus_by_hash(corpus, 8, seed=20260813)

    assert first == second
    assert len(first) == 8
    assert "qrels" not in inspect.signature(select_corpus_by_hash).parameters
    assert qrels_a != qrels_b  # neither object participates in selection


def test_hash_corpus_selection_rejects_non_string_document_ids() -> None:
    with pytest.raises(TypeError, match="string"):
        select_corpus_by_hash({1: {"text": "one"}, "1": {"text": "string one"}}, 2)


def test_canonical_hash_is_order_independent_for_mapping_keys() -> None:
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})


def test_test_split_rejects_tuning_or_a_mismatched_validation_lock() -> None:
    hashes = {
        name: character * 64
        for name, character in zip(
            (
                "config_sha256",
                "dataset_sha256",
                "manifest_sha256",
                "generator_sha256",
                "runner_sha256",
                "evaluator_sha256",
                "pipeline_sha256",
                "runtime_source_closure_sha256",
                "source_diff_sha256",
                "source_sha256",
                "commit_state_sha256",
            ),
            "123456789ab",
        )
    }
    lock = {
        "schema_version": "retrieval-v2-validation-lock/1",
        "origin_protocol": "validation",
        "seed": 20260813,
        "metrics": {"recall": [5, 10, 20], "deduplication": "raw_cutoff"},
        "config": {"top_k": 20},
        "sources": {"dataset": "synthetic-v1"},
        "protocol": {"split_policy": "fixed"},
        "backends": ["legacy-sqlite", "v2-sqlite"],
        "hashes": hashes,
    }

    with pytest.raises(ValueError, match="tuning"):
        validate_split_protocol(
            "test",
            tuning_allowed=True,
            config_sha256=hashes["config_sha256"],
            dataset_sha256=hashes["dataset_sha256"],
            validation_lock=lock,
        )
    with pytest.raises(ValueError, match="config"):
        validate_split_protocol(
            "test",
            tuning_allowed=False,
            config_sha256="a" * 64,
            dataset_sha256=hashes["dataset_sha256"],
            validation_lock=lock,
        )
    with pytest.raises(ValueError, match="dataset"):
        validate_split_protocol(
            "test",
            tuning_allowed=False,
            config_sha256=hashes["config_sha256"],
            dataset_sha256="b" * 64,
            validation_lock=lock,
        )
    validate_split_protocol(
        "test",
        tuning_allowed=False,
        config_sha256=hashes["config_sha256"],
        dataset_sha256=hashes["dataset_sha256"],
        validation_lock=lock,
        expected_lock=lock,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("schema_version", "bad"), "schema"),
        (("origin_protocol", "test"), "origin"),
        (("seed", 7), "protocol state"),
    ],
)
def test_test_split_rejects_lock_schema_origin_and_protocol_drift(
    mutation: tuple[str, object], message: str
) -> None:
    digest_names = (
        "config_sha256",
        "dataset_sha256",
        "manifest_sha256",
        "generator_sha256",
        "runner_sha256",
        "evaluator_sha256",
        "pipeline_sha256",
        "runtime_source_closure_sha256",
        "source_diff_sha256",
        "source_sha256",
        "commit_state_sha256",
    )
    expected = {
        "schema_version": "retrieval-v2-validation-lock/1",
        "origin_protocol": "validation",
        "seed": 20260813,
        "metrics": {"cutoffs": [5, 10, 20]},
        "config": {"top_k": 20},
        "sources": {"manifest": "frozen"},
        "protocol": {"split_policy": "fixed"},
        "backends": ["legacy-sqlite", "v2-sqlite"],
        "hashes": {name: "a" * 64 for name in digest_names},
    }
    actual = deepcopy(expected)
    actual[mutation[0]] = mutation[1]

    with pytest.raises(ValueError, match=message):
        validate_split_protocol(
            "test",
            tuning_allowed=False,
            config_sha256="a" * 64,
            dataset_sha256="a" * 64,
            validation_lock=actual,
            expected_lock=expected,
        )


def test_test_split_rejects_empty_or_malformed_lock_digests() -> None:
    lock = {
        "schema_version": "retrieval-v2-validation-lock/1",
        "origin_protocol": "validation",
        "seed": 20260813,
        "metrics": {"cutoffs": [20]},
        "config": {"top_k": 20},
        "sources": {"manifest": "frozen"},
        "protocol": {"split_policy": "fixed"},
        "backends": ["v2-sqlite"],
        "hashes": {
            name: "a" * 64
            for name in (
                "config_sha256",
                "dataset_sha256",
                "manifest_sha256",
                "generator_sha256",
                "runner_sha256",
                "evaluator_sha256",
                "pipeline_sha256",
                "runtime_source_closure_sha256",
                "source_diff_sha256",
                "source_sha256",
                "commit_state_sha256",
            )
        },
    }
    lock["hashes"]["runner_sha256"] = ""

    with pytest.raises(ValueError, match="SHA256"):
        validate_split_protocol(
            "test",
            tuning_allowed=False,
            config_sha256="a" * 64,
            dataset_sha256="a" * 64,
            validation_lock=lock,
        )


def test_json_writer_rejects_nonfinite_numbers_and_round_trips(tmp_path: Path) -> None:
    destination = tmp_path / "report.json"
    payload = {"schema_version": "2.0", "value": 1.0}

    write_json_report(destination, payload)

    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    with pytest.raises(ValueError, match="finite|JSON"):
        write_json_report(destination, {"value": math.inf})


def test_json_writer_rejects_non_string_keys_without_replacing_the_old_report(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "report.json"
    original = {"status": "complete"}
    write_json_report(destination, original)

    with pytest.raises(TypeError, match="string"):
        write_json_report(destination, {"colliding": {1: "int", "1": "string"}})

    assert json.loads(destination.read_text(encoding="utf-8")) == original


def test_duplicate_identity_namespace_cannot_collide_with_a_fingerprint() -> None:
    fingerprints = {"a": ("document-id", "b")}

    assert duplicate_rate_at_k(["a", "b"], fingerprints, 2) == 0.0


def test_runner_rejects_cutoffs_below_the_required_recall_at_20() -> None:
    runner = _load_runner_module()

    with pytest.raises(SystemExit):
        runner._parse_arguments(["--scale", "24", "--top-k", "19"])


@pytest.mark.parametrize(
    "arguments",
    [
        ["--warmup", "10001"],
        ["--repetitions", "1001"],
        ["--bootstrap-samples", "1000001"],
    ],
)
def test_runner_bounds_expensive_repetition_controls(arguments) -> None:
    runner = _load_runner_module()

    with pytest.raises(SystemExit):
        runner._parse_arguments(arguments)


def test_test_protocol_rejects_ablation_or_tuning_variants(tmp_path: Path) -> None:
    runner = _load_runner_module()
    lock = tmp_path / "lock.json"
    lock.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit):
        runner._parse_arguments(
            [
                "--protocol",
                "test",
                "--validation-lock",
                str(lock),
                "--ablations",
                "all",
            ]
        )


def test_invalid_test_lock_is_rejected_before_qrels_are_materialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner_module()
    lock = tmp_path / "invalid-lock.json"
    lock.write_text("{}", encoding="utf-8")
    called = False

    def forbidden_materialization(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("test qrels were materialized before lock validation")

    monkeypatch.setattr(runner, "_materialize_queries", forbidden_materialization)
    with pytest.raises(ValueError, match="validation lock"):
        runner.run(
            [
                "--protocol",
                "test",
                "--validation-lock",
                str(lock),
                "--scale",
                "24",
                "--top-k",
                "20",
                "--channel-k",
                "20",
                "--output",
                str(tmp_path / "unused.json"),
            ]
        )

    assert called is False


def _lock_inputs(runner, tmp_path: Path):
    args = runner._parse_arguments(
        [
            "--protocol",
            "validation",
            "--scale",
            "24",
            "--top-k",
            "20",
            "--channel-k",
            "20",
            "--warmup",
            "0",
            "--repetitions",
            "1",
            "--bootstrap-samples",
            "100",
            "--output",
            str(tmp_path / "validation.json"),
        ]
    )
    manifest = runner._load_manifest()
    corpus = runner._generate_corpus(args.scale, manifest["seed"])
    _, manifest_sha256, dataset_sha256 = runner._dataset_identity(manifest, corpus)
    config = runner._common_config(args, args.backend_values, args.ablation_values)
    lock = runner._validation_lock_for(
        args,
        manifest,
        config,
        manifest_sha256=manifest_sha256,
        dataset_sha256=dataset_sha256,
    )
    return args, manifest, corpus, lock


def test_validation_lock_hashes_the_complete_runtime_source_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner_module()
    args, manifest, _, lock = _lock_inputs(runner, tmp_path)
    required = {
        f"rag3d/{name}.py"
        for name in (
            "engine",
            "store",
            "pgstore",
            "pgvector_store",
            "encoders",
            "fusion",
            "diversity",
            "rerank",
            "textproc",
            "ingest",
            "holo",
            "retrieve",
            "retrieval_v2",
            "backend",
            "config",
            "reader",
            "llm",
            "memory",
        )
    }
    runtime_sources = lock["sources"]["runtime_source_files_sha256"]
    assert required <= set(runtime_sources)
    assert lock["hashes"]["runtime_source_closure_sha256"] == canonical_sha256(
        runtime_sources
    )
    assert len(lock["hashes"]["source_diff_sha256"]) == 64

    original_sha256_file = runner._sha256_file

    def changed_encoder(path: Path) -> str:
        if Path(path).name == "encoders.py":
            return "f" * 64
        return original_sha256_file(path)

    monkeypatch.setattr(runner, "_sha256_file", changed_encoder)
    changed = runner._validation_lock_for(
        args,
        manifest,
        runner._common_config(args, args.backend_values, args.ablation_values),
        manifest_sha256=lock["hashes"]["manifest_sha256"],
        dataset_sha256=lock["hashes"]["dataset_sha256"],
    )
    assert changed["hashes"]["runtime_source_closure_sha256"] != lock["hashes"][
        "runtime_source_closure_sha256"
    ]
    assert changed["hashes"]["source_sha256"] != lock["hashes"]["source_sha256"]


def test_runtime_source_drift_rejects_test_before_qrels_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner_module()
    _, _, _, lock = _lock_inputs(runner, tmp_path)
    lock_path = tmp_path / "validation-lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    original_sha256_file = runner._sha256_file

    def changed_encoder(path: Path) -> str:
        if Path(path).name == "encoders.py":
            return "e" * 64
        return original_sha256_file(path)

    materialized = False

    def forbidden_materialization(*args, **kwargs):
        nonlocal materialized
        materialized = True
        raise AssertionError("source-drifted test qrels were materialized")

    monkeypatch.setattr(runner, "_sha256_file", changed_encoder)
    monkeypatch.setattr(runner, "_materialize_queries", forbidden_materialization)
    with pytest.raises(ValueError, match="protocol state"):
        runner.run(
            [
                "--protocol",
                "test",
                "--validation-lock",
                str(lock_path),
                "--scale",
                "24",
                "--top-k",
                "20",
                "--channel-k",
                "20",
                "--warmup",
                "0",
                "--repetitions",
                "1",
                "--bootstrap-samples",
                "100",
                "--output",
                str(tmp_path / "must-not-exist.json"),
            ]
        )
    assert materialized is False


def test_test_report_binds_the_validated_lock(tmp_path: Path) -> None:
    runner = _load_runner_module()
    lock_path = tmp_path / "validation-lock.json"
    common = [
        "--scale",
        "24",
        "--top-k",
        "20",
        "--channel-k",
        "20",
        "--dense-dim",
        "32",
        "--structural-dim",
        "8",
        "--warmup",
        "0",
        "--repetitions",
        "1",
        "--bootstrap-samples",
        "100",
    ]
    runner.run(
        [
            "--protocol",
            "validation",
            "--write-validation-lock",
            str(lock_path),
            "--output",
            str(tmp_path / "validation.json"),
            *common,
        ]
    )
    report = runner.run(
        [
            "--protocol",
            "test",
            "--validation-lock",
            str(lock_path),
            "--output",
            str(tmp_path / "test.json"),
            *common,
        ]
    )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))

    assert report["protocol"]["validation_lock_sha256"] == canonical_sha256(lock)
    assert report["protocol"]["validation_lock_identity"] == {
        "schema_version": lock["schema_version"],
        "origin_protocol": "validation",
        "config_sha256": lock["hashes"]["config_sha256"],
        "dataset_sha256": lock["hashes"]["dataset_sha256"],
        "runtime_source_closure_sha256": lock["hashes"][
            "runtime_source_closure_sha256"
        ],
        "source_diff_sha256": lock["hashes"]["source_diff_sha256"],
    }


def test_synthetic_materialized_splits_have_disjoint_qrels_and_valid_references() -> None:
    runner = _load_runner_module()
    manifest = runner._load_manifest()
    corpus = runner._generate_corpus(60, manifest["seed"])
    materialized = {
        split: runner._materialize_queries(manifest, split, corpus)
        for split in ("calibration", "validation", "test")
    }

    qrel_ids = {
        split: {
            document_id
            for query in queries
            for document_id in query["qrels"]
        }
        for split, queries in materialized.items()
    }
    assert qrel_ids["calibration"].isdisjoint(qrel_ids["validation"])
    assert qrel_ids["calibration"].isdisjoint(qrel_ids["test"])
    assert qrel_ids["validation"].isdisjoint(qrel_ids["test"])
    assert all(ids <= set(corpus) for ids in qrel_ids.values())


class _FakeAnnEncoder:
    def encode(self, texts, is_query=False):
        return [type("Vector", (), {"dense": [1.0, 0.0]})() for _ in texts]


class _FakeAnnStore:
    def __init__(self, exact, ann, hnsw_used=True):
        self.exact = exact
        self.ann = ann
        self.hnsw_used = hnsw_used

    def dense_search(self, vector, k, *, filters=None, exact=None):
        return list(self.exact if exact else self.ann)[:k]

    def explain_dense(self, vector, k, *, filters=None, exact=None):
        return {"plan": {"hnsw_used": self.hnsw_used}, "mode": "ann"}


@pytest.mark.parametrize(
    ("exact", "ann", "hnsw_used", "expected_status"),
    [
        ([(1, 1.0), (2, 1.0)], [(2, 1.0), (1, 1.0)], True, "passed"),
        ([(1, 1.0), (2, 1.0)], [(3, 1.0), (4, 1.0)], True, "failed"),
        ([(1, 1.0)], [(1, 1.0)], True, "not_evaluated"),
        ([(1, 1.0), (2, 0.9)], [(1, 1.0), (2, 0.9)], False, "not_evaluated"),
    ],
)
def test_hnsw_audit_requires_full_exact_and_ann_results_and_a_natural_hnsw_plan(
    exact, ann, hnsw_used: bool, expected_status: str
) -> None:
    runner = _load_runner_module()
    result = runner._evaluate_hnsw_recall(
        _FakeAnnStore(exact, ann, hnsw_used),
        _FakeAnnEncoder(),
        [{"query_id": "q1", "text": "secret query"}],
        k=2,
    )

    assert result["status"] == expected_status
    assert result["minimum_recall_at_k"] == pytest.approx(0.98)
    assert "secret query" not in json.dumps(result)


def test_hnsw_audit_rejects_eighty_percent_recall_as_non_promotable() -> None:
    runner = _load_runner_module()
    exact = [(item, 1.0) for item in range(1, 21)]
    ann = [(item, 1.0) for item in range(1, 17)] + [
        (item, 1.0) for item in range(21, 25)
    ]

    result = runner._evaluate_hnsw_recall(
        _FakeAnnStore(exact, ann, hnsw_used=True),
        _FakeAnnEncoder(),
        [{"query_id": "q1", "text": "bounded query"}],
        k=20,
    )

    assert result["recall_at_k"] == pytest.approx(0.8)
    assert result["status"] == "failed"
    assert result["queries"][0]["reason"] == "recall_below_preregistered_threshold"


def test_successful_qps_never_counts_failures_and_requires_a_sustained_clean_window() -> None:
    runner = _load_runner_module()
    failures = runner._throughput_report(
        completed=0,
        failed=100,
        measured_wall_seconds=0.1,
        concurrency=1,
        cache_mode="warm",
        process_isolation=True,
        preregistered_window=True,
    )
    shared = runner._throughput_report(
        completed=100,
        failed=0,
        measured_wall_seconds=5.0,
        concurrency=1,
        cache_mode="warm",
        process_isolation=False,
        preregistered_window=True,
    )
    eligible = runner._throughput_report(
        completed=100,
        failed=0,
        measured_wall_seconds=5.0,
        concurrency=1,
        cache_mode="warm",
        process_isolation=True,
        preregistered_window=True,
    )

    assert failures["successful_qps_sample"] == 0.0
    assert failures["serial_attempt_rate_sample"] == pytest.approx(1_000.0)
    assert failures["error_rate"] == 1.0
    assert failures["qps"] is None
    assert failures["claim_eligible"] is False
    assert "errors_present" in failures["non_claim_reasons"]
    assert shared["qps"] is None
    assert shared["claim_eligible"] is False
    assert shared["non_claim_reasons"] == ["process_not_isolated"]
    assert eligible["qps"] == 20.0
    assert eligible["claim_eligible"] is True
    assert eligible["non_claim_reasons"] == []


def test_qps_claim_requires_preregistered_load_metadata() -> None:
    runner = _load_runner_module()

    result = runner._throughput_report(
        completed=100,
        failed=0,
        measured_wall_seconds=5.0,
        concurrency=None,
        cache_mode=None,
        process_isolation=None,
        preregistered_window=False,
    )

    assert result["qps"] is None
    assert result["claim_eligible"] is False
    assert result["non_claim_reasons"] == [
        "concurrency_not_preregistered",
        "cache_mode_not_preregistered",
        "process_isolation_not_proven",
        "measurement_window_not_preregistered",
    ]


class _UnavailableStorageDb:
    def execute(self, statement):
        raise RuntimeError("storage unavailable")


def test_storage_failure_is_null_not_zero(tmp_path: Path) -> None:
    runner = _load_runner_module()
    store = SimpleNamespace(backend_name="pgvector", db=_UnavailableStorageDb())

    result = runner._index_size(store, tmp_path)

    assert result["index_bytes"] is None
    assert result["size_per_chunk_bytes"] is None
    assert result["status"] == "unavailable"
    assert result["error_type"] == "RuntimeError"


def test_storage_size_per_chunk_is_derived_only_from_available_bytes() -> None:
    runner = _load_runner_module()

    available = runner._storage_per_chunk(
        {"status": "available", "index_bytes": 101}, chunks=4
    )
    unavailable = runner._storage_per_chunk(
        {"status": "unavailable", "index_bytes": None}, chunks=4
    )

    assert available["size_per_chunk_bytes"] == pytest.approx(25.25)
    assert unavailable["size_per_chunk_bytes"] is None


class _FakeHnswBuilder:
    def __init__(self) -> None:
        self.options = None

    def create_hnsw_index(self, **options):
        self.options = options
        return {"ready": True, "created_by_caller": True}


def test_hnsw_build_time_is_measured_separately_from_ingest() -> None:
    runner = _load_runner_module()
    builder = _FakeHnswBuilder()
    ticks = iter((10.0, 12.5))

    result = runner._timed_hnsw_build(
        builder,
        m=16,
        ef_construction=64,
        concurrently=False,
        clock=lambda: next(ticks),
    )

    assert result == {
        "status": "measured",
        "hnsw_build_seconds": 2.5,
        "index_status": {"ready": True, "created_by_caller": True},
        "created_by_runner": True,
    }
    assert builder.options == {"m": 16, "ef_construction": 64, "concurrently": False}


def test_hnsw_build_reports_a_preexisting_compatible_index_as_not_measured() -> None:
    runner = _load_runner_module()

    class ExistingBuilder:
        def refresh_capabilities(self):
            return {
                "exists": True,
                "valid": True,
                "ready": True,
                "definition_valid": True,
                "options": {"m": 16, "ef_construction": 64},
            }

        def create_hnsw_index(self, **_options):
            raise AssertionError("an existing compatible index is not a build")

    result = runner._timed_hnsw_build(
        ExistingBuilder(), m=16, ef_construction=64, concurrently=False
    )

    assert result["status"] == "already_present"
    assert result["hnsw_build_seconds"] is None
    assert result["created_by_runner"] is False


def test_hnsw_build_race_recovery_is_not_measured_or_owned_by_runner() -> None:
    runner = _load_runner_module()

    class RaceRecoveredBuilder:
        def refresh_capabilities(self):
            return {"exists": False}

        def create_hnsw_index(self, **_options):
            return {
                "ready": True,
                "created_by_caller": False,
            }

    result = runner._timed_hnsw_build(
        RaceRecoveredBuilder(),
        m=16,
        ef_construction=64,
        concurrently=False,
        clock=iter((10.0, 12.5)).__next__,
    )

    assert result == {
        "status": "race_recovered",
        "hnsw_build_seconds": None,
        "index_status": {"ready": True, "created_by_caller": False},
        "created_by_runner": False,
    }


def test_cold_sample_runs_before_warmup_and_not_as_a_warm_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    events = []

    class Resettable:
        def reset_capture(self):
            pass

    class Recorder:
        def reset(self):
            pass

    class Retriever:
        def search(self, text, *, top_k, channel_k):
            events.append("warmup")

    system = {
        "id": "legacy-sqlite",
        "specification": {"id": "legacy-sqlite"},
        "backend_view": Resettable(),
        "recorder": Recorder(),
        "retriever": Retriever(),
    }
    query = {"query_id": "q1", "text": "not serialized", "qrels": {}, "unanswerable_gold": None}

    def cold_once(*args, **kwargs):
        events.append("cold")
        return {"status": "measured", "cold_latency_ms": 1.0}

    monkeypatch.setattr(runner, "_cold_search_once", cold_once, raising=False)
    runner._measure_systems(
        SimpleNamespace(warmup=1, repetitions=0, top_k=20, channel_k=20),
        [system],
        [query],
        {},
        {},
        {"sqlite": {}},
    )

    assert events == ["cold", "warmup"]
    assert system["cold_measurement"] == {
        "status": "measured",
        "cold_latency_ms": 1.0,
    }
    assert system["completed"] == 0
    assert system["failed"] == 0


@pytest.mark.parametrize("failure_mode", ["runtime_error", "ranking_drift"])
def test_measurement_integrity_fails_closed_on_partial_or_drifting_runs(
    failure_mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner_module()

    class BackendView:
        def reset_capture(self):
            pass

    class Retriever:
        def __init__(self):
            self.calls = 0

        def search(self, _text, *, top_k, channel_k):
            self.calls += 1
            if failure_mode == "runtime_error" and self.calls > 1:
                raise RuntimeError("injected repetition failure")
            document_id = 1 if self.calls == 1 else 2
            return SimpleNamespace(
                fused=[{"id": document_id, "doc_id": document_id}],
                stats={},
            )

    system = {
        "id": "v2-sqlite",
        "specification": {"id": "v2-sqlite", "pipeline": "v2"},
        "backend_view": BackendView(),
        "recorder": StageRecorder(),
        "retriever": Retriever(),
        "document_id_map": {1: "doc-a", 2: "doc-b"},
    }
    query = {
        "query_id": "q1",
        "text": "not serialized",
        "qrels": {"doc-a": 1},
        "required_facts": None,
        "unanswerable_gold": None,
    }
    monkeypatch.setattr(
        runner,
        "_cold_search_once",
        lambda *_args, **_kwargs: {"status": "measured"},
    )

    with pytest.raises(RuntimeError, match="measurement integrity"):
        runner._measure_systems(
            SimpleNamespace(
                warmup=0,
                repetitions=3,
                top_k=20,
                channel_k=20,
                structural_depth=20,
            ),
            [system],
            [query],
            {},
            {},
            {"sqlite": {1: "doc-a", 2: "doc-b"}},
        )


def test_measurement_integrity_fails_closed_on_warmup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()

    class BackendView:
        def reset_capture(self):
            pass

    class Retriever:
        def search(self, _text, *, top_k, channel_k):
            raise RuntimeError("injected warmup failure")

    system = {
        "id": "v2-sqlite",
        "specification": {"id": "v2-sqlite", "pipeline": "v2"},
        "backend_view": BackendView(),
        "recorder": StageRecorder(),
        "retriever": Retriever(),
        "document_id_map": {},
    }
    query = {
        "query_id": "q1",
        "text": "not serialized",
        "qrels": {},
        "required_facts": None,
        "unanswerable_gold": None,
    }
    monkeypatch.setattr(
        runner,
        "_cold_search_once",
        lambda *_args, **_kwargs: {"status": "measured"},
    )

    with pytest.raises(RuntimeError, match="measurement integrity"):
        runner._measure_systems(
            SimpleNamespace(warmup=1, repetitions=0, top_k=20, channel_k=20),
            [system],
            [query],
            {},
            {},
            {"sqlite": {}},
        )


@pytest.mark.parametrize("backend", ["postgres-holo", "pgvector"])
def test_remote_backend_requires_explicit_write_authorization(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner_module()
    monkeypatch.setenv("RAG3D_BENCHMARK_PG_DSN", "postgresql:///rag3d_bench")
    monkeypatch.delenv("RAG3D_BENCHMARK_ALLOW_WRITE", raising=False)

    with pytest.raises(SystemExit):
        runner._parse_arguments(["--backends", backend, "--allow-remote"])


def test_remote_backend_rejects_a_non_test_database_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    monkeypatch.setenv("RAG3D_BENCHMARK_PG_DSN", "postgresql:///production")
    monkeypatch.setenv("RAG3D_BENCHMARK_ALLOW_WRITE", "1")

    with pytest.raises(SystemExit):
        runner._parse_arguments(["--backends", "pgvector", "--allow-remote"])


@pytest.mark.parametrize("database", ["production_contest", "benchmarker"])
def test_remote_backend_requires_a_delimited_test_or_bench_database_token(
    database: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner_module()
    monkeypatch.setenv("RAG3D_BENCHMARK_PG_DSN", f"postgresql:///{database}")
    monkeypatch.setenv("RAG3D_BENCHMARK_ALLOW_WRITE", "1")

    with pytest.raises(SystemExit):
        runner._parse_arguments(["--backends", "pgvector", "--allow-remote"])


@pytest.mark.parametrize(
    "dsn",
    [
        "dbname=rag3d_bench dbname=production",
        "postgresql:///rag3d_bench?dbname=production",
    ],
)
def test_remote_guard_uses_the_effective_libpq_database_override(
    dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner_module()
    monkeypatch.setenv("RAG3D_BENCHMARK_PG_DSN", dsn)
    monkeypatch.setenv("RAG3D_BENCHMARK_ALLOW_WRITE", "1")

    with pytest.raises(SystemExit):
        runner._parse_arguments(["--backends", "pgvector", "--allow-remote"])


def test_remote_guard_uses_libpq_percent_decoding_and_never_prints_password(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    runner = _load_runner_module()
    secret = "never-print-this-password"
    monkeypatch.setenv(
        "RAG3D_BENCHMARK_PG_DSN",
        f"postgresql://user:{secret}@localhost/rag3d%5Fbench",
    )
    monkeypatch.setenv("RAG3D_BENCHMARK_ALLOW_WRITE", "1")

    args = runner._parse_arguments(["--backends", "pgvector", "--allow-remote"])

    assert args.backend_values == ["pgvector"]
    assert secret not in capsys.readouterr().err


def test_remote_guard_fails_closed_without_echoing_a_malformed_secret_dsn(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    runner = _load_runner_module()
    secret = "never-print-this-malformed-password"
    monkeypatch.setenv(
        "RAG3D_BENCHMARK_PG_DSN",
        f"postgresql://user:{secret}@[invalid/rag3d_bench",
    )
    monkeypatch.setenv("RAG3D_BENCHMARK_ALLOW_WRITE", "1")

    with pytest.raises(SystemExit):
        runner._parse_arguments(["--backends", "pgvector", "--allow-remote"])

    assert secret not in capsys.readouterr().err


def test_remote_backend_accepts_double_authorization_for_a_benchmark_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    monkeypatch.setenv(
        "RAG3D_BENCHMARK_PG_DSN", "dbname=rag3d_test host=/private/tmp"
    )
    monkeypatch.setenv("RAG3D_BENCHMARK_ALLOW_WRITE", "1")

    args = runner._parse_arguments(["--backends", "pgvector", "--allow-remote"])

    assert args.backend_values == ["pgvector"]


@pytest.mark.parametrize(
    ("documents", "chunks", "sparse_postings"),
    [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
)
def test_remote_ingest_requires_all_pgvector_relations_to_be_empty(
    documents: int, chunks: int, sparse_postings: int
) -> None:
    runner = _load_runner_module()
    store = SimpleNamespace(
        backend_name="pgvector",
        metrics=lambda: {
            "status": "ok",
            "counts": {
                "documents": documents,
                "chunks": chunks,
                "sparse_postings": sparse_postings,
            },
        },
    )

    with pytest.raises(RuntimeError, match="non-empty"):
        runner._assert_remote_store_empty(store)


def test_remote_ingest_requires_holographic_spectrum_to_be_empty() -> None:
    runner = _load_runner_module()

    class Database:
        def execute(self, _statement):
            return self

        def fetchone(self):
            return (0, 0, 1)

    store = SimpleNamespace(backend_name="postgres-holo", db=Database())

    with pytest.raises(RuntimeError, match="non-empty"):
        runner._assert_remote_store_empty(store)


def test_partial_remote_backend_build_deletes_committed_docs_and_closes_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner_module()
    events = []

    class Store:
        backend_name = "postgres-holo"

        class _Database:
            def execute(self, _sql):
                return self

            def fetchone(self):
                return (0, 0, 0)

        db = _Database()

        def metrics(self):
            return {
                "status": "ok",
                "counts": {
                    "documents": 0,
                    "chunks": 0,
                    "sparse_postings": 0,
                },
            }

        def delete_document(self, document_id):
            events.append(("delete", document_id))

        def close(self):
            events.append(("close", None))

        def commit(self):
            events.append(("commit", None))

    class Rag:
        def __init__(self, cfg, llm):
            self.store = Store()
            self.calls = 0

        def ingest(self, text, source, title):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("injected ingest failure")
            return {"doc_id": 41, "chunks": 1}

    monkeypatch.setattr(runner, "TriRag", Rag)
    monkeypatch.setenv("RAG3D_BENCHMARK_PG_DSN", "postgresql:///rag3d_bench")
    monkeypatch.setenv("RAG3D_BENCHMARK_ALLOW_WRITE", "1")
    args = runner._parse_arguments(
        ["--backends", "postgres-holo", "--allow-remote"]
    )
    corpus = {
        "doc-a": {"text": "first"},
        "doc-b": {"text": "second"},
    }

    with pytest.raises(RuntimeError, match="injected ingest failure"):
        runner._build_backend(
            args,
            "postgres-holo",
            corpus,
            tmp_path,
            "postgresql:///rag3d_bench",
        )

    assert events == [("delete", 41), ("commit", None), ("close", None)]


def test_remote_cleanup_failure_is_not_suppressed_and_always_closes() -> None:
    runner = _load_runner_module()
    events = []

    class Store:
        backend_name = "pgvector"

        def delete_document(self, document_id):
            events.append(("delete", document_id))
            raise RuntimeError("contains a secret that must not be echoed")

        def commit(self):
            events.append(("commit", None))

        def metrics(self):
            return {
                "status": "ok",
                "counts": {
                    "documents": 1,
                    "chunks": 1,
                    "sparse_postings": 1,
                },
            }

        def close(self):
            events.append(("close", None))

    state = {
        "rag": SimpleNamespace(store=Store()),
        "persisted_document_ids": [41],
        "hnsw_build": {"created_by_runner": False},
    }

    with pytest.raises(RuntimeError, match="remote benchmark cleanup failed") as captured:
        runner._cleanup_backend_states({"pgvector": state})

    assert "secret" not in str(captured.value)
    assert events[-1] == ("close", None)


def test_remote_cleanup_drops_only_an_hnsw_index_created_by_the_runner() -> None:
    runner = _load_runner_module()
    events = []

    class Store:
        backend_name = "pgvector"

        def delete_document(self, document_id):
            events.append(("delete", document_id))

        def commit(self):
            events.append(("commit", None))

        def metrics(self):
            return {
                "status": "ok",
                "counts": {
                    "documents": 0,
                    "chunks": 0,
                    "sparse_postings": 0,
                },
            }

        def drop_hnsw_index(self):
            events.append(("drop_hnsw_index", None))

        def close(self):
            events.append(("close", None))

    state = {
        "rag": SimpleNamespace(store=Store()),
        "persisted_document_ids": [41],
        "hnsw_build": {"created_by_runner": True},
    }

    runner._cleanup_backend_states({"pgvector": state})

    assert ("drop_hnsw_index", None) in events
    assert events[-1] == ("close", None)


def test_remote_cleanup_never_drops_a_race_recovered_hnsw_index() -> None:
    runner = _load_runner_module()
    events = []

    class Store:
        backend_name = "pgvector"

        def commit(self):
            events.append(("commit", None))

        def metrics(self):
            return {
                "status": "ok",
                "counts": {
                    "documents": 0,
                    "chunks": 0,
                    "sparse_postings": 0,
                },
            }

        def drop_hnsw_index(self):
            raise AssertionError("race-recovered index belongs to another caller")

        def close(self):
            events.append(("close", None))

    state = {
        "rag": SimpleNamespace(store=Store()),
        "persisted_document_ids": [],
        "hnsw_build": {
            "status": "race_recovered",
            "created_by_runner": False,
        },
    }

    runner._cleanup_backend_states({"pgvector": state})

    assert events[-1] == ("close", None)


def test_runner_smoke_compares_legacy_and_v2_and_emits_auditable_schema(
    tmp_path: Path,
) -> None:
    output = tmp_path / "retrieval-v2.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--scale",
            "24",
            "--top-k",
            "20",
            "--channel-k",
            "20",
            "--dense-dim",
            "32",
            "--structural-dim",
            "8",
            "--warmup",
            "0",
            "--repetitions",
            "1",
            "--bootstrap-samples",
            "100",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "2.0"
    assert payload["dataset"]["id"] == "retrieval-v2-synthetic-v1"
    assert payload["dataset"]["documents"] == 24
    assert payload["config"]["top_k"] == 20
    assert payload["config"]["channel_k"] == 20
    assert payload["protocol"]["repetitions"] == 1
    assert payload["protocol"]["warmup"] == 0
    assert payload["protocol"]["bootstrap_samples"] == 100
    assert payload["environment"]["hardware"]
    assert payload["run"]["commit"]
    assert payload["run"]["process_cpu_seconds"] >= 0.0
    assert payload["run"]["cpu_scope"] == "whole_shared_python_process_not_attributable_per_system"
    systems = {item["id"]: item for item in payload["systems"]}
    assert {"legacy-sqlite", "v2-sqlite"} <= set(systems)
    assert systems["v2-sqlite"]["queries"]
    query = systems["v2-sqlite"]["queries"][0]
    assert tuple(query["stages"]) == RETRIEVAL_STAGES
    assert all(
        item["lineage"]["status"] == "passed"
        and tuple(item["lineage"]["observed_stages"]) == RETRIEVAL_STAGES
        for item in systems["v2-sqlite"]["queries"]
    )
    assert "total" in query["timings_ms"]
    assert len(query["timings_ms"]["total"]) == 1
    performance = systems["v2-sqlite"]["aggregate"]["performance"]
    assert set(performance["latency_ms"]) == {"p50", "p95", "p99"}
    assert performance["qps"]["completed"] > 0
    assert performance["qps"]["qps"] is None
    assert performance["qps"]["successful_qps_sample"] > 0.0
    assert performance["qps"]["claim_eligible"] is False
    assert performance["cold_latency_ms"]["sample_count"] == 1
    assert performance["cold_latency_ms"]["claim_eligible"] is False
    assert performance["cold_latency_ms"]["single_run"] is True
    assert performance["latency_p95_bootstrap"]["unit"] == "query_cluster"
    assert performance["memory"]["rss_peak_bytes"] >= 0
    assert performance["memory"]["comparable_across_systems"] is False
    assert performance["storage"]["index_bytes"] >= 0
    assert performance["storage"]["size_per_chunk_bytes"] >= 0.0
    assert performance["storage"]["hnsw_build_seconds"] is None
    assert performance["ingest"]["documents"] == 24
    assert payload["comparisons"]


def test_runner_legacy_baseline_uses_the_public_legacy_diversity_default(
    tmp_path: Path,
) -> None:
    runner = _load_runner_module()
    args = runner._parse_arguments(
        [
            "--scale",
            "24",
            "--dense-dim",
            "32",
            "--structural-dim",
            "8",
            "--max-structural-tokens",
            "8",
        ]
    )
    specification = runner._system_specifications("sqlite", [])[0]

    config = runner._system_config(args, "sqlite", specification, tmp_path, "")
    public_legacy = runner.TriRagConfig(retrieval_pipeline="legacy")

    assert specification["id"] == "legacy-sqlite"
    assert specification["diversity"] == "legacy-default"
    assert config.diversity == public_legacy.diversity == 0.35


def test_synthetic_dataset_manifest_freezes_seed_and_disjoint_splits() -> None:
    manifest_path = ROOT / "benchmarks" / "datasets" / "retrieval_v2_synthetic_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["seed"] == 20260813
    split_ids = [set(item["id"] for item in manifest["splits"][name]) for name in ("calibration", "validation", "test")]
    assert split_ids[0].isdisjoint(split_ids[1])
    assert split_ids[0].isdisjoint(split_ids[2])
    assert split_ids[1].isdisjoint(split_ids[2])
