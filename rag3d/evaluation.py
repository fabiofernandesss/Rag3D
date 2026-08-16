"""Pure, reproducible evaluation helpers for Retrieval Engine V2.

The quality unit is a unique query.  Repeated latency measurements stay
grouped under that query and never become extra quality observations.  The
module intentionally has no storage or model dependency so benchmark reports
can be audited and recalculated without opening an index.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import (
    Any,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)


BOOTSTRAP_SEED = 20260813
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
MAX_BOOTSTRAP_SAMPLES = 1_000_000
# Finite axes are not sufficient when their product is impractical.  This
# bounds the dominant resampling work while still allowing the default 10k
# bootstrap on several thousand BEIR-style queries.
MAX_BOOTSTRAP_WORK = 100_000_000
MAX_BOOTSTRAP_QUERIES = 100_000
MAX_BOOTSTRAP_OBSERVATIONS = 1_000_000
RETRIEVAL_STAGES = (
    "dense",
    "sparse",
    "union",
    "fusion",
    "structural",
    "reranker",
    "final",
)
VALIDATION_LOCK_SCHEMA = "retrieval-v2-validation-lock/1"
REQUIRED_VALIDATION_HASHES = (
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

_DOCUMENT_ID_NAMESPACE = object()
_FINGERPRINT_NAMESPACE = object()


def _validate_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer, not bool")
    if k < 0:
        raise ValueError("k must be non-negative")
    return k


def _validate_finite_number(name: str, value: Any, *, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, not bool")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if non_negative and normalized < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def deduplicate_ranking(
    ranking: Iterable[Hashable],
) -> Tuple[List[Hashable], int]:
    """Keep the first occurrence of each identifier.

    This utility does not apply a cutoff. Metrics take their raw prefix first,
    so a duplicate consumes a rank even though it receives no second gain.
    """

    if isinstance(ranking, (str, bytes)):
        raise TypeError("ranking must be an iterable of document identifiers")
    unique: List[Hashable] = []
    seen = set()
    removed = 0
    for document_id in ranking:
        try:
            duplicate = document_id in seen
        except TypeError as exc:
            raise TypeError("document identifiers must be hashable") from exc
        if duplicate:
            removed += 1
            continue
        seen.add(document_id)
        unique.append(document_id)
    return unique, removed


def _raw_prefix(ranking: Iterable[Hashable], k: int) -> List[Hashable]:
    """Take at most ``k`` raw ranks without backfilling duplicate IDs."""

    cutoff = _validate_k(k)
    if isinstance(ranking, (str, bytes)):
        raise TypeError("ranking must be an iterable of document identifiers")
    try:
        prefix = list(itertools.islice(iter(ranking), cutoff))
    except TypeError as exc:
        raise TypeError("ranking must be an iterable of document identifiers") from exc
    for document_id in prefix:
        try:
            hash(document_id)
        except TypeError as exc:
            raise TypeError("document identifiers must be hashable") from exc
    return prefix


def _positive_relevance(relevance: Mapping[Hashable, Any]) -> Dict[Hashable, float]:
    if not isinstance(relevance, Mapping):
        raise TypeError("relevance must be a mapping of document ID to grade")
    positive: Dict[Hashable, float] = {}
    for document_id, raw_grade in relevance.items():
        grade = _validate_finite_number("relevance grade", raw_grade)
        if grade > 0.0:
            positive[document_id] = grade
    return positive


def recall_at_k(
    ranking: Iterable[Hashable], relevance: Mapping[Hashable, Any], k: int
) -> Optional[float]:
    """Return relevance Recall@k, or ``None`` when labels have no positives."""

    cutoff = _validate_k(k)
    positive = _positive_relevance(relevance)
    if not positive:
        return None
    if cutoff == 0:
        return 0.0
    prefix = _raw_prefix(ranking, cutoff)
    recovered = len(set(prefix).intersection(positive))
    return min(1.0, max(0.0, recovered / len(positive)))


def mrr_at_k(
    ranking: Iterable[Hashable], relevance: Mapping[Hashable, Any], k: int
) -> Optional[float]:
    """Return reciprocal rank truncated at ``k`` for one query."""

    cutoff = _validate_k(k)
    positive = _positive_relevance(relevance)
    if not positive:
        return None
    prefix = _raw_prefix(ranking, cutoff)
    for index, document_id in enumerate(prefix):
        if document_id in positive:
            return 1.0 / (index + 1)
    return 0.0


def ndcg_at_k(
    ranking: Iterable[Hashable], relevance: Mapping[Hashable, Any], k: int
) -> Optional[float]:
    """Return graded nDCG@k with raw ranks and zero gain for duplicates."""

    cutoff = _validate_k(k)
    positive = _positive_relevance(relevance)
    if not positive:
        return None
    if cutoff == 0:
        return 0.0
    prefix = _raw_prefix(ranking, cutoff)
    maximum_grade = max(positive.values())

    def scaled_gain(grade: float) -> float:
        # Equivalent to (2**grade - 1) * 2**(-maximum_grade), evaluated in a
        # form that preserves subnormal/tiny grades and avoids huge overflow.
        return math.pow(2.0, grade - maximum_grade) * -math.expm1(
            -math.log(2.0) * grade
        )

    seen = set()
    dcg_terms = []
    for index, document_id in enumerate(prefix):
        grade = 0.0 if document_id in seen else positive.get(document_id, 0.0)
        seen.add(document_id)
        if grade > 0.0:
            dcg_terms.append(scaled_gain(grade) / math.log2(index + 2))
    dcg = math.fsum(dcg_terms)
    ideal_grades = sorted(positive.values(), reverse=True)[:cutoff]
    idcg = math.fsum(
        scaled_gain(grade) / math.log2(index + 2)
        for index, grade in enumerate(ideal_grades)
    )
    if idcg <= 0.0:
        return None
    # The minimum/maximum also absorbs harmless floating-point overshoot.
    return min(1.0, max(0.0, dcg / idcg))


def coverage_at_k(
    ranking: Iterable[Hashable],
    document_facts: Mapping[Hashable, Iterable[Hashable]],
    required_facts: Optional[Iterable[Hashable]],
    k: int,
) -> Optional[float]:
    """Return distinct fact coverage, or ``None`` without fact labels."""

    cutoff = _validate_k(k)
    if required_facts is None:
        return None
    if isinstance(required_facts, (str, bytes)):
        raise TypeError("required_facts must be an iterable of fact identifiers")
    required = set(required_facts)
    if not required:
        return None
    prefix = _raw_prefix(ranking, cutoff)
    covered = set()
    for document_id in prefix:
        facts = document_facts.get(document_id, ())
        if isinstance(facts, (str, bytes)):
            facts = (facts,)
        covered.update(fact for fact in facts if fact in required)
    return min(1.0, max(0.0, len(covered) / len(required)))


def duplicate_rate_at_k(
    ranking: Iterable[Hashable],
    document_fingerprints: Mapping[Hashable, Hashable],
    k: int,
) -> float:
    """Return exact-content duplicate pairs divided by all pairs in top-k.

    Unlike relevance metrics this intentionally inspects the raw prefix: two
    different document IDs with the same normalized-content fingerprint are
    the redundancy the metric is meant to expose.  Repeated IDs also count as
    duplicate results.
    """

    cutoff = _validate_k(k)
    if isinstance(ranking, (str, bytes)):
        raise TypeError("ranking must be an iterable of document identifiers")
    prefix = _raw_prefix(ranking, cutoff)
    if len(prefix) < 2:
        return 0.0
    identities = []
    for document_id in prefix:
        try:
            if document_id in document_fingerprints:
                identity = (_FINGERPRINT_NAMESPACE, document_fingerprints[document_id])
            else:
                identity = (_DOCUMENT_ID_NAMESPACE, document_id)
            hash(identity)
        except TypeError as exc:
            raise TypeError("document fingerprints must be hashable") from exc
        identities.append(identity)
    duplicate_pairs = sum(count * (count - 1) // 2 for count in Counter(identities).values())
    all_pairs = len(prefix) * (len(prefix) - 1) // 2
    return min(1.0, max(0.0, duplicate_pairs / all_pairs))


def no_answer_correct(
    unanswerable_gold: Optional[bool], abstain_pred: Optional[bool]
) -> Optional[float]:
    """Score an explicit abstention decision; missing either label is null."""

    if unanswerable_gold is None or abstain_pred is None:
        return None
    if not isinstance(unanswerable_gold, bool) or not isinstance(abstain_pred, bool):
        raise TypeError("no-answer labels and predictions must be bool or None")
    return 1.0 if unanswerable_gold == abstain_pred else 0.0


def citation_precision(
    citation_support_labels: Optional[Iterable[bool]],
) -> Optional[float]:
    """Return supported/total citations, remaining null without labels."""

    if citation_support_labels is None:
        return None
    if isinstance(citation_support_labels, (str, bytes)):
        raise TypeError("citation labels must be an iterable of booleans")
    labels = list(citation_support_labels)
    if not labels:
        return None
    if any(not isinstance(label, bool) for label in labels):
        raise TypeError("citation labels must be booleans")
    return sum(1 for label in labels if label) / len(labels)


def evaluate_query(
    ranking: Iterable[Hashable],
    relevance: Mapping[Hashable, Any],
    *,
    top_k: int = 20,
    document_facts: Optional[Mapping[Hashable, Iterable[Hashable]]] = None,
    required_facts: Optional[Iterable[Hashable]] = None,
    document_fingerprints: Optional[Mapping[Hashable, Hashable]] = None,
    unanswerable_gold: Optional[bool] = None,
    abstain_pred: Optional[bool] = None,
    citation_support_labels: Optional[Iterable[bool]] = None,
) -> Dict[str, Optional[float]]:
    """Evaluate one ranking using the fixed Retrieval V2 metric suite."""

    raw_ranking = _raw_prefix(ranking, top_k)
    _, removed = deduplicate_ranking(raw_ranking)
    facts = document_facts or {}
    fingerprints = document_fingerprints or {}
    return {
        "recall_at_5": recall_at_k(raw_ranking, relevance, 5),
        "recall_at_10": recall_at_k(raw_ranking, relevance, 10),
        "recall_at_20": recall_at_k(raw_ranking, relevance, 20),
        "mrr_at_20": mrr_at_k(raw_ranking, relevance, 20),
        "ndcg_at_10": ndcg_at_k(raw_ranking, relevance, 10),
        "coverage_at_20": coverage_at_k(raw_ranking, facts, required_facts, 20),
        "duplicate_rate_at_20": duplicate_rate_at_k(
            raw_ranking, fingerprints, 20
        ),
        "duplicate_ids_removed": removed,
        "no_answer_correct": no_answer_correct(unanswerable_gold, abstain_pred),
        "citation_precision": citation_precision(citation_support_labels),
    }


class StageRecorder:
    """Callable observer that snapshots bounded rankings emitted by V2."""

    def __init__(self) -> None:
        self._rankings: Dict[str, Tuple[Hashable, ...]] = {}

    def __call__(self, stage: str, ids: Sequence[Hashable]) -> None:
        if stage not in RETRIEVAL_STAGES:
            raise ValueError("unknown retrieval stage")
        if isinstance(ids, (str, bytes)):
            raise TypeError("stage IDs must be a sequence")
        self._rankings[stage] = tuple(ids)

    def reset(self) -> None:
        self._rankings.clear()

    def snapshot(self) -> Dict[str, Tuple[Hashable, ...]]:
        return {
            stage: tuple(self._rankings[stage])
            for stage in RETRIEVAL_STAGES
            if stage in self._rankings
        }

    def report(
        self,
        relevance: Mapping[Hashable, Any],
        *,
        limits: Optional[Mapping[str, int]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        limits = limits or {}
        report: Dict[str, Dict[str, Any]] = {}
        for stage in RETRIEVAL_STAGES:
            observed = stage in self._rankings
            raw = list(self._rankings.get(stage, ()))
            unique, removed = deduplicate_ranking(raw)
            limit = limits.get(stage, len(unique))
            _validate_k(limit)
            report[stage] = {
                "status": "observed" if observed else "unobserved",
                "limit": limit,
                "candidate_count": len(unique) if observed else None,
                "duplicate_ids_removed": removed if observed else None,
                "recall": recall_at_k(raw, relevance, limit) if observed else None,
            }
        return report


def validate_stage_lineage(
    stage_rankings: Mapping[str, Sequence[Hashable]],
    *,
    limits: Optional[Mapping[str, int]] = None,
    top_k: int,
) -> Dict[str, Any]:
    """Validate observed candidate lineage without manufacturing missing stages."""

    if not isinstance(stage_rankings, Mapping):
        raise TypeError("stage_rankings must be a mapping")
    limits = limits or {}
    final_limit = _validate_k(top_k)
    snapshots: Dict[str, Tuple[Hashable, ...]] = {}
    violations: List[Dict[str, Any]] = []
    for stage in RETRIEVAL_STAGES:
        if stage not in stage_rankings:
            violations.append({"stage": stage, "code": "missing_stage"})
    for stage, raw_ids in stage_rankings.items():
        if stage not in RETRIEVAL_STAGES:
            raise ValueError(f"unknown retrieval stage: {stage}")
        if isinstance(raw_ids, (str, bytes)):
            raise TypeError("stage IDs must be sequences")
        ids = tuple(raw_ids)
        snapshots[stage] = ids
        unique, removed = deduplicate_ranking(ids)
        if removed:
            violations.append(
                {"stage": stage, "code": "duplicate_id", "count": removed}
            )
        limit = limits.get(stage)
        if limit is not None:
            checked_limit = _validate_k(limit)
            if len(ids) > checked_limit:
                violations.append(
                    {
                        "stage": stage,
                        "code": "limit_exceeded",
                        "candidate_count": len(ids),
                        "limit": checked_limit,
                    }
                )
        for document_id in unique:
            if not isinstance(document_id, str):
                violations.append(
                    {"stage": stage, "code": "non_string_id", "value_type": type(document_id).__name__}
                )

    def require_subset(child: str, parents: Sequence[str]) -> None:
        if child not in snapshots or any(parent not in snapshots for parent in parents):
            return
        parent_ids = set()
        for parent in parents:
            parent_ids.update(snapshots[parent])
        invented = set(snapshots[child]).difference(parent_ids)
        if invented:
            violations.append(
                {
                    "stage": child,
                    "code": "invented_id",
                    "count": len(invented),
                    "parent_stages": list(parents),
                }
            )

    require_subset("union", ("dense", "sparse"))
    require_subset("fusion", ("union",))
    require_subset("structural", ("fusion",))
    require_subset("reranker", ("structural",))
    require_subset("final", ("reranker",))
    if "final" in snapshots and len(snapshots["final"]) > final_limit:
        violations.append(
            {
                "stage": "final",
                "code": "limit_exceeded",
                "candidate_count": len(snapshots["final"]),
                "limit": final_limit,
            }
        )
    hashes = {
        stage: canonical_sha256(list(ids))
        for stage, ids in snapshots.items()
        if all(isinstance(document_id, str) for document_id in ids)
    }
    return {
        "status": "failed" if violations else "passed",
        "observed_stages": [stage for stage in RETRIEVAL_STAGES if stage in snapshots],
        "stage_ids_sha256": hashes,
        "violations": violations,
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def latency_percentiles(samples_ms: Iterable[float]) -> Dict[str, Optional[float]]:
    """Calculate p50/p95/p99 using deterministic linear interpolation."""

    samples = [
        _validate_finite_number("latency sample", value, non_negative=True)
        for value in samples_ms
    ]
    if not samples:
        return {"p50": None, "p95": None, "p99": None}
    return {
        "p50": _percentile(samples, 0.50),
        "p95": _percentile(samples, 0.95),
        "p99": _percentile(samples, 0.99),
    }


def _stable_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    # Scaling before fsum avoids overflowing on repeated values near max-float.
    count = len(values)
    result = math.fsum(value / count for value in values)
    if not math.isfinite(result):
        raise ValueError("mean is not finite")
    return result


def _bounded_mapping_keys(mapping: Mapping[Any, Any], label: str) -> List[Any]:
    """Materialize mapping keys with honest and adversarial cardinality gates."""

    try:
        if len(mapping) > MAX_BOOTSTRAP_QUERIES:
            raise ValueError(
                f"{label} cannot exceed {MAX_BOOTSTRAP_QUERIES} queries"
            )
    except ValueError:
        raise
    except Exception:
        raise ValueError(f"{label} has invalid cardinality") from None
    try:
        keys = list(
            itertools.islice(iter(mapping), MAX_BOOTSTRAP_QUERIES + 1)
        )
    except Exception:
        raise ValueError(f"{label} has invalid query IDs") from None
    if len(keys) > MAX_BOOTSTRAP_QUERIES:
        raise ValueError(
            f"{label} cannot exceed {MAX_BOOTSTRAP_QUERIES} queries"
        )
    return keys


def aggregate_query_metrics(
    query_metrics: Iterable[Mapping[str, Optional[float]]],
) -> Dict[str, Any]:
    """Macro-average available query labels; undefined metrics remain null."""

    rows = list(query_metrics)
    keys = sorted({key for row in rows for key in row})
    aggregate: Dict[str, Any] = {}
    support: Dict[str, int] = {}
    for key in keys:
        values = []
        for row in rows:
            raw = row.get(key)
            if raw is None:
                continue
            values.append(_validate_finite_number(key, raw))
        aggregate[key] = _stable_mean(values) if values else None
        support[key] = len(values)
    aggregate["support"] = support
    return aggregate


def paired_bootstrap(
    baseline: Mapping[str, Optional[float]],
    candidate: Mapping[str, Optional[float]],
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
    higher_is_better: bool = True,
) -> Dict[str, Any]:
    """Paired percentile bootstrap over unique query IDs.

    Absolute deltas are always ``candidate - baseline``.  Relative gain uses
    the requested direction, and is null when the baseline mean is zero.
    """

    if isinstance(samples, bool) or not isinstance(samples, int):
        raise TypeError("samples must be an integer, not bool")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if samples > MAX_BOOTSTRAP_SAMPLES:
        raise ValueError(
            f"samples cannot exceed {MAX_BOOTSTRAP_SAMPLES}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer, not bool")
    confidence_value = _validate_finite_number("confidence", confidence)
    if not 0.0 < confidence_value < 1.0:
        raise ValueError("confidence must be between zero and one")
    if not isinstance(higher_is_better, bool):
        raise TypeError("higher_is_better must be bool")
    if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
        raise TypeError("baseline and candidate must be mappings")
    baseline_keys = _bounded_mapping_keys(baseline, "baseline")
    candidate_keys = _bounded_mapping_keys(candidate, "candidate")
    if any(not isinstance(query_id, str) for query_id in baseline_keys) or any(
        not isinstance(query_id, str) for query_id in candidate_keys
    ):
        raise TypeError("baseline and candidate query IDs must be strings")
    if set(baseline_keys) != set(candidate_keys):
        raise ValueError("baseline and candidate must contain the same query IDs")
    if samples * len(baseline_keys) > MAX_BOOTSTRAP_WORK:
        raise ValueError(
            f"bootstrap work cannot exceed {MAX_BOOTSTRAP_WORK} draws"
        )

    pairs: List[Tuple[float, float]] = []
    for query_id in sorted(baseline_keys):
        base_raw = baseline[query_id]
        candidate_raw = candidate[query_id]
        if base_raw is None or candidate_raw is None:
            continue
        pairs.append(
            (
                _validate_finite_number("baseline metric", base_raw),
                _validate_finite_number("candidate metric", candidate_raw),
            )
        )
    common_query_count = len(baseline_keys)
    discarded_query_count = common_query_count - len(pairs)
    if not pairs:
        return {
            "n_queries": 0,
            "input_queries": common_query_count,
            "discarded_queries": discarded_query_count,
            "samples": samples,
            "seed": seed,
            "confidence": confidence_value,
            "direction": "candidate_minus_baseline",
            "improvement_direction": (
                "higher_is_better" if higher_is_better else "lower_is_better"
            ),
            "absolute_delta": None,
            "absolute_ci": None,
            "relative_gain": None,
            "relative_ci": None,
            "relative_defined_samples": 0,
            "relative_undefined_samples": samples,
        }

    baseline_mean = _stable_mean([pair[0] for pair in pairs])
    deltas = []
    for baseline_value, candidate_value in pairs:
        delta = candidate_value - baseline_value
        if not math.isfinite(delta):
            raise ValueError("paired metric delta must be finite")
        deltas.append(delta)
    absolute_delta = _stable_mean(deltas)
    if baseline_mean == 0.0:
        relative_gain = None
    elif higher_is_better:
        relative_gain = absolute_delta / baseline_mean
    else:
        relative_gain = -absolute_delta / baseline_mean
    if relative_gain is not None and not math.isfinite(relative_gain):
        relative_gain = None

    rng = random.Random(seed)
    absolute_distribution: List[float] = []
    relative_distribution: List[float] = []
    relative_undefined_samples = 0
    pair_count = len(pairs)
    for _ in range(samples):
        indices = [rng.randrange(pair_count) for _index in range(pair_count)]
        sampled_baseline = _stable_mean([pairs[index][0] for index in indices])
        sampled_delta = _stable_mean([deltas[index] for index in indices])
        absolute_distribution.append(sampled_delta)
        if sampled_baseline != 0.0:
            relative = sampled_delta / sampled_baseline
            if not higher_is_better:
                relative = -relative
            if math.isfinite(relative):
                relative_distribution.append(relative)
            else:
                relative_undefined_samples += 1
        else:
            relative_undefined_samples += 1

    alpha = (1.0 - confidence_value) / 2.0
    absolute_ci = [
        _percentile(absolute_distribution, alpha),
        _percentile(absolute_distribution, 1.0 - alpha),
    ]
    relative_ci = (
        [
            _percentile(relative_distribution, alpha),
            _percentile(relative_distribution, 1.0 - alpha),
        ]
        if relative_gain is not None
        and relative_distribution
        and relative_undefined_samples == 0
        else None
    )
    return {
        "n_queries": pair_count,
        "input_queries": common_query_count,
        "discarded_queries": discarded_query_count,
        "samples": samples,
        "seed": seed,
        "confidence": confidence_value,
        "direction": "candidate_minus_baseline",
        "improvement_direction": (
            "higher_is_better" if higher_is_better else "lower_is_better"
        ),
        "absolute_delta": absolute_delta,
        "absolute_ci": absolute_ci,
        "relative_gain": relative_gain,
        "relative_ci": relative_ci,
        "relative_defined_samples": len(relative_distribution),
        "relative_undefined_samples": relative_undefined_samples,
    }


def clustered_percentile_bootstrap(
    latency_by_query: Mapping[str, Sequence[float]],
    *,
    percentile: float = 0.95,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Bootstrap a latency percentile with query as the resampling cluster."""

    probability = _validate_finite_number("percentile", percentile)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentile must be between zero and one")
    confidence_value = _validate_finite_number("confidence", confidence)
    if not 0.0 < confidence_value < 1.0:
        raise ValueError("confidence must be between zero and one")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("samples must be a positive integer")
    if samples > MAX_BOOTSTRAP_SAMPLES:
        raise ValueError(
            f"samples cannot exceed {MAX_BOOTSTRAP_SAMPLES}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer, not bool")
    if not isinstance(latency_by_query, Mapping):
        raise TypeError("latency_by_query must be a mapping")

    query_ids = _bounded_mapping_keys(latency_by_query, "latency_by_query")
    clusters: List[Tuple[float, ...]] = []
    observation_count = 0
    for query_id in sorted(query_ids):
        if not isinstance(query_id, str):
            raise TypeError("query IDs must be strings")
        raw_cluster = latency_by_query[query_id]
        remaining = MAX_BOOTSTRAP_OBSERVATIONS - observation_count
        if isinstance(raw_cluster, Sequence) and len(raw_cluster) > remaining:
            raise ValueError(
                "latency observations exceed the bootstrap safety bound"
            )
        try:
            bounded_cluster = list(
                itertools.islice(iter(raw_cluster), remaining + 1)
            )
        except Exception:
            raise ValueError("latency samples must be iterable") from None
        if len(bounded_cluster) > remaining:
            raise ValueError(
                "latency observations exceed the bootstrap safety bound"
            )
        cluster = tuple(
            _validate_finite_number("latency sample", value, non_negative=True)
            for value in bounded_cluster
        )
        if not cluster:
            raise ValueError("each query cluster must contain a latency sample")
        clusters.append(cluster)
        observation_count += len(cluster)
    if not clusters:
        return {
            "estimate": None,
            "ci": None,
            "confidence": confidence_value,
            "percentile": probability,
            "samples": samples,
            "seed": seed,
            "queries": 0,
            "observations": 0,
            "unit": "query_cluster",
        }

    maximum_cluster_size = max(len(cluster) for cluster in clusters)
    if samples * len(clusters) * maximum_cluster_size > MAX_BOOTSTRAP_WORK:
        raise ValueError(
            f"bootstrap work cannot exceed {MAX_BOOTSTRAP_WORK} draws"
        )
    flattened = [value for cluster in clusters for value in cluster]
    estimate = _percentile(flattened, probability)
    rng = random.Random(seed)
    distribution = []
    cluster_count = len(clusters)
    for _ in range(samples):
        resampled: List[float] = []
        for _cluster in range(cluster_count):
            resampled.extend(clusters[rng.randrange(cluster_count)])
        distribution.append(_percentile(resampled, probability))
    alpha = (1.0 - confidence_value) / 2.0
    return {
        "estimate": estimate,
        "ci": [
            _percentile(distribution, alpha),
            _percentile(distribution, 1.0 - alpha),
        ],
        "confidence": confidence_value,
        "percentile": probability,
        "samples": samples,
        "seed": seed,
        "queries": cluster_count,
        "observations": len(flattened),
        "unit": "query_cluster",
    }


def canonical_sha256(value: Any) -> str:
    """Hash canonical UTF-8 JSON without accepting non-finite numbers."""

    _assert_json_value(value)
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_corpus_by_hash(
    corpus: Mapping[str, Any], limit: int, *, seed: int = BOOTSTRAP_SEED
) -> Dict[str, Any]:
    """Select a stable smoke corpus using IDs only, never relevance labels."""

    cutoff = _validate_k(limit)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer, not bool")
    if not isinstance(corpus, Mapping):
        raise TypeError("corpus must be a mapping keyed by document ID")
    if any(not isinstance(document_id, str) for document_id in corpus):
        raise TypeError("corpus document IDs must be strings")
    ranked = sorted(
        corpus.items(),
        key=lambda item: (
            hashlib.sha256(f"{seed}\0{item[0]}".encode("utf-8")).digest(),
            str(item[0]),
        ),
    )
    return dict(ranked[: min(cutoff, len(ranked))])


def validate_split_protocol(
    protocol: str,
    *,
    tuning_allowed: bool,
    config_sha256: str,
    dataset_sha256: str,
    validation_lock: Optional[Mapping[str, Any]] = None,
    expected_lock: Optional[Mapping[str, Any]] = None,
) -> None:
    """Enforce the frozen validation-to-test boundary."""

    if protocol not in {"calibration", "validation", "test"}:
        raise ValueError("protocol must be calibration, validation, or test")
    if not isinstance(tuning_allowed, bool):
        raise TypeError("tuning_allowed must be bool")
    if protocol != "test":
        return
    if tuning_allowed:
        raise ValueError("tuning is forbidden in the test split")
    if validation_lock is None:
        raise ValueError("test split requires a validation lock")
    _validate_validation_lock_shape(validation_lock)
    hashes = validation_lock["hashes"]
    if hashes.get("config_sha256") != config_sha256:
        raise ValueError("test config does not match the validation lock")
    if hashes.get("dataset_sha256") != dataset_sha256:
        raise ValueError("test dataset does not match the validation lock")
    if expected_lock is not None:
        _validate_validation_lock_shape(expected_lock)
        if canonical_sha256(validation_lock) != canonical_sha256(expected_lock):
            raise ValueError("test protocol state does not match the validation lock")


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_validation_lock_shape(lock: Mapping[str, Any]) -> None:
    if not isinstance(lock, Mapping):
        raise TypeError("validation lock must be a mapping")
    if lock.get("schema_version") != VALIDATION_LOCK_SCHEMA:
        raise ValueError("validation lock schema is invalid")
    if lock.get("origin_protocol") != "validation":
        raise ValueError("validation lock origin must be validation")
    seed = lock.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("validation lock seed is invalid")
    for field in ("metrics", "config", "sources", "protocol"):
        if not isinstance(lock.get(field), Mapping) or not lock[field]:
            raise ValueError(f"validation lock {field} must be a non-empty mapping")
    backends = lock.get("backends")
    if not isinstance(backends, list) or not backends:
        raise ValueError("validation lock backends must be a non-empty list")
    hashes = lock.get("hashes")
    if not isinstance(hashes, Mapping):
        raise ValueError("validation lock hashes must be a mapping")
    for name in REQUIRED_VALIDATION_HASHES:
        if not _is_sha256(hashes.get(name)):
            raise ValueError(f"validation lock {name} must be a SHA256 digest")
    _assert_json_value(lock)


def _assert_json_value(value: Any, path: str = "$") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"JSON number at {path} must be finite")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object key at {path} must be a string")
            _assert_json_value(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_json_value(item, f"{path}[{index}]")
        return
    raise TypeError(f"value at {path} is not JSON serializable")


def write_json_report(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write strict JSON only after the complete run is available."""

    destination = Path(path)
    _assert_json_value(payload)
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=False,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(destination.parent),
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
