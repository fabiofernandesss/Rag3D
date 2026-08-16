"""Standalone Retrieval Engine V2 pipeline.

The V2 path intentionally leaves :class:`rag3d.retrieve.TriRetriever` intact.
Dense and sparse retrieval are the only global candidate generators.  The
multi-vector structural signal is a bounded late-interaction reranker over the
already fused pool, with a rank-only contribution whose baseline weight is
explicitly ``1.0``.  That value is a neutral, untuned baseline rather than a
quality optimum.
"""
from __future__ import annotations

import math
import time
from collections.abc import Sized
from dataclasses import asdict
from itertools import islice
from numbers import Integral, Real
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .backend import (
    DEFAULT_RETRIEVAL_LIMITS,
    SearchDiagnostics,
    SearchFilters,
    normalize_sparse_weights,
    validate_query_text,
)
from .config import MAX_STITCH_RADIUS, TriRagConfig
from .diversity import diversify
from .encoders import BaseEncoder, TriVec
from .fusion import FusedHit, fuse
from .rerank import NoOpReranker
from .retrieve import CHANNELS, TriResult, _SYS_EXPAND
from .textproc import normalize

StageObserver = Callable[[str, Sequence[int]], None]

_DIAGNOSTIC_IDENTIFIER_ALLOWLISTS = {
    "backend": frozenset({"sqlite", "postgres-holo", "pgvector"}),
    "pipeline": frozenset({"v2"}),
    "encoder": frozenset({"base", "bge-m3", "fallback", "hash"}),
    "fusion": frozenset({"rrf", "quantum"}),
    "reranker": frozenset({"none", "llm", "cross-encoder"}),
    "diversity": frozenset({"none", "mmr", "dpp"}),
}


def _elapsed_ms(start_ns: int) -> float:
    return max(0.0, (time.perf_counter_ns() - start_ns) / 1_000_000.0)


def _diagnostic_identifier(kind: str, value: Any, fallback: str = "unknown") -> str:
    """Return only a known public identifier, never a normalized input value."""
    if not isinstance(value, str):
        return fallback
    candidate = value.strip().lower()
    return (
        candidate
        if candidate in _DIAGNOSTIC_IDENTIFIER_ALLOWLISTS[kind]
        else fallback
    )


def _bounded_backend_rows(rows: Any, maximum: int, label: str) -> List[Any]:
    """Consume an adapter result through a strict cardinality boundary.

    Backends are extension points, so a buggy adapter may return a generator,
    an infinite iterable, or a ``Sized`` object that lies about its length.
    Honest oversized results fail before iteration; every other result is cut
    off at the first row beyond the request rather than being materialized.
    """

    if isinstance(rows, Sized):
        try:
            if len(rows) > maximum:
                raise ValueError(f"{label} exceeds requested cardinality")
        except ValueError:
            raise
        except Exception:
            raise ValueError(f"{label} returned invalid rows") from None
    try:
        materialized = list(islice(iter(rows), maximum + 1))
    except Exception:
        raise ValueError(f"{label} returned invalid rows") from None
    if len(materialized) > maximum:
        raise ValueError(f"{label} exceeds requested cardinality")
    return materialized


def _object_diagnostic_identifier(
    kind: str, obj: Any, attribute: str, fallback: str = "unknown"
) -> str:
    try:
        value = getattr(obj, attribute)
    except Exception:
        return fallback
    return _diagnostic_identifier(kind, value, fallback)


def _validate_public_limit(name: str, value: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not bool")
    if value < 0 or value > maximum:
        raise ValueError(f"{name} must be between 0 and {maximum}")
    return int(value)


class RetrievalV2:
    """Sequential, bounded V2 retrieval pipeline returning legacy ``TriResult``."""

    pipeline_name = "v2"

    def __init__(
        self,
        backend: Any,
        encoder: BaseEncoder,
        cfg: TriRagConfig,
        reranker: Optional[Any] = None,
        llm: Optional[Any] = None,
        *,
        structural_rank_weight: float = 1.0,
        stage_observer: Optional[StageObserver] = None,
    ):
        if isinstance(structural_rank_weight, bool) or not isinstance(
            structural_rank_weight, Real
        ):
            raise TypeError("structural_rank_weight must be a real number, not bool")
        structural_rank_weight = float(structural_rank_weight)
        if not math.isfinite(structural_rank_weight) or structural_rank_weight < 0.0:
            raise ValueError("structural_rank_weight must be finite and non-negative")
        if stage_observer is not None and not callable(stage_observer):
            raise TypeError("stage_observer must be callable")
        self.backend = backend
        self.encoder = encoder
        self.cfg = cfg
        self.reranker = reranker if reranker is not None else NoOpReranker()
        self.llm = llm
        self.structural_rank_weight = structural_rank_weight
        self.stage_observer = stage_observer
        self._last_stitch_status = "skipped"
        self._last_stitch_reason = "not-run"

    def _observe(self, stage: str, ids: Sequence[int]) -> None:
        if self.stage_observer is None:
            return
        snapshot = tuple(int(cid) for cid in ids)
        try:
            self.stage_observer(stage, snapshot)
        except Exception:
            # Benchmark/diagnostic callbacks must never change retrieval.
            pass

    def _reranker_enabled(self) -> bool:
        if isinstance(self.reranker, NoOpReranker):
            return False
        try:
            return bool(self.reranker.available())
        except Exception:
            return False

    @staticmethod
    def _candidate_ranking(
        ranking: Sequence[Tuple[int, float]], limit: int
    ) -> List[Tuple[int, float]]:
        """Validate, deduplicate and deterministically order retriever output."""
        raw = _bounded_backend_rows(ranking, limit, "candidate ranking")
        clean: Dict[int, float] = {}
        for item in raw:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("candidate ranking contains an invalid row")
            cid, score = item
            if isinstance(cid, bool) or not isinstance(cid, Integral):
                raise ValueError("candidate ranking contains an invalid ID")
            if isinstance(score, bool) or not isinstance(score, Real):
                raise ValueError("candidate ranking contains an invalid score")
            value = float(score)
            if not math.isfinite(value):
                raise ValueError("candidate ranking contains a non-finite score")
            clean.setdefault(int(cid), value)
        return sorted(clean.items(), key=lambda item: (-item[1], item[0]))

    @staticmethod
    def _structural_ranking(
        ranking: Sequence[Tuple[int, float]], allowed_ids: Sequence[int], limit: int
    ) -> Optional[List[Tuple[int, float]]]:
        allowed = set(allowed_ids)
        seen = set()
        clean: List[Tuple[int, float]] = []
        try:
            raw = _bounded_backend_rows(
                ranking, limit, "structural ranking"
            )
        except Exception:
            return None
        for item in raw:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                return None
            cid, score = item
            if isinstance(cid, bool) or not isinstance(cid, Integral):
                return None
            normalized_id = int(cid)
            if normalized_id not in allowed or normalized_id in seen:
                return None
            if isinstance(score, bool) or not isinstance(score, Real):
                return None
            value = float(score)
            if not math.isfinite(value):
                return None
            seen.add(normalized_id)
            clean.append((normalized_id, value))
        return clean

    def _expand(self, normalized_query: str) -> List[str]:
        if not self.cfg.expand_query or self.llm is None:
            return []
        try:
            if not self.llm.available():
                return []
            raw = self.llm.complete(
                _SYS_EXPAND.format(N=self.cfg.expand_query_max),
                [{"role": "user", "content": normalized_query}],
                max_tokens=200,
            )
        except Exception:
            return []
        try:
            raw = validate_query_text(raw)
        except (TypeError, ValueError):
            return []
        variants: List[str] = []
        seen = {normalized_query}
        for line in raw.splitlines():
            candidate = normalize(line)
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            variants.append(candidate)
            if len(variants) >= self.cfg.expand_query_max:
                break
        return variants

    def _validated_query_vector(self, vector: Any) -> TriVec:
        if not isinstance(vector, TriVec):
            raise TypeError("encoder outputs must be TriVec instances")
        dense_raw = vector.dense
        tokens_raw = vector.tokens
        if not isinstance(dense_raw, np.ndarray):
            raise TypeError("dense query vectors must be NumPy arrays")
        if not isinstance(tokens_raw, np.ndarray):
            raise TypeError("structural query vectors must be NumPy arrays")
        if dense_raw.ndim != 1 or dense_raw.shape[0] != int(self.cfg.dense_dim):
            raise ValueError("dense query vector dimension mismatch")
        if dense_raw.shape[0] > DEFAULT_RETRIEVAL_LIMITS.max_dense_dim:
            raise ValueError("dense query vector exceeds the public dimension bound")
        if tokens_raw.ndim != 2 or tokens_raw.shape[1] != int(self.cfg.colbert_dim):
            raise ValueError("structural query vector dimension mismatch")
        if tokens_raw.shape[0] > int(self.cfg.max_colbert_tokens):
            raise ValueError("structural query vector exceeds the configured token bound")
        if tokens_raw.shape[1] > DEFAULT_RETRIEVAL_LIMITS.max_structural_dim:
            raise ValueError("structural query vector exceeds the public dimension bound")
        try:
            dense = np.asarray(dense_raw, dtype=np.float32).copy()
            tokens = np.asarray(tokens_raw, dtype=np.float32).copy()
        except (TypeError, ValueError, OverflowError):
            raise TypeError("query vectors must contain numeric values") from None
        if not np.isfinite(dense).all():
            raise ValueError("dense query vector values must be finite")
        if not np.isfinite(tokens).all():
            raise ValueError("structural query vector values must be finite")
        sparse = normalize_sparse_weights(vector.sparse)
        return TriVec(dense=dense, sparse=sparse, tokens=tokens)

    def _encode_query(self, texts: Sequence[str]) -> TriVec:
        expected = len(texts)
        raw_vectors = self.encoder.encode(list(texts), is_query=True)
        if isinstance(raw_vectors, Sized) and len(raw_vectors) != expected:
            raise ValueError("encoder must return exactly one vector per query text")
        vectors = list(islice(iter(raw_vectors), expected + 1))
        if len(vectors) != expected:
            raise ValueError("encoder must return exactly one vector per query text")
        validated = [self._validated_query_vector(vector) for vector in vectors]
        if len(validated) == 1:
            return validated[0]

        dense = np.zeros(int(self.cfg.dense_dim), dtype=np.float64)
        for vector in validated:
            dense += vector.dense
        norm = float(np.linalg.norm(dense))
        if math.isfinite(norm) and norm > 0.0:
            dense = dense / norm
        sparse: Dict[int, float] = {}
        sparse_limit = DEFAULT_RETRIEVAL_LIMITS.max_sparse_terms
        for vector in validated:
            for term, weight in vector.sparse.items():
                if term not in sparse and len(sparse) >= sparse_limit:
                    raise ValueError(
                        f"expanded sparse query exceeds maximum of {sparse_limit} terms"
                    )
                if term not in sparse or sparse[term] < weight:
                    sparse[term] = weight
        token_limit = int(self.cfg.max_colbert_tokens)
        token_buffer = np.empty(
            (token_limit, int(self.cfg.colbert_dim)), dtype=np.float32
        )
        token_count = 0
        for vector in validated:
            remaining = token_limit - token_count
            if remaining <= 0:
                break
            take = min(remaining, vector.tokens.shape[0])
            if take:
                token_buffer[token_count : token_count + take] = vector.tokens[:take]
                token_count += take
        tokens = token_buffer[:token_count].copy()
        dense32 = np.asarray(dense, dtype=np.float32)
        if not np.isfinite(dense32).all():
            raise ValueError("expanded dense query vector must remain finite")
        return TriVec(
            dense=dense32,
            sparse=sparse,
            tokens=tokens,
        )

    @staticmethod
    def _fusion_metadata(hits: Sequence[FusedHit], rrf_k: int) -> List[dict]:
        out: List[dict] = []
        for index, hit in enumerate(hits):
            rank = index + 1
            score = float(hit.score)
            out.append(
                {
                    "id": int(hit.chunk_id),
                    "score": score,
                    "fusion_score": score,
                    "fusion_rank": rank,
                    "classical": float(hit.classical),
                    "interference": float(hit.interference),
                    "channels": list(hit.channels),
                    "per_channel": dict(hit.per_channel),
                    "blend_score": 1.0 / (rrf_k + rank),
                }
            )
        return out

    def _blend_structural(
        self, candidates: Sequence[dict], ranking: Sequence[Tuple[int, float]]
    ) -> List[dict]:
        if self.structural_rank_weight == 0.0:
            return [dict(hit) for hit in candidates]
        by_id = {int(hit["id"]): dict(hit) for hit in candidates}
        returned: List[dict] = []
        seen = set()
        for structural_index, (cid, raw_score) in enumerate(ranking):
            if cid not in by_id:
                continue
            hit = dict(by_id[cid])
            structural_rank = structural_index + 1
            base_rank = int(hit["fusion_rank"])
            contribution = self.structural_rank_weight / (
                self.cfg.rrf_k + structural_rank
            )
            hit["structural_rank"] = structural_rank
            hit["structural_score"] = float(raw_score)
            hit["structural_rank_score"] = contribution
            hit["blend_score"] = 1.0 / (self.cfg.rrf_k + base_rank) + contribution
            returned.append(hit)
            seen.add(cid)
        returned.sort(
            key=lambda hit: (
                -float(hit["blend_score"]),
                int(hit["fusion_rank"]),
                int(hit["id"]),
            )
        )
        # Structural omissions stay behind the evaluated results and retain
        # their exact fusion-relative order.
        returned.extend(dict(hit) for hit in candidates if int(hit["id"]) not in seen)
        return returned

    @staticmethod
    def _reconcile_reranker(
        candidates: Sequence[dict], reranked: Any
    ) -> Optional[List[dict]]:
        base = {int(hit["id"]): dict(hit) for hit in candidates}
        expected = len(candidates)
        if isinstance(reranked, Sized) and len(reranked) > expected:
            return None
        try:
            rows = list(islice(iter(reranked), expected + 1))
        except Exception:
            return None
        if len(rows) > expected:
            return None
        out: List[dict] = []
        seen = set()
        for row in rows:
            if not isinstance(row, Mapping):
                return None
            cid = row.get("id")
            if isinstance(cid, bool) or not isinstance(cid, Integral):
                return None
            normalized_id = int(cid)
            if normalized_id not in base or normalized_id in seen:
                return None
            merged = dict(base[normalized_id])
            # A reranker controls order and may add annotations, but it cannot
            # rewrite fusion/structural metadata that belongs to prior stages.
            merged.update(
                {
                    key: value
                    for key, value in row.items()
                    if key not in base[normalized_id] and key != "id"
                }
            )
            merged["id"] = normalized_id
            out.append(merged)
            seen.add(normalized_id)
        out.extend(dict(hit) for hit in candidates if int(hit["id"]) not in seen)
        return out

    def _prefetch(
        self,
        ids: Sequence[int],
        cached_rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
    ) -> Tuple[Dict[int, Mapping[str, Any]], Dict[int, Mapping[str, Any]]]:
        unique_ids = list(dict.fromkeys(int(cid) for cid in ids))[
            : DEFAULT_RETRIEVAL_LIMITS.max_pool
        ]
        wanted = set(unique_ids)
        by_id: Dict[int, Mapping[str, Any]] = {}
        if cached_rows:
            # Probe only the bounded requested IDs instead of trusting a custom
            # Mapping's potentially unbounded ``items()`` iterator.
            for cid in unique_ids:
                row = cached_rows.get(cid)
                if isinstance(row, Mapping):
                    row_id = row.get("id")
                    if (
                        isinstance(row_id, bool)
                        or not isinstance(row_id, Integral)
                        or int(row_id) != cid
                    ):
                        raise ValueError("cached chunk lookup returned an invalid id")
                    by_id[cid] = row
        missing_ids = [cid for cid in unique_ids if cid not in by_id]
        rows = (
            _bounded_backend_rows(
                self.backend.get_chunks(missing_ids),
                len(missing_ids),
                "chunk lookup",
            )
            if missing_ids
            else []
        )
        missing_set = set(missing_ids)
        returned_ids = set()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("chunk lookup returned an invalid row")
            row_id = row.get("id")
            if isinstance(row_id, bool) or not isinstance(row_id, Integral):
                raise ValueError("chunk lookup returned an invalid id")
            normalized_id = int(row_id)
            if normalized_id not in missing_set or normalized_id in returned_ids:
                raise ValueError("chunk lookup returned an unexpected or duplicate id")
            returned_ids.add(normalized_id)
            by_id[normalized_id] = row
        parent_ids = list(
            dict.fromkeys(
                int(row["parent_id"])
                for row in by_id.values()
                if row.get("parent_id") is not None
            )
        )[: DEFAULT_RETRIEVAL_LIMITS.max_pool]
        parent_rows = (
            _bounded_backend_rows(
                self.backend.get_chunks(parent_ids),
                len(parent_ids),
                "parent lookup",
            )
            if parent_ids
            else []
        )
        expected_parent_documents: Dict[int, set] = {}
        for child in by_id.values():
            parent_id = child.get("parent_id")
            document_id = child.get("doc_id")
            if parent_id is None or document_id is None:
                continue
            expected_parent_documents.setdefault(int(parent_id), set()).add(
                int(document_id)
            )
        parents: Dict[int, Mapping[str, Any]] = {}
        expected_parent_ids = set(parent_ids)
        for row in parent_rows:
            if not isinstance(row, Mapping):
                raise ValueError("parent lookup returned an invalid row")
            raw_parent_id = row.get("id")
            if isinstance(raw_parent_id, bool) or not isinstance(
                raw_parent_id, Integral
            ):
                raise ValueError("parent lookup returned an invalid id")
            parent_id = int(raw_parent_id)
            if parent_id not in expected_parent_ids or parent_id in parents:
                raise ValueError(
                    "parent lookup returned an unexpected or duplicate id"
                )
            parent_document_id = row.get("doc_id")
            if parent_document_id is None:
                continue
            if int(parent_document_id) not in expected_parent_documents.get(
                parent_id, set()
            ):
                continue
            parents[parent_id] = row
        return by_id, parents

    @staticmethod
    def _assemble(
        cid: int,
        by_id: Mapping[int, Mapping[str, Any]],
        parents: Mapping[int, Mapping[str, Any]],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Optional[dict]:
        row = by_id.get(cid)
        if row is None:
            return None
        parent_id = row.get("parent_id")
        parent = parents.get(int(parent_id)) if parent_id is not None else None
        if parent is not None and parent.get("doc_id") != row.get("doc_id"):
            parent = None
        text = str(row.get("text", ""))
        hit = {
            "id": int(row["id"]),
            "kind": row.get("kind", "chunk"),
            "doc_id": row.get("doc_id"),
            "pos": row.get("pos"),
            "text": text,
            "wide": str(parent.get("text", text)) if parent is not None else text,
            "n_tokens": row.get("n_tokens", 0),
            "turn_no": row.get("turn_no"),
            "accessed_turn": row.get("accessed_turn"),
            "created": row.get("created"),
            "importance": row.get("importance", 0.5),
            "score": None,
        }
        if metadata:
            hit.update(dict(metadata))
            hit["id"] = int(row["id"])
        return hit

    def _stitch(
        self, hits: List[dict], filters: Optional[SearchFilters] = None
    ) -> List[dict]:
        self._last_stitch_status = "skipped"
        self._last_stitch_reason = "not-run"
        configured_radius = self.cfg.stitch_radius
        if isinstance(configured_radius, bool) or not isinstance(
            configured_radius, Integral
        ):
            raise TypeError("stitch_radius must be an integer, not bool")
        radius = int(configured_radius)
        if radius < 0 or radius > MAX_STITCH_RADIUS:
            raise ValueError(
                f"stitch_radius must be between 0 and {MAX_STITCH_RADIUS}"
            )
        # ``neighbors`` has no filter parameter in the shared backend contract.
        # Fail closed instead of appending text that may sit outside an active
        # tenant/parent/metadata scope.
        if radius == 0:
            self._last_stitch_reason = "not-configured"
            return hits
        if filters is not None and not filters.is_empty:
            self._last_stitch_reason = "filters-applied"
            return hits
        positions_by_doc: Dict[int, set] = {}
        remaining_positions = DEFAULT_RETRIEVAL_LIMITS.max_pool
        for hit in hits:
            if remaining_positions <= 0:
                break
            doc_id = hit.get("doc_id")
            position = hit.get("pos")
            if doc_id is None or position is None or hit.get("kind") != "chunk":
                continue
            positions = positions_by_doc.setdefault(int(doc_id), set())
            for offset in range(-radius, radius + 1):
                candidate_position = int(position) + offset
                if candidate_position in positions:
                    continue
                if remaining_positions <= 0:
                    break
                positions.add(candidate_position)
                remaining_positions -= 1
        text_by_position: Dict[Tuple[int, int], str] = {}
        if not positions_by_doc:
            self._last_stitch_reason = "empty-pool"
            return hits
        remaining_neighbor_rows = DEFAULT_RETRIEVAL_LIMITS.max_pool
        for doc_id, positions in positions_by_doc.items():
            if remaining_neighbor_rows <= 0:
                break
            try:
                neighbors = _bounded_backend_rows(
                    self.backend.neighbors(doc_id, sorted(positions)),
                    remaining_neighbor_rows,
                    "neighbor lookup",
                )
            except ValueError:
                self._last_stitch_status = "fallback"
                self._last_stitch_reason = "invalid-output"
                return hits
            except Exception:
                self._last_stitch_status = "fallback"
                self._last_stitch_reason = "stage-error"
                return hits
            remaining_neighbor_rows -= len(neighbors)
            for row in neighbors:
                if not isinstance(row, Mapping):
                    self._last_stitch_status = "fallback"
                    self._last_stitch_reason = "invalid-output"
                    return hits
                row_id = row.get("id")
                row_doc_id = row.get("doc_id")
                row_position = row.get("pos")
                row_kind = row.get("kind")
                row_text = row.get("text")
                if (
                    isinstance(row_id, bool)
                    or not isinstance(row_id, Integral)
                    or isinstance(row_doc_id, bool)
                    or not isinstance(row_doc_id, Integral)
                    or int(row_doc_id) != doc_id
                    or isinstance(row_position, bool)
                    or not isinstance(row_position, Integral)
                    or int(row_position) not in positions
                    or row_kind != "chunk"
                    or not isinstance(row_text, str)
                ):
                    self._last_stitch_status = "fallback"
                    self._last_stitch_reason = "invalid-output"
                    return hits
                key = (doc_id, int(row_position))
                if key in text_by_position:
                    self._last_stitch_status = "fallback"
                    self._last_stitch_reason = "invalid-output"
                    return hits
                text_by_position[key] = row_text
        for hit in hits:
            doc_id = hit.get("doc_id")
            position = hit.get("pos")
            if doc_id is None or position is None or hit.get("kind") != "chunk":
                continue
            parts = [
                text_by_position[(int(doc_id), int(position) + offset)]
                for offset in range(-radius, radius + 1)
                if (int(doc_id), int(position) + offset) in text_by_position
            ]
            if parts:
                hit["wide"] = " ".join(parts)
        self._last_stitch_status = "applied"
        self._last_stitch_reason = "none"
        return hits

    def _stats(
        self,
        timings: Mapping[str, float],
        counts: Mapping[str, int],
        filters: Optional[SearchFilters],
        *,
        structural_depth_requested: int = 0,
        structural_candidates_attempted: int = 0,
        structural_candidates_evaluated: int = 0,
        structural_status: str = "skipped",
        structural_reason: str = "not-run",
        reranker_status: str = "skipped",
        reranker_reason: str = "not-run",
        diversity_status: str = "skipped",
        diversity_reason: str = "not-run",
        stitch_status: str = "skipped",
        stitch_reason: str = "not-run",
    ) -> dict:
        backend_name = _object_diagnostic_identifier(
            "backend", self.backend, "backend_name"
        )
        encoder_name = _object_diagnostic_identifier(
            "encoder", self.encoder, "name"
        )
        reranker_name = (
            _object_diagnostic_identifier("reranker", self.reranker, "name")
            if self._reranker_enabled()
            else "none"
        )
        filter_count = filters.predicate_count if filters is not None else 0
        diagnostics = SearchDiagnostics(
            **{name: float(timings.get(name, 0.0)) for name in SearchDiagnostics._TIMINGS},
            **{
                name: int(counts.get(name, 0))
                for name in SearchDiagnostics._COUNTS
                if name != "filter_count"
            },
            backend=backend_name,
            pipeline=_diagnostic_identifier("pipeline", self.pipeline_name),
            encoder=encoder_name,
            fusion=_diagnostic_identifier("fusion", self.cfg.fusion),
            reranker=reranker_name,
            diversity=_diagnostic_identifier(
                "diversity", self.cfg.diversity_method
            ),
            filters_applied=filter_count > 0,
            filter_count=filter_count,
            structural_status=structural_status,
            structural_reason=structural_reason,
            reranker_status=reranker_status,
            reranker_reason=reranker_reason,
            diversity_status=diversity_status,
            diversity_reason=diversity_reason,
            stitch_status=stitch_status,
            stitch_reason=stitch_reason,
        )
        stats = asdict(diagnostics)
        stats.update(
            {
                "candidatos": {
                    "semantico": diagnostics.dense_candidates,
                    "lexico": diagnostics.sparse_candidates,
                    "estrutural": diagnostics.structural_candidates,
                },
                "pool": diagnostics.union_candidates,
                "fusao": diagnostics.fusion,
                "structural_candidate_depth_requested": max(
                    0, int(structural_depth_requested)
                ),
                "structural_candidates_evaluated": max(
                    0, int(structural_candidates_evaluated)
                ),
                "structural_candidates_attempted": max(
                    0, int(structural_candidates_attempted)
                ),
            }
        )
        return stats

    def _empty_result(
        self,
        query: str,
        start_ns: int,
        timings: Optional[Dict[str, float]] = None,
        filters: Optional[SearchFilters] = None,
    ) -> TriResult:
        timings = dict(timings or {})
        elapsed = _elapsed_ms(start_ns)
        timings.setdefault("total_retrieval_ms", elapsed)
        timings["total_ms"] = elapsed
        for stage in ("dense", "sparse", "union", "fusion", "structural", "reranker", "final"):
            self._observe(stage, ())
        return TriResult(
            query=query,
            fused=[],
            views={name: [] for name in CHANNELS},
            stats=self._stats(
                timings,
                {},
                filters,
                structural_depth_requested=(
                    int(self.cfg.structural_candidate_depth)
                    if self.cfg.structural_rerank
                    else 0
                ),
            ),
        )

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        channel_k: Optional[int] = None,
        filters: Optional[SearchFilters] = None,
    ) -> TriResult:
        """Run V2 stages sequentially; no backend connection is shared by threads."""
        query = validate_query_text(query)
        if filters is not None and not isinstance(filters, SearchFilters):
            raise TypeError("filters must be SearchFilters")
        final_k = _validate_public_limit(
            "top_k",
            self.cfg.top_k if top_k is None else top_k,
            DEFAULT_RETRIEVAL_LIMITS.max_top_k,
        )
        per_channel_k = _validate_public_limit(
            "channel_k",
            self.cfg.channel_k if channel_k is None else channel_k,
            DEFAULT_RETRIEVAL_LIMITS.max_channel_k,
        )
        total_start = time.perf_counter_ns()
        timings: Dict[str, float] = {}
        if final_k == 0 or per_channel_k == 0:
            return self._empty_result(query, total_start, timings, filters)

        stage_start = time.perf_counter_ns()
        normalized_query = normalize(query)
        timings["normalize_ms"] = _elapsed_ms(stage_start)
        if not normalized_query:
            return self._empty_result(query, total_start, timings, filters)

        stage_start = time.perf_counter_ns()
        variants = self._expand(normalized_query)
        timings["expand_ms"] = _elapsed_ms(stage_start)

        stage_start = time.perf_counter_ns()
        query_vector = self._encode_query([normalized_query] + variants)
        timings["encode_ms"] = _elapsed_ms(stage_start)

        stage_start = time.perf_counter_ns()
        dense_raw = self.backend.dense_search(
            query_vector.dense, per_channel_k, filters=filters, exact=None
        )
        dense = self._candidate_ranking(dense_raw, per_channel_k)
        timings["dense_ms"] = _elapsed_ms(stage_start)
        self._observe("dense", [cid for cid, _ in dense])

        stage_start = time.perf_counter_ns()
        sparse_raw = self.backend.sparse_search(
            query_vector.sparse, per_channel_k, filters=filters
        )
        sparse = self._candidate_ranking(sparse_raw, per_channel_k)
        timings["sparse_ms"] = _elapsed_ms(stage_start)
        self._observe("sparse", [cid for cid, _ in sparse])

        stage_start = time.perf_counter_ns()
        union_ids = sorted({cid for cid, _ in dense} | {cid for cid, _ in sparse})
        timings["union_ms"] = _elapsed_ms(stage_start)
        self._observe("union", union_ids)

        reranker_enabled = self._reranker_enabled()
        diversity_method = self.cfg.diversity_method
        structural_depth_requested = (
            int(self.cfg.structural_candidate_depth)
            if self.cfg.structural_rerank
            else 0
        )
        structural_capability = bool(
            getattr(
                getattr(self.backend, "capabilities", None),
                "structural_rerank",
                False,
            )
        )
        structural_configured = bool(
            self.cfg.structural_rerank and structural_capability
        )
        # Keep one final-sized reserve for hydration backfill. The fused pool
        # remains absolutely bounded even when a stale index omits rows.
        required_pool = min(
            DEFAULT_RETRIEVAL_LIMITS.max_pool,
            max(final_k, final_k * 2),
        )
        if structural_configured:
            required_pool = max(required_pool, structural_depth_requested)
        if reranker_enabled:
            required_pool = max(required_pool, int(self.cfg.rerank_pool))
        if diversity_method != "none":
            required_pool = max(required_pool, int(self.cfg.diversity_pool))
        pool_size = min(
            required_pool,
            DEFAULT_RETRIEVAL_LIMITS.max_pool,
            len(union_ids),
        )

        stage_start = time.perf_counter_ns()
        channels = {"semantico": dense, "lexico": sparse}
        if self.cfg.fusion == "rrf":
            # Untuned V2 baseline: independent retrievers, exactly equal weight.
            fusion_weights = (1.0, 1.0)
        else:
            configured = tuple(self.cfg.channel_weights)
            if len(configured) < 2:
                raise ValueError("channel_weights must provide dense and sparse weights")
            fusion_weights = configured[:2]
        fused_hits = fuse(
            channels,
            fusion_weights,
            pool_size,
            method=self.cfg.fusion,
            interference_strength=self.cfg.interference_strength,
            rrf_k=self.cfg.rrf_k,
            # Coherence-weighted RRF remains available only through the
            # legacy fusion facade; the V2 baseline is exactly equal-weight.
            coherence_strength=(
                0.0 if self.cfg.fusion == "rrf" else self.cfg.coherence_strength
            ),
        )
        candidates = self._fusion_metadata(fused_hits, self.cfg.rrf_k)
        timings["fusion_ms"] = _elapsed_ms(stage_start)
        self._observe("fusion", [hit["id"] for hit in candidates])

        structural: List[Tuple[int, float]] = []
        structural_candidates_attempted = 0
        structural_candidates_evaluated = 0
        if not self.cfg.structural_rerank:
            structural_status, structural_reason = "skipped", "not-configured"
        elif not structural_capability:
            structural_status, structural_reason = "skipped", "unsupported"
        elif not candidates:
            structural_status, structural_reason = "skipped", "empty-pool"
        else:
            structural_status, structural_reason = "skipped", "not-run"
        structural_enabled = bool(
            structural_configured and candidates
        )
        if structural_enabled:
            stage_start = time.perf_counter_ns()
            structural_limit = min(
                len(candidates),
                structural_depth_requested,
                DEFAULT_RETRIEVAL_LIMITS.max_pool,
            )
            structural_ids = [
                int(hit["id"]) for hit in candidates[:structural_limit]
            ]
            structural_candidates_attempted = len(structural_ids)
            try:
                raw_structural = self.backend.structural_rerank(
                    query_vector.tokens,
                    structural_ids,
                    structural_limit,
                    filters=filters,
                )
                validated = self._structural_ranking(
                    raw_structural,
                    structural_ids,
                    structural_limit,
                )
                if validated is not None:
                    structural = validated
                    candidates = self._blend_structural(candidates, structural)
                    structural_candidates_evaluated = len(structural_ids)
                    structural_status, structural_reason = "applied", "none"
                else:
                    structural_status, structural_reason = (
                        "fallback",
                        "invalid-output",
                    )
            except Exception:
                structural = []
                structural_status, structural_reason = "fallback", "stage-error"
            timings["structural_ms"] = _elapsed_ms(stage_start)
        else:
            timings["structural_ms"] = 0.0
        self._observe("structural", [hit["id"] for hit in candidates])

        snippet_rows: Dict[int, Mapping[str, Any]] = {}
        reranker_candidates_evaluated = 0
        if self.cfg.reranker == "none":
            reranker_status, reranker_reason = "skipped", "not-configured"
        elif not reranker_enabled:
            reranker_status, reranker_reason = "skipped", "unavailable"
        elif not candidates:
            reranker_status, reranker_reason = "skipped", "empty-pool"
        else:
            reranker_status, reranker_reason = "skipped", "not-run"
        if reranker_enabled and candidates:
            stage_start = time.perf_counter_ns()
            previous = [dict(hit) for hit in candidates]
            rerank_depth = min(
                len(previous),
                int(self.cfg.rerank_pool),
                DEFAULT_RETRIEVAL_LIMITS.max_pool,
            )
            prefix = previous[:rerank_depth]
            tail = previous[rerank_depth:]
            try:
                snippet_ids = [int(hit["id"]) for hit in prefix]
                rows = _bounded_backend_rows(
                    self.backend.get_chunks(snippet_ids),
                    len(snippet_ids),
                    "reranker snippet lookup",
                )
                snippet_rows = {
                    int(row["id"]): row
                    for row in rows
                    if isinstance(row, Mapping) and "id" in row
                }
                enriched: List[dict] = []
                for hit in prefix:
                    cid = int(hit["id"])
                    row = snippet_rows.get(cid)
                    text = str(row.get("text", "")) if row is not None else ""
                    if not text.strip():
                        raise ValueError("reranker snippet is missing")
                    candidate = dict(hit)
                    candidate["text"] = text
                    enriched.append(candidate)
                reranked = self.reranker.rerank(
                    normalized_query, enriched, top_k=len(enriched)
                )
                reported_status = getattr(self.reranker, "last_status", None)
                reported_reason = getattr(self.reranker, "last_reason", None)
                if reported_status in {"fallback", "skipped"}:
                    candidates = previous
                    reranker_status = reported_status
                    reranker_reason = (
                        reported_reason
                        if reported_reason in SearchDiagnostics._STAGE_REASONS
                        else "invalid-output"
                    )
                else:
                    reconciled = self._reconcile_reranker(enriched, reranked)
                    if reconciled is None:
                        candidates = previous
                        reranker_status, reranker_reason = (
                            "fallback",
                            "invalid-output",
                        )
                    else:
                        candidates = reconciled + tail
                        reranker_candidates_evaluated = len(enriched)
                        reranker_status, reranker_reason = "applied", "none"
            except Exception:
                candidates = previous
                reranker_status, reranker_reason = "fallback", "stage-error"
            timings["rerank_ms"] = _elapsed_ms(stage_start)
        else:
            timings["rerank_ms"] = 0.0
        self._observe("reranker", [hit["id"] for hit in candidates])

        stage_start = time.perf_counter_ns()
        diversity_depth = min(
            len(candidates),
            DEFAULT_RETRIEVAL_LIMITS.max_pool,
            max(final_k, int(self.cfg.diversity_pool))
            if diversity_method != "none"
            else final_k,
        )
        diversity_candidates = candidates[:diversity_depth]
        rank_items = [
            (int(hit["id"]), (len(diversity_candidates) - index) / len(diversity_candidates))
            for index, hit in enumerate(diversity_candidates)
        ] if diversity_candidates else []
        fallback_ids = [int(hit["id"]) for hit in candidates[:final_k]]
        vectors: Mapping[int, np.ndarray] = {}
        if diversity_method == "none":
            selected_ids = fallback_ids
            diversity_status, diversity_reason = "skipped", "not-configured"
        elif not diversity_candidates:
            selected_ids = []
            diversity_status, diversity_reason = "skipped", "empty-pool"
        else:
            try:
                vector_ids = [int(hit["id"]) for hit in diversity_candidates]
                raw_vectors = self.backend.dense_vectors(vector_ids)
                if not isinstance(raw_vectors, Mapping):
                    raise ValueError("dense vector lookup returned an invalid mapping")
                vectors = {
                    cid: raw_vectors[cid]
                    for cid in vector_ids
                    if cid in raw_vectors
                }
                if len(vectors) != len(vector_ids):
                    selected_ids = fallback_ids
                    diversity_status, diversity_reason = (
                        "fallback",
                        "missing-vectors",
                    )
                else:
                    for cid in vector_ids:
                        vector = vectors[cid]
                        norm = math.inf
                        if isinstance(vector, np.ndarray) and vector.ndim == 1:
                            with np.errstate(over="ignore", invalid="ignore"):
                                norm = float(np.linalg.norm(vector))
                        if (
                            not isinstance(vector, np.ndarray)
                            or vector.ndim != 1
                            or vector.shape[0] != int(self.cfg.dense_dim)
                            or vector.shape[0]
                            > DEFAULT_RETRIEVAL_LIMITS.max_dense_dim
                            or not np.isfinite(vector).all()
                            or not math.isfinite(norm)
                            or norm <= 0.0
                        ):
                            raise ValueError(
                                "diversity vectors contain invalid output"
                            )
                    proposed_ids = diversify(
                        rank_items,
                        vectors,
                        final_k,
                        method=diversity_method,
                    )
                    allowed_ids = {cid for cid, _score in rank_items}
                    if isinstance(proposed_ids, Sized) and len(proposed_ids) > final_k:
                        raise ValueError("diversity output exceeds top_k")
                    bounded_ids = list(islice(iter(proposed_ids), final_k + 1))
                    if len(bounded_ids) > final_k:
                        raise ValueError("diversity output exceeds top_k")
                    if any(
                        isinstance(cid, bool)
                        or not isinstance(cid, Integral)
                        or int(cid) not in allowed_ids
                        for cid in bounded_ids
                    ):
                        raise ValueError("diversity output contains an invalid ID")
                    selected_ids = list(
                        dict.fromkeys(int(cid) for cid in bounded_ids)
                    )
                    diversity_status, diversity_reason = "applied", "none"
            except (TypeError, ValueError, OverflowError):
                selected_ids = fallback_ids
                diversity_status, diversity_reason = (
                    "fallback",
                    "invalid-output",
                )
            except Exception:
                selected_ids = fallback_ids
                diversity_status, diversity_reason = "fallback", "stage-error"
        candidates_by_id = {int(hit["id"]): hit for hit in candidates}
        selected_set = set(selected_ids)
        ordered_candidate_ids = [
            cid for cid in selected_ids if cid in candidates_by_id
        ]
        ordered_candidate_ids.extend(
            int(hit["id"])
            for hit in candidates
            if int(hit["id"]) not in selected_set
        )
        ordered_metadata = [
            dict(candidates_by_id[cid])
            for cid in ordered_candidate_ids[: DEFAULT_RETRIEVAL_LIMITS.max_pool]
        ]
        timings["diversity_ms"] = _elapsed_ms(stage_start)

        view_depth = min(
            DEFAULT_RETRIEVAL_LIMITS.max_pool,
            max(final_k, final_k * 2),
        )
        view_rankings = {
            "semantico": dense[:view_depth],
            "lexico": sparse[:view_depth],
            "estrutural": structural[:view_depth],
        }
        # Priority is selected output, then view backfill, then fused reserve.
        # `_prefetch` applies the final aggregate max_pool clamp.
        hydration_ids = [cid for cid in selected_ids if cid in candidates_by_id]
        hydration_ids.extend(
            cid for ranking in view_rankings.values() for cid, _ in ranking
        )
        hydration_ids.extend(ordered_candidate_ids)
        stage_start = time.perf_counter_ns()
        by_id, parents = self._prefetch(hydration_ids, cached_rows=snippet_rows)
        fused: List[dict] = []
        for metadata in ordered_metadata:
            hit = self._assemble(int(metadata["id"]), by_id, parents, metadata)
            if hit is not None:
                fused.append(hit)
            if len(fused) >= final_k:
                break
        views: Dict[str, List[dict]] = {}
        for name, ranking in view_rankings.items():
            hydrated_view: List[dict] = []
            for cid, score in ranking:
                hit = self._assemble(cid, by_id, parents, {"score": score})
                if hit is not None:
                    hydrated_view.append(hit)
                if len(hydrated_view) >= final_k:
                    break
            views[name] = hydrated_view

        # ``score`` remains the compatibility field consumed by ChatMemory.
        # Publish a finite ordinal score matching the final V2 order and keep
        # the raw fusion value separately in ``fusion_score``.
        for index, hit in enumerate(fused):
            hit["final_rank"] = index + 1
            hit["score"] = 1.0 / (index + 1)
        timings["hydrate_ms"] = _elapsed_ms(stage_start)

        stage_start = time.perf_counter_ns()
        self._stitch(fused, filters=filters)
        stitch_status = self._last_stitch_status
        stitch_reason = self._last_stitch_reason
        timings["stitch_ms"] = _elapsed_ms(stage_start)
        # ``search`` ends here (there is no reader stage in this method), so
        # both totals enclose the complete retrieval, hydration and stitch
        # path.  A higher-level engine may publish a separate reader/E2E total.
        timings["total_retrieval_ms"] = _elapsed_ms(total_start)
        timings["total_ms"] = _elapsed_ms(total_start)
        self._observe("final", [hit["id"] for hit in fused])

        counts = {
            "dense_candidates": len(dense),
            "sparse_candidates": len(sparse),
            "union_candidates": len(union_ids),
            "fused_candidates": len(fused_hits),
            "structural_candidates": len(structural),
            "reranked_candidates": reranker_candidates_evaluated,
            "final_candidates": len(fused),
        }
        return TriResult(
            query=query,
            fused=fused,
            views=views,
            stats=self._stats(
                timings,
                counts,
                filters,
                structural_depth_requested=structural_depth_requested,
                structural_candidates_attempted=structural_candidates_attempted,
                structural_candidates_evaluated=structural_candidates_evaluated,
                structural_status=structural_status,
                structural_reason=structural_reason,
                reranker_status=reranker_status,
                reranker_reason=reranker_reason,
                diversity_status=diversity_status,
                diversity_reason=diversity_reason,
                stitch_status=stitch_status,
                stitch_reason=stitch_reason,
            ),
        )


# Descriptive alias for callers that prefer an engine-style name.
RetrievalEngineV2 = RetrievalV2
