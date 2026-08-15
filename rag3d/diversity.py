"""Bounded diversity selection for Retrieval Engine V2.

``none`` is an exact ranking identity, ``mmr`` is the standard greedy
relevance/novelty baseline, and ``dpp`` is greedy k-DPP selection.  The latter
is an approximation to global MAP inference; no global-optimum claim is made.
"""
from __future__ import annotations

import math
from collections.abc import Sized
from numbers import Integral, Real
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .backend import DEFAULT_RETRIEVAL_LIMITS

RankedItems = Sequence[Tuple[int, float]]


class _DiversityBoundError(ValueError):
    """Resource boundary violation that must not degrade into ranking fallback."""


def _validate_configuration(
    top_k: int,
    method: str,
    mmr_lambda: float,
    dpp_alpha: float,
    jitter: float,
) -> Tuple[int, str, float, float, float]:
    if isinstance(top_k, bool) or not isinstance(top_k, Integral):
        raise TypeError("top_k must be an integer, not bool")
    if top_k < 0 or top_k > DEFAULT_RETRIEVAL_LIMITS.max_pool:
        raise ValueError(
            f"top_k must be between 0 and {DEFAULT_RETRIEVAL_LIMITS.max_pool}"
        )
    if not isinstance(method, str) or method not in {"none", "mmr", "dpp"}:
        raise ValueError("method must be 'none', 'mmr', or 'dpp'")
    values = {
        "mmr_lambda": mmr_lambda,
        "dpp_alpha": dpp_alpha,
        "jitter": jitter,
    }
    normalized = {}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number, not bool")
        normalized[name] = float(value)
        if not math.isfinite(normalized[name]):
            raise ValueError(f"{name} must be finite")
    if not 0.0 <= normalized["mmr_lambda"] <= 1.0:
        raise ValueError("mmr_lambda must be between 0 and 1")
    if normalized["dpp_alpha"] < 0.0:
        raise ValueError("dpp_alpha must be non-negative")
    if normalized["jitter"] < 0.0:
        raise ValueError("jitter must be non-negative")
    return (
        int(top_k),
        method,
        normalized["mmr_lambda"],
        normalized["dpp_alpha"],
        normalized["jitter"],
    )


def _first_occurrences(items: RankedItems) -> List[Tuple[int, float]]:
    if isinstance(items, (str, bytes)):
        raise TypeError("items must be a sequence of (id, relevance) pairs")
    limit = DEFAULT_RETRIEVAL_LIMITS.max_pool
    if isinstance(items, Sized) and len(items) > limit:
        raise ValueError(f"items exceed maximum of {limit}")
    out: List[Tuple[int, float]] = []
    seen = set()
    for item_count, item in enumerate(items, start=1):
        if item_count > limit:
            raise ValueError(f"items exceed maximum of {limit}")
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError("each item must be an (id, relevance) pair")
        cid, relevance = item
        if isinstance(cid, bool) or not isinstance(cid, Integral):
            raise TypeError("item id must be an integer, not bool")
        normalized_id = int(cid)
        if normalized_id in seen:
            continue
        seen.add(normalized_id)
        try:
            normalized_relevance = float(relevance)
        except (TypeError, ValueError, OverflowError):
            normalized_relevance = math.nan
        out.append((normalized_id, normalized_relevance))
    return out


def _normalized_vectors(
    ids: Sequence[int], vectors: Mapping[int, np.ndarray]
) -> Optional[np.ndarray]:
    if not isinstance(vectors, Mapping):
        return None
    rows: List[np.ndarray] = []
    dimension: Optional[int] = None
    try:
        for cid in ids:
            raw = vectors.get(cid)
            if raw is None:
                return None
            if isinstance(raw, np.ndarray):
                if raw.ndim != 1:
                    return None
                raw_size = int(raw.size)
            elif isinstance(raw, Sized):
                raw_size = len(raw)
            else:
                return None
            if raw_size > DEFAULT_RETRIEVAL_LIMITS.max_dense_dim:
                raise _DiversityBoundError(
                    "diversity vector dimension exceeds maximum of "
                    f"{DEFAULT_RETRIEVAL_LIMITS.max_dense_dim}"
                )
            vector = np.asarray(raw, dtype=np.float64)
            if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
                return None
            if dimension is None:
                dimension = int(vector.size)
                total_values = len(ids) * dimension
                if total_values > DEFAULT_RETRIEVAL_LIMITS.max_diversity_values:
                    raise _DiversityBoundError(
                        "diversity vector values exceed maximum of "
                        f"{DEFAULT_RETRIEVAL_LIMITS.max_diversity_values}"
                    )
            elif vector.size != dimension:
                return None
            norm = float(np.linalg.norm(vector))
            if not math.isfinite(norm) or norm <= 1e-15:
                return None
            rows.append(vector / norm)
    except _DiversityBoundError:
        raise
    except Exception:
        return None
    if not rows:
        return np.empty((0, 0), dtype=np.float64)
    matrix = np.vstack(rows).astype(np.float64, copy=False)
    return matrix if np.isfinite(matrix).all() else None


def shifted_cosine_kernel(matrix: np.ndarray, jitter: float = 1e-9) -> np.ndarray:
    """Build the PSD shifted-cosine kernel ``0.5 * (Gram + 1)`` in float64."""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a NumPy array")
    if matrix.ndim != 2:
        raise ValueError("matrix must be a finite two-dimensional array")
    rows, dimension = (int(value) for value in matrix.shape)
    if rows > DEFAULT_RETRIEVAL_LIMITS.max_pool:
        raise ValueError(
            f"matrix rows exceed maximum of {DEFAULT_RETRIEVAL_LIMITS.max_pool}"
        )
    if dimension > DEFAULT_RETRIEVAL_LIMITS.max_dense_dim:
        raise ValueError(
            "matrix dimension exceeds maximum of "
            f"{DEFAULT_RETRIEVAL_LIMITS.max_dense_dim}"
        )
    if rows * dimension > DEFAULT_RETRIEVAL_LIMITS.max_diversity_values:
        raise ValueError(
            "matrix values exceed maximum of "
            f"{DEFAULT_RETRIEVAL_LIMITS.max_diversity_values}"
        )
    vectors = np.asarray(matrix, dtype=np.float64)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("matrix must be a finite two-dimensional array")
    if not math.isfinite(float(jitter)) or jitter < 0.0:
        raise ValueError("jitter must be finite and non-negative")
    # Scale before taking row norms so a finite vector near float64 limits
    # cannot overflow the norm/Gram and violate the finite PSD contract.
    scales = np.max(np.abs(vectors), axis=1)
    if not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise ValueError("matrix rows must have finite positive norm")
    scaled = vectors / scales[:, None]
    norms = np.linalg.norm(scaled, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise ValueError("matrix rows must have finite positive norm")
    normalized = scaled / norms[:, None]
    gram = normalized @ normalized.T
    similarity = 0.5 * (gram + 1.0)
    similarity = 0.5 * (similarity + similarity.T)
    if jitter:
        similarity = similarity + float(jitter) * np.eye(len(vectors), dtype=np.float64)
    if not np.isfinite(similarity).all():
        raise ValueError("shifted cosine kernel must be finite")
    return similarity.astype(np.float64, copy=False)


def _mmr_select(
    ids: Sequence[int], relevance: np.ndarray, vectors: np.ndarray, top_k: int, weight: float
) -> List[int]:
    n = len(ids)
    selected: List[int] = []
    available = np.ones(n, dtype=bool)
    max_similarity = np.zeros(n, dtype=np.float64)
    has_selected = False
    while bool(np.any(available)) and len(selected) < top_k:
        candidates = [index for index in range(n) if available[index]]
        scored = weight * relevance
        if has_selected:
            scored = scored - (1.0 - weight) * max_similarity
        chosen = min(
            candidates,
            key=lambda index: (-float(scored[index]), index, ids[index]),
        )
        selected.append(chosen)
        available[chosen] = False

        # One matrix-vector product per selected pivot updates the maximum
        # similarity for every remaining candidate. This is O(N*k*D), rather
        # than rescanning all prior pivots for every candidate and step.
        similarities = np.clip(vectors @ vectors[chosen], -1.0, 1.0)
        if has_selected:
            max_similarity = np.maximum(max_similarity, similarities)
        else:
            max_similarity = similarities.astype(np.float64, copy=True)
            has_selected = True
    return [ids[index] for index in selected]


def _greedy_dpp_select(
    ids: Sequence[int], relevance: np.ndarray, vectors: np.ndarray, top_k: int,
    alpha: float, jitter: float
) -> List[int]:
    # DPP quality must be non-negative. Finite negative ranking scores have no
    # probability interpretation and conservatively receive zero quality.
    with np.errstate(all="ignore"):
        quality = np.power(
            np.maximum(relevance, 0.0), alpha, dtype=np.float64
        )
    if not np.isfinite(quality).all():
        return list(ids[:top_k])

    n = len(ids)
    k = min(top_k, n)
    # S_ii = 1 + jitter for normalized valid vectors. Only the diagonal and
    # the current pivot column are needed by greedy Cholesky; no N x N Gram or
    # kernel is materialized.
    with np.errstate(all="ignore"):
        residual = quality * quality * (1.0 + jitter)
    if not np.isfinite(residual).all():
        return list(ids[:top_k])
    residual = np.maximum(residual.astype(np.float64, copy=False), 0.0)
    factors = np.zeros((n, k), dtype=np.float64)
    available = np.ones(n, dtype=bool)
    chosen: List[int] = []
    numerical_floor = max(np.finfo(np.float64).eps, jitter * 1e-6)

    for step in range(k):
        candidates = [index for index in range(n) if available[index]]
        if not candidates:
            break
        pivot = min(candidates, key=lambda index: (-residual[index], index, ids[index]))
        if not math.isfinite(float(residual[pivot])) or residual[pivot] <= numerical_floor:
            chosen.extend(candidates)
            break
        chosen.append(pivot)
        available[pivot] = False
        if len(chosen) == k:
            break

        denominator = math.sqrt(max(float(residual[pivot]), numerical_floor))
        remaining = np.flatnonzero(available)
        if not len(remaining):
            continue
        cosine = np.clip(vectors[remaining] @ vectors[pivot], -1.0, 1.0)
        similarity = 0.5 * (cosine + 1.0)
        kernel_column = quality[remaining] * quality[pivot] * similarity
        if step:
            projection = factors[remaining, :step] @ factors[pivot, :step]
        else:
            projection = np.zeros(len(remaining), dtype=np.float64)
        coefficients = (kernel_column - projection) / denominator
        if not np.isfinite(coefficients).all():
            return list(ids[:top_k])
        factors[remaining, step] = coefficients
        residual[remaining] = np.maximum(
            residual[remaining] - coefficients * coefficients,
            0.0,
        )
    return [ids[index] for index in chosen[:k]]


def diversify(
    items: RankedItems,
    vectors: Mapping[int, np.ndarray],
    top_k: int,
    *,
    method: str = "none",
    mmr_lambda: float = 0.5,
    dpp_alpha: float = 1.0,
    jitter: float = 1e-9,
) -> List[int]:
    """Select up to ``top_k`` unique IDs with deterministic fallback behavior."""
    top_k, method, mmr_lambda, dpp_alpha, jitter = _validate_configuration(
        top_k, method, mmr_lambda, dpp_alpha, jitter
    )
    ranked = _first_occurrences(items)
    ids = [cid for cid, _ in ranked]
    if top_k == 0 or not ids:
        return []
    if method == "none":
        return ids[:top_k]

    relevance = np.asarray([score for _, score in ranked], dtype=np.float64)
    if not np.isfinite(relevance).all():
        return ids[:top_k]
    normalized = _normalized_vectors(ids, vectors)
    if normalized is None:
        return ids[:top_k]
    if method == "mmr":
        return _mmr_select(ids, relevance, normalized, min(top_k, len(ids)), mmr_lambda)
    return _greedy_dpp_select(
        ids,
        relevance,
        normalized,
        min(top_k, len(ids)),
        dpp_alpha,
        jitter,
    )
