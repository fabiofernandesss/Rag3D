#!/usr/bin/env python3
"""Reproducible Retrieval Engine V2 calibration and benchmark runner.

The safe default builds a 1,000-document frozen synthetic corpus once and
compares the legacy and V2 pipelines on the same SQLite index.  It writes no
artifact until all requested systems have completed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import re
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag3d.config import TriRagConfig
from rag3d.backend import DEFAULT_RETRIEVAL_LIMITS
from rag3d.engine import TriRag
from rag3d.evaluation import (
    BOOTSTRAP_SEED,
    MAX_BOOTSTRAP_SAMPLES,
    RETRIEVAL_STAGES,
    StageRecorder,
    aggregate_query_metrics,
    canonical_sha256,
    clustered_percentile_bootstrap,
    evaluate_query,
    latency_percentiles,
    paired_bootstrap,
    select_corpus_by_hash,
    validate_stage_lineage,
    validate_split_protocol,
    write_json_report,
)
from rag3d.llm import NoLLM
from rag3d.retrieve import TriRetriever
from rag3d.retrieval_v2 import RetrievalV2


DATASET_MANIFEST = ROOT / "benchmarks" / "datasets" / "retrieval_v2_synthetic_v1.json"
DEFAULT_SCALE = 1_000
MAX_SCALE = 100_000
MAX_WARMUP = 10_000
MAX_REPETITIONS = 1_000
MAX_DENSE_DIM = DEFAULT_RETRIEVAL_LIMITS.max_dense_dim
MAX_STRUCTURAL_DIM = DEFAULT_RETRIEVAL_LIMITS.max_structural_dim
MAX_STRUCTURAL_TOKENS = DEFAULT_RETRIEVAL_LIMITS.max_structural_tokens
MAX_STRUCTURAL_VALUES_PER_CHUNK = DEFAULT_RETRIEVAL_LIMITS.max_structural_values
MAX_ESTIMATED_EMBEDDING_BYTES = 512 * 1024 * 1024
MIN_HNSW_RECALL_AT_K = 0.98
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
_SUPPORTED_BACKENDS = ("sqlite", "postgres-holo", "pgvector")
_SUPPORTED_ABLATIONS = ("dense", "sparse", "rrf", "quantum", "structural", "mmr", "dpp")
_REMOTE_WRITE_ENV = "RAG3D_BENCHMARK_ALLOW_WRITE"


def _normalized_text_fingerprint(text: str) -> str:
    normalized = " ".join(text.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _load_manifest() -> Dict[str, Any]:
    payload = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    if payload.get("id") != "retrieval-v2-synthetic-v1":
        raise ValueError("synthetic manifest ID is invalid")
    if not isinstance(payload.get("version"), str) or not payload["version"]:
        raise ValueError("synthetic manifest version is required")
    if payload.get("seed") != BOOTSTRAP_SEED:
        raise ValueError("synthetic manifest seed does not match the frozen protocol")
    splits = payload.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"calibration", "validation", "test"}:
        raise ValueError("synthetic manifest must define three fixed splits")
    identifiers = []
    query_sources = []
    allowed_kinds = {"single", "multi", "duplicate", "empty", "no_answer"}
    for split, rows in splits.items():
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"synthetic split {split} must be a non-empty list")
        for item in rows:
            if not isinstance(item, dict):
                raise ValueError("synthetic query specifications must be objects")
            identifier = item.get("id")
            kind = item.get("kind")
            if not isinstance(identifier, str) or not identifier:
                raise ValueError("synthetic query IDs must be non-empty strings")
            if kind not in allowed_kinds:
                raise ValueError("synthetic query kind is invalid")
            source_field = "query" if kind in {"empty", "no_answer"} else "query_template"
            source = item.get(source_field)
            if not isinstance(source, str):
                raise ValueError(f"synthetic {kind} query source is invalid")
            if kind == "single":
                slot = item.get("target_slot")
                if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
                    raise ValueError("single-query target_slot must be non-negative")
            if kind == "multi":
                slots = item.get("target_slots")
                if (
                    not isinstance(slots, list)
                    or len(slots) < 2
                    or any(isinstance(slot, bool) or not isinstance(slot, int) or slot < 0 for slot in slots)
                    or len(slots) != len(set(slots))
                ):
                    raise ValueError("multi-query target_slots must be distinct non-negative integers")
            identifiers.append(identifier)
            query_sources.append(source)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("synthetic query IDs must be disjoint across splits")
    if len(query_sources) != len(set(query_sources)):
        raise ValueError("synthetic query contents/templates must be disjoint across splits")
    return payload


def _generate_corpus(scale: int, seed: int) -> Dict[str, Dict[str, Any]]:
    topics = (
        "auditoria ambiental",
        "segurança de redes",
        "pesquisa clínica",
        "mobilidade urbana",
        "gestão de contratos",
        "energia renovável",
        "preservação histórica",
        "educação científica",
    )
    corpus: Dict[str, Dict[str, Any]] = {}
    for index in range(scale):
        # Every four-document block contains one exact-content duplicate. Both
        # IDs are generated before labels, so redundancy is observable without
        # relevance-aware corpus construction. Content-key modulo three gives
        # each fixed split an exclusive label namespace.
        content_index = index - 1 if index % 4 == 1 else index
        topic = topics[content_index % len(topics)]
        code = f"R3D-{content_index:06d}"
        deadline = f"{1 + content_index % 28:02d}/09/{2027 + content_index % 5}"
        limit = 10_000 + ((content_index * 7919 + seed) % 900_000)
        text = (
            f"Registro técnico {code}. O projeto de {topic} termina em {deadline}. "
            f"O limite operacional aprovado é {limit} unidades. "
            "A revisão foi registrada pelo comitê responsável."
        )
        document_id = f"doc-{index:06d}"
        corpus[document_id] = {
            "document_id": document_id,
            "text": text,
            "code": code,
            "topic": topic,
            "deadline": deadline,
            "limit": limit,
            "content_key": content_index,
            "facts": [f"deadline-{content_index}", f"limit-{content_index}"],
            "fingerprint": _normalized_text_fingerprint(text),
        }
    # Even a reduced synthetic run exercises the exact same ID-only sampler
    # used for public smoke subsets.  Qrels do not exist at this point.
    return select_corpus_by_hash(corpus, scale, seed=seed)


def _documents_with_content(
    documents: Sequence[Mapping[str, Any]], content_key: int
) -> List[Mapping[str, Any]]:
    return [document for document in documents if document["content_key"] == content_key]


def _materialize_queries(
    manifest: Mapping[str, Any],
    protocol: str,
    corpus: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    documents = list(corpus.values())
    if not documents:
        raise ValueError("synthetic corpus must not be empty")
    content_groups: Dict[int, List[Mapping[str, Any]]] = {}
    for document in documents:
        content_groups.setdefault(int(document["content_key"]), []).append(document)
    split_owner = {"calibration": 0, "validation": 1, "test": 2}[protocol]
    eligible_keys = sorted(
        content_key for content_key in content_groups if content_key % 3 == split_owner
    )
    duplicate_keys = [
        content_key for content_key in eligible_keys if len(content_groups[content_key]) > 1
    ]
    if not eligible_keys or not duplicate_keys:
        raise ValueError(
            "synthetic scale is too small for disjoint targets and duplicate labels"
        )

    def target_for_slot(slot: int) -> Mapping[str, Any]:
        key = eligible_keys[slot % len(eligible_keys)]
        return content_groups[key][0]

    def distinct_targets(slots: Sequence[int]) -> List[Mapping[str, Any]]:
        selected = []
        used = set()
        for slot in slots:
            for offset in range(len(eligible_keys)):
                key = eligible_keys[(slot + offset) % len(eligible_keys)]
                if key not in used:
                    used.add(key)
                    selected.append(content_groups[key][0])
                    break
            else:
                raise ValueError("synthetic scale cannot provide distinct multi-query targets")
        return selected

    queries: List[Dict[str, Any]] = []
    for specification in manifest["splits"][protocol]:
        kind = specification["kind"]
        query: Dict[str, Any] = {
            "query_id": specification["id"],
            "qrels": {},
            "required_facts": None,
            "unanswerable_gold": None,
        }
        if kind == "no_answer":
            query["text"] = specification["query"]
            query["unanswerable_gold"] = True
        elif kind == "empty":
            query["text"] = specification["query"]
        elif kind == "duplicate":
            duplicate_group = content_groups[duplicate_keys[0]]
            target = duplicate_group[0]
            query["text"] = specification["query_template"].format(**target)
            query["qrels"] = {
                str(document["document_id"]): 2 for document in duplicate_group
            }
            query["required_facts"] = list(target["facts"])
            query["unanswerable_gold"] = False
        elif kind == "single":
            target = target_for_slot(int(specification["target_slot"]))
            relevant = _documents_with_content(documents, int(target["content_key"]))
            query["text"] = specification["query_template"].format(**target)
            query["qrels"] = {
                str(document["document_id"]): 2 for document in relevant
            }
            query["required_facts"] = list(target["facts"])
            query["unanswerable_gold"] = False
        elif kind == "multi":
            targets = distinct_targets(specification["target_slots"])
            format_values = {
                f"code_{index}": target["code"] for index, target in enumerate(targets)
            }
            query["text"] = specification["query_template"].format(**format_values)
            qrels: Dict[str, int] = {}
            required_facts = []
            for target_index, target in enumerate(targets):
                grade = 2 if target_index == 0 else 1
                for document in _documents_with_content(documents, int(target["content_key"])):
                    qrels[str(document["document_id"])] = max(
                        grade, qrels.get(str(document["document_id"]), 0)
                    )
                required_facts.append(target["facts"][0])
            query["qrels"] = qrels
            query["required_facts"] = required_facts
            query["unanswerable_gold"] = False
        else:
            raise ValueError(f"unsupported synthetic query kind: {kind}")
        queries.append(query)
    _validate_materialized_queries(queries, corpus, protocol)
    return queries


def _validate_materialized_queries(
    queries: Sequence[Mapping[str, Any]],
    corpus: Mapping[str, Mapping[str, Any]],
    protocol: str,
) -> None:
    query_ids = [query.get("query_id") for query in queries]
    if any(not isinstance(query_id, str) or not query_id for query_id in query_ids):
        raise ValueError("materialized query IDs must be non-empty strings")
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("materialized query IDs must be unique")
    query_contents = [query.get("text") for query in queries]
    if any(not isinstance(text, str) for text in query_contents):
        raise ValueError("materialized query contents must be strings")
    if len(query_contents) != len(set(query_contents)):
        raise ValueError("materialized query contents must be unique within a split")
    owner = {"calibration": 0, "validation": 1, "test": 2}[protocol]
    for query in queries:
        qrels = query.get("qrels")
        if not isinstance(qrels, Mapping):
            raise ValueError("materialized qrels must be mappings")
        for document_id in qrels:
            if not isinstance(document_id, str) or document_id not in corpus:
                raise ValueError("materialized qrel references an unknown document")
            if int(corpus[document_id]["content_key"]) % 3 != owner:
                raise ValueError("materialized qrel crosses the fixed split boundary")


def _safe_git(*arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_diff_sha256(paths: Sequence[Path]) -> str:
    relative_paths = [str(path.relative_to(ROOT)) for path in paths]
    try:
        completed = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--", *relative_paths],
            cwd=ROOT,
            check=True,
            capture_output=True,
            timeout=10,
        )
        payload = completed.stdout
    except (OSError, subprocess.SubprocessError):
        payload = b"git-diff-unavailable"
    return hashlib.sha256(payload).hexdigest()


def _dataset_identity(
    manifest: Mapping[str, Any],
    corpus: Mapping[str, Mapping[str, Any]],
) -> Tuple[str, str, str]:
    manifest_sha256 = _sha256_file(DATASET_MANIFEST)
    corpus_sha256 = canonical_sha256(
        [
            {"document_id": key, "text": value["text"]}
            for key, value in corpus.items()
        ]
    )
    dataset_sha256 = canonical_sha256(
        {
            "id": manifest["id"],
            "version": manifest["version"],
            "seed": manifest["seed"],
            "corpus_sha256": corpus_sha256,
            "manifest_sha256": manifest_sha256,
        }
    )
    return corpus_sha256, manifest_sha256, dataset_sha256


def _validation_lock_for(
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    common_config: Mapping[str, Any],
    *,
    manifest_sha256: str,
    dataset_sha256: str,
) -> Dict[str, Any]:
    runner_path = Path(__file__).resolve()
    evaluator_path = ROOT / "rag3d" / "evaluation.py"
    runtime_paths = sorted((ROOT / "rag3d").rglob("*.py"))
    runtime_source_files = {
        str(path.relative_to(ROOT)): _sha256_file(path) for path in runtime_paths
    }
    runtime_source_closure_sha256 = canonical_sha256(runtime_source_files)
    # Retain the historical field name while making it a closed package hash.
    pipeline_files = dict(runtime_source_files)
    pipeline_sha256 = runtime_source_closure_sha256
    runner_sha256 = _sha256_file(runner_path)
    evaluator_sha256 = _sha256_file(evaluator_path)
    generator_sha256 = hashlib.sha256(
        (
            inspect.getsource(_generate_corpus)
            + inspect.getsource(_materialize_queries)
            + inspect.getsource(_validate_materialized_queries)
        ).encode("utf-8")
    ).hexdigest()
    source_files = {
        "runner": runner_sha256,
        "evaluator": evaluator_sha256,
        "runtime_source_closure": runtime_source_closure_sha256,
        "generator": generator_sha256,
        "manifest": manifest_sha256,
    }
    source_sha256 = canonical_sha256(source_files)
    source_diff_sha256 = _git_diff_sha256(
        [runner_path, *runtime_paths, DATASET_MANIFEST]
    )
    commit_state_sha256 = canonical_sha256(
        {
            "commit": _safe_git("rev-parse", "HEAD"),
            "source_diff_sha256": source_diff_sha256,
            "source_sha256": source_sha256,
        }
    )
    config_sha256 = canonical_sha256(common_config)
    return {
        "schema_version": "retrieval-v2-validation-lock/1",
        "origin_protocol": "validation",
        "seed": int(manifest["seed"]),
        "metrics": {
            "recall_cutoffs": [5, 10, 20],
            "mrr_cutoff": 20,
            "ndcg_cutoff": 10,
            "coverage_cutoff": 20,
            "duplicate_rate_cutoff": 20,
            "deduplication_policy": "raw_cutoff; duplicate occupies rank and has zero repeated gain",
            "bootstrap_unit": "paired query",
            "latency_bootstrap_unit": "query cluster with all repetitions",
        },
        "config": dict(common_config),
        "sources": {
            "dataset_id": manifest["id"],
            "dataset_version": manifest["version"],
            "manifest": str(DATASET_MANIFEST.relative_to(ROOT)),
            "generator": "_generate_corpus+_materialize_queries",
            "source_files_sha256": source_files,
            "pipeline_files_sha256": pipeline_files,
            "runtime_source_files_sha256": runtime_source_files,
        },
        "protocol": {
            "split_policy": "fixed disjoint content-key modulo three",
            "corpus_selection": "sha256(seed + NUL + document_id), before qrels",
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "confidence": 0.95,
        },
        "backends": list(common_config["backends"]),
        "hashes": {
            "config_sha256": config_sha256,
            "dataset_sha256": dataset_sha256,
            "manifest_sha256": manifest_sha256,
            "generator_sha256": generator_sha256,
            "runner_sha256": runner_sha256,
            "evaluator_sha256": evaluator_sha256,
            "pipeline_sha256": pipeline_sha256,
            "runtime_source_closure_sha256": runtime_source_closure_sha256,
            "source_diff_sha256": source_diff_sha256,
            "source_sha256": source_sha256,
            "commit_state_sha256": commit_state_sha256,
        },
    }


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _total_memory_bytes() -> Optional[int]:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _rss_peak_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def _environment(backends: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    cpu_count = os.cpu_count()
    total_memory = _total_memory_bytes()
    machine = platform.machine() or "unknown"
    processor = platform.processor() or "unknown"
    hardware = f"{machine}; cpu_count={cpu_count}; memory_bytes={total_memory}; processor={processor}"
    return {
        "hardware": hardware,
        "os": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            "rag3d": _package_version("rag3d") or "workspace",
            "numpy": _package_version("numpy"),
            "psycopg": _package_version("psycopg"),
            "pgvector_python": _package_version("pgvector"),
        },
        "backend_health": {
            name: dict(payload) for name, payload in sorted(backends.items())
        },
    }


class _BackendView:
    """Non-owning channel/mode view over one benchmark index."""

    def __init__(
        self,
        backend: Any,
        *,
        dense_enabled: bool = True,
        sparse_enabled: bool = True,
        dense_exact: Optional[bool] = None,
    ) -> None:
        self._backend = backend
        self._dense_enabled = dense_enabled
        self._sparse_enabled = sparse_enabled
        self._dense_exact = dense_exact
        self.last_dense: List[Tuple[int, float]] = []
        self.last_sparse: List[Tuple[int, float]] = []
        self.last_structural: List[Tuple[int, float]] = []

    @property
    def backend_name(self) -> str:
        return str(self._backend.backend_name)

    @property
    def capabilities(self) -> Any:
        capabilities = self._backend.capabilities
        if self._dense_enabled and self._sparse_enabled:
            return capabilities
        return replace(
            capabilities,
            exact_dense_search=(
                capabilities.exact_dense_search and self._dense_enabled
            ),
            ann_dense_search=(
                capabilities.ann_dense_search and self._dense_enabled
            ),
            sparse_search=(capabilities.sparse_search and self._sparse_enabled),
        )

    def reset_capture(self) -> None:
        self.last_dense = []
        self.last_sparse = []
        self.last_structural = []

    def dense_search(
        self,
        query_vector: Any,
        k: int,
        *,
        filters: Any = None,
        exact: Optional[bool] = None,
    ) -> List[Tuple[int, float]]:
        if not self._dense_enabled:
            self.last_dense = []
            return []
        requested_exact = self._dense_exact if self._dense_exact is not None else exact
        rows = list(
            self._backend.dense_search(
                query_vector, k, filters=filters, exact=requested_exact
            )
        )
        self.last_dense = rows
        return rows

    def sparse_search(
        self, query_weights: Any, k: int, *, filters: Any = None
    ) -> List[Tuple[int, float]]:
        if not self._sparse_enabled:
            self.last_sparse = []
            return []
        rows = list(self._backend.sparse_search(query_weights, k, filters=filters))
        self.last_sparse = rows
        return rows

    def structural_rerank(
        self,
        query_vectors: Any,
        candidate_ids: Sequence[int],
        k: int,
        *,
        filters: Any = None,
    ) -> List[Tuple[int, float]]:
        rows = list(
            self._backend.structural_rerank(
                query_vectors, candidate_ids, k, filters=filters
            )
        )
        self.last_structural = rows
        return rows

    def colbert_scores(
        self, query_vectors: Any, candidate_ids: Sequence[int]
    ) -> List[Tuple[int, float]]:
        rows = list(self._backend.colbert_scores(query_vectors, candidate_ids))
        self.last_structural = rows
        return rows

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)


def _parse_csv(raw: str, allowed: Sequence[str], name: str) -> List[str]:
    values = [value.strip().lower() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    invalid = sorted(set(values) - set(allowed))
    if invalid:
        raise ValueError(f"invalid {name}: {', '.join(invalid)}")
    return list(dict.fromkeys(values))


def _ablation_values(raw: str) -> List[str]:
    if not raw.strip():
        return []
    if raw.strip().lower() == "all":
        return list(_SUPPORTED_ABLATIONS)
    return _parse_csv(raw, _SUPPORTED_ABLATIONS, "ablations")


def _system_specifications(backend: str, ablations: Sequence[str]) -> List[Dict[str, Any]]:
    specifications = [
        {
            "id": f"legacy-{backend}",
            "pipeline": "legacy",
            "fusion": "quantum",
            "dense": True,
            "sparse": True,
            "structural": True,
            "diversity": "legacy-default",
            "ablation": "legacy",
        },
        {
            "id": f"v2-{backend}",
            "pipeline": "v2",
            "fusion": "rrf",
            "dense": True,
            "sparse": True,
            "structural": True,
            "diversity": "none",
            "ablation": "candidate_default",
        },
    ]
    variants = {
        "dense": ("rrf", True, False, False, "none"),
        "sparse": ("rrf", False, True, False, "none"),
        "rrf": ("rrf", True, True, False, "none"),
        "quantum": ("quantum", True, True, False, "none"),
        "structural": ("rrf", True, True, True, "none"),
        "mmr": ("rrf", True, True, True, "mmr"),
        "dpp": ("rrf", True, True, True, "dpp"),
    }
    for ablation in ablations:
        fusion, dense, sparse, structural, diversity = variants[ablation]
        specifications.append(
            {
                "id": f"v2-{backend}-ablation-{ablation}",
                "pipeline": "v2",
                "fusion": fusion,
                "dense": dense,
                "sparse": sparse,
                "structural": structural,
                "diversity": diversity,
                "ablation": ablation,
            }
        )
    return specifications


def _system_config(
    args: argparse.Namespace, backend: str, specification: Mapping[str, Any], data_dir: Path, dsn: str
) -> TriRagConfig:
    legacy = str(specification["pipeline"]) == "legacy"
    return TriRagConfig(
        data_dir=data_dir,
        pg_dsn=dsn,
        backend=backend,
        retrieval_pipeline=str(specification["pipeline"]),
        encoder="hash",
        dense_dim=args.dense_dim,
        colbert_dim=args.structural_dim,
        max_colbert_tokens=args.max_structural_tokens,
        contextual_enrich=False,
        tiny_doc_tokens=100_000,
        huge_doc_tokens=1_000_000,
        small_corpus_tokens=0,
        top_k=args.top_k,
        channel_k=args.channel_k,
        fusion=str(specification["fusion"]),
        rrf_k=args.rrf_k,
        structural_rerank=bool(specification["structural"]),
        structural_candidate_depth=args.structural_depth,
        rerank=False,
        reranker="none",
        diversity=(TriRagConfig().diversity if legacy else 0.0),
        diversity_method=(
            "none" if legacy else str(specification["diversity"])
        ),
        diversity_pool=min(1_000, max(args.top_k * 2, args.top_k)),
        stitch_radius=0,
        expand_query=False,
        pgvector_search_mode=(
            "ann" if backend == "pgvector" and args.pgvector_mode == "hnsw" else "exact"
        ),
    )


def _safe_backend_health(store: Any) -> Dict[str, Any]:
    try:
        payload = dict(store.health())
    except Exception as exc:
        return {"status": "error", "backend": store.backend_name, "error_type": type(exc).__name__}
    # Backend health contracts are already secret-safe; keep only JSON scalar,
    # mapping and sequence values and never attach exception messages.
    return payload


def _index_size(store: Any, data_dir: Path) -> Dict[str, Any]:
    backend = store.backend_name
    if backend == "sqlite":
        database = data_dir / "trirag.db"
        paths = [database, Path(str(database) + "-wal"), Path(str(database) + "-shm")]
        size = sum(path.stat().st_size for path in paths if path.exists())
        return {
            "status": "available",
            "index_bytes": int(size),
            "size_per_chunk_bytes": None,
            "definition": "SQLite database file plus present WAL and SHM sidecars",
        }
    if backend == "postgres-holo":
        sql = (
            "SELECT COALESCE(pg_total_relation_size(to_regclass('holo_docs')),0) + "
            "COALESCE(pg_total_relation_size(to_regclass('holo_grams')),0) + "
            "COALESCE(pg_total_relation_size(to_regclass('holo_spectrum')),0) + "
            "COALESCE(pg_total_relation_size(to_regclass('holo_meta')),0)"
        )
        definition = "sum pg_total_relation_size for fixed holo_* benchmark relations"
    else:
        sql = (
            "SELECT COALESCE(pg_total_relation_size(to_regclass('rag3d_v2_documents')),0) + "
            "COALESCE(pg_total_relation_size(to_regclass('rag3d_v2_chunks')),0) + "
            "COALESCE(pg_total_relation_size(to_regclass('rag3d_v2_sparse_postings')),0) + "
            "COALESCE(pg_total_relation_size(to_regclass('rag3d_v2_meta')),0)"
        )
        definition = "sum pg_total_relation_size for fixed rag3d_v2_* benchmark relations"
    try:
        size = int(store.db.execute(sql).fetchone()[0])
    except Exception as exc:
        return {
            "status": "unavailable",
            "index_bytes": None,
            "size_per_chunk_bytes": None,
            "definition": definition,
            "error_type": type(exc).__name__,
        }
    return {
        "status": "available",
        "index_bytes": max(0, size),
        "size_per_chunk_bytes": None,
        "definition": definition,
    }


def _storage_per_chunk(storage: Mapping[str, Any], *, chunks: int) -> Dict[str, Any]:
    if isinstance(chunks, bool) or not isinstance(chunks, int) or chunks < 0:
        raise ValueError("chunks must be a non-negative integer")
    result = dict(storage)
    index_bytes = result.get("index_bytes")
    if (
        result.get("status") == "available"
        and isinstance(index_bytes, int)
        and not isinstance(index_bytes, bool)
        and index_bytes >= 0
        and chunks > 0
    ):
        result["size_per_chunk_bytes"] = index_bytes / chunks
    else:
        result["size_per_chunk_bytes"] = None
    result["size_per_chunk_definition"] = "index_bytes / ingested_chunks"
    return result


def _timed_hnsw_build(
    store: Any,
    *,
    m: int,
    ef_construction: int,
    concurrently: bool,
    clock: Any = time.perf_counter,
) -> Dict[str, Any]:
    requested_options = {"m": int(m), "ef_construction": int(ef_construction)}
    refresh = getattr(store, "refresh_capabilities", None)
    if callable(refresh):
        before = dict(refresh())
        if before.get("exists"):
            compatible = bool(
                before.get("valid")
                and before.get("ready")
                and before.get("definition_valid")
                and before.get("options") == requested_options
            )
            if not compatible:
                raise RuntimeError(
                    "preexisting HNSW index is not compatible with benchmark options"
                )
            return {
                "status": "already_present",
                "hnsw_build_seconds": None,
                "index_status": before,
                "created_by_runner": False,
            }
    started = clock()
    index_status = dict(
        store.create_hnsw_index(
            m=m,
            ef_construction=ef_construction,
            concurrently=concurrently,
        )
    )
    elapsed = max(0.0, float(clock()) - float(started))
    created_by_runner = index_status.get("created_by_caller") is True
    return {
        "status": "measured" if created_by_runner else "race_recovered",
        "hnsw_build_seconds": elapsed if created_by_runner else None,
        "index_status": index_status,
        "created_by_runner": created_by_runner,
    }


def _assert_remote_store_empty(store: Any) -> None:
    backend = str(store.backend_name)
    if backend == "pgvector":
        payload = dict(store.metrics())
        counts = payload.get("counts")
        if payload.get("status") != "ok" or not isinstance(counts, Mapping):
            raise RuntimeError("could not verify empty pgvector benchmark relations")
        relation_counts = {
            "documents": counts.get("documents"),
            "chunks": counts.get("chunks"),
            "sparse_postings": counts.get("sparse_postings"),
        }
    elif backend == "postgres-holo":
        row = store.db.execute(
            "SELECT (SELECT COUNT(*) FROM holo_docs), "
            "(SELECT COUNT(*) FROM holo_grams), "
            "(SELECT COUNT(*) FROM holo_spectrum)"
        ).fetchone()
        relation_counts = {
            "documents": row[0],
            "chunks": row[1],
            "sparse_postings": row[2],
        }
    else:
        raise ValueError("remote emptiness check requires a PostgreSQL backend")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for value in relation_counts.values()
    ):
        raise RuntimeError("could not verify empty PostgreSQL benchmark relations")
    if any(value != 0 for value in relation_counts.values()):
        raise RuntimeError(
            "refusing to benchmark non-empty PostgreSQL retrieval relations"
        )


def _cleanup_backend_states(backend_states: Mapping[str, Mapping[str, Any]]) -> None:
    """Close every backend and prove that remote benchmark writes were removed."""

    failures: List[str] = []
    for backend, state in backend_states.items():
        rag = state.get("rag")
        store = getattr(rag, "store", None)
        if store is None:
            failures.append(f"{backend}:missing-store")
            continue
        if backend != "sqlite":
            for document_id in reversed(list(state.get("persisted_document_ids", ()))):
                try:
                    store.delete_document(document_id)
                except Exception as exc:
                    failures.append(f"{backend}:delete:{type(exc).__name__}")
            try:
                store.commit()
            except Exception as exc:
                failures.append(f"{backend}:commit:{type(exc).__name__}")
            try:
                _assert_remote_store_empty(store)
            except Exception as exc:
                failures.append(f"{backend}:postcondition:{type(exc).__name__}")
            if bool(state.get("hnsw_build", {}).get("created_by_runner")):
                try:
                    store.drop_hnsw_index()
                    store.commit()
                except Exception as exc:
                    failures.append(f"{backend}:drop-index:{type(exc).__name__}")
        try:
            store.close()
        except Exception as exc:
            failures.append(f"{backend}:close:{type(exc).__name__}")
    if failures:
        raise RuntimeError(
            "remote benchmark cleanup failed (" + ", ".join(failures) + ")"
        )


def _external_id(hit: Mapping[str, Any], document_id_map: Mapping[int, str]) -> str:
    raw_document_id = hit.get("doc_id")
    if isinstance(raw_document_id, int) and raw_document_id in document_id_map:
        return document_id_map[raw_document_id]
    return f"unmapped-chunk-{int(hit['id'])}"


def _map_stage_ids(ids: Iterable[int], chunk_id_map: Mapping[int, str]) -> List[str]:
    return [chunk_id_map.get(int(chunk_id), f"unmapped-chunk-{int(chunk_id)}") for chunk_id in ids]


def _legacy_stage_report(
    system: Mapping[str, Any],
    result: Any,
    qrels: Mapping[str, int],
    chunk_id_map: Mapping[int, str],
    limits: Mapping[str, int],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    backend_view = system["backend_view"]
    dense = _map_stage_ids((row[0] for row in backend_view.last_dense), chunk_id_map)
    sparse = _map_stage_ids((row[0] for row in backend_view.last_sparse), chunk_id_map)
    final = [_external_id(hit, system["document_id_map"]) for hit in result.fused]
    recorder = StageRecorder()
    recorder("dense", dense)
    recorder("sparse", sparse)
    recorder("final", final)
    snapshots = recorder.snapshot()
    return (
        recorder.report(qrels, limits=limits),
        {
            "status": "unknown",
            "reason": "legacy pipeline has no native full-stage observer",
            "observed_stages": list(snapshots),
            "stage_ids_sha256": {
                stage: canonical_sha256(list(ids)) for stage, ids in snapshots.items()
            },
            "violations": None,
        },
    )


def _stage_limits(args: argparse.Namespace) -> Dict[str, int]:
    pool = min(1_000, max(args.top_k * 2, args.structural_depth))
    return {
        "dense": args.channel_k,
        "sparse": args.channel_k,
        "union": min(2 * args.channel_k, 2_000),
        "fusion": pool,
        # The observer sees the full ranking after the structurally scored
        # prefix is blended back with its untouched tail.
        "structural": pool,
        "reranker": pool,
        "final": args.top_k,
    }


def _evaluate_hnsw_recall(
    store: Any,
    encoder: Any,
    queries: Sequence[Mapping[str, Any]],
    *,
    k: int,
    snapshot_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Audit ANN against exact dense retrieval on one snapshot and filter."""

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("HNSW audit k must be a positive integer")
    per_query = []
    recalls = []
    for query in queries:
        query_id = query.get("query_id")
        if not isinstance(query_id, str):
            raise ValueError("HNSW audit query IDs must be strings")
        item: Dict[str, Any] = {
            "query_id": query_id,
            "status": "not_evaluated",
            "exact_count": None,
            "ann_count": None,
            "underfill": None,
            "recall_at_k": None,
            "natural_plan": None,
        }
        try:
            vector = list(encoder.encode([str(query["text"])], is_query=True))[0]
            exact_rows = list(store.dense_search(vector.dense, k, filters=None, exact=True))
            audit = dict(store.explain_dense(vector.dense, k, filters=None, exact=False))
            ann_rows = list(store.dense_search(vector.dense, k, filters=None, exact=False))
            exact_ids = [int(row[0]) for row in exact_rows]
            ann_ids = [int(row[0]) for row in ann_rows]
            exact_unique = set(exact_ids)
            ann_unique = set(ann_ids)
            hnsw_used = bool(
                isinstance(audit.get("plan"), Mapping)
                and audit["plan"].get("hnsw_used") is True
            )
            underfill = (
                len(exact_ids) != k
                or len(ann_ids) != k
                or len(exact_unique) != k
                or len(ann_unique) != k
            )
            item.update(
                {
                    "exact_count": len(exact_ids),
                    "ann_count": len(ann_ids),
                    "underfill": underfill,
                    "natural_plan": {
                        "hnsw_used": hnsw_used,
                        "configured_search_mode": audit.get("configured_search_mode"),
                        "selected_mode": audit.get("selected_mode", audit.get("mode")),
                    },
                }
            )
            if not underfill and hnsw_used:
                recall = len(exact_unique.intersection(ann_unique)) / k
                item["recall_at_k"] = recall
                item["status"] = (
                    "passed" if recall >= MIN_HNSW_RECALL_AT_K else "failed"
                )
                if item["status"] == "failed":
                    item["reason"] = "recall_below_preregistered_threshold"
                recalls.append(recall)
            else:
                item["reason"] = (
                    "exact_or_ann_underfill" if underfill else "natural_plan_not_hnsw"
                )
        except Exception as exc:
            item["reason"] = "exact_ann_or_explain_unavailable"
            item["error_type"] = type(exc).__name__
        per_query.append(item)
    evaluated_all = len(recalls) == len(queries) and bool(queries)
    threshold_failed = any(
        recall < MIN_HNSW_RECALL_AT_K for recall in recalls
    )
    if threshold_failed:
        status = "failed"
        reason = "ANN recall is below the preregistered threshold"
    elif evaluated_all:
        status = "passed"
        reason = None
    else:
        status = "not_evaluated"
        reason = "requires full exact and ANN top-k plus natural HNSW EXPLAIN"
    return {
        "status": status,
        "reason": reason,
        "k": k,
        "minimum_recall_at_k": MIN_HNSW_RECALL_AT_K,
        "definition": "|ANN top-k intersect exact top-k| / k",
        "snapshot_sha256": snapshot_sha256,
        "filters": {"policy": "none", "sha256": canonical_sha256({"policy": "none"})},
        "recall_at_k": math.fsum(recalls) / len(recalls) if recalls else None,
        "support": len(recalls),
        "queries": per_query,
    }


def _build_backend(
    args: argparse.Namespace,
    backend: str,
    corpus: Mapping[str, Mapping[str, Any]],
    work_root: Path,
    dsn: str,
) -> Dict[str, Any]:
    cleanup_state: Dict[str, Any] = {}
    try:
        return _build_backend_state(
            args, backend, corpus, work_root, dsn, cleanup_state
        )
    except Exception:
        rag = cleanup_state.get("rag")
        if rag is not None:
            _cleanup_backend_states(
                {
                    backend: {
                        "rag": rag,
                        "persisted_document_ids": cleanup_state.get(
                            "persisted_document_ids", ()
                        ),
                        "hnsw_build": cleanup_state.get("hnsw_build", {}),
                    }
                }
            )
        raise


def _build_backend_state(
    args: argparse.Namespace,
    backend: str,
    corpus: Mapping[str, Mapping[str, Any]],
    work_root: Path,
    dsn: str,
    cleanup_state: Dict[str, Any],
) -> Dict[str, Any]:
    data_dir = work_root / backend
    specification = _system_specifications(backend, [])[1]
    cfg = _system_config(args, backend, specification, data_dir, dsn)
    rss_before = _rss_peak_bytes()
    rag = TriRag(cfg, llm=NoLLM())
    cleanup_state["rag"] = rag
    if backend != "sqlite":
        _assert_remote_store_empty(rag.store)

    document_id_map: Dict[int, str] = {}
    persisted_document_ids: List[int] = []
    cleanup_state["persisted_document_ids"] = persisted_document_ids
    chunks = 0
    ingest_start = time.perf_counter()
    for document_id, document in corpus.items():
        result = rag.ingest(
            str(document["text"]), source=document_id, title=document_id
        )
        database_document_id = int(result["doc_id"])
        persisted_document_ids.append(database_document_id)
        document_id_map[database_document_id] = document_id
        chunks += int(result["chunks"])
    rag.store.commit()
    ingest_seconds = max(0.0, time.perf_counter() - ingest_start)

    hnsw_build = {
        "status": "not_applicable",
        "hnsw_build_seconds": None,
        "index_status": None,
        "created_by_runner": False,
    }
    if backend == "pgvector" and args.pgvector_mode == "hnsw":
        rag.store.ef_search = args.hnsw_ef_search
        rag.store.iterative_scan = args.hnsw_iterative_scan
        rag.store.max_scan_tuples = args.hnsw_max_scan_tuples
        rag.store.scan_mem_multiplier = args.hnsw_scan_mem_multiplier
        hnsw_build = _timed_hnsw_build(
            rag.store,
            m=args.hnsw_m,
            ef_construction=args.hnsw_ef_construction,
            concurrently=args.hnsw_concurrently,
        )
        cleanup_state["hnsw_build"] = hnsw_build

    all_rows = rag.store.all_texts()
    hydrated = _get_chunks_batched(
        rag.store, (int(row["id"]) for row in all_rows)
    )
    chunk_id_map = {
        int(row["id"]): document_id_map[int(row["doc_id"])]
        for row in hydrated
        if isinstance(row.get("doc_id"), int) and int(row["doc_id"]) in document_id_map
    }
    rss_after = _rss_peak_bytes()
    storage = _storage_per_chunk(_index_size(rag.store, data_dir), chunks=chunks)
    storage.update(
        {
            "hnsw_build_seconds": hnsw_build["hnsw_build_seconds"],
            "hnsw_build_status": hnsw_build["status"],
        }
    )
    return {
        "rag": rag,
        "data_dir": data_dir,
        "document_id_map": document_id_map,
        "chunk_id_map": chunk_id_map,
        "persisted_document_ids": persisted_document_ids,
        "ingest": {
            "documents": len(corpus),
            "chunks": chunks,
            "seconds": ingest_seconds,
            "documents_per_second": (
                len(corpus) / ingest_seconds if ingest_seconds > 0.0 else None
            ),
            "chunks_per_second": chunks / ingest_seconds if ingest_seconds > 0.0 else None,
        },
        "memory": {
            "rss_baseline_bytes": rss_before,
            "rss_after_ingest_bytes": rss_after,
            "rss_ingest_delta_bytes": max(0, rss_after - rss_before),
        },
        "storage": storage,
        "hnsw_build": hnsw_build,
        "health": _safe_backend_health(rag.store),
    }


def _get_chunks_batched(store: Any, chunk_ids: Iterable[int]) -> List[dict]:
    """Hydrate arbitrary benchmark scales through bounded adapter calls."""

    maximum = DEFAULT_RETRIEVAL_LIMITS.max_pool
    hydrated: List[dict] = []
    batch: List[int] = []
    for chunk_id in chunk_ids:
        batch.append(int(chunk_id))
        if len(batch) == maximum:
            hydrated.extend(store.get_chunks(batch))
            batch = []
    if batch:
        hydrated.extend(store.get_chunks(batch))
    return hydrated


def _build_systems(
    args: argparse.Namespace,
    backend_state: Mapping[str, Any],
    backend: str,
    ablations: Sequence[str],
) -> List[Dict[str, Any]]:
    rag = backend_state["rag"]
    dense_exact: Optional[bool] = None
    if backend == "pgvector":
        dense_exact = args.pgvector_mode == "exact"
    systems = []
    for specification in _system_specifications(backend, ablations):
        cfg = _system_config(
            args,
            backend,
            specification,
            backend_state["data_dir"],
            "" if backend == "sqlite" else "configured-out-of-band",
        )
        backend_view = _BackendView(
            rag.store,
            dense_enabled=bool(specification["dense"]),
            sparse_enabled=bool(specification["sparse"]),
            dense_exact=dense_exact,
        )
        recorder = StageRecorder()
        if specification["pipeline"] == "legacy":
            retriever = TriRetriever(
                backend_view, rag.encoder, cfg, reranker=None, llm=NoLLM()
            )
            observer_kind = "derived_legacy_adapter"
        else:
            chunk_id_map = backend_state["chunk_id_map"]

            def observe(
                stage: str,
                ids: Sequence[int],
                *,
                target: StageRecorder = recorder,
                mapping: Mapping[int, str] = chunk_id_map,
            ) -> None:
                target(stage, _map_stage_ids(ids, mapping))

            retriever = RetrievalV2(
                backend_view,
                rag.encoder,
                cfg,
                stage_observer=observe,
            )
            observer_kind = "native_v2_stage_observer"
        systems.append(
            {
                "id": specification["id"],
                "specification": dict(specification),
                "cfg": cfg,
                "backend_view": backend_view,
                "retriever": retriever,
                "recorder": recorder,
                "observer_kind": observer_kind,
                "document_id_map": backend_state["document_id_map"],
                "measurements": {},
                "active_seconds": 0.0,
                "completed": 0,
                "failed": 0,
                "warmup_errors": 0,
            }
        )
    return systems


def _empty_measurement(query: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "query_id": query["query_id"],
        "relevant_count": len(query["qrels"]),
        "unanswerable_gold": query["unanswerable_gold"],
        "ranking": None,
        "stages": None,
        "lineage": None,
        "timings_ms": {"total": []},
        "errors": [],
    }


def _run_one(
    args: argparse.Namespace,
    system: Dict[str, Any],
    query: Mapping[str, Any],
    document_facts: Mapping[str, Iterable[str]],
    document_fingerprints: Mapping[str, str],
) -> None:
    measurement = system["measurements"][query["query_id"]]
    system["backend_view"].reset_capture()
    system["recorder"].reset()
    started_ns = time.perf_counter_ns()
    try:
        result = system["retriever"].search(
            str(query["text"]), top_k=args.top_k, channel_k=args.channel_k
        )
    except Exception as exc:
        elapsed_ms = max(0.0, (time.perf_counter_ns() - started_ns) / 1_000_000.0)
        system["active_seconds"] += elapsed_ms / 1_000.0
        system["failed"] += 1
        measurement["errors"].append({"type": type(exc).__name__})
        return
    elapsed_ms = max(0.0, (time.perf_counter_ns() - started_ns) / 1_000_000.0)
    system["active_seconds"] += elapsed_ms / 1_000.0
    system["completed"] += 1
    measurement["timings_ms"]["total"].append(elapsed_ms)
    for name, value in result.stats.items():
        if not name.endswith("_ms") or name == "total_ms":
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            normalized = float(value)
            if math.isfinite(normalized) and normalized >= 0.0:
                measurement["timings_ms"].setdefault(name[:-3], []).append(normalized)

    ranking = [
        _external_id(hit, system["document_id_map"]) for hit in result.fused[: args.top_k]
    ]
    limits = _stage_limits(args)
    if system["specification"]["pipeline"] == "legacy":
        stages, lineage = _legacy_stage_report(
            system,
            result,
            query["qrels"],
            system["backend_view"]._backend_state_chunk_map
            if hasattr(system["backend_view"], "_backend_state_chunk_map")
            else {},
            limits,
        )
    else:
        stages = system["recorder"].report(query["qrels"], limits=limits)
        lineage = validate_stage_lineage(
            system["recorder"].snapshot(), limits=limits, top_k=args.top_k
        )

    if measurement["ranking"] is None:
        measurement["ranking"] = ranking
        measurement["stages"] = stages
        measurement["lineage"] = lineage
    elif measurement["ranking"] != ranking:
        measurement["errors"].append({"type": "RankingDrift"})


def _cold_search_once(
    system: Mapping[str, Any],
    query: Mapping[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    system["backend_view"].reset_capture()
    system["recorder"].reset()
    started_ns = time.perf_counter_ns()
    error_type = None
    try:
        system["retriever"].search(
            str(query["text"]), top_k=args.top_k, channel_k=args.channel_k
        )
        status = "measured"
    except Exception as exc:
        status = "error"
        error_type = type(exc).__name__
    elapsed_ms = max(0.0, (time.perf_counter_ns() - started_ns) / 1_000_000.0)
    return {
        "status": status,
        "cold_latency_ms": elapsed_ms,
        "sample_count": 1,
        "single_run": True,
        "claim_eligible": False,
        "comparable_across_systems": False,
        "cache_mode": "best_effort_cold_before_warmup",
        "reason": "single shared-backend observation; descriptive only",
        "error_type": error_type,
    }


def _measure_systems(
    args: argparse.Namespace,
    systems: List[Dict[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    document_facts: Mapping[str, Iterable[str]],
    document_fingerprints: Mapping[str, str],
    chunk_id_maps: Mapping[str, Mapping[int, str]],
) -> None:
    for system in systems:
        system.setdefault("active_seconds", 0.0)
        system.setdefault("completed", 0)
        system.setdefault("failed", 0)
        system.setdefault("warmup_errors", 0)
        system["measurements"] = {
            query["query_id"]: _empty_measurement(query) for query in queries
        }
        # The legacy adapter needs the same storage-ID mapping that the native
        # V2 observer applies inside its callback.
        system["backend_view"]._backend_state_chunk_map = chunk_id_maps[
            system["specification"]["id"].split("-", 1)[1].split("-ablation", 1)[0]
        ]

    cold_query = queries[0]
    for system in systems:
        system["cold_measurement"] = _cold_search_once(system, cold_query, args)

    if args.warmup:
        for warmup_index in range(args.warmup):
            query = queries[warmup_index % len(queries)]
            order = systems if warmup_index % 2 == 0 else list(reversed(systems))
            for system in order:
                system["backend_view"].reset_capture()
                system["recorder"].reset()
                try:
                    system["retriever"].search(
                        str(query["text"]), top_k=args.top_k, channel_k=args.channel_k
                    )
                except Exception:
                    system["warmup_errors"] += 1

    for repetition in range(args.repetitions):
        for query_index, query in enumerate(queries):
            order = systems if (repetition + query_index) % 2 == 0 else list(reversed(systems))
            for system in order:
                _run_one(
                    args,
                    system,
                    query,
                    document_facts,
                    document_fingerprints,
                )

    # A ranking retained from an earlier successful repetition must never mask
    # a later failure or drift.  Benchmark evidence is all-or-nothing for the
    # preregistered repetition matrix; partial measurements are not finalized.
    expected_per_system = len(queries) * int(args.repetitions)
    integrity_failed = False
    for system in systems:
        if int(system["warmup_errors"]) != 0 or int(system["failed"]) != 0:
            integrity_failed = True
        if int(system["completed"]) != expected_per_system:
            integrity_failed = True
        for measurement in system["measurements"].values():
            if measurement["errors"]:
                integrity_failed = True
            if len(measurement["timings_ms"].get("total", ())) != int(
                args.repetitions
            ):
                integrity_failed = True
    if integrity_failed:
        raise RuntimeError("benchmark measurement integrity validation failed")


def _mean_optional(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [float(value) for value in values if value is not None]
    return (
        math.fsum(value / len(present) for value in present) if present else None
    )


def _throughput_report(
    *,
    completed: int,
    failed: int,
    measured_wall_seconds: float,
    concurrency: Optional[int],
    cache_mode: Optional[str],
    process_isolation: Optional[bool],
    preregistered_window: bool,
) -> Dict[str, Any]:
    for name, value in (("completed", completed), ("failed", failed)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if (
        isinstance(measured_wall_seconds, bool)
        or not isinstance(measured_wall_seconds, (int, float))
        or not math.isfinite(float(measured_wall_seconds))
        or measured_wall_seconds < 0.0
    ):
        raise ValueError("measured_wall_seconds must be finite and non-negative")
    if concurrency is not None and (
        isinstance(concurrency, bool)
        or not isinstance(concurrency, int)
        or concurrency <= 0
    ):
        raise ValueError("concurrency must be a positive integer or None")
    if cache_mode is not None and not isinstance(cache_mode, str):
        raise TypeError("cache_mode must be a string or None")
    if process_isolation is not None and not isinstance(process_isolation, bool):
        raise TypeError("process_isolation must be bool or None")
    if not isinstance(preregistered_window, bool):
        raise TypeError("preregistered_window must be bool")

    duration = float(measured_wall_seconds)
    attempts = completed + failed
    successful_rate = completed / duration if duration > 0.0 else None
    attempt_rate = attempts / duration if duration > 0.0 else None
    reasons = []
    if completed < 100:
        reasons.append("successful_calls_below_100")
    if duration < 5.0:
        reasons.append("measurement_window_below_5_seconds")
    if failed:
        reasons.append("errors_present")
    if concurrency is None:
        reasons.append("concurrency_not_preregistered")
    if cache_mode is None:
        reasons.append("cache_mode_not_preregistered")
    elif cache_mode != "warm":
        reasons.append("cache_mode_not_warm")
    if process_isolation is None:
        reasons.append("process_isolation_not_proven")
    elif not process_isolation:
        reasons.append("process_not_isolated")
    if not preregistered_window:
        reasons.append("measurement_window_not_preregistered")
    claim_eligible = not reasons
    return {
        "qps": successful_rate if claim_eligible else None,
        "successful_qps_sample": successful_rate,
        "serial_attempt_rate_sample": attempt_rate,
        "completed": completed,
        "errors": failed,
        "calls_including_errors": attempts,
        "error_rate": failed / attempts if attempts else 0.0,
        "elapsed_active_seconds": duration,
        "concurrency": concurrency,
        "cache_mode": cache_mode,
        "process_isolation": process_isolation,
        "preregistered_window": preregistered_window,
        "claim_eligible": claim_eligible,
        "status": "measured" if claim_eligible else "non_claim",
        "non_claim_reasons": reasons,
        "reason": None if claim_eligible else ";".join(reasons),
    }


def _finalize_system(
    args: argparse.Namespace,
    system: Mapping[str, Any],
    queries: Sequence[Mapping[str, Any]],
    document_facts: Mapping[str, Iterable[str]],
    document_fingerprints: Mapping[str, str],
    backend_state: Mapping[str, Any],
) -> Dict[str, Any]:
    per_query = []
    metric_rows = []
    for query in queries:
        measurement = system["measurements"][query["query_id"]]
        ranking = list(measurement["ranking"] or [])
        if measurement["ranking"] is None:
            metrics = {
                "recall_at_5": None,
                "recall_at_10": None,
                "recall_at_20": None,
                "mrr_at_20": None,
                "ndcg_at_10": None,
                "coverage_at_20": None,
                "duplicate_rate_at_20": None,
                "duplicate_ids_removed": None,
                "no_answer_correct": None,
                "citation_precision": None,
            }
            stages = {
                stage: {
                    "status": "unobserved",
                    "limit": _stage_limits(args)[stage],
                    "candidate_count": None,
                    "duplicate_ids_removed": None,
                    "recall": None,
                }
                for stage in RETRIEVAL_STAGES
            }
            lineage = {
                "status": "unknown",
                "reason": "retrieval call did not complete",
                "observed_stages": [],
                "stage_ids_sha256": {},
                "violations": None,
            }
        else:
            metrics = evaluate_query(
                ranking,
                query["qrels"],
                top_k=args.top_k,
                document_facts=document_facts,
                required_facts=query["required_facts"],
                document_fingerprints=document_fingerprints,
                unanswerable_gold=query["unanswerable_gold"],
                # Retrieval does not expose a calibrated abstention decision.
                abstain_pred=None,
                # There is no reader or calibrated citation label in this run.
                citation_support_labels=None,
            )
            stages = measurement["stages"]
            lineage = measurement["lineage"]
        metric_rows.append(metrics)
        per_query.append(
            {
                "query_id": query["query_id"],
                "relevant_count": measurement["relevant_count"],
                "unanswerable_gold": measurement["unanswerable_gold"],
                "ranking": ranking,
                "duplicate_ids_removed": metrics["duplicate_ids_removed"],
                "stages": stages,
                "lineage": lineage,
                "metrics": metrics,
                "timings_ms": measurement["timings_ms"],
                "errors": measurement["errors"],
            }
        )

    quality = aggregate_query_metrics(metric_rows)
    quality["duplicate_ids_removed_total"] = sum(
        int(row.get("duplicate_ids_removed") or 0) for row in metric_rows
    )
    stage_recall = {
        stage: _mean_optional(
            query["stages"][stage]["recall"] for query in per_query
        )
        for stage in RETRIEVAL_STAGES
    }
    all_latencies = [
        sample
        for query in per_query
        for sample in query["timings_ms"].get("total", [])
    ]
    latency_by_query = {
        query["query_id"]: list(query["timings_ms"].get("total", []))
        for query in per_query
        if query["timings_ms"].get("total")
    }
    timing_names = sorted(
        {
            name
            for query in per_query
            for name in query["timings_ms"]
            if name != "total"
        }
    )
    stage_latency = {
        name: latency_percentiles(
            sample
            for query in per_query
            for sample in query["timings_ms"].get(name, [])
        )
        for name in timing_names
    }
    rss_peak = _rss_peak_bytes()
    memory = dict(backend_state["memory"])
    memory.update(
        {
            "rss_peak_bytes": rss_peak,
            "rss_peak_delta_bytes": max(
                0, rss_peak - int(memory["rss_baseline_bytes"])
            ),
            "scope": "benchmark Python process; ru_maxrss high-water mark",
            "python_process_rss_peak_bytes": rss_peak,
            "postgres_server_rss_peak_bytes": None,
            "shared_process": True,
            "comparable_across_systems": False,
            "comparison_reason": "systems share one Python process and ru_maxrss high-water state",
        }
    )
    cfg = system["cfg"]
    specification = system["specification"]
    config = {
        "backend": cfg.backend,
        "pipeline": specification["pipeline"],
        "ablation": specification["ablation"],
        "encoder": "hash",
        "dense_dim": cfg.dense_dim,
        "structural_dim": cfg.colbert_dim,
        "top_k": cfg.top_k,
        "channel_k": cfg.channel_k,
        "fusion": cfg.fusion,
        "rrf_k": cfg.rrf_k,
        "channel_weights": list(cfg.channel_weights),
        "dense_enabled": specification["dense"],
        "sparse_enabled": specification["sparse"],
        "structural": specification["structural"],
        "structural_candidate_depth": cfg.structural_candidate_depth,
        "reranker": "none",
        "diversity": specification["diversity"],
        "mmr_lambda": 0.5 if specification["diversity"] == "mmr" else None,
        "dpp_alpha": 1.0 if specification["diversity"] == "dpp" else None,
        "dpp_jitter": 1e-9 if specification["diversity"] == "dpp" else None,
        "pgvector_dense_mode": args.pgvector_mode if cfg.backend == "pgvector" else None,
        "stage_observer": system["observer_kind"],
    }
    lineage_violations = sum(
        len(query["lineage"]["violations"] or []) for query in per_query
    )
    duplicate_violations = int(quality["duplicate_ids_removed_total"])
    gate_failed = lineage_violations > 0 or duplicate_violations > 0
    throughput = _throughput_report(
        completed=int(system["completed"]),
        failed=int(system["failed"]),
        measured_wall_seconds=float(system["active_seconds"]),
        concurrency=1,
        cache_mode="warm",
        process_isolation=False,
        preregistered_window=False,
    )
    throughput["measurement_window"] = (
        "serial active calls in balanced interleaved order; not duration-controlled"
    )
    return {
        "id": system["id"],
        "config": config,
        "queries": per_query,
        "aggregate": {
            "quality": quality,
            "stage_recall": stage_recall,
            "performance": {
                "latency_ms": latency_percentiles(all_latencies),
                "latency_p95_bootstrap": clustered_percentile_bootstrap(
                    latency_by_query,
                    percentile=0.95,
                    samples=args.bootstrap_samples,
                    seed=BOOTSTRAP_SEED,
                ),
                "stage_latency_ms": stage_latency,
                "qps": throughput,
                "cold_latency_ms": dict(system["cold_measurement"]),
                "memory": memory,
                "storage": dict(backend_state["storage"]),
                "ingest": dict(backend_state["ingest"]),
            },
            "quality_gate": {
                "status": "failed" if gate_failed else "not_evaluated",
                "claim_scope": "synthetic",
                "reason": (
                    "duplicate IDs or stage-lineage violations detected"
                    if gate_failed
                    else "synthetic calibration data cannot establish a global quality gain"
                ),
                "duplicate_ids_removed_total": duplicate_violations,
                "stage_lineage_violations": lineage_violations,
            },
        },
    }


def _metric_by_query(system: Mapping[str, Any], metric: str) -> Dict[str, Optional[float]]:
    return {
        query["query_id"]: query["metrics"].get(metric) for query in system["queries"]
    }


def _latency_by_query(system: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    values: Dict[str, Optional[float]] = {}
    for query in system["queries"]:
        samples = query["timings_ms"].get("total", [])
        values[query["query_id"]] = median(samples) if samples else None
    return values


def _comparisons(
    systems: Sequence[Mapping[str, Any]], bootstrap_samples: int
) -> List[Dict[str, Any]]:
    by_id = {system["id"]: system for system in systems}
    comparisons = []
    for candidate in systems:
        if candidate["config"]["pipeline"] != "v2":
            continue
        baseline_id = f"legacy-{candidate['config']['backend']}"
        baseline = by_id.get(baseline_id)
        if baseline is None:
            continue
        paired = {}
        for metric in (
            "ndcg_at_10",
            "recall_at_5",
            "recall_at_10",
            "recall_at_20",
            "mrr_at_20",
            "coverage_at_20",
            "duplicate_rate_at_20",
        ):
            paired[metric] = paired_bootstrap(
                _metric_by_query(baseline, metric),
                _metric_by_query(candidate, metric),
                samples=bootstrap_samples,
                seed=BOOTSTRAP_SEED,
                higher_is_better=metric != "duplicate_rate_at_20",
            )
        paired["median_query_latency_ms"] = paired_bootstrap(
            _latency_by_query(baseline),
            _latency_by_query(candidate),
            samples=bootstrap_samples,
            seed=BOOTSTRAP_SEED,
            higher_is_better=False,
        )
        comparisons.append(
            {
                "baseline": baseline_id,
                "candidate": candidate["id"],
                "paired_delta": paired,
                "claim": None,
            }
        )
    return comparisons


def _common_config(
    args: argparse.Namespace, backends: Sequence[str], ablations: Sequence[str]
) -> Dict[str, Any]:
    return {
        "backends": list(backends),
        "pipelines": ["legacy", "v2"],
        "encoder": "hash",
        "dense_dim": args.dense_dim,
        "structural_dim": args.structural_dim,
        "max_structural_tokens": args.max_structural_tokens,
        "top_k": args.top_k,
        "channel_k": args.channel_k,
        "rrf_k": args.rrf_k,
        "structural_candidate_depth": args.structural_depth,
        "ablations": list(ablations),
        "remote_write_guard": (
            "allow_remote + explicit write env + test/bench database name"
            if any(backend != "sqlite" for backend in backends)
            else None
        ),
        "pgvector_mode": args.pgvector_mode if "pgvector" in backends else None,
        "hnsw_minimum_recall_at_k": (
            MIN_HNSW_RECALL_AT_K if "pgvector" in backends else None
        ),
        "hnsw": (
            {
                "m": args.hnsw_m,
                "ef_construction": args.hnsw_ef_construction,
                "ef_search": args.hnsw_ef_search,
                "iterative_scan": args.hnsw_iterative_scan,
                "max_scan_tuples": args.hnsw_max_scan_tuples,
                "scan_mem_multiplier": args.hnsw_scan_mem_multiplier,
                "concurrently": args.hnsw_concurrently,
            }
            if "pgvector" in backends and args.pgvector_mode == "hnsw"
            else None
        ),
    }


def _database_name_from_dsn(dsn: str) -> str:
    if not isinstance(dsn, str) or not dsn.strip():
        raise ValueError("PostgreSQL DSN is missing")
    try:
        from psycopg.conninfo import conninfo_to_dict
    except ImportError:
        raise ValueError(
            "PostgreSQL benchmark DSN validation requires the postgres extra"
        ) from None
    try:
        # libpq permits duplicate keyword parameters and URI query overrides;
        # only its canonical parser identifies the database that will actually
        # receive benchmark writes.
        database = str(conninfo_to_dict(dsn.strip()).get("dbname", ""))
    except Exception:
        # Never echo the DSN: it may contain a password.
        raise ValueError("PostgreSQL DSN is invalid") from None
    if not database or "/" in database or "\x00" in database:
        raise ValueError("PostgreSQL DSN must name one database")
    return database


def _is_benchmark_database_name(database: str) -> bool:
    """Require an explicit, delimited ``test`` or ``bench`` name token."""

    return bool(
        re.search(r"(?:^|[_-])(?:test|bench)(?:$|[_-])", database.casefold())
    )


def _parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=int, default=DEFAULT_SCALE)
    parser.add_argument(
        "--protocol",
        choices=("calibration", "validation", "test"),
        default="calibration",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--channel-k", type=int, default=100)
    parser.add_argument("--dense-dim", type=int, default=128)
    parser.add_argument("--structural-dim", type=int, default=32)
    parser.add_argument("--max-structural-tokens", type=int, default=64)
    parser.add_argument("--structural-depth", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--backends", "--backend", dest="backends", default="sqlite")
    parser.add_argument("--ablations", default="")
    parser.add_argument("--postgres-dsn-env", default="RAG3D_BENCHMARK_PG_DSN")
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--pgvector-mode", choices=("exact", "hnsw"), default="exact")
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=64)
    parser.add_argument("--hnsw-ef-search", type=int, default=40)
    parser.add_argument(
        "--hnsw-iterative-scan",
        choices=("off", "strict_order", "relaxed_order"),
        default="off",
    )
    parser.add_argument("--hnsw-max-scan-tuples", type=int, default=20_000)
    parser.add_argument("--hnsw-scan-mem-multiplier", type=float, default=1.0)
    parser.add_argument("--hnsw-concurrently", action="store_true")
    parser.add_argument("--validation-lock", type=Path)
    parser.add_argument("--write-validation-lock", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if not 20 <= args.scale <= MAX_SCALE:
        parser.error(f"--scale must be between 20 and {MAX_SCALE}")
    for name in ("top_k", "channel_k", "dense_dim", "structural_dim", "max_structural_tokens", "structural_depth", "rrf_k", "repetitions", "bootstrap_samples"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.warmup > MAX_WARMUP:
        parser.error(f"--warmup cannot exceed {MAX_WARMUP}")
    if args.repetitions > MAX_REPETITIONS:
        parser.error(f"--repetitions cannot exceed {MAX_REPETITIONS}")
    if args.bootstrap_samples > MAX_BOOTSTRAP_SAMPLES:
        parser.error(
            f"--bootstrap-samples cannot exceed {MAX_BOOTSTRAP_SAMPLES}"
        )
    if args.top_k < 20:
        parser.error("--top-k must be at least 20 for Recall@20")
    if args.top_k > args.channel_k:
        parser.error("--top-k cannot exceed --channel-k")
    if args.scale < args.top_k:
        parser.error("--scale cannot be smaller than --top-k")
    if args.top_k > 100 or args.channel_k > 1_000 or args.structural_depth > 1_000:
        parser.error("retrieval limits exceed the public V2 bounds")
    if args.dense_dim > MAX_DENSE_DIM:
        parser.error(f"--dense-dim cannot exceed {MAX_DENSE_DIM}")
    if args.structural_dim > MAX_STRUCTURAL_DIM:
        parser.error(f"--structural-dim cannot exceed {MAX_STRUCTURAL_DIM}")
    if args.max_structural_tokens > MAX_STRUCTURAL_TOKENS:
        parser.error(
            "--max-structural-tokens cannot exceed "
            f"{MAX_STRUCTURAL_TOKENS}"
        )
    structural_values = args.structural_dim * args.max_structural_tokens
    if structural_values > MAX_STRUCTURAL_VALUES_PER_CHUNK:
        parser.error(
            "structural tensor cannot exceed "
            f"{MAX_STRUCTURAL_VALUES_PER_CHUNK} values per chunk"
        )
    # Conservative peak estimate: encoder arrays and the persisted adapter
    # representation coexist during ingestion.  Eight bytes/value covers the
    # float32 working copy plus float16/float32 storage and bounded overhead;
    # the fixed term covers per-chunk sparse/row metadata.
    estimated_embedding_bytes = args.scale * (
        8 * args.dense_dim + 8 * structural_values + 1_024
    )
    if estimated_embedding_bytes > MAX_ESTIMATED_EMBEDDING_BYTES:
        parser.error(
            "estimated embedding payload exceeds the 512 MiB benchmark safety bound"
        )
    if not _ENV_NAME.fullmatch(args.postgres_dsn_env):
        parser.error("--postgres-dsn-env must be an uppercase environment variable name")
    try:
        args.backend_values = _parse_csv(args.backends, _SUPPORTED_BACKENDS, "backends")
        args.ablation_values = _ablation_values(args.ablations)
    except ValueError as exc:
        parser.error(str(exc))
    if args.protocol == "test" and args.ablation_values:
        parser.error("--protocol test forbids ablations and tuning variants")
    if not 1 <= args.hnsw_ef_search <= 1_000:
        parser.error("--hnsw-ef-search must be between 1 and 1000")
    if not 2 <= args.hnsw_m <= 100:
        parser.error("--hnsw-m must be between 2 and 100")
    if not 4 <= args.hnsw_ef_construction <= 1_000:
        parser.error("--hnsw-ef-construction must be between 4 and 1000")
    if args.hnsw_ef_construction < 2 * args.hnsw_m:
        parser.error("--hnsw-ef-construction must be at least 2 times --hnsw-m")
    if not 1 <= args.hnsw_max_scan_tuples <= 1_000_000:
        parser.error("--hnsw-max-scan-tuples must be between 1 and 1000000")
    if (
        not math.isfinite(args.hnsw_scan_mem_multiplier)
        or not 1.0 <= args.hnsw_scan_mem_multiplier <= 1_000.0
    ):
        parser.error("--hnsw-scan-mem-multiplier must be finite and between 1 and 1000")
    if any(backend != "sqlite" for backend in args.backend_values):
        if not args.allow_remote:
            parser.error("PostgreSQL backends require --allow-remote and a dedicated empty database")
        dsn = os.environ.get(args.postgres_dsn_env, "").strip()
        if not dsn:
            parser.error(f"PostgreSQL backends require DSN in {args.postgres_dsn_env}")
        if os.environ.get(_REMOTE_WRITE_ENV) != "1":
            parser.error(
                "PostgreSQL benchmark writes require RAG3D_BENCHMARK_ALLOW_WRITE=1"
            )
        try:
            database_name = _database_name_from_dsn(dsn)
        except ValueError:
            parser.error("PostgreSQL benchmark DSN must identify a database")
        if not _is_benchmark_database_name(database_name):
            parser.error(
                "PostgreSQL benchmark database name must contain a delimited "
                "test or bench token"
            )
    if args.protocol == "test" and args.validation_lock is None:
        parser.error("--protocol test requires --validation-lock")
    if args.write_validation_lock is not None and args.protocol != "validation":
        parser.error("--write-validation-lock is valid only for --protocol validation")
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = ROOT / "benchmarks" / "results" / f"retrieval-v2-{stamp}.json"
    return args


def run(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    process_cpu_started = time.process_time()
    args = _parse_arguments(argv)
    manifest = _load_manifest()
    corpus = _generate_corpus(args.scale, int(manifest["seed"]))
    corpus_sha256, manifest_sha256, dataset_identity_sha256 = _dataset_identity(
        manifest, corpus
    )
    common_config = _common_config(args, args.backend_values, args.ablation_values)
    expected_validation_lock = _validation_lock_for(
        args,
        manifest,
        common_config,
        manifest_sha256=manifest_sha256,
        dataset_sha256=dataset_identity_sha256,
    )
    config_sha256 = expected_validation_lock["hashes"]["config_sha256"]
    validation_lock = None
    if args.validation_lock is not None:
        validation_lock = json.loads(args.validation_lock.read_text(encoding="utf-8"))
    # This boundary deliberately precedes _materialize_queries: a malformed,
    # stale, or validation-incompatible lock cannot cause test qrels to exist.
    validate_split_protocol(
        args.protocol,
        tuning_allowed=args.protocol != "test",
        config_sha256=config_sha256,
        dataset_sha256=dataset_identity_sha256,
        validation_lock=validation_lock,
        expected_lock=expected_validation_lock,
    )
    validation_lock_sha256 = None
    validation_lock_identity = None
    if args.protocol == "test":
        assert validation_lock is not None
        validation_lock_sha256 = canonical_sha256(validation_lock)
        validation_lock_identity = {
            "schema_version": validation_lock["schema_version"],
            "origin_protocol": validation_lock["origin_protocol"],
            "config_sha256": validation_lock["hashes"]["config_sha256"],
            "dataset_sha256": validation_lock["hashes"]["dataset_sha256"],
            "runtime_source_closure_sha256": validation_lock["hashes"][
                "runtime_source_closure_sha256"
            ],
            "source_diff_sha256": validation_lock["hashes"][
                "source_diff_sha256"
            ],
        }
    queries = _materialize_queries(manifest, args.protocol, corpus)
    document_facts = {
        document_id: list(document["facts"]) for document_id, document in corpus.items()
    }
    document_fingerprints = {
        document_id: str(document["fingerprint"])
        for document_id, document in corpus.items()
    }
    queries_sha256 = canonical_sha256(
        [{"query_id": query["query_id"], "text": query["text"]} for query in queries]
    )
    qrels_sha256 = canonical_sha256(
        {query["query_id"]: query["qrels"] for query in queries}
    )

    dsn = os.environ.get(args.postgres_dsn_env, "").strip()
    backend_states: Dict[str, Dict[str, Any]] = {}
    temporary = tempfile.TemporaryDirectory(prefix="rag3d-retrieval-v2-")
    work_root = Path(temporary.name)
    cleanup_attempted = False
    try:
        for backend in args.backend_values:
            backend_states[backend] = _build_backend(
                args,
                backend,
                corpus,
                work_root,
                dsn if backend != "sqlite" else "",
            )
            if backend == "pgvector" and args.pgvector_mode == "hnsw":
                backend_states[backend]["ann_validation"] = _evaluate_hnsw_recall(
                    backend_states[backend]["rag"].store,
                    backend_states[backend]["rag"].encoder,
                    queries,
                    k=args.top_k,
                    snapshot_sha256=corpus_sha256,
                )
            else:
                backend_states[backend]["ann_validation"] = {
                    "status": "not_applicable",
                    "reason": "backend is not pgvector HNSW mode",
                }
        systems: List[Dict[str, Any]] = []
        for backend in args.backend_values:
            systems.extend(
                _build_systems(
                    args, backend_states[backend], backend, args.ablation_values
                )
            )
        _measure_systems(
            args,
            systems,
            queries,
            document_facts,
            document_fingerprints,
            {
                backend: state["chunk_id_map"]
                for backend, state in backend_states.items()
            },
        )
        finalized = [
            _finalize_system(
                args,
                system,
                queries,
                document_facts,
                document_fingerprints,
                backend_states[system["specification"]["id"].split("-", 1)[1].split("-ablation", 1)[0]],
            )
            for system in systems
        ]
        commit = _safe_git("rev-parse", "HEAD")
        dirty = _safe_git("status", "--porcelain") != "unknown" and bool(
            _safe_git("status", "--porcelain")
        )
        created_at = datetime.now(timezone.utc).isoformat()
        comparisons = _comparisons(finalized, args.bootstrap_samples)
        process_cpu_seconds = max(0.0, time.process_time() - process_cpu_started)
        report = {
            "schema_version": "2.0",
            "run": {
                "id": f"retrieval-v2-{config_sha256[:12]}-{int(time.time())}",
                "created_at": created_at,
                "commit": commit,
                "dirty": dirty,
                "process_cpu_seconds": process_cpu_seconds,
                "cpu_scope": "whole_shared_python_process_not_attributable_per_system",
            },
            "dataset": {
                "id": manifest["id"],
                "version": manifest["version"],
                "claim_scope": "synthetic",
                "split": args.protocol,
                "corpus_sha256": corpus_sha256,
                "queries_sha256": queries_sha256,
                "qrels_sha256": qrels_sha256,
                "manifest_sha256": manifest_sha256,
                "dataset_sha256": dataset_identity_sha256,
                "documents": len(corpus),
                "chunks": max(state["ingest"]["chunks"] for state in backend_states.values()),
                "queries": len(queries),
                "queries_with_positive_qrels": sum(bool(query["qrels"]) for query in queries),
                "seed": manifest["seed"],
                "selection": "sha256(seed + NUL + document_id), before qrels materialization",
            },
            "environment": _environment(
                {backend: state["health"] for backend, state in backend_states.items()}
            ),
            "config": common_config,
            "protocol": {
                "tuning_allowed": args.protocol != "test",
                "warmup": args.warmup,
                "repetitions": args.repetitions,
                "bootstrap_samples": args.bootstrap_samples,
                "confidence": 0.95,
                "quality_unit": "query",
                "latency_unit": "query repetition grouped by query",
                "order": "balanced interleaving, reversed on alternating query/repetition",
                "seeds": {
                    "dataset": BOOTSTRAP_SEED,
                    "bootstrap": BOOTSTRAP_SEED,
                },
                "config_sha256": config_sha256,
                "validation_lock_sha256": validation_lock_sha256,
                "validation_lock_identity": validation_lock_identity,
            },
            "systems": finalized,
            "comparisons": comparisons,
            "ann_validation": {
                backend: state["ann_validation"]
                for backend, state in backend_states.items()
            },
            "claims": {
                "quality_gain": None,
                "meta_20_percent": "not_evaluated",
                "reason": "runner output only; synthetic data cannot establish general improvement",
            },
        }
        # A report is successful only after every remote write has been
        # removed and the empty-store postcondition has been verified.
        cleanup_attempted = True
        _cleanup_backend_states(backend_states)
        write_json_report(args.output, report)
        if args.write_validation_lock is not None:
            write_json_report(args.write_validation_lock, expected_validation_lock)
        print(f"wrote retrieval evaluation report: {args.output}")
        print("no quality-gain claim was computed; inspect paired intervals in the JSON")
        return report
    finally:
        try:
            if not cleanup_attempted:
                cleanup_attempted = True
                _cleanup_backend_states(backend_states)
        finally:
            temporary.cleanup()


def main() -> int:
    try:
        run()
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"benchmark failed ({type(exc).__name__})", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
