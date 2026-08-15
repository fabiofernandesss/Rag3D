# ADR 0001: Retrieval Engine V2 behind an opt-in pipeline

## Status

Accepted for opt-in evaluation; not a replacement for the legacy pipeline.

## Date

2026-08-14

## Context

The original retrieval path couples storage details, three-channel fusion,
reranking and diversity. SQLite and the PostgreSQL holographic backend must
remain operational, and Node/Java depend on the existing holographic schema.
At the same time, pgvector exact/HNSW evaluation requires a native-vector
adapter and trustworthy diagnostics.

Code inspection showed that structural MaxSim operates only on the union of
dense and sparse candidates. Treating it as a third independent retriever
overstated its recall and made weighted raw-score fusion difficult to reason
about. Existing benchmark shortcuts also used test qrels to select corpus
documents, so they cannot support quality claims.

## Decision

1. Preserve `TriRetriever` as `legacy`; introduce a separate `RetrievalV2`.
2. Use typed, capability-aware backend adapters for SQLite, postgres-holo and
   optional pgvector.
3. Make dense and sparse the global retrievers. Use equal-weight RRF with
   `k0=60` as the untuned V2 baseline.
4. Run structural scoring as a bounded rank-based late reranker after fusion.
5. Keep quantum fusion available, but experimental in V2 and unchanged in
   legacy.
6. Apply optional listwise/cross-encoder reranking before `none`/MMR/greedy
   DPP diversity.
7. Fail closed on incompatible fingerprints and silent encoder fallback in V2.
8. Keep pgvector tables and Docker profile separate; do not install extensions
   at runtime.
9. Gate recommendation on paired quality/performance measurements. Architecture
   and tests may ship behind the flag even when the 20% target is not met.
10. Add and validate named foreign keys to the shared holographic schema in all
    three ports. Run the additive DDL with finite timeouts and fail closed on
    orphans or incompatible constraints.
11. Treat the complete V2 holographic fingerprint as Python-only. Node/Java
    retain legacy Hash parity but refuse V2-certified indexes rather than
    writing data under an incomplete certificate.

## Alternatives considered

### Replace the legacy pipeline in place

Rejected because it would broaden rollback and cross-language risk. A separate
pipeline makes adoption and comparison explicit.

### Continue using quantum fusion as the V2 default

Rejected as a default because the available baseline does not show superiority
over RRF. Quantum remains falsifiable and available for ablation.

### Treat structural MaxSim as a global third retriever

Rejected because the implementation never scans the corpus structurally. It
cannot recover candidates lost before the structural stage.

### Normalize and sum raw dense/sparse/structural scores

Rejected as the baseline because their scales are not interchangeable. RRF and
rank blending avoid uncalibrated raw-score arithmetic.

### Make pgvector mandatory

Rejected because SQLite is the zero-server path and postgres-holo is an
existing supported deployment. pgvector remains an optional extra/profile.

### Parallelize dense and sparse immediately

Rejected by profiling: dense was a small fraction of local latency, SQLite is
thread-affine and the PostgreSQL adapters use one connection. The expected
Amdahl benefit did not justify the concurrency risk.

## Consequences

- There are two explicit pipelines during rollout.
- V2 diagnostics and stage recall can locate candidate loss.
- Structural depth, ANN mode and diversity parameters are measurable knobs,
  not claims of optimality.
- A V2 fingerprint mismatch requires reindexing or rollback to legacy; it is
  not silently accepted.
- Node/Java remain compatible with a legacy Hash/holographic index, but do not
  gain pgvector/V2 parity and cannot open a V2-certified holographic index.
- Existing holographic columns/data remain portable, but startup now adds and
  validates three FKs. Orphaned legacy data blocks startup until repaired by a
  separately reviewed administrative operation.
- More code and contract tests exist, and the feature flag contains code-path
  adoption risk. Data rollback still requires a current legacy index because
  this change does not dual-write across backends.

## Reversibility

Set `RAG3D_RETRIEVAL_PIPELINE=legacy`, select `sqlite` or `postgres-holo`, and
restore `RAG3D_FUSION=quantum`. The additive `rag3d_v2_*` schema can remain in
place. The additive holographic FKs remain in place after traffic rollback;
they are not removed automatically. Before switching traffic, verify the
legacy ingestion watermark and backfill/reindex if it is stale. Node/Java
rollback from a V2-certified holographic database requires a separate legacy
Hash index built from source documents; never delete or rewrite fingerprint
metadata to bypass the guard.
