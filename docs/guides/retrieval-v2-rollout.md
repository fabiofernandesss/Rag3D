# Retrieval V2 rollout

Retrieval Engine V2 is opt-in. Keep the legacy path available until a frozen,
representative test set satisfies the quality and operational gates for the
deployment that will use it.

## Configuration precedence

An explicit `RAG3D_*` value wins over its historical equivalent. Historical
`TRIRAG_*` values are accepted where they have the same meaning and emit a
deprecation warning. A safe pipeline-dependent default is used last.

```bash
export RAG3D_BACKEND=sqlite               # sqlite | postgres-holo | pgvector
export RAG3D_RETRIEVAL_PIPELINE=v2        # legacy | v2
export RAG3D_FUSION=rrf                   # rrf | quantum
export RAG3D_STRUCTURAL_RERANK=true
export RAG3D_RERANKER=none                # none | llm | cross-encoder
export RAG3D_DIVERSITY_METHOD=none        # none | mmr | dpp
export RAG3D_ALLOW_ENCODER_FALLBACK=false
```

Install `rag3d[reranker]` before selecting `cross-encoder`. The adapter is
optional and fail-closed: an unavailable model or inference failure preserves
the pre-rerank order, and no quality gain is claimed without evaluation.

The historical `RAG3D_PG`/`TRIRAG_PG` DSN still selects postgres-holo when no
new backend is explicit. `RAG3D_BACKEND=pgvector` requires a DSN and the
optional pgvector dependencies. V2 does not silently replace a requested BGE
encoder with the Hash encoder.

## Holographic schema preflight

All three language ports add and validate the same named foreign keys. Before
upgrading a populated `postgres-holo` database, take a backup and run these
read-only checks. Every count must be zero:

```sql
SELECT COUNT(*) FROM holo_grams g
LEFT JOIN holo_docs d ON d.id = g.doc_id
WHERE g.doc_id IS NOT NULL AND d.id IS NULL;

SELECT COUNT(*) FROM holo_grams g
LEFT JOIN holo_grams p ON p.id = g.parent_id
WHERE g.parent_id IS NOT NULL AND p.id IS NULL;

SELECT COUNT(*) FROM holo_spectrum s
LEFT JOIN holo_grams g ON g.id = s.gram_id
WHERE g.id IS NULL;
```

Schedule first startup in a coordinated maintenance window. Adding/validating
FKs takes table locks and scans existing rows; the adapters use finite lock and
statement timeouts and fail closed rather than waiting indefinitely. Repairing
orphans or conflicting constraints is an explicit DBA operation, not an
automatic migration.

The V2 holographic fingerprint is Python-only. Node/Java continue to share a
legacy Hash index, but intentionally reject a V2-certified database. If
cross-language traffic is required, provision and synchronize a separate
legacy index before rollout.

## Staged adoption

1. Freeze encoder, chunking and pipeline settings. Ingest a new index; do not
   reuse an index with a different fingerprint. If rollback must remain
   immediately available, keep the legacy index synchronized explicitly; the
   library does not dual-write across backends.
2. Run legacy and V2 against the same labeled validation set. Tune only on
   calibration/validation, then write the validation lock.
3. Run the locked test split. Check nDCG@10, Recall@20, MRR, p95, QPS, memory,
   storage, errors and recall at every candidate stage.
4. Exercise empty query/corpus, filter bounds, reranker failure, connection
   loss, restart, the 64 KiB UTF-8 query boundary and rollback before
   production traffic.
5. Enable V2 for a bounded cohort while preserving aggregate diagnostics. Do
   not log queries, chunks, embeddings, filter values or DSNs.

Ingestion rejects document bodies above 16 MiB, titles/sources above 4 KiB, or
more than 512 chunks and encodes in batches of at most 32 before opening the
write transaction. These are safety boundaries, not throughput claims; size
and batch settings are intentionally not tuned by the synthetic benchmark.

The included synthetic runner validates mechanics and local performance only:

```bash
.venv/bin/python benchmarks/run_retrieval_v2.py \
  --protocol validation \
  --write-validation-lock /tmp/rag3d-v2-validation-lock.json
```

A test run must use the frozen configuration and dataset identity:

```bash
.venv/bin/python benchmarks/run_retrieval_v2.py \
  --protocol test \
  --validation-lock /tmp/rag3d-v2-validation-lock.json
```

## Promotion gate

Recommend V2 only when all tests pass, security and rollback checks pass, no
primary quality metric regresses more than 2%, and the result is reproducible
on the target corpus and infrastructure. Synthetic calibration results do not
establish a general quality claim.
