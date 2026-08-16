# pgvector backend

The pgvector adapter is optional and Python-only. SQLite and the existing
PostgreSQL holographic backend do not import or require pgvector.

## Install and provision

```bash
pip install -e '.[pgvector]'
```

An administrator must install the server extension separately:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
SELECT extversion FROM pg_extension WHERE extname = 'vector';
```

The runtime validates the extension and never executes `CREATE EXTENSION`.
For local development, the pinned profile binds only to loopback and requires
an explicit password:

```bash
export RAG3D_PGVECTOR_PASSWORD='choose-a-local-password'
docker compose --profile pgvector up -d pgvector
```

Apply the additive, idempotent schema as an operator. Dimensions must match the
encoder fingerprint:

```bash
export RAG3D_PG='postgresql://postgres:PASSWORD@127.0.0.1:5434/rag3d'
psql "$RAG3D_PG" \
  -v dense_dim=1024 \
  -v structural_dim=128 \
  -f migrations/pgvector/001_retrieval_v2.sql
```

The migration takes a transaction-scoped advisory lock, validates inputs and
catalog state, and touches only `rag3d_v2_*`. It fails on incompatible
dimensions instead of rewriting stored metadata.

## Configure

```bash
export RAG3D_BACKEND=pgvector
export RAG3D_RETRIEVAL_PIPELINE=v2
export RAG3D_FUSION=rrf
export RAG3D_ENCODER=bge-m3
export RAG3D_ALLOW_ENCODER_FALLBACK=false
export RAG3D_PGVECTOR_SEARCH_MODE=exact
export RAG3D_PGVECTOR_STATEMENT_TIMEOUT_MS=5000
```

Search modes are explicit:

- `exact` is the safe default and ANN ground truth;
- `ann` requires a compatible HNSW index and fails if the natural plan does
  not use it;
- `auto` may select ANN only when the adapter verifies the index/plan, otherwise
  it uses exact search.

Cosine distance uses `<=>` with `vector_cosine_ops`; returned similarity is
`1 - distance`. This remains meaningful when a caller fails to provide a
perfectly normalized vector, while normalized vectors retain the expected
cosine ordering.

## Build and measure HNSW

HNSW is not created by the base migration. Build it deliberately after loading
the corpus so `m`, `ef_construction` and build cost are recorded. At query time,
measure `ef_search`, iterative scan mode, `max_scan_tuples` and
`scan_mem_multiplier` against exact top-k on the same filters.

Do not treat successful SQL execution as proof that ANN was used. Inspect the
adapter's `EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)` result and record plan node,
rows removed by filter, buffers and warm/cold latency. The `ann` mode is
fail-closed when the natural planner chooses a sequential scan.

Filtered evaluation should cover no filter and low, medium and high
selectivity. Compare auxiliary B-tree indexes and iterative scan before
considering partial indexes or partitioning; those designs are corpus-specific.

## Operations and rollback

`health()` returns a bounded `ok`/`error` object and never returns the DSN.
Fingerprint mismatch is a reindex or rollback event, not permission to update
metadata in place. To leave this backend without deleting data:

```bash
export RAG3D_RETRIEVAL_PIPELINE=legacy
export RAG3D_BACKEND=postgres-holo   # or sqlite
export RAG3D_FUSION=quantum
```

See the [rollback guide](retrieval-v2-rollback.md) for the full procedure.
