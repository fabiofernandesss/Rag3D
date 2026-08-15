# Retrieval V2 rollback

The code-path switch is configuration-only, but data-plane rollback is safe
only when the legacy index has been kept current. This change does **not**
dual-write V2 and legacy backends. The holographic migration adds validated
foreign keys; they remain after traffic rollback and are not removed by these
environment changes.

Before switching traffic, compare an application-owned ingestion watermark or
document count/checksum and run a known query against the legacy index. If the
legacy index is stale, backfill or reindex it first; changing environment
variables does not copy pgvector data into SQLite or `holo_*`.

For local SQLite:

```bash
export RAG3D_RETRIEVAL_PIPELINE=legacy
export RAG3D_BACKEND=sqlite
export RAG3D_FUSION=quantum
```

For the existing PostgreSQL holographic backend:

```bash
export RAG3D_RETRIEVAL_PIPELINE=legacy
export RAG3D_BACKEND=postgres-holo
export RAG3D_FUSION=quantum
export RAG3D_PG='postgresql://USER:PASSWORD@HOST:PORT/DATABASE'
```

Restart the application processes after changing their environment. Validate
`health()`, the ingestion watermark and one known retrieval query. The legacy path ignores the
additive `rag3d_v2_*` pgvector tables, so rollback does not require dropping
them or rebuilding `holo_*`.

Python can return to the legacy pipeline on the same holographic database only
when its full stored V2 fingerprint matches the requested configuration.
Node/Java intentionally reject any V2-certified holographic database because
they cannot validate all V2 chunking/pipeline fields. For cross-language
rollback, export the source documents and build a separate legacy Hash index,
then validate its watermark and a known query before switching traffic. Never
delete or rewrite V2 fingerprint keys to make a port accept an index.

Do not delete V2 data during an incident. If removal is later required, take a
database backup and use a separately reviewed administrative migration; no
destructive rollback migration is part of this change.

If rollback is caused by an encoder/index mismatch, keep fallback disabled and
either return to the matching index configuration or build a new index. Never
rewrite the stored fingerprint to bypass the check.
