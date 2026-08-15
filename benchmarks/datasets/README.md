# Retrieval V2 benchmark datasets

`retrieval_v2_synthetic_v1.json` freezes the local mechanism-check dataset at
seed `20260813`. The runner first creates and hash-orders the corpus from
document IDs, then materializes query labels. Relevance judgments are never an
input to corpus selection. Reduced runs are therefore smoke subsets, not the
oracle subset implemented by the historical `tests/beir_ablation.py --max_docs`.

The calibration, validation, and test query IDs/templates are disjoint and
versioned in the manifest. Generated label-bearing contents are assigned to an
exclusive split by `content_key % 3`, so qrels cannot cross split boundaries.
Synthetic scores only validate metric mathematics, pipeline behavior, stage
accounting, and local performance; they are not BEIR results and do not support
a general quality claim. Metric cutoffs apply to the raw ranking: a duplicate
occupies its rank and receives no repeated relevance gain.

Default local comparison (1,000 documents, legacy SQLite versus V2 SQLite):

```bash
python benchmarks/run_retrieval_v2.py
```

Use `--ablations all` to add dense, sparse, RRF, quantum, structural, MMR, and
greedy-DPP variants. PostgreSQL backends require a dedicated empty benchmark
database, `--allow-remote`, `RAG3D_BENCHMARK_ALLOW_WRITE=1`, and a DSN supplied
indirectly through the variable named by `--postgres-dsn-env`; the DSN is never
serialized. The database name in that DSN must contain `test` or `bench`, and
the runner verifies both document and chunk relations are empty before ingest.
The flag is therefore only one part of the write authorization, not an override
for the environment, database-name, or emptiness gates. `pgvector` accepts
`--pgvector-mode exact` or `--pgvector-mode hnsw`. HNSW mode records
`ef_search`, iterative-scan, max-scan and scan-memory parameters, then compares
ANN top-k with exact top-k on the same snapshot and filter. It is reported as
`not_evaluated` unless both searches are full and natural `EXPLAIN` selects the
verified HNSW index. HNSW construction is an explicit, durable database change
and its build time is recorded separately from ingest.

The default protocol is `calibration`. A `test` run forbids ablations and
requires a validation-origin lock. Before any test qrel is materialized, the
runner validates its schema and SHA-256 digests for the full configuration,
metric/dedup policy, seeds, manifest/generator, evaluator/runner, every local
`rag3d/**/*.py` runtime source, repository diff/state and backend parameters.
Test reports bind the canonical hash and identity of the validated lock.
Reports include per-query rankings and timings but never query text, document
text, embeddings, filter values, or connection strings. The single pre-warmup
cold observation, serial success-rate sample, shared-process RSS/CPU and storage
density are descriptive; the default runner marks them ineligible for a
cross-system performance claim.
