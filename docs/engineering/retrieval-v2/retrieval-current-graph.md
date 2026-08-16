# Grafo do retrieval atual

```mermaid
flowchart TD
    Q[consulta] --> EXP{query expansion habilitada e LLM disponível?}
    EXP -->|não| ENCODE[encoder.encode]
    EXP -->|sim| VAR[variações via LLM]
    VAR --> ENCODE
    ENCODE --> DENSE[dense_search global]
    ENCODE --> SPARSE[sparse_search global]
    DENSE --> UNION[união de ids]
    SPARSE --> UNION
    UNION --> STRUCT[structural MaxSim sobre candidatos]
    DENSE --> CHANNELS[três rankings]
    SPARSE --> CHANNELS
    STRUCT --> CHANNELS
    CHANNELS --> FUSE[quantum ou RRF]
    FUSE --> HYDRATE[hidratação em lote]
    HYDRATE --> PARENT[small-to-big / pai]
    PARENT --> RR{reranker LLM disponível?}
    RR -->|sim| LISTWISE[reordenação listwise]
    RR -->|não| DIV
    LISTWISE --> DIV{diversidade > 0?}
    DIV -->|sim| DPP[greedy-DPP]
    DIV -->|não| CUT[corte top-k]
    DPP --> STITCH[costura opcional]
    CUT --> STITCH
    STITCH --> RESULT[TriResult + três views]
```

O nó estrutural é um **late-interaction reranker**: suas consultas SQL/SQLite usam apenas ids da união. A view “estrutural” é independente como apresentação, mas não como geração global de candidatos.

## Decomposição a instrumentar

```text
T_total = T_normalize + T_expand + T_encode + T_dense + T_sparse
        + T_union + T_fusion + T_structural + T_rerank
        + T_diversity + T_hydrate + T_stitch + T_reader
```

O código-base não mede esses termos separadamente. Qualquer paralelismo dense/sparse deve comparar `max(T_dense,T_sparse)+overhead` com a soma sequencial e respeitar conexões/thread-safety.
