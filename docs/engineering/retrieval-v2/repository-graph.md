# Grafo do repositório

O grafo usa arestas observadas no código-base; documentação não é usada para inferir chamadas ausentes.

```mermaid
flowchart LR
    API["API Python: Rag3D / TriRag"] -->|instancia| CFG[TriRagConfig]
    API -->|instancia| ENC[BaseEncoder]
    API -->|instancia| ING[Ingestor]
    API -->|instancia| RET[TriRetriever]
    API -->|instancia| MEM[ChatMemory]
    API -->|instancia| READ[Reader]
    CFG -->|seleciona| SQLITE[TriStore / SQLite]
    CFG -->|seleciona por PG| HOLO[PgHoloStore]
    ING -->|grava| SQLITE
    ING -->|grava| HOLO
    RET -->|dense/sparse/structural| SQLITE
    RET -->|dense/sparse/structural| HOLO
    RET -->|chama| FUSION[quantum / RRF]
    RET -->|chama| RERANK[Reranker LLM]
    RET -->|chama| DPP[fermionic_select]
    MEM -->|chama| RET
    MEM -->|lê/grava| SQLITE
    MEM -->|lê/grava| HOLO
    READ -->|lê contexto| MEM
    JS[Port Node] -->|mantém compatibilidade Hash/schema| HOLO_SCHEMA[(holo_*)]
    JAVA[Port Java] -->|mantém compatibilidade Hash/schema| HOLO_SCHEMA
    HOLO -->|grava| HOLO_SCHEMA
    TESTS[Testes/scripts] -->|validam| API
    TESTS -->|validam| FUSION
    BENCH[Benchmarks históricos] -->|medem| RET
```

## Relações sensíveis

| Origem | Aresta | Destino | Risco |
| --- | --- | --- | --- |
| `engine.py` | instancia | store/encoder/retriever | alto: escolha de backend e fingerprint |
| `ingest.py` | grava | schemas SQLite/holo | alto: atomicidade e paridade |
| `retrieve.py` | chama | dense/sparse/structural/fusion | alto: recall e ordem pública |
| `holo.py` | implementa | bytes portáveis | alto: cross-language |
| `fusion.py` | mede/ordena | hits finais | alto: determinismo |
| Node/Java | compartilha | `holo_*` | alto: schema/encoder |
| `memory.py` | depende de | `TriResult` | médio: retrieval-only/contexto |
