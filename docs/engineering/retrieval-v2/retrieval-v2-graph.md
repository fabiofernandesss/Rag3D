# Grafo da Retrieval Engine V2

O grafo alvo é `G=(V,E)`. Nós representam configuração, estágio, backend,
métrica e invariante; arestas nomeiam a relação executável. RRF recebe somente
rankings de recuperadores independentes. O canal estrutural é uma interação
tardia limitada e não pode introduzir um documento ausente da união.

```mermaid
flowchart LR
  Q[consulta] -->|é normalizada por| N[normalização]
  N -->|pode chamar| X[expansão opcional]
  X -->|é codificada por| E[encoder + fingerprint]
  E -->|chama| D[dense retrieval]
  E -->|chama| S[sparse retrieval]
  D -->|contribui ranking| U[união limitada]
  S -->|contribui ranking| U
  D -->|é fundido por rank| R[RRF equal-weight]
  S -->|é fundido por rank| R
  U -->|limita candidatos de| T[structural late-interaction rerank]
  R -->|fornece ordem-base| T
  T -->|pode ser reordenado por| RR[reranker opcional]
  RR -->|é diversificado por| V[none / MMR / greedy-DPP]
  V -->|é hidratado por| H[backend]
  H -->|expande para| B[small-to-big]
  B -->|é costurado por| C[contiguidade]
  C -->|produz| CTX[contexto]
  CTX -->|é consumido por| Reader[reader opcional]

  CFG[RAG3D_RETRIEVAL_PIPELINE=v2] -->|seleciona| R
  CFG -->|consulta| CAP[BackendCapabilities]
  CAP -->|protege chamada| D
  CAP -->|protege chamada| S
  CAP -->|protege chamada| T
  FP[IndexFingerprint] -->|protege| H
  DIAG[SearchDiagnostics] -->|mede| N
  DIAG -->|mede| D
  DIAG -->|mede| S
  DIAG -->|mede| R
  DIAG -->|mede| T
  DIAG -->|mede| RR
  DIAG -->|mede| V
  DIAG -->|mede| H
```

## Backends e capabilities

```mermaid
flowchart TB
  P[RetrievalBackend Protocol] -->|implementado por| SQ[SQLite / TriStore]
  P -->|implementado por| PH[PostgreSQL holográfico]
  P -->|implementado por| PV[PostgreSQL + pgvector]
  SQ -->|mantém compatibilidade com| L[pipeline legacy]
  PH -->|mantém compatibilidade com| L
  PH -->|mantém compatibilidade com| X[Node e Java / holo_*]
  PV -->|grava| NS[rag3d_v2_*]
  PV -->|mede ground truth com| EX[exact scan]
  PV -->|pode usar| ANN[HNSW]
  ANN -->|é validado contra| EX
```

Rollback global: `RAG3D_RETRIEVAL_PIPELINE=legacy`. Rollbacks de estágio são
`RAG3D_STRUCTURAL_RERANK=false`, `RAG3D_RERANKER=none` e
`RAG3D_DIVERSITY_METHOD=none`.
