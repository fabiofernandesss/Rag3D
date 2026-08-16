# Grafo de riscos

```mermaid
flowchart LR
  F[filters seletivos] -->|pode degradar| H[recall HNSW]
  H -->|é medido por| AR[ANN recall vs exact]
  POOL[pool pré-structural] -->|pode degradar| SR[recall structural/final]
  SR -->|é medido por| ST[recall por estágio]
  Q[quantização int8] -->|pode degradar| RANK[ordenação/PSD]
  RANK -->|é protegido por| NORM[normalização + fallback identidade]
  DPP[greedy-DPP] -->|pode degradar| REL[nDCG/MRR]
  DPP -->|pode falhar por| NUM[duplicatas/NaN/kernel indefinido]
  NUM -->|é protegido por| J[float64 + jitter + validação]
  RR[reranker LLM] -->|pode degradar| REL
  RR -->|é revertido por| NOOP[reranker=none]
  FB[fallback de encoder] -->|pode corromper| FP[index compatibility]
  FP -->|é protegido por| DIG[fingerprint canônico]
  DSN[DSN/segredo] -->|pode vazar por| LOG[stats/erro/log]
  LOG -->|é protegido por| RED[redação + testes negativos]
  TX[autocommit composto] -->|pode gravar| PART[estado parcial]
  PART -->|é protegido por| TR[transação por documento]
  BEIR[qrels de test] -->|pode enviesar| CLAIM[claim de qualidade]
  CLAIM -->|é protegido por| SPLIT[calibration/validation/test lock]
  X[holo_* / meta encoder] -->|pode quebrar| PORTS[Node/Java]
  PORTS -->|é protegido por| PAR[golden parity + schema intacto]
```

| Risco | Probabilidade | Impacto | Evidência atual | Gate | Rollback |
| --- | --- | --- | --- | --- | --- |
| HNSW com filtro devolve menos que `k` | alta | alta | filtro ocorre durante scan aproximado | quatro seletividades + exact | exact ou holo/SQLite |
| fingerprint insuficiente mistura índices | alta | alta | chave atual tem só três campos | mismatch tests | legacy/reindex |
| fallback silencioso troca BGE por Hash | alta | alta | factory captura exceções amplas | flag explícita | Hash explícito ou legacy |
| transação PostgreSQL parcial | alta | alta | autocommit por statement | fault injection | adapter anterior |
| DPP instável/claim MAP incorreto | média | alta | eco int8 pode gerar kernel indefinido | PSD/jitter/property tests | `none`/MMR |
| DSN em stats/erro | alta | alta | API atual devolve valor cru | redaction tests | saída sanitizada |
| tuning/leakage no test | alta | alta | `--max_docs` lê qrels antes do corpus | config/checksum lock | sem claim |
| regressão cross-language | média | alta | schema/meta compartilhados | Node+Java golden | legacy/holo |
| threads sem ganho | média | média | teto Amdahl observado 3,69% | não implementar | sequencial |
| benchmark sintético generalizado | alta | alta | scripts históricos são controlados | claim scope explícito | meta não atingida |
