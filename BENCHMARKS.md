# RAG3D — Benchmarks e Ablação (honesto)

> **Resumo:** neste momento **não há evidência de que a fusão quântica supere o
> RRF**. No harness sintético embutido, todos os métodos de fusão empatam. Esta
> página existe para dizer isso com clareza e mostrar como medir de verdade.

Isto responde à crítica legítima de que "as alegações de superioridade precisam
de evidência experimental". Elas precisam mesmo. Abaixo está o que já dá para
medir com o que vem no repositório, o que ele mostra, e o que ainda falta.

---

## 1. Ablação de fusão (dado real, reproduzível)

Rode você mesmo:

```bash
python3 tests/bench_fusion.py 1800
```

Corpus sintético multilíngue (PT/EN/ZH), encoder **fallback** (hashing de
n-gramas — o encoder fraco embutido), 120 consultas com documento-alvo conhecido.

| estratégia          | Recall@5 | MRR   |
|---------------------|:--------:|:-----:|
| eixo semântico      |  83.3%   | 0.833 |
| eixo léxico         |  83.3%   | 0.750 |
| eixo estrutural     |  83.3%   | 0.833 |
| CombSUM (λ=0)       |  83.3%   | 0.833 |
| **quântica (λ=1)**  |  83.3%   | 0.833 |
| RRF (k=60)          |  83.3%   | 0.833 |

**Leitura honesta:**

- **quântica == RRF == CombSUM.** A fusão por interferência **não vence** o RRF
  aqui. `λ=0` (sem interferência) dá o mesmo resultado que `λ=1` — coerente com
  a garantia de que `λ=0` colapsa no clássico.
- O único sinal real: o eixo léxico tem MRR menor (0.750); a fusão recupera para
  o nível do melhor eixo (0.833). Isso mostra que **fundir ≥ pior eixo**, não que
  fundir com física quântica seja melhor que fundir com RRF.
- **Por que empatam:** o corpus sintético é fácil — cada "agulha" é lexicalmente
  única, então todo método acha o alvo. Um benchmark que não separa os métodos
  não prova nada sobre qual é melhor. Este é justamente o ponto da crítica.

**Conclusão prática:** use `--rrf` (RRF k=60) como padrão de produção. A fusão
quântica é oferecida como opção falsificável (`interference_strength`), não como
alegação de superioridade. Meça nos SEUS dados antes de ligá-la.

---

## 2. O que este número NÃO prova

- Não é benchmark público (BEIR/MIRACL/MS MARCO/LoTTE).
- Usa o encoder de hashing (léxico fraco), não o BGE-M3 (semântico real). Sem
  semântica de verdade, o eixo "semântico" não diverge do léxico de forma
  interessante — e é justamente na divergência entre eixos que a fusão importaria.
- Sem curvas recall × latência contra pgvector / FAISS / Qdrant.
- Sem ablação com dados onde os eixos discordam (distratores adversariais).
- Sem repetição em vários datasets nem intervalos de confiança.

Tudo isso está em aberto. Nenhuma dessas lacunas é escondida.

---

## 3. Como medir de verdade (roteiro)

Para uma avaliação que a comunidade aceite:

1. **Encoder real:** trocar o fallback pelo BGE-M3 (`pip install FlagEmbedding`,
   `RAG3D_ENCODER=bge-m3`). Só aí o eixo semântico vira semântico de verdade.
2. **Base pública:** rodar BEIR (nDCG@10) e MIRACL (multilíngue). Ingerir o
   corpus, usar as queries/qrels oficiais.
3. **Ablação completa** por dataset:
   - só léxico · só semântico · só estrutural
   - CombSUM · RRF · quântica (λ ∈ {0, 0.5, 1})
   - com/sem costura de contiguidade · com/sem reranker
4. **Baseline forte:** comparar com pgvector (HNSW) e FAISS nos mesmos dados —
   recall e latência.
5. **Estatística:** repetir, reportar média ± IC, teste de significância par a par.

**Hipótese honesta a testar:** o valor do RAG3D provavelmente está na
**engenharia** (3 eixos + Postgres sem pgvector + paridade Python/JS + defesas
de grounding), não em a fusão quântica bater o RRF. O experimento acima diria
se essa hipótese se sustenta.

---

## 4. O que já é medível e verdadeiro hoje

Estas alegações são testadas pelas suítes do repositório (`tests/`), não são
retóricas:

- **Paridade Python/JS bit a bit:** `Hamming = 0/1024` — assinatura holográfica
  idêntica entre as duas linguagens (`tests/xlang_parity.mjs`).
- **Postgres sem pgvector funciona:** ingestão, 3 eixos, fusão, memória e
  persistência entre conexões, tudo em Postgres puro (`tests/test_pg.py`).
- **`λ=0` = fusão clássica:** garantido por teste (`test_quantum_fusion_math`).
- **Agnóstico de língua:** PT/EN/中文/árabe passam nos testes de texto.
- **Latência local:** ~11 ms/consulta e ingestão de doc de 200 páginas em ~0.7 s
  (medido, corpus local — não é benchmark de recall).

A diferença: estes são fatos de **funcionamento e engenharia**, verificáveis. As
alegações de **qualidade de recuperação superior** continuam por provar.
