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

## 1b. Seleção fermiônica (MAP-DPP) — aqui SIM há ganho medido

A fusão decide *quais* documentos são relevantes, mas nada decidia *qual
conjunto* devolver — e o top-k enchia de quase-duplicatas (sobreposição de
chunks e costura produzem near-dupes), deixando os outros fatos de fora.

A seleção fermiônica escolhe o conjunto maximizando **relevância × volume**
(determinante de Slater / DPP): dois trechos quase idênticos são duas
partículas no mesmo estado — o determinante zera e um deles é excluído.

```bash
python3 tests/bench_coverage.py 20 8
```

Corpus: 20 tópicos × (3 fatos distintos + 8 duplicatas quase idênticas do
fato 1). **Cobertura@6** = fração dos 3 fatos presentes no top-6 (se o fato
não entra no contexto, a IA não tem como acertar).

| configuração | cobertura@6 | rank-1 útil |
|---|:--:|:--:|
| ranking puro (diversidade=0) | 46.7% | 100% |
| **fermiônica 0.3** | **100%** | 100% |
| **fermiônica 0.5** | **100%** | 100% |
| fermiônica 0.7 | 100% | 100% |
| RRF puro | 48.3% | 100% |
| **RRF + fermiônica 0.5** | **100%** | 100% |

**+53 pontos de cobertura, sem perder o topo** (rank-1 continua 100%
relevante). Vale igual para a fusão quântica e para o RRF — é ortogonal ao
método de fusão.

**Sem regressão** onde não há redundância (benchmark de agulha, 900 docs):

| | Recall@5 | MRR |
|---|:--:|:--:|
| diversidade = 0 | 83.3% | 0.833 |
| fermiônica 0.35 | 83.3% | 0.833 |
| fermiônica 0.5 | 83.3% | 0.833 |

Por isso está **ligada por padrão** (`diversity = 0.35`); `diversity = 0`
reproduz exatamente o comportamento anterior.

**Honestidade sobre este número:** o benchmark foi desenhado para medir
redundância — é o modo de falha que a técnica ataca. Ele não diz que o RAG3D
recupera *melhor* em geral; diz que, quando o corpus tem trechos repetidos
(o caso comum com overlap/costura), o conjunto entregue à IA cobre muito mais
fatos distintos. Referência: Chen et al., NeurIPS 2018 (guloso exato O(k²N)
com Cholesky incremental); Kulesza & Taskar (DPP).

### Pesos por coerência — implementado, mas DESLIGADO

Também implementei modular o peso de cada eixo pela "pureza" Tr(ρ²) da sua
distribuição de pontuações (canal indeciso pesaria menos). A pesquisa
desaconselha ligar por padrão, e concordo:

1. Tr(ρ²) numa ρ diagonal **é exatamente a entropia de Rényi-2** — "pureza" é
   entropia com outro nome, não um conceito novo.
2. A literatura de QPP mostra correlação ~0.09 para prever *qual ranker está
   certo* nesta consulta (prevê bem dificuldade, mal acerto).
3. Um canal pode estar **decidido e errado**.
4. Canal com poucos candidatos parece trivialmente decidido (mitigado com um
   piso de 3 candidatos, mas o viés não some).

Fica disponível como `coherence_strength` para experimentação, **default 0**.
No benchmark de cobertura não mudou nada (100% com e sem).

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
