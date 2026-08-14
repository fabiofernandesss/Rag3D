# Fontes da Retrieval Engine V2

Consulta realizada em 2026-08-13. Resultados de outros sistemas são tratados como hipóteses; nenhuma prática entra no RAG3D sem teste local, métrica, critério de aceite e rollback.

## pgvector

- **Título/projeto:** pgvector README e changelog oficiais.
- **Versão:** extensão 0.8.6, release estável de 2026-07-29; cliente Python 0.5.0. O backend deve confirmar `pg_extension.extversion` em runtime.
- **Fonte:** https://github.com/pgvector/pgvector e https://github.com/pgvector/pgvector/blob/master/CHANGELOG.md
- **Problema:** busca vetorial exata e aproximada no PostgreSQL, HNSW, filtros e iterative scans.
- **Regra extraída:** exact é ground truth; HNSW precisa de curva recall-latência; filtros ANN são pós-scan e podem exigir `iterative_scan`, índice B-tree, índice parcial ou particionamento. `hnsw.ef_search` e demais GUCs de consulta devem usar `SET LOCAL`.
- **Limitações:** defaults não são ótimos universais; plano e comportamento mudam por versão/dados; índice criado não garante uso.
- **Teste no RAG3D:** exact versus HNSW, quatro seletividades, `EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)`, tamanho/build/RSS, restart e persistência.
- **Hipótese/aceite/rollback:** HNSW reduz p95 sem perda >2% de qualidade; aceitar somente ponto Pareto com Recall@20 ANN >=0,98 contra exact; rollback para `search_mode=exact` ou backend anterior.

## HNSW

- **Título/autores:** *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs*, Yu. A. Malkov e D. A. Yashunin.
- **Versão/data:** arXiv:1603.09320; submissão inicial 2016, publicação TPAMI.
- **Fonte:** https://arxiv.org/abs/1603.09320
- **Problema:** ANN baseado em grafo hierárquico com hierarquia de escalas.
- **Regra extraída:** medir `M`, `efConstruction` e `ef` conjuntamente; maior conectividade/ef tende a trocar memória/build/latência por recall.
- **Limitações:** datasets/hardware do paper não determinam os parâmetros do RAG3D.
- **Teste no RAG3D:** grid de build `m={8,16,32}`, `ef_construction={64,128,256}` e busca `ef_search={20,40,80,160,320,640,1000}`, apenas calibration/validation antes do test.
- **Hipótese/aceite/rollback:** existe ponto não dominado versus exact; promover somente após gate global; remover índice/usar exact para rollback.

## Reciprocal Rank Fusion

- **Título/autores:** *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*, Gordon V. Cormack, Charles L. A. Clarke e Stefan Büttcher.
- **Versão/data:** SIGIR 2009.
- **Fonte:** https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf
- **Problema:** fundir rankings heterogêneos sem somar scores de escalas incompatíveis.
- **Regra extraída:** baseline equal-weight, rank iniciado em 1, ausente contribui zero e `k0=60` congelado inicialmente; pesos/`k0` são calibração, não defaults invisíveis.
- **Limitações:** coleções de 2009 e foco em MAP; não prova superioridade universal nem no RAG3D.
- **Teste no RAG3D:** dense, sparse, RRF equal-weight, weighted RRF, quantum e ablação estrutural no mesmo pool; sensibilidade de `k0` apenas em validation.
- **Hipótese/aceite/rollback:** RRF não degrada primárias >2% e oferece baseline determinístico; rollback por `RAG3D_RETRIEVAL_PIPELINE=legacy` ou `RAG3D_FUSION=quantum`.

## BM25

- **Título/autores:** *The Probabilistic Relevance Framework: BM25 and Beyond*, Stephen Robertson e Hugo Zaragoza.
- **Versão/data:** Foundations and Trends in IR 3(4), 2009.
- **Fonte:** https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf
- **Problema:** retrieval lexical com saturação de TF e normalização por comprimento.
- **Regra extraída:** não chamar a soma `learned_sparse × IDF` existente de BM25; uma baseline BM25 real precisa registrar tokenizer, `k1`, `b`, `dl` e `avgdl`.
- **Limitações:** hipóteses probabilísticas simplificadoras e parâmetros dependentes de coleção.
- **Teste no RAG3D:** comparar dense, sparse BGE nativo, sparse+IDF atual e BM25 real apenas se implementado.
- **Hipótese/aceite/rollback:** sparse complementa dense em termos raros; capability/config retorna à função lexical anterior.

## BGE-M3

- **Título/autores:** *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation*, Jianlv Chen et al.
- **Versão/data:** arXiv:2402.03216, consultado na revisão vigente em 2026-08-13.
- **Fonte:** https://arxiv.org/abs/2402.03216 e documentação oficial https://bge-model.com/bge/bge_m3.html
- **Problema:** dense, learned sparse e multi-vector/late interaction num encoder multilíngue.
- **Regra extraída:** tratar as três funções separadamente; fixar modelo/revision/dimensões/normalização no fingerprint; multi-vector pode reranquear candidatos, mas tem custo maior.
- **Limitações:** resultados do modelo não transferem para a projeção 1024→128, truncamento, int8/LSH ou hardware do RAG3D.
- **Teste no RAG3D:** ablação por função, batch encoding, línguas distintas e fingerprint/fallback explícito.
- **Hipótese/aceite/rollback:** BGE melhora qualidade versus Hash sem violar limites operacionais; rollback para índice Hash separado e fallback somente quando autorizado.

## ColBERTv2

- **Título/autores:** *ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction*, Keshav Santhanam et al.
- **Versão/data:** arXiv:2112.01488, 2021.
- **Fonte:** https://arxiv.org/abs/2112.01488
- **Problema:** qualidade e custo de representações multi-vetoriais/late interaction.
- **Regra extraída:** MaxSim é um estágio de interação tardia; custo de armazenamento e candidate generation precisam ser explícitos.
- **Limitações:** compressão residual/treinamento ColBERTv2 não existem no RAG3D; não atribuir seus ganhos à projeção/assinatura local.
- **Teste no RAG3D:** `Recall_union`, `Recall_after_structural`, nDCG/MRR, `structural_ms`, memória por chunk e ablação float versus binário.
- **Hipótese/aceite/rollback:** structural melhora ordem dentro do pool sem regressão >2%; desativar com `RAG3D_STRUCTURAL_RERANK=false`.

## Sentence Transformers CrossEncoder

- **Título/projeto:** Sentence Transformers, documentação oficial de CrossEncoder e pacote oficial.
- **Versão/data:** 5.7.0 para Python >=3.10, consultada em 2026-08-14; a linha Python 3.9 permanece isolada no extra compatível 3.4.x.
- **Fonte:** https://www.sbert.net/docs/cross_encoder/usage/usage.html e https://pypi.org/project/sentence-transformers/
- **Problema:** reranking supervisionado de pares consulta-documento após geração de candidatos.
- **Regra extraída:** carregar como extra opcional, limitar texto/pool, preservar a ordem em indisponibilidade ou erro e nunca supor que uma saída válida melhora qualidade.
- **Limitações:** custo cresce com o pool e depende do modelo/domínio; suporte da interface não demonstra ganho no RAG3D.
- **Teste no RAG3D:** parser/ordem/falha/bounds unitários; qualidade e latência somente em validation/test congelados com o modelo identificado.
- **Hipótese/aceite/rollback:** o reranker melhora nDCG/MRR sem exceder o gate de p95; aceitar apenas com IC pareado, rollback por `RAG3D_RERANKER=none`.

## BEIR

- **Título/autores:** *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*, Nandan Thakur et al.
- **Versão/data:** arXiv:2104.08663 / NeurIPS 2021 Datasets and Benchmarks.
- **Fonte:** https://arxiv.org/abs/2104.08663 e https://github.com/beir-cellar/beir
- **Problema:** avaliação heterogênea zero-shot e trade-off entre BM25, dense, sparse, late interaction e reranking.
- **Regra extraída:** resultado sintético ou de uma coleção não é claim global; reportar datasets individualmente. Reranking/late interaction pode custar mais.
- **Limitações:** julgamentos incompletos, viés de pooling e cobertura linguística/domain diferente do RAG3D.
- **Teste no RAG3D:** corpus integral/qrels congelados; proibir seleção de corpus após ler qrels para claim de qualidade; nDCG@10, Recall@20 e MRR por query.
- **Hipótese/aceite/rollback:** V2 generaliza sem regressão primária >2%; se a escala não couber, publicar somente teste funcional/performance, sem claim.

## Greedy DPP

- **Título/autores:** *Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity*, Laming Chen, Guoxin Zhang e Eric Zhou.
- **Versão/data:** NeurIPS 2018.
- **Fonte:** https://proceedings.neurips.cc/paper/2018/hash/dbbf603ff0e99629dda5d75b6f75f966-Abstract.html
- **Problema:** acelerar inferência gulosa de DPP com atualização incremental de Cholesky.
- **Regra extraída:** DPP MAP global é difícil/NP-hard; o algoritmo rápido implementa greedy, não ótimo global. Usar float64, kernel PSD/jitter e testar duplicatas/vetores ausentes.
- **Limitações:** aplicação original e datasets não são RAG; melhor diversidade pode sacrificar relevância.
- **Teste no RAG3D:** ranking puro, MMR e DPP em nDCG/Recall/cobertura/duplicidade/latência; validar estabilidade e custo observado.
- **Hipótese/aceite/rollback:** DPP/MMR reduzem duplicidade ou aumentam cobertura sem queda primária >2%; `diversity_method=none` é rollback-identidade.

## MMR

- **Título/autores:** *Summarization: (1) Using MMR for Diversity-Based Reranking and (2) Evaluating Summaries*, Jade Goldstein e Jaime Carbonell.
- **Versão/data:** TIPSTER 1998.
- **Fonte:** https://aclanthology.org/X98-1025/
- **Problema:** equilibrar relevância de consulta e novidade/redundância.
- **Regra extraída:** MMR é baseline simples e explícito para comparar com DPP.
- **Limitações:** resultados originais são preliminares/anteriores a embeddings modernos; `lambda` depende da tarefa.
- **Teste no RAG3D:** grade de `lambda` somente em validation; mesma matriz de similaridade/candidate pool do DPP.
- **Hipótese/aceite/rollback:** MMR oferece trade-off competitivo com menor custo; `none` reverte exatamente a ordem.

## Random-hyperplane LSH

- **Título/autor:** *Similarity Estimation Techniques from Rounding Algorithms*, Moses Charikar.
- **Versão/data:** STOC 2002.
- **Fonte:** https://www.cs.princeton.edu/courses/archive/spr04/cos598B/bib/CharikarEstim.pdf
- **Problema:** estimar similaridade angular por sinais de hiperplanos aleatórios.
- **Regra extraída:** validar `P[bit diferente]=acos(cos(x,y))/pi`, independência efetiva, distribuição de Hamming e probabilidade real do banding; documentar que somente 128/1024 bits formam bandas hoje.
- **Limitações:** hiperplanos pseudoaleatórios/projeções finitas e banding local podem violar aproximações independentes.
- **Teste no RAG3D:** simulação por faixa de cosseno, recall de prefilter, fallback/full scan, multi-probe e erro/Recall após int8.
- **Hipótese/aceite/rollback:** prefilter reduz custo mantendo recall; full scan holográfico ou pgvector exact é rollback.

## ARES

- **Título/autores:** *ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems*, Jon Saad-Falcon et al.
- **Versão/data:** NAACL 2024 / arXiv:2311.09476.
- **Fonte:** https://aclanthology.org/2024.naacl-long.20/
- **Problema:** avaliar relevância do contexto, fidelidade e relevância da resposta com judges e prediction-powered inference.
- **Regra extraída:** ARES é complementar e requer anotações humanas in-domain; não substitui qrels/Recall/MRR/nDCG.
- **Limitações:** custo de modelo/dados, drift de domínio/língua e erro do judge.
- **Teste no RAG3D:** somente com conjunto humano separado e IC PPI; fora do quality gate primário desta entrega se esses dados não existirem.
- **Hipótese/aceite/rollback:** judge correlaciona com humanos; sem essa validação, remover o claim e manter métricas IR.

## PostgreSQL

- **Título/projeto:** documentação oficial PostgreSQL, `EXPLAIN`, particionamento, transações e locking.
- **Versão/data:** PostgreSQL 16/17/18, consultada em 2026-08-13; testes devem registrar a versão real.
- **Fonte:** https://www.postgresql.org/docs/current/sql-explain.html, https://www.postgresql.org/docs/current/ddl-partitioning.html e https://www.postgresql.org/docs/current/transaction-iso.html
- **Problema:** verificar plano, buffers, seletividade, atomicidade e alternativas físicas a filtros ANN.
- **Regra extraída:** `EXPLAIN ANALYZE` executa a consulta e tem overhead; usar fora das amostras de latência. Transações devem envolver escrita por documento; particionamento só com filtro estável e benefício medido.
- **Limitações:** planner/estatísticas/cache alteram o plano; forçar `enable_seqscan=off` serve apenas para diagnóstico.
- **Teste no RAG3D:** planos naturais, warm/cold identificados, rollback de falha no meio da ingestão, filtros e restart.
- **Hipótese/aceite/rollback:** schema/índices escolhidos melhoram estágio sem dano global; migrations são aditivas e a seleção de backend faz rollback.

## Significância e ferramenta de referência

- **Título/autores/projeto:** *A Comparison of Statistical Significance Tests for Information Retrieval Evaluation*, Mark Smucker, James Allan e Ben Carterette; NIST `trec_eval`.
- **Versão/data:** CIKM 2007; `trec_eval` 9.0.7 disponível na consulta.
- **Fonte:** https://ciir-publications.cs.umass.edu/getpdf.php?id=744 e https://trec.nist.gov/trec_eval/
- **Problema:** pareamento por query e implementação correta de métricas.
- **Regra extraída:** bootstrap/randomização devem preservar pares e usar a mesma métrica reportada; fixtures próprias devem ser cruzadas com referência.
- **Limitações:** significância não corrige leakage, múltiplas comparações nem dataset inadequado.
- **Teste no RAG3D:** bootstrap pareado com seed/número de amostras exportados; sistema idêntico produz delta zero/IC contendo zero; qrels/run pequenos cruzados com referência.
- **Hipótese/aceite/rollback:** IC sustenta o delta observado; se não, resultado é inconclusivo e nenhuma meta é declarada.
