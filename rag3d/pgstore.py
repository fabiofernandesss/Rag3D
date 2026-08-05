"""Backend Postgres PURO — sem pgvector, sem extensão nenhuma.

Cada chunk vira um Holograma Textual (ver holo.py). O que o Postgres guarda
é só o que ele já sabe indexar há 30 anos: BIGINT, INT[], BYTEA, REAL.

  eixo 1 (semântico)  : assinatura em coluna BIT(1024); distância de Hamming
                        = bit_count(sig # consulta) — XOR e popcount NATIVOS
                        do Postgres 14+. Em bases grandes, o pré-filtro por
                        facetas (INT[] + índice GIN, operador &&) corta a
                        varredura. Depois, re-pontuação exata pelo eco int8
                        em Python.
  eixo 2 (léxico)     : tabela invertida comum (term BIGINT, weight REAL)
                        com B-tree — SQL clássico.
  eixo 3 (estrutural) : constelação binária em BYTEA; MaxSim por XOR +
                        popcount sobre os candidatos, em numpy.

Interface idêntica ao TriStore (SQLite): o resto do TriRAG nem percebe.
"""
from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .encoders import TriVec
from .holo import HOLO_BITS, Holographer

_KINDS_SEARCH = "('chunk','turn','summary')"

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS holo_meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS holo_docs(
  id BIGSERIAL PRIMARY KEY, source TEXT, title TEXT,
  created DOUBLE PRECISION, n_tokens INT, meta TEXT
);
CREATE TABLE IF NOT EXISTS holo_grams(
  id BIGSERIAL PRIMARY KEY, doc_id BIGINT, parent_id BIGINT,
  kind TEXT NOT NULL DEFAULT 'chunk',
  pos INT, text TEXT NOT NULL, ctx TEXT,
  n_tokens INT, created DOUBLE PRECISION,
  importance REAL DEFAULT 0.5, turn_no INT, accessed_turn INT,
  sig BIT({HOLO_BITS}),
  bands INT[],
  echo BYTEA,
  constellation BYTEA, n_tok INT
);
CREATE INDEX IF NOT EXISTS idx_grams_kind ON holo_grams(kind);
CREATE INDEX IF NOT EXISTS idx_grams_doc ON holo_grams(doc_id);
CREATE INDEX IF NOT EXISTS idx_grams_bands ON holo_grams USING GIN(bands);
CREATE TABLE IF NOT EXISTS holo_spectrum(
  term BIGINT NOT NULL, gram_id BIGINT NOT NULL, weight REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_spectrum_term ON holo_spectrum(term);
CREATE INDEX IF NOT EXISTS idx_spectrum_gram ON holo_spectrum(gram_id);
"""


class PgHoloStore:
    """Mesma interface do TriStore, sobre Postgres comum."""

    # acima disso, a busca semântica usa o pré-filtro de facetas
    BAND_PREFILTER_THRESHOLD = 20000
    # candidatos re-pontuados pelo eco (multiplicador do k pedido)
    PREFETCH_FACTOR = 4
    PREFETCH_MIN = 200

    def __init__(self, dsn: str, dense_dim: int, colbert_dim: int):
        import psycopg  # dependência só deste backend

        # autocommit=True: leituras não deixam a conexão "idle in transaction"
        # (um `chat` fica parado no prompt sem segurar transação aberta). As
        # escritas compostas usam `with self.db.transaction()` para atomicidade.
        self.db = psycopg.connect(dsn, autocommit=True)
        with self.db.cursor() as cur:
            cur.execute(SCHEMA)
        self.holo = Holographer(dense_dim, colbert_dim)
        self._n_grams_cache: Optional[int] = None

    # ------------------------------------------------------------- meta ---

    def get_meta(self, key: str) -> Optional[str]:
        row = self.db.execute("SELECT value FROM holo_meta WHERE key=%s", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute(
            "INSERT INTO holo_meta VALUES(%s,%s) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
            (key, value),
        )
        self.db.commit()

    # ------------------------------------------------------------ escrita ---

    def add_doc(self, source: str, title: str, n_tokens: int, meta: Optional[dict] = None) -> int:
        row = self.db.execute(
            "INSERT INTO holo_docs(source,title,created,n_tokens,meta) VALUES(%s,%s,%s,%s,%s) RETURNING id",
            (source, title, time.time(), n_tokens, json.dumps(meta or {})),
        ).fetchone()
        return int(row[0])

    def add_chunk(
        self,
        doc_id: Optional[int],
        text: str,
        ctx: str,
        n_tokens: int,
        vec: TriVec,
        kind: str = "chunk",
        pos: int = 0,
        parent_id: Optional[int] = None,
        importance: float = 0.5,
        turn_no: Optional[int] = None,
    ) -> int:
        sig = self.holo.sign_dense(vec.dense)
        bits = self.holo.sig_to_bitstring(sig)
        bands = self.holo.bands_of(sig)
        echo = self.holo.quantize(vec.dense)
        const = self.holo.sign_tokens(vec.tokens)
        # gram + espectro numa ÚNICA query via CTE: um round-trip por chunk (era
        # 2) e atômica por si só (sem BEGIN/COMMIT avulso).
        terms = list(vec.sparse.keys())
        weights = [vec.sparse[t] for t in terms]
        row = self.db.execute(
            f"WITH g AS ("
            f"  INSERT INTO holo_grams(doc_id,parent_id,kind,pos,text,ctx,n_tokens,created,"
            f"    importance,turn_no,sig,bands,echo,constellation,n_tok) VALUES("
            f"    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::bit({HOLO_BITS}),%s,%s,%s,%s) RETURNING id"
            f"), s AS ("
            f"  INSERT INTO holo_spectrum(term, gram_id, weight)"
            f"  SELECT u.term, (SELECT id FROM g), u.weight"
            f"  FROM unnest(%s::bigint[], %s::real[]) AS u(term, weight)"
            f") SELECT id FROM g",
            (
                doc_id, parent_id, kind, pos, text, ctx, n_tokens, time.time(),
                importance, turn_no, bits, bands, echo, const, int(vec.tokens.shape[0]),
                terms, weights,
            ),
        ).fetchone()
        self._n_grams_cache = None
        return int(row[0])

    def add_parent(self, doc_id: int, text: str, n_tokens: int, pos: int) -> int:
        row = self.db.execute(
            "INSERT INTO holo_grams(doc_id,kind,pos,text,ctx,n_tokens,created)"
            " VALUES(%s,'parent',%s,%s,%s,%s,%s) RETURNING id",
            (doc_id, pos, text, text, n_tokens, time.time()),
        ).fetchone()
        return int(row[0])

    def commit(self) -> None:
        self.db.commit()

    def delete_chunk(self, chunk_id: int) -> None:
        self.db.execute("DELETE FROM holo_grams WHERE id=%s", (chunk_id,))
        self.db.execute("DELETE FROM holo_spectrum WHERE gram_id=%s", (chunk_id,))
        self._n_grams_cache = None

    # ------------------------------------------------------------ leitura ---

    _COLS = "id,doc_id,parent_id,kind,pos,text,ctx,n_tokens,created,importance,turn_no,accessed_turn"

    def get_chunks(self, ids: Sequence[int]) -> List[dict]:
        if not ids:
            return []
        rows = self.db.execute(
            f"SELECT {self._COLS} FROM holo_grams WHERE id = ANY(%s)", (list(ids),)
        ).fetchall()
        cols = self._COLS.split(",")
        by_id = {r[0]: dict(zip(cols, r)) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def dense_vecs(self, ids: Sequence[int]) -> Dict[int, np.ndarray]:
        """Vetores densos (eco int8 desquantizado) — para a seleção fermiônica."""
        if not ids:
            return {}
        rows = self.db.execute(
            "SELECT id, echo FROM holo_grams WHERE id = ANY(%s) AND echo IS NOT NULL", (list(ids),)
        ).fetchall()
        return {int(cid): self.holo.dequantize(bytes(echo)) for cid, echo in rows}

    def touch_access(self, ids: Sequence[int], turn_no: int) -> None:
        if not ids:
            return
        self.db.execute(
            "UPDATE holo_grams SET accessed_turn=%s WHERE id = ANY(%s)", (turn_no, list(ids))
        )
        self.db.commit()

    def corpus_tokens(self, kinds: Sequence[str] = ("chunk", "turn")) -> int:
        row = self.db.execute(
            "SELECT COALESCE(SUM(n_tokens),0) FROM holo_grams WHERE kind = ANY(%s)", (list(kinds),)
        ).fetchone()
        return int(row[0])

    def all_texts(self, kinds: Sequence[str] = ("chunk", "turn")) -> List[dict]:
        rows = self.db.execute(
            "SELECT id,kind,text,created,turn_no FROM holo_grams WHERE kind = ANY(%s) ORDER BY id",
            (list(kinds),),
        ).fetchall()
        return [dict(zip(["id", "kind", "text", "created", "turn_no"], r)) for r in rows]

    def n_chunks(self) -> int:
        if self._n_grams_cache is None:
            row = self.db.execute(
                f"SELECT COUNT(*) FROM holo_grams WHERE kind IN {_KINDS_SEARCH}"
            ).fetchone()
            self._n_grams_cache = int(row[0])
        return self._n_grams_cache

    def last_turn_no(self) -> int:
        row = self.db.execute(
            "SELECT COALESCE(MAX(turn_no),0) FROM holo_grams WHERE kind='turn'"
        ).fetchone()
        return int(row[0])

    def neighbors(self, doc_id, positions):
        """Chunks vizinhos do mesmo doc em posições dadas (costura Rag3D)."""
        if doc_id is None or not positions:
            return []
        rows = self.db.execute(
            "SELECT pos, text FROM holo_grams WHERE doc_id=%s AND kind='chunk' AND pos = ANY(%s) ORDER BY pos",
            (doc_id, list(positions)),
        ).fetchall()
        return [{"pos": p, "text": t} for p, t in rows]

    # ------------------------------------------------- eixo 1: semântico ---

    def dense_search(self, qvec: np.ndarray, k: int) -> List[Tuple[int, float]]:
        sig = self.holo.sign_dense(qvec)
        bits = self.holo.sig_to_bitstring(sig)
        prefetch = max(self.PREFETCH_FACTOR * k, self.PREFETCH_MIN)
        ham_expr = f"bit_count(sig # %s::bit({HOLO_BITS}))"

        def scan(use_bands: bool):
            where = f"kind IN {_KINDS_SEARCH} AND sig IS NOT NULL"
            params: list = [bits]
            if use_bands:
                # cast explícito: psycopg adapta int pequeno como smallint[],
                # mas a coluna bands é int[] (senão o operador && não casa)
                where += " AND bands && %s::int[]"
                params.append(self.holo.bands_of(sig))
            return self.db.execute(
                f"SELECT id, echo, ({ham_expr}) AS ham FROM holo_grams"
                f" WHERE {where} ORDER BY ham ASC LIMIT %s",
                (*params, prefetch),
            ).fetchall()

        big = self.n_chunks() > self.BAND_PREFILTER_THRESHOLD
        rows = scan(use_bands=big)
        # o pré-filtro de facetas (LSH banding) pode descartar vizinhos reais
        # para similaridade moderada; se ele devolver poucos candidatos, cai
        # para o full-scan Hamming (que ainda usa só popcount nativo).
        if big and len(rows) < prefetch // 2:
            rows = scan(use_bands=False)
        if not rows:
            return []

        # re-pontuação exata pelo eco int8 (cosseno real, não Hamming).
        # concatena os bytes de todos os ecos e faz um frombuffer/reshape só,
        # em vez de dequantize+stack por linha (menos alocação).
        q = qvec.astype(np.float32)
        dim = q.shape[0]
        blob = b"".join(bytes(r[1]) for r in rows)
        echoes = np.frombuffer(blob, dtype=np.int8).reshape(len(rows), dim).astype(np.float32) / 127.0
        with np.errstate(all="ignore"):  # flags FPE espúrias do Accelerate/macOS
            sims = echoes @ q
        # desempate por id (asc) — argsort é instável; JS ordena estável. Mesmo
        # critério nas duas linguagens para o corte top-k coincidir no empate.
        ids = [int(r[0]) for r in rows]
        order = sorted(range(len(rows)), key=lambda i: (-float(sims[i]), ids[i]))[:k]
        return [(ids[i], float(sims[i])) for i in order]

    # --------------------------------------------------- eixo 2: léxico ----

    def sparse_search(self, qsparse: Dict[int, float], k: int) -> List[Tuple[int, float]]:
        if not qsparse:
            return []
        terms = list(qsparse.keys())
        n_docs = max(1, self.n_chunks())
        # uma query só, SEM join: só grams pesquisáveis têm postings (parents
        # não recebem spectrum), então o join+filtro de kind era redundante.
        # df por termo via window COUNT(*) OVER (PARTITION BY term).
        rows = self.db.execute(
            "SELECT term, gram_id, weight, COUNT(*) OVER (PARTITION BY term) AS df"
            " FROM holo_spectrum WHERE term = ANY(%s)",
            (terms,),
        ).fetchall()
        scores: Dict[int, float] = {}
        for term, gid, dw, df in rows:
            t = int(term)
            idf = float(np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)))
            scores[gid] = scores.get(gid, 0.0) + qsparse[t] * float(dw) * idf
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
        return [(int(g), float(s)) for g, s in ranked]

    # ----------------------------------------------- eixo 3: estrutural ----

    def colbert_scores(self, qtokens: np.ndarray, candidate_ids: Sequence[int]) -> List[Tuple[int, float]]:
        if len(candidate_ids) == 0 or qtokens.size == 0:
            return []
        q_const = self.holo.sign_tokens(qtokens)
        rows = self.db.execute(
            "SELECT id, constellation FROM holo_grams WHERE id = ANY(%s) AND constellation IS NOT NULL",
            (list(candidate_ids),),
        ).fetchall()
        out = [(int(gid), self.holo.binary_maxsim(q_const, bytes(const))) for gid, const in rows]
        out.sort(key=lambda x: (-x[1], x[0]))  # desempate por id, igual ao JS
        return out
