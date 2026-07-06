"""Armazenamento tridimensional em SQLite + numpy.

Cada chunk é salvo nas três formas na hora da ingestão:
  dvecs    -> vetor denso (BLOB float32)
  postings -> termos esparsos invertidos (term -> chunk, peso)
  colvecs  -> matriz token a token (BLOB float16)

Sem servidor, sem dependência externa: um arquivo .db + cache em RAM.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .encoders import TriVec

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS docs(
  id INTEGER PRIMARY KEY, source TEXT, title TEXT,
  created REAL, n_tokens INTEGER, meta TEXT
);
CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY, doc_id INTEGER, parent_id INTEGER,
  kind TEXT NOT NULL DEFAULT 'chunk',   -- chunk | parent | summary | turn | rolling_summary
  pos INTEGER, text TEXT NOT NULL, ctx TEXT,
  n_tokens INTEGER, created REAL,
  importance REAL DEFAULT 0.5, turn_no INTEGER,
  accessed_turn INTEGER                 -- último turno em que a memória foi usada
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_kind ON chunks(kind);
CREATE TABLE IF NOT EXISTS dvecs(chunk_id INTEGER PRIMARY KEY, data BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS postings(term INTEGER NOT NULL, chunk_id INTEGER NOT NULL, weight REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_postings_term ON postings(term);
CREATE INDEX IF NOT EXISTS idx_postings_chunk ON postings(chunk_id);
CREATE TABLE IF NOT EXISTS colvecs(chunk_id INTEGER PRIMARY KEY, n_tok INTEGER, dim INTEGER, data BLOB NOT NULL);
"""


class TriStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        # migração: bases criadas antes da coluna accessed_turn
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(chunks)")}
        if "accessed_turn" not in cols:
            self.db.execute("ALTER TABLE chunks ADD COLUMN accessed_turn INTEGER")
            self.db.commit()
        self._dense_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (ids, matriz)
        self._dense_cache_ver: int = -1  # PRAGMA data_version do cache atual

    # ------------------------------------------------------------- meta ---

    def get_meta(self, key: str) -> Optional[str]:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (key, value))
        self.db.commit()

    # ------------------------------------------------------------ escrita ---

    def add_doc(self, source: str, title: str, n_tokens: int, meta: Optional[dict] = None) -> int:
        cur = self.db.execute(
            "INSERT INTO docs(source,title,created,n_tokens,meta) VALUES(?,?,?,?,?)",
            (source, title, time.time(), n_tokens, json.dumps(meta or {})),
        )
        return int(cur.lastrowid)

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
        cur = self.db.execute(
            "INSERT INTO chunks(doc_id,parent_id,kind,pos,text,ctx,n_tokens,created,importance,turn_no)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (doc_id, parent_id, kind, pos, text, ctx, n_tokens, time.time(), importance, turn_no),
        )
        cid = int(cur.lastrowid)
        self.db.execute(
            "INSERT INTO dvecs VALUES(?,?)", (cid, vec.dense.astype(np.float32).tobytes())
        )
        self.db.executemany(
            "INSERT INTO postings VALUES(?,?,?)",
            [(t, cid, w) for t, w in vec.sparse.items()],
        )
        col = vec.tokens.astype(np.float16)
        self.db.execute(
            "INSERT INTO colvecs VALUES(?,?,?,?)", (cid, col.shape[0], col.shape[1], col.tobytes())
        )
        self._dense_cache = None
        return cid

    def add_parent(self, doc_id: int, text: str, n_tokens: int, pos: int) -> int:
        """Nó pai (small-to-big): não é indexado, só recuperado por expansão."""
        cur = self.db.execute(
            "INSERT INTO chunks(doc_id,kind,pos,text,ctx,n_tokens,created) VALUES(?,?,?,?,?,?,?)",
            (doc_id, "parent", pos, text, text, n_tokens, time.time()),
        )
        return int(cur.lastrowid)

    def commit(self) -> None:
        self.db.commit()

    def delete_chunk(self, chunk_id: int) -> None:
        for table in ("chunks", "dvecs", "colvecs"):
            self.db.execute(f"DELETE FROM {table} WHERE {'id' if table=='chunks' else 'chunk_id'}=?", (chunk_id,))
        self.db.execute("DELETE FROM postings WHERE chunk_id=?", (chunk_id,))
        self._dense_cache = None

    # ------------------------------------------------------------ leitura ---

    def get_chunks(self, ids: Sequence[int]) -> List[dict]:
        if not ids:
            return []
        q = ",".join("?" * len(ids))
        rows = self.db.execute(
            f"SELECT id,doc_id,parent_id,kind,pos,text,ctx,n_tokens,created,importance,turn_no,accessed_turn"
            f" FROM chunks WHERE id IN ({q})",
            list(ids),
        ).fetchall()
        cols = ["id", "doc_id", "parent_id", "kind", "pos", "text", "ctx", "n_tokens", "created", "importance", "turn_no", "accessed_turn"]
        by_id = {r[0]: dict(zip(cols, r)) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def touch_access(self, ids: Sequence[int], turn_no: int) -> None:
        """Recência conta do último USO, não da criação (fórmula de Stanford)."""
        if not ids:
            return
        q = ",".join("?" * len(ids))
        self.db.execute(f"UPDATE chunks SET accessed_turn=? WHERE id IN ({q})", [turn_no] + list(ids))
        self.db.commit()

    def corpus_tokens(self, kinds: Sequence[str] = ("chunk", "turn")) -> int:
        q = ",".join("?" * len(kinds))
        row = self.db.execute(
            f"SELECT COALESCE(SUM(n_tokens),0) FROM chunks WHERE kind IN ({q})", list(kinds)
        ).fetchone()
        return int(row[0])

    def all_texts(self, kinds: Sequence[str] = ("chunk", "turn")) -> List[dict]:
        q = ",".join("?" * len(kinds))
        rows = self.db.execute(
            f"SELECT id,kind,text,created,turn_no FROM chunks WHERE kind IN ({q}) ORDER BY id", list(kinds)
        ).fetchall()
        return [dict(zip(["id", "kind", "text", "created", "turn_no"], r)) for r in rows]

    def n_chunks(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM chunks WHERE kind IN ('chunk','turn','summary')").fetchone()[0])

    def last_turn_no(self) -> int:
        row = self.db.execute("SELECT COALESCE(MAX(turn_no),0) FROM chunks WHERE kind='turn'").fetchone()
        return int(row[0])

    def neighbors(self, doc_id, positions):
        """Chunks vizinhos do mesmo doc em posições dadas (costura Rag3D)."""
        if doc_id is None or not positions:
            return []
        q = ",".join("?" * len(positions))
        rows = self.db.execute(
            f"SELECT pos, text FROM chunks WHERE doc_id=? AND kind='chunk' AND pos IN ({q}) ORDER BY pos",
            [doc_id] + list(positions),
        ).fetchall()
        return [{"pos": r[0], "text": r[1]} for r in rows]

    # ------------------------------------------------- eixo 1: denso -------

    def _dense_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        # PRAGMA data_version muda quando OUTRA conexão grava no arquivo — assim
        # um `chat` de longa duração enxerga um `ingest` feito noutro terminal.
        dv = self.db.execute("PRAGMA data_version").fetchone()[0]
        if dv != self._dense_cache_ver:
            self._dense_cache = None
            self._dense_cache_ver = dv
        if self._dense_cache is None:
            rows = self.db.execute(
                "SELECT d.chunk_id, d.data FROM dvecs d JOIN chunks c ON c.id=d.chunk_id"
                " WHERE c.kind IN ('chunk','turn','summary') ORDER BY d.chunk_id"
            ).fetchall()
            if not rows:
                self._dense_cache = (np.zeros(0, dtype=np.int64), np.zeros((0, 1), dtype=np.float32))
            else:
                ids = np.array([r[0] for r in rows], dtype=np.int64)
                # concatena os BLOBs uma vez e reinterpreta — evita o loop
                # frombuffer+stack por linha (byte-idêntico ao np.stack). O
                # resultado é read-only, mas dense_search só faz `mat @ qvec`.
                buf = b"".join(r[1] for r in rows)
                mat = np.frombuffer(buf, dtype=np.float32).reshape(len(rows), -1)
                self._dense_cache = (ids, mat)
        return self._dense_cache

    def dense_search(self, qvec: np.ndarray, k: int) -> List[Tuple[int, float]]:
        ids, mat = self._dense_matrix()
        if len(ids) == 0:
            return []
        # errstate: numpy+Accelerate no macOS emite flags FPE espúrias em matmul
        # (resultados corretos; ver numpy#25864-família). Valores validados finitos.
        with np.errstate(all="ignore"):
            sims = mat @ qvec.astype(np.float32)
        k = min(k, len(ids))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(ids[i]), float(sims[i])) for i in top]

    # ------------------------------------------------ eixo 2: esparso ------

    def sparse_search(self, qsparse: Dict[int, float], k: int) -> List[Tuple[int, float]]:
        if not qsparse:
            return []
        n_docs = max(1, self.n_chunks())
        scores: Dict[int, float] = {}
        for term, qw in qsparse.items():
            rows = self.db.execute(
                "SELECT p.chunk_id, p.weight FROM postings p JOIN chunks c ON c.id=p.chunk_id"
                " WHERE p.term=? AND c.kind IN ('chunk','turn','summary')",
                (term,),
            ).fetchall()
            df = len(rows)
            if df == 0:
                continue
            idf = float(np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)))
            for cid, dw in rows:
                scores[cid] = scores.get(cid, 0.0) + qw * dw * idf
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return [(int(c), float(s)) for c, s in ranked]

    # ------------------------------------- eixo 3: estrutural (MaxSim) -----

    def colbert_scores(self, qtokens: np.ndarray, candidate_ids: Sequence[int]) -> List[Tuple[int, float]]:
        """MaxSim: para cada token da consulta, o melhor token do chunk; média."""
        if len(candidate_ids) == 0 or qtokens.size == 0:
            return []
        q = ",".join("?" * len(candidate_ids))
        rows = self.db.execute(
            f"SELECT chunk_id,n_tok,dim,data FROM colvecs WHERE chunk_id IN ({q})",
            list(candidate_ids),
        ).fetchall()
        qt = qtokens.astype(np.float32)
        out: List[Tuple[int, float]] = []
        for cid, n_tok, dim, blob in rows:
            dt = np.frombuffer(blob, dtype=np.float16).reshape(n_tok, dim).astype(np.float32)
            with np.errstate(all="ignore"):  # flags FPE espúrias do Accelerate/macOS
                sim = qt @ dt.T                 # (Tq, Td)
            out.append((int(cid), float(sim.max(axis=1).mean())))
        out.sort(key=lambda x: -x[1])
        return out
