// Encoders tridimensionais — espelha o HashEncoder de trirag/encoders.py.
//
// Fallback por hashing (crc32) de n-gramas de caracteres: zero dependências,
// qualquer escrita. Produz as MESMAS três representações que o Python para o
// mesmo texto (crc32 e tokenização idênticos), então os hologramas batem.
//
// dense  : hashing trick sobre n-gramas de caracteres
// sparse : palavras Unicode (CJK vira bigramas), peso 1+ln(tf)
// tokens : um vetor hasheado por palavra (MaxSim aproximado)

import { crc32str } from "./portable.js";
import { charNgrams, wordTokens } from "./textproc.js";

const hash = crc32str; // crc32 dos bytes UTF-8, sem alocar TextEncoder por chamada

function l2normalize(v) {
  let n = 0;
  for (let i = 0; i < v.length; i++) n += v[i] * v[i];
  n = Math.sqrt(n);
  if (n > 0) for (let i = 0; i < v.length; i++) v[i] /= n;
  return v;
}

export class HashEncoder {
  constructor({ denseDim = 1024, colbertDim = 128, maxTokens = 256 } = {}) {
    this.name = "hash";
    this.denseDim = denseDim;
    this.colbertDim = colbertDim;
    this.maxTokens = maxTokens;
  }

  _dense(text) {
    const v = new Float32Array(this.denseDim);
    for (const g of charNgrams(text)) {
      const h = hash(g);
      v[h % this.denseDim] += (h >>> 31) & 1 ? 1.0 : -1.0;
    }
    return l2normalize(v);
  }

  _sparse(text) {
    const tf = new Map();
    for (const w of wordTokens(text)) {
      const tid = hash("w:" + w);
      tf.set(tid, (tf.get(tid) || 0) + 1);
    }
    const out = new Map();
    for (const [t, c] of tf) out.set(t, 1.0 + Math.log(c));
    return out;
  }

  _tokenVec(word) {
    const v = new Float32Array(this.colbertDim);
    for (const g of charNgrams(word, 2, 4)) {
      const h = hash("t:" + g);
      v[h % this.colbertDim] += (h >>> 31) & 1 ? 1.0 : -1.0;
    }
    return l2normalize(v);
  }

  // devolve [{dense: Float32Array, sparse: Map, tokens: Float32Array[]}]
  encode(texts, _isQuery = false) {
    return texts.map((t) => {
      const words = wordTokens(t).slice(0, this.maxTokens);
      const tokens = words.length
        ? words.map((w) => this._tokenVec(w))
        : [new Float32Array(this.colbertDim)];
      return { dense: this._dense(t), sparse: this._sparse(t), tokens };
    });
  }
}

export function makeEncoder(cfg) {
  // JS traz só o fallback portável; BGE-M3 vive no lado Python (mesmos planos)
  return new HashEncoder({
    denseDim: cfg.denseDim,
    colbertDim: cfg.colbertDim,
    maxTokens: cfg.maxColbertTokens,
  });
}
