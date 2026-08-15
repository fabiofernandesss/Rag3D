// Holograma Textual — espelha trirag/holo.py.
//
// Projeta o TriVec em assinatura LSH (BIT), facetas (INT[]), eco int8 (BYTEA) e
// constelação de tokens (BYTEA). Os hiperplanos vêm do portable.js, idênticos
// aos do Python — então as assinaturas batem entre linguagens.

import { hyperplanes } from "./portable.js";

export const HOLO_BITS = 1024;
export const TOKEN_BITS = 128;
export const BAND_BITS = 8;
export const N_BANDS = 16;
const SEED = 7742;

// round-half-to-even (banker's), igual ao np.round — para o eco int8 casar
function roundHalfEven(x) {
  const r = Math.round(x);
  if (Math.abs(x - Math.trunc(x)) === 0.5) {
    // Math.round arredonda .5 pra cima; ajusta para o par mais próximo
    const f = Math.floor(x);
    return f % 2 === 0 ? f : f + 1;
  }
  return r;
}

// popcount SWAR de 32 bits (correto p/ qualquer padrão int32, inclui negativo)
function popcount32(n) {
  n = n - ((n >> 1) & 0x55555555);
  n = (n & 0x33333333) + ((n >> 2) & 0x33333333);
  return (((n + (n >> 4)) & 0x0f0f0f0f) * 0x01010101) >> 24;
}

// constelação (bytes) -> Uint32Array, 4 palavras por token de 128 bits
function toU32(bytes) {
  const n = bytes.length >> 2;
  const u = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    const o = i << 2;
    u[i] = ((bytes[o] << 24) | (bytes[o + 1] << 16) | (bytes[o + 2] << 8) | bytes[o + 3]) >>> 0;
  }
  return u;
}

export class Holographer {
  constructor(denseDim, colbertDim) {
    this.denseDim = denseDim;
    this.colbertDim = colbertDim;
    this.planes = hyperplanes(SEED, denseDim, HOLO_BITS); // Float32Array [d*BITS + bit]
    this.tokPlanes = hyperplanes(SEED ^ 0x5d, colbertDim, TOKEN_BITS);
  }

  // vetor denso -> Uint8Array(128) (1024 bits LSH), packbits big-endian
  signDense(vec) {
    const bits = this._project(vec, this.planes, this.denseDim, HOLO_BITS);
    return packbits(bits);
  }

  // bits do projetor (>=0) na ordem [0..OUT). Itera só dims não-zero do vetor
  // (o encoder por hashing é esparso) e por linha contígua dos planos — muito
  // mais rápido em JS, e a soma por bit fica na MESMA ordem, então os bits são
  // idênticos ao laço ingênuo (e à versão Python/BLAS).
  _project(vec, planes, dim, out) {
    const acc = new Float64Array(out);
    for (let d = 0; d < dim; d++) {
      const v = vec[d];
      if (v === 0) continue;
      const base = d * out;
      for (let b = 0; b < out; b++) acc[b] += v * planes[base + b];
    }
    const bits = new Uint8Array(out);
    for (let b = 0; b < out; b++) bits[b] = acc[b] >= 0 ? 1 : 0;
    return bits;
  }

  // 128 bytes -> '0101...' (1024) para coluna BIT(1024) do Postgres
  static sigToBitstring(sig) {
    let s = "";
    for (let i = 0; i < sig.length; i++) s += sig[i].toString(2).padStart(8, "0");
    return s;
  }

  // facetas: banda i = (i<<8)|byte_i (o índice entra no valor, sem colisão)
  static bandsOf(sig) {
    const out = [];
    for (let i = 0; i < N_BANDS; i++) out.push((i << 8) | sig[i]);
    return out;
  }

  // vetor normalizado -> int8 (eco para re-pontuação aproximada) — Int8Array bytes
  static quantize(vec) {
    const out = new Int8Array(vec.length);
    for (let i = 0; i < vec.length; i++) {
      let q = roundHalfEven(vec[i] * 127.0);
      out[i] = Math.max(-127, Math.min(127, q));
    }
    return new Uint8Array(out.buffer, out.byteOffset, out.byteLength);
  }

  static dequantize(bytes) {
    const i8 = new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const out = new Float32Array(i8.length);
    for (let i = 0; i < i8.length; i++) out[i] = i8[i] / 127.0;
    return out;
  }

  // tokens (Float32Array[]) -> Uint8Array(T*16), cada linha 128 bits
  signTokens(tokens) {
    const nb = TOKEN_BITS / 8;
    const out = new Uint8Array(tokens.length * nb);
    for (let t = 0; t < tokens.length; t++) {
      const bits = this._project(tokens[t], this.tokPlanes, this.colbertDim, TOKEN_BITS);
      const packed = packbits(bits);
      out.set(packed, t * nb);
    }
    return out;
  }

  // MaxSim binário: para cada token da consulta, menor Hamming no doc.
  // 4 palavras uint32 por token + popcount SWAR (idêntico ao laço de bytes).
  binaryMaxsim(qConst, dConst) {
    const q = toU32(qConst);
    const d = toU32(dConst);
    const Tq = q.length >> 2;
    const Td = d.length >> 2;
    if (Tq === 0 || Td === 0) return 0.0;
    let sum = 0;
    for (let i = 0; i < Tq; i++) {
      const i4 = i << 2, a0 = q[i4], a1 = q[i4 + 1], a2 = q[i4 + 2], a3 = q[i4 + 3];
      let best = TOKEN_BITS;
      for (let j = 0; j < Td; j++) {
        const j4 = j << 2;
        const ham = popcount32(a0 ^ d[j4]) + popcount32(a1 ^ d[j4 + 1])
          + popcount32(a2 ^ d[j4 + 2]) + popcount32(a3 ^ d[j4 + 3]);
        if (ham < best) best = ham;
      }
      sum += 1.0 - (2.0 * best) / TOKEN_BITS;
    }
    return sum / Tq;
  }

  hamming(a, b) {
    let h = 0;
    for (let i = 0; i < a.length; i++) h += popcount32(a[i] ^ b[i]);
    return h;
  }
}

// empacota bits (0/1) em bytes, big-endian (bit 0 -> MSB), igual np.packbits
function packbits(bits) {
  const out = new Uint8Array(Math.ceil(bits.length / 8));
  for (let i = 0; i < bits.length; i++) {
    if (bits[i]) out[i >> 3] |= 1 << (7 - (i & 7));
  }
  return out;
}

export { packbits };
