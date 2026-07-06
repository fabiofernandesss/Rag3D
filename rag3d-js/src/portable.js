// Núcleo determinístico portável — bit a bit igual ao trirag/portable.py.
//
// Mesmos hiperplanos LSH, mesma assinatura, mesmo crc32 que a versão Python.
// Um Holograma Textual salvo por Python é idêntico ao salvo por JS: retrieval
// cruzado no mesmo Postgres funciona. É o que faz o TriRAG valer para todas as
// linguagens de verdade.
//
// splitmix64 em BigInt (64 bits exatos); Box-Muller para normais; crc32 IEEE.

const MASK64 = (1n << 64n) - 1n;
const GOLDEN = 0x9e3779b97f4a7c15n;
const C1 = 0xbf58476d1ce4e5b9n;
const C2 = 0x94d049bb133111ebn;

// splitmix64 counter-based: idêntico ao sequencial (estado_k = seed + k*GOLDEN)
function splitmix64(seed, k) {
  let z = (seed + k * GOLDEN) & MASK64;
  z = ((z ^ (z >> 30n)) * C1) & MASK64;
  z = ((z ^ (z >> 27n)) * C2) & MASK64;
  z = (z ^ (z >> 31n)) & MASK64;
  return z;
}

// n floats em [0,1) — top 53 bits (igual ao Python)
export function uniforms(seed, n) {
  const s = BigInt(seed) & MASK64;
  const out = new Float64Array(n);
  const denom = Number(1n << 53n);
  for (let i = 0; i < n; i++) {
    const z = splitmix64(s, BigInt(i + 1));
    out[i] = Number(z >> 11n) / denom;
  }
  return out;
}

// n normais padrão via Box-Muller (ramo do cosseno), idêntico ao Python
export function normals(seed, n) {
  const u = uniforms(seed, 2 * n);
  const out = new Float32Array(n);
  for (let j = 0; j < n; j++) {
    const u1 = Math.max(u[2 * j], 1e-12);
    const u2 = u[2 * j + 1];
    out[j] = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  }
  return out;
}

// matriz (rows x cols) achatada em ordem C (row-major) — mesma de numpy
export function hyperplanes(seed, rows, cols) {
  return normals(seed, rows * cols); // Float32Array de rows*cols; índice i*cols+j
}

// ---------------------------------------------------------------- crc32 ---

let CRC_TABLE = null;
function crcTable() {
  if (CRC_TABLE) return CRC_TABLE;
  CRC_TABLE = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    CRC_TABLE[n] = c >>> 0;
  }
  return CRC_TABLE;
}

// CRC-32 IEEE — mesmo resultado de zlib.crc32 e de portable.crc32 (Python)
export function crc32(bytes) {
  const tbl = crcTable();
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = tbl[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

// crc32 dos bytes UTF-8 de uma string, SEM alocar (TextEncoder por chamada é
// caro no hot path do encoder). Resultado idêntico a crc32(utf8_bytes(s)).
export function crc32str(s) {
  const tbl = crcTable();
  let c = 0xffffffff;
  for (let i = 0; i < s.length; i++) {
    let cp = s.charCodeAt(i);
    if (cp >= 0xd800 && cp <= 0xdbff && i + 1 < s.length) {
      const lo = s.charCodeAt(i + 1);
      if (lo >= 0xdc00 && lo <= 0xdfff) { cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00); i++; }
    }
    if (cp < 0x80) {
      c = tbl[(c ^ cp) & 0xff] ^ (c >>> 8);
    } else if (cp < 0x800) {
      c = tbl[(c ^ (0xc0 | (cp >> 6))) & 0xff] ^ (c >>> 8);
      c = tbl[(c ^ (0x80 | (cp & 0x3f))) & 0xff] ^ (c >>> 8);
    } else if (cp < 0x10000) {
      c = tbl[(c ^ (0xe0 | (cp >> 12))) & 0xff] ^ (c >>> 8);
      c = tbl[(c ^ (0x80 | ((cp >> 6) & 0x3f))) & 0xff] ^ (c >>> 8);
      c = tbl[(c ^ (0x80 | (cp & 0x3f))) & 0xff] ^ (c >>> 8);
    } else {
      c = tbl[(c ^ (0xf0 | (cp >> 18))) & 0xff] ^ (c >>> 8);
      c = tbl[(c ^ (0x80 | ((cp >> 12) & 0x3f))) & 0xff] ^ (c >>> 8);
      c = tbl[(c ^ (0x80 | ((cp >> 6) & 0x3f))) & 0xff] ^ (c >>> 8);
      c = tbl[(c ^ (0x80 | (cp & 0x3f))) & 0xff] ^ (c >>> 8);
    }
  }
  return (c ^ 0xffffffff) >>> 0;
}
