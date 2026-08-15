// Fusão dos três eixos — o novo cálculo (interferência quântica).
// Espelha trirag/fusion.py.
//
//   a_c(d) = sqrt(w_c * s_c(d)) * e^(i*phi_c(d))
//   P(d)   = |sum_c a_c(d)|^2
//          = clássico (CombSUM) + interferência entre canais
//
// Canais que concordam no rank interferem construtivamente; discordam,
// destrutivamente. interferenceStrength=0 colapsa no clássico. RRF de base.

function minmax(ranking) {
  const m = new Map();
  if (!ranking.length) return m;
  const vals = ranking.map(([, s]) => s);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  if (hi - lo < 1e-12) {
    for (const [cid] of ranking) m.set(cid, 1.0);
  } else {
    // Scale before subtracting so opposite finite extremes do not overflow.
    const scale = Math.max(Math.abs(lo), Math.abs(hi), 1.0);
    const loScaled = lo / scale;
    const span = hi / scale - loScaled;
    for (const [cid, s] of ranking) m.set(cid, (s / scale - loScaled) / span);
  }
  return m;
}

function phases(ranking) {
  const m = new Map();
  const n = ranking.length;
  if (n <= 1) {
    for (const [cid] of ranking) m.set(cid, 0.0);
    return m;
  }
  ranking.forEach(([cid], i) => m.set(cid, (Math.PI * i) / (n - 1)));
  return m;
}

// Coerência do canal — quão DECIDIDO ele está nesta consulta (pureza Tr(rho²)
// da distribuição de pontuações, reescalada para [0,1]). Espelha fusion.py.
// EXPERIMENTAL — desligado por padrão. Tr(rho²) é a pureza e determina
// H₂=-log Tr(rho²); não é a própria entropia. Decisão de pontuação prediz mal QUAL canal
// está certo; e canal com poucos candidatos parece trivialmente decidido —
// por isso o piso de 3 candidatos. Ver fusion.py para a nota completa.
export function coherence(normMap) {
  const vals = [...normMap.values()].filter((s) => s > 0);
  const n = vals.length;
  if (n <= 2) return 0.0;
  const total = vals.reduce((a, b) => a + b, 0);
  if (total <= 0) return 0.0;
  let purity = 0;
  for (const v of vals) { const p = v / total; purity += p * p; }
  return Math.max(0, Math.min(1, (purity * n - 1) / (n - 1)));
}

function coherentWeights(base, norm, strength) {
  if (strength <= 0) return base;
  const kappa = {}; let kmax = 0;
  for (const name of Object.keys(norm)) { kappa[name] = coherence(norm[name]); kmax = Math.max(kmax, kappa[name]); }
  if (kmax <= 0) return base;
  const out = {};
  for (const name of Object.keys(norm)) {
    out[name] = (base[name] ?? 1.0) * ((1 - strength) + strength * (kappa[name] / kmax));
  }
  return out;
}

export function quantumFuse(channels, weights, topK, interferenceStrength = 1.0, coherenceStrength = 0.0) {
  const names = Object.keys(channels);
  const norm = {}, phase = {};
  for (const name of names) {
    norm[name] = minmax(channels[name]);
    phase[name] = phases(channels[name]);
  }
  weights = coherentWeights(weights, norm, coherenceStrength);
  const allIds = new Set();
  for (const name of names) for (const [cid] of channels[name]) allIds.add(cid);

  const hits = [];
  for (const cid of allIds) {
    const amps = []; // [amplitude, fase]
    const perChannel = {};
    const chans = [];
    for (const name of names) {
      const s = norm[name].get(cid);
      if (s === undefined || s <= 0) continue;
      perChannel[name] = s;
      amps.push([Math.sqrt((weights[name] ?? 1.0) * s), phase[name].get(cid)]);
      chans.push(name);
    }
    let classical = 0;
    for (const [a] of amps) classical += a * a;
    let interf = 0;
    for (let i = 0; i < amps.length; i++)
      for (let j = i + 1; j < amps.length; j++)
        interf += 2.0 * amps[i][0] * amps[j][0] * Math.cos(amps[i][1] - amps[j][1]);
    hits.push({
      chunkId: cid,
      score: classical + interferenceStrength * interf,
      classical,
      interference: interf,
      channels: chans,
      perChannel,
    });
  }
  // desempate determinístico por id (idêntico ao Python) — sets de JS e Python
  // iteram em ordens diferentes, empates precisam de critério estável e igual
  hits.sort((a, b) => b.score - a.score || a.chunkId - b.chunkId);
  return hits.slice(0, topK);
}

export function rrfFuse(channels, weights, topK, rrfK = 60) {
  const scores = new Map(), found = new Map();
  for (const name of Object.keys(channels)) {
    const w = weights[name] ?? 1.0;
    const seen = new Set();
    channels[name].forEach(([cid], rank) => {
      if (seen.has(cid)) return;
      seen.add(cid);
      scores.set(cid, (scores.get(cid) || 0) + w / (rrfK + rank + 1));
      found.set(cid, [...(found.get(cid) || []), name]);
    });
  }
  return [...scores.entries()]
    .sort((a, b) => b[1] - a[1] || a[0] - b[0])
    .slice(0, topK)
    .map(([cid, s]) => ({
      chunkId: cid, score: s, classical: s, interference: 0, channels: found.get(cid),
      perChannel: {},
    }));
}

export function fuse(channels, weightsArr, topK, method = "quantum", interferenceStrength = 1.0, rrfK = 60, coherenceStrength = 0.0) {
  const names = Object.keys(channels);
  let w = {};
  names.forEach((n, i) => (w[n] = weightsArr[i]));
  if (coherenceStrength > 0) {
    const norm = {};
    for (const n of names) norm[n] = minmax(channels[n]);
    w = coherentWeights(w, norm, coherenceStrength);
  }
  return method === "rrf"
    ? rrfFuse(channels, w, topK, rrfK)
    : quantumFuse(channels, w, topK, interferenceStrength);
}

// ---------------------------------------------- seleção fermiônica (DPP) ---
//
// A fusão acima é BOSÔNICA: amplitudes somam, concordância amplifica — decide
// QUAIS documentos são relevantes. Falta decidir QUAL CONJUNTO devolver.
//
// Um conjunto de k documentos é um estado de k partículas. Se for
// ANTISSIMÉTRICO (determinante de Slater), a amplitude é
//     |psi_S|² = det(Gram(v_S)) = Vol²(v_1..v_k)
// Dois documentos idênticos = duas partículas no mesmo estado: o determinante
// zera (EXCLUSÃO DE PAULI). Redundância proibida por construção.
//
// É um Determinantal Point Process (Kulesza & Taskar); o guloso aproxima o
// objetivo log-det com Cholesky incremental, sem resolver o MAP global.
// O custo inclui similaridades O(NkD), além das atualizações O(Nk²).
// bit a bit (mesmo float64, mesma ordem de operações).
export function fermionicSelect(items, vectors, topK, diversity = 0.0) {
  const ids = items.map(([i]) => i);
  const n = ids.length;
  if (diversity <= 0 || n <= 1 || topK <= 0) return ids.slice(0, topK);
  const k = Math.min(topK, n);
  const eps = 1e-12;

  const rel = items.map(([, s]) => s);
  const lo = Math.min(...rel), hi = Math.max(...rel);
  const logQ2 = rel.map((s) => 2.0 * Math.log(Math.max(hi - lo > eps ? (s - lo) / (hi - lo) : 1.0, eps)));

  let dim = 0;
  for (const v of vectors.values()) if (v) { dim = v.length; break; }
  const V = ids.map((cid) => {
    const v = vectors.get(cid);
    return v && v.length === dim ? v : new Float64Array(dim);
  });

  const theta = Math.max(0, Math.min(1, 1.0 - diversity));
  const d2 = new Float64Array(n).fill(1.0);          // volume residual
  const C = Array.from({ length: n }, () => new Float64Array(k));
  const avail = new Array(n).fill(true);
  const chosen = [];

  for (let t = 0; t < k; t++) {
    let best = -Infinity, j = -1;
    for (let i = 0; i < n; i++) {
      if (!avail[i]) continue;
      const g = theta * logQ2[i] + (1 - theta) * Math.log(Math.max(d2[i], eps));
      if (g > best) { best = g; j = i; }
    }
    if (j < 0) break;
    // sem volume restante: completa por relevância pura (Cholesky instável)
    if (d2[j] < eps) {
      for (let i = 0; i < n && chosen.length < k; i++) if (avail[i]) chosen.push(ids[i]);
      return chosen.slice(0, k);
    }
    chosen.push(ids[j]);
    avail[j] = false;
    if (chosen.length === k) break;
    const sj = Math.sqrt(Math.max(d2[j], eps));
    for (let i = 0; i < n; i++) {
      if (!avail[i]) continue;
      let dot = 0;
      for (let d = 0; d < dim; d++) dot += V[i][d] * V[j][d];   // similaridade i·j
      let proj = 0;
      for (let s = 0; s < t; s++) proj += C[i][s] * C[j][s];
      const e = (dot - proj) / sj;
      C[i][t] = e;
      d2[i] = Math.max(d2[i] - e * e, 0);
    }
  }
  return chosen;
}
