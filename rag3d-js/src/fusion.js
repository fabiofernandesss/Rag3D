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
    for (const [cid, s] of ranking) m.set(cid, (s - lo) / (hi - lo));
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

export function quantumFuse(channels, weights, topK, interferenceStrength = 1.0) {
  const names = Object.keys(channels);
  const norm = {}, phase = {};
  for (const name of names) {
    norm[name] = minmax(channels[name]);
    phase[name] = phases(channels[name]);
  }
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
    channels[name].forEach(([cid], rank) => {
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

export function fuse(channels, weightsArr, topK, method = "quantum", interferenceStrength = 1.0, rrfK = 60) {
  const names = Object.keys(channels);
  const w = {};
  names.forEach((n, i) => (w[n] = weightsArr[i]));
  return method === "rrf"
    ? rrfFuse(channels, w, topK, rrfK)
    : quantumFuse(channels, w, topK, interferenceStrength);
}
