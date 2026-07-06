// Processamento de texto agnóstico de língua — espelha trirag/textproc.py.
//
// Precisa produzir os MESMOS tokens/n-gramas que a versão Python para que os
// hologramas batam entre linguagens. Iteração por code point (Array.from) para
// casar com o fatiamento por code point do Python.

// Faixas CJK (han, ext-A, hiragana, katakana, hangul, compat) — mesmas do Python
const CJK = "[\\u4e00-\\u9fff\\u3400-\\u4dbf\\u3040-\\u309f\\u30a0-\\u30ff\\uac00-\\ud7af\\uf900-\\ufaff]";
const CJK_RE = new RegExp(CJK, "gu");
const CJK_ONE = new RegExp("^" + CJK + "$", "u");
// \w Unicode do Python ~= letras + números + underscore
const WORD_RE = /[\p{L}\p{N}_]+/gu;

// pontuação de fim de sentença multi-escrita (mesma lista do Python).
// . ! ? são literais dentro de [] — sem escape (inválido no modo unicode).
const SENT_END = ".!?。！？؟۔।॥։።၊။…";
const SENT_RE = new RegExp(`[^${SENT_END}\\n]+[${SENT_END}]*\\s*`, "gu");

// sentinela p/ proteger pontos internos de número/seção (13.243, 5.1.2)
const SENT_GUARD = String.fromCharCode(0xe000);

export function normalize(text) {
  text = text.normalize("NFKC");
  // conserta números partidos pela extração de PDF: "13. 243" -> "13.243".
  // [0-9] e (?![0-9]) explícitos (não \d/\b) — \d é ASCII no JS e Unicode no
  // Python; a forma explícita é idêntica nas duas linguagens (mantém paridade).
  text = text.replace(/([0-9])\.\s+([0-9]{3})(?![0-9])/g, "$1.$2");
  text = text.replace(/[ \t]+/g, " ");
  text = text.replace(/\n{3,}/g, "\n\n");
  return text.trim();
}

function codepoints(s) {
  return Array.from(s);
}

export function estimateTokens(text) {
  if (!text) return 0;
  const cjk = (text.match(CJK_RE) || []).length;
  const rest = codepoints(text).length - cjk;
  return cjk + Math.max(1, Math.floor(rest / 4));
}

export function splitSentences(text) {
  // não quebrar sentença no ponto entre dígitos (13.243, 5.1.2, R$ 1.000).
  // [0-9] explícito (não \d) — paridade JS/Python.
  text = text.replace(/(?<=[0-9])\.(?=[0-9])/g, SENT_GUARD);
  const restore = (s) => s.split(SENT_GUARD).join(".");
  const out = [];
  for (const paraRaw of text.split("\n\n")) {
    const para = paraRaw.trim();
    if (!para) continue;
    let matched = false;
    for (const m of para.matchAll(SENT_RE)) {
      const s = m[0].trim();
      if (s) {
        out.push(restore(s));
        matched = true;
      }
    }
    if (!matched && para) out.push(restore(para));
  }
  if (out.length) return out;
  const t = restore(text).trim();
  return t ? [t] : [];
}

// tokens lexicais: palavras minúsculas; CJK vira caracteres + bigramas
export function wordTokens(text, cjkBigrams = true) {
  text = text.toLowerCase();
  const toks = [];
  for (const m of text.matchAll(WORD_RE)) {
    const w = m[0];
    const parts = w.split(CJK_RE);
    const cjks = w.match(CJK_RE) || [];
    for (const p of parts) if (p) toks.push(p);
    if (cjks.length) {
      for (const c of cjks) toks.push(c);
      if (cjkBigrams && cjks.length > 1) {
        for (let i = 0; i + 1 < cjks.length; i++) toks.push(cjks[i] + cjks[i + 1]);
      }
    }
  }
  return toks;
}

// n-gramas de caracteres — base do encoder fallback, qualquer escrita
export function charNgrams(text, nLo = 3, nHi = 5) {
  text = " " + text.toLowerCase().replace(/\s+/g, " ") + " ";
  const cp = codepoints(text);
  const out = [];
  const L = cp.length;
  for (let n = nLo; n <= nHi; n++) {
    for (let i = 0; i + n <= L; i++) out.push(cp.slice(i, i + n).join(""));
  }
  return out;
}

export { CJK_ONE };
