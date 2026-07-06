"""Processamento de texto agnóstico de língua.

Nada aqui depende de modelos por idioma: segmentação de sentenças por
pontuação Unicode de múltiplas escritas, tokenização por propriedades
Unicode, e CJK tratado por caractere (com bigramas no eixo léxico).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List

# Pontuação de fim de sentença em várias escritas:
# latim/cirílico/grego (.!?), CJK (。！？), árabe (؟ ۔), devanágari (। ॥),
# armênio (։), etíope (።), birmanês (၊ ။), tailandês usa espaço (coberto por quebra).
_SENT_END = "\\.\\!\\?。！？؟۔।॥։።၊။…"
_SENT_RE = re.compile(rf"[^{_SENT_END}\n]+[{_SENT_END}]*\s*", re.UNICODE)

# Faixas CJK (han, hiragana, katakana, hangul)
_CJK_RE = re.compile(
    "[一-鿿㐀-䶿぀-ゟ゠-ヿ가-힯豈-﫿]"
)
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def normalize(text: str) -> str:
    """NFKC + colapso de espaço. Preserva quebras de parágrafo."""
    text = unicodedata.normalize("NFKC", text)
    # conserta números partidos pela extração de PDF: "13. 243" -> "13.243".
    # [0-9] e (?![0-9]) explícitos (não \d/\b) — \d é Unicode no Python e ASCII
    # no JS; a forma explícita é idêntica nas duas linguagens (mantém paridade).
    text = re.sub(r"([0-9])\.\s+([0-9]{3})(?![0-9])", r"\1.\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def estimate_tokens(text: str) -> int:
    """Estimativa de tokens sem tokenizer: CJK ~1 token/char, resto ~1 token/4 chars."""
    cjk = len(_CJK_RE.findall(text))
    rest = len(text) - cjk
    return cjk + max(1, rest // 4) if text else 0


_SENT_GUARD = chr(0xE000)  # sentinela p/ ponto interno de numero (13.243)


def split_sentences(text: str) -> List[str]:
    """Divide em sentenças por pontuação multi-escrita; parágrafos são limites duros.
    Não quebra no ponto entre dígitos (13.243, 5.1.2) — comum em leis/seções."""
    text = re.sub(r"(?<=[0-9])\.(?=[0-9])", _SENT_GUARD, text)
    restore = lambda s: s.replace(_SENT_GUARD, ".")
    out: List[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        matched = False
        for m in _SENT_RE.finditer(para):
            s = m.group().strip()
            if s:
                out.append(restore(s))
                matched = True
        # parágrafo sem pontuação final (título, item de lista) — mesma lógica
        # do JS (empurra o parágrafo só se nenhuma sentença casou).
        if not matched:
            out.append(restore(para))
    if out:
        return out
    t = restore(text).strip()
    return [t] if t else []


def word_tokens(text: str, cjk_bigrams: bool = True) -> List[str]:
    """Tokens lexicais: palavras Unicode minúsculas; CJK vira caracteres + bigramas.

    Bigramas CJK são a técnica clássica agnóstica de língua para busca em
    chinês/japonês sem segmentador.
    """
    text = text.lower()
    toks: List[str] = []
    pos = 0
    for m in _WORD_RE.finditer(text):
        w = m.group()
        # separa trechos CJK dentro da "palavra"
        parts = _CJK_RE.split(w)
        cjks = _CJK_RE.findall(w)
        for p in parts:
            if p:
                toks.append(p)
        if cjks:
            toks.extend(cjks)
            if cjk_bigrams and len(cjks) > 1:
                toks.extend(a + b for a, b in zip(cjks, cjks[1:]))
        pos = m.end()
    return toks


def char_ngrams(text: str, n_lo: int = 3, n_hi: int = 5) -> Iterable[str]:
    """N-gramas de caracteres — base do encoder fallback, funciona em qualquer escrita."""
    text = " " + re.sub(r"\s+", " ", text.lower()) + " "
    L = len(text)
    for n in range(n_lo, n_hi + 1):
        for i in range(L - n + 1):
            yield text[i : i + n]
