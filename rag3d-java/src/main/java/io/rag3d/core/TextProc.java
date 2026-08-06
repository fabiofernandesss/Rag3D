package io.rag3d.core;

import java.text.Normalizer;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Processamento de texto agnóstico de língua — espelha rag3d/textproc.py e
 * rag3d-js/src/textproc.js.
 *
 * <p>Precisa produzir os MESMOS tokens/n-gramas que as outras linguagens para
 * que os hologramas batam. Iteração por CODE POINT (não char UTF-16) para casar
 * com o fatiamento do Python.
 */
public final class TextProc {

    // Faixas CJK (han, ext-A, hiragana, katakana, hangul, compat)
    private static final String CJK =
            "[\\u4e00-\\u9fff\\u3400-\\u4dbf\\u3040-\\u309f\\u30a0-\\u30ff\\uac00-\\ud7af\\uf900-\\ufaff]";
    private static final Pattern CJK_RE = Pattern.compile(CJK);
    private static final Pattern WORD_RE = Pattern.compile("[\\p{L}\\p{N}_]+", Pattern.UNICODE_CHARACTER_CLASS);

    // pontuação de fim de sentença multi-escrita (mesma lista das outras linguagens)
    private static final String SENT_END = ".!?。！？؟۔।॥։።၊။…";
    private static final Pattern SENT_RE =
            Pattern.compile("[^" + Pattern.quote(SENT_END) + "\\n]+[" + Pattern.quote(SENT_END) + "]*\\s*");

    // conserta número partido pela extração de PDF: "13. 243" -> "13.243"
    private static final Pattern NUM_FIX = Pattern.compile("([0-9])\\.\\s+([0-9]{3})(?![0-9])");
    private static final Pattern SPACES = Pattern.compile("[ \\t]+");
    private static final Pattern BLANKS = Pattern.compile("\\n{3,}");

    // sentinela p/ proteger ponto interno de número/seção (13.243, 5.1.2)
    private static final String SENT_GUARD = "\uE000";
    private static final Pattern INNER_DOT = Pattern.compile("(?<=[0-9])\\.(?=[0-9])");

    private TextProc() {}

    public static String normalize(String text) {
        String s = Normalizer.normalize(text, Normalizer.Form.NFKC);
        s = NUM_FIX.matcher(s).replaceAll("$1.$2");
        s = SPACES.matcher(s).replaceAll(" ");
        s = BLANKS.matcher(s).replaceAll("\n\n");
        return s.trim();
    }

    private static int[] codePoints(String s) {
        return s.codePoints().toArray();
    }

    private static String fromCodePoints(int[] cp, int from, int to) {
        return new String(cp, from, to - from);
    }

    public static int estimateTokens(String text) {
        if (text == null || text.isEmpty()) return 0;
        int cjk = 0;
        Matcher m = CJK_RE.matcher(text);
        while (m.find()) cjk++;
        int rest = codePoints(text).length - cjk;
        return cjk + Math.max(1, rest / 4);
    }

    /** Divide em sentenças; não quebra no ponto entre dígitos (13.243, 5.1.2). */
    public static List<String> splitSentences(String text) {
        String guarded = INNER_DOT.matcher(text).replaceAll(SENT_GUARD);
        List<String> out = new ArrayList<>();
        for (String paraRaw : guarded.split("\n\n")) {
            String para = paraRaw.trim();
            if (para.isEmpty()) continue;
            boolean matched = false;
            Matcher m = SENT_RE.matcher(para);
            while (m.find()) {
                String s = m.group().trim();
                if (!s.isEmpty()) {
                    out.add(s.replace(SENT_GUARD, "."));
                    matched = true;
                }
            }
            if (!matched) out.add(para.replace(SENT_GUARD, "."));
        }
        if (!out.isEmpty()) return out;
        String t = guarded.replace(SENT_GUARD, ".").trim();
        if (!t.isEmpty()) out.add(t);
        return out;
    }

    /** Tokens lexicais: palavras minúsculas; CJK vira caracteres + bigramas. */
    public static List<String> wordTokens(String text) {
        String lower = text.toLowerCase();
        List<String> toks = new ArrayList<>();
        Matcher m = WORD_RE.matcher(lower);
        while (m.find()) {
            String w = m.group();
            // partes não-CJK, na ordem (split como Python/JS)
            for (String p : CJK_RE.split(w, -1)) if (!p.isEmpty()) toks.add(p);
            List<String> cjks = new ArrayList<>();
            Matcher cm = CJK_RE.matcher(w);
            while (cm.find()) cjks.add(cm.group());
            if (!cjks.isEmpty()) {
                toks.addAll(cjks);
                for (int i = 0; i + 1 < cjks.size(); i++) toks.add(cjks.get(i) + cjks.get(i + 1));
            }
        }
        return toks;
    }

    /** N-gramas de caracteres (por code point) — base do encoder fallback. */
    public static List<String> charNgrams(String text, int nLo, int nHi) {
        String s = " " + text.toLowerCase().replaceAll("\\s+", " ") + " ";
        int[] cp = codePoints(s);
        List<String> out = new ArrayList<>();
        int L = cp.length;
        for (int n = nLo; n <= nHi; n++) {
            for (int i = 0; i + n <= L; i++) out.add(fromCodePoints(cp, i, i + n));
        }
        return out;
    }

    public static List<String> charNgrams(String text) {
        return charNgrams(text, 3, 5);
    }
}
