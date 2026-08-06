package io.rag3d.core;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Encoder tridimensional — espelha o HashEncoder de rag3d/encoders.py e
 * rag3d-js/src/encoders.js.
 *
 * <p>Fallback por hashing (crc32) de n-gramas de caracteres: zero dependências,
 * qualquer escrita. Produz as MESMAS três representações que Python/JS para o
 * mesmo texto, então os hologramas batem.
 *
 * <ul>
 *   <li>dense  — hashing trick sobre n-gramas de caracteres
 *   <li>sparse — palavras Unicode (CJK vira bigramas), peso 1+ln(tf)
 *   <li>tokens — um vetor hasheado por palavra (MaxSim aproximado)
 * </ul>
 */
public final class Encoders {

    /** As três projeções de um mesmo texto. */
    public static final class TriVec {
        public final float[] dense;
        public final Map<Long, Float> sparse;
        public final List<float[]> tokens;

        public TriVec(float[] dense, Map<Long, Float> sparse, List<float[]> tokens) {
            this.dense = dense;
            this.sparse = sparse;
            this.tokens = tokens;
        }
    }

    public static final class HashEncoder {
        public final String name = "hash";
        private final int denseDim;
        private final int colbertDim;
        private final int maxTokens;

        public HashEncoder(int denseDim, int colbertDim, int maxTokens) {
            this.denseDim = denseDim;
            this.colbertDim = colbertDim;
            this.maxTokens = maxTokens;
        }

        public HashEncoder() {
            this(1024, 128, 256);
        }

        private static float[] l2(float[] v) {
            double n = 0;
            for (float x : v) n += (double) x * x;
            n = Math.sqrt(n);
            if (n > 0) for (int i = 0; i < v.length; i++) v[i] = (float) (v[i] / n);
            return v;
        }

        private float[] dense(String text) {
            float[] v = new float[denseDim];
            for (String g : TextProc.charNgrams(text)) {
                long h = Portable.crc32(g);
                v[(int) (h % denseDim)] += ((h >> 31) & 1L) != 0 ? 1.0f : -1.0f;
            }
            return l2(v);
        }

        private Map<Long, Float> sparse(String text) {
            Map<Long, Integer> tf = new LinkedHashMap<>();
            for (String w : TextProc.wordTokens(text)) {
                long tid = Portable.crc32("w:" + w);
                tf.merge(tid, 1, Integer::sum);
            }
            Map<Long, Float> out = new LinkedHashMap<>();
            for (Map.Entry<Long, Integer> e : tf.entrySet()) {
                out.put(e.getKey(), (float) (1.0 + Math.log(e.getValue())));
            }
            return out;
        }

        private float[] tokenVec(String word) {
            float[] v = new float[colbertDim];
            for (String g : TextProc.charNgrams(word, 2, 4)) {
                long h = Portable.crc32("t:" + g);
                v[(int) (h % colbertDim)] += ((h >> 31) & 1L) != 0 ? 1.0f : -1.0f;
            }
            return l2(v);
        }

        public TriVec encode(String text) {
            List<String> words = TextProc.wordTokens(text);
            if (words.size() > maxTokens) words = words.subList(0, maxTokens);
            List<float[]> tokens = new ArrayList<>();
            if (words.isEmpty()) {
                tokens.add(new float[colbertDim]);
            } else {
                for (String w : words) tokens.add(tokenVec(w));
            }
            return new TriVec(dense(text), sparse(text), tokens);
        }
    }

    private Encoders() {}
}
