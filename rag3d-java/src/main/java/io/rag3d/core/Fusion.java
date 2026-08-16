package io.rag3d.core;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Fusão dos três eixos — espelha rag3d/fusion.py e rag3d-js/src/fusion.js.
 *
 * <p>BOSÔNICA (quem é relevante): cada eixo entra como amplitude complexa
 * a_c = sqrt(w_c*s_c)*e^(i*phi_c) e a pontuação é |soma|^2 = clássico +
 * interferência. Eixos que concordam interferem construtivamente.
 * interferenceStrength=0 colapsa no CombSUM clássico. RRF (k=60) de base.
 *
 * <p>FERMIÔNICA (qual conjunto): o conjunto de k documentos é antissimetrizado
 * (determinante de Slater), |psi_S|^2 = det(Gram) = Vol^2. Dois documentos
 * idênticos = duas partículas no mesmo estado -> determinante zera (exclusão de
 * Pauli). É um DPP; o guloso aproxima o objetivo log-det com Cholesky
 * incremental O(k^2 N), sem resolver o MAP global (Chen et al., NeurIPS 2018).
 */
public final class Fusion {

    /** Um par (id, pontuação) de um ranking de canal. */
    public record Scored(long id, double score) {}

    /** Resultado da fusão de um documento. */
    public static final class Hit {
        public final long chunkId;
        public final double score;
        public final double classical;
        public final double interference;
        public final List<String> channels;

        Hit(long chunkId, double score, double classical, double interference, List<String> channels) {
            this.chunkId = chunkId;
            this.score = score;
            this.classical = classical;
            this.interference = interference;
            this.channels = channels;
        }
    }

    private Fusion() {}

    private static Map<Long, Double> minmax(List<Scored> ranking) {
        Map<Long, Double> m = new LinkedHashMap<>();
        if (ranking.isEmpty()) return m;
        double lo = Double.MAX_VALUE, hi = -Double.MAX_VALUE;
        for (Scored s : ranking) { lo = Math.min(lo, s.score()); hi = Math.max(hi, s.score()); }
        if (hi - lo < 1e-12) {
            for (Scored s : ranking) m.put(s.id(), 1.0);
        } else {
            // Scale before subtracting so opposite finite extremes do not overflow.
            double scale = Math.max(Math.max(Math.abs(lo), Math.abs(hi)), 1.0);
            double loScaled = lo / scale;
            double span = hi / scale - loScaled;
            for (Scored s : ranking) {
                m.put(s.id(), (s.score() / scale - loScaled) / span);
            }
        }
        return m;
    }

    private static Map<Long, Double> phases(List<Scored> ranking) {
        Map<Long, Double> m = new LinkedHashMap<>();
        int n = ranking.size();
        if (n <= 1) {
            for (Scored s : ranking) m.put(s.id(), 0.0);
            return m;
        }
        for (int i = 0; i < n; i++) m.put(ranking.get(i).id(), Math.PI * i / (n - 1));
        return m;
    }

    /** Fusão por interferência quântica. */
    public static List<Hit> quantumFuse(Map<String, List<Scored>> channels,
                                        Map<String, Double> weights,
                                        int topK, double interferenceStrength) {
        Map<String, Map<Long, Double>> norm = new LinkedHashMap<>();
        Map<String, Map<Long, Double>> phase = new LinkedHashMap<>();
        Set<Long> allIds = new LinkedHashSet<>();
        for (Map.Entry<String, List<Scored>> e : channels.entrySet()) {
            norm.put(e.getKey(), minmax(e.getValue()));
            phase.put(e.getKey(), phases(e.getValue()));
            for (Scored s : e.getValue()) allIds.add(s.id());
        }

        List<Hit> hits = new ArrayList<>();
        for (long cid : allIds) {
            List<double[]> amps = new ArrayList<>();   // [amplitude, fase]
            List<String> chans = new ArrayList<>();
            for (String name : channels.keySet()) {
                Double s = norm.get(name).get(cid);
                if (s == null || s <= 0.0) continue;
                double w = weights.getOrDefault(name, 1.0);
                amps.add(new double[]{Math.sqrt(w * s), phase.get(name).get(cid)});
                chans.add(name);
            }
            double classical = 0;
            for (double[] a : amps) classical += a[0] * a[0];
            double interf = 0;
            for (int i = 0; i < amps.size(); i++) {
                for (int j = i + 1; j < amps.size(); j++) {
                    interf += 2.0 * amps.get(i)[0] * amps.get(j)[0]
                            * Math.cos(amps.get(i)[1] - amps.get(j)[1]);
                }
            }
            hits.add(new Hit(cid, classical + interferenceStrength * interf, classical, interf, chans));
        }
        // desempate determinístico por id — idêntico a Python/JS
        hits.sort(Comparator.<Hit>comparingDouble(h -> -h.score).thenComparingLong(h -> h.chunkId));
        return hits.subList(0, Math.min(topK, hits.size()));
    }

    /** Reciprocal Rank Fusion — linha de base. */
    public static List<Hit> rrfFuse(Map<String, List<Scored>> channels,
                                    Map<String, Double> weights, int topK, int rrfK) {
        Map<Long, Double> scores = new LinkedHashMap<>();
        Map<Long, List<String>> found = new HashMap<>();
        for (Map.Entry<String, List<Scored>> e : channels.entrySet()) {
            double w = weights.getOrDefault(e.getKey(), 1.0);
            List<Scored> r = e.getValue();
            Set<Long> seen = new HashSet<>();
            for (int rank = 0; rank < r.size(); rank++) {
                long cid = r.get(rank).id();
                if (!seen.add(cid)) continue;
                scores.merge(cid, w / (rrfK + rank + 1), Double::sum);
                found.computeIfAbsent(cid, k -> new ArrayList<>()).add(e.getKey());
            }
        }
        List<Hit> hits = new ArrayList<>();
        for (Map.Entry<Long, Double> e : scores.entrySet()) {
            hits.add(new Hit(e.getKey(), e.getValue(), e.getValue(), 0.0, found.get(e.getKey())));
        }
        hits.sort(Comparator.<Hit>comparingDouble(h -> -h.score).thenComparingLong(h -> h.chunkId));
        return hits.subList(0, Math.min(topK, hits.size()));
    }

    /**
     * Greedy DPP: aproxima um conjunto com relevância e diversidade;
     * não é solução MAP global exata. Equilibra relevância x volume com
     * Cholesky incremental — espelha
     * fermionic_select/fermionicSelect bit a bit.
     *
     * @param items     pares (id, relevância) em ordem decrescente
     * @param vectors   vetor denso por id (define o volume)
     * @param topK      tamanho do conjunto
     * @param diversity 0 = ranking puro; 1 = só volume
     */
    public static List<Long> fermionicSelect(List<Scored> items, Map<Long, float[]> vectors,
                                             int topK, double diversity) {
        int n = items.size();
        List<Long> ids = new ArrayList<>(n);
        for (Scored s : items) ids.add(s.id());
        if (diversity <= 0 || n <= 1 || topK <= 0) return ids.subList(0, Math.min(topK, n));

        int k = Math.min(topK, n);
        double eps = 1e-12;

        double lo = Double.MAX_VALUE, hi = -Double.MAX_VALUE;
        for (Scored s : items) { lo = Math.min(lo, s.score()); hi = Math.max(hi, s.score()); }
        double[] logQ2 = new double[n];
        for (int i = 0; i < n; i++) {
            double q = (hi - lo > eps) ? (items.get(i).score() - lo) / (hi - lo) : 1.0;
            logQ2[i] = 2.0 * Math.log(Math.max(q, eps));
        }

        int dim = 0;
        for (float[] v : vectors.values()) if (v != null) { dim = v.length; break; }
        float[][] V = new float[n][];
        for (int i = 0; i < n; i++) {
            float[] v = vectors.get(ids.get(i));
            V[i] = (v != null && v.length == dim) ? v : new float[dim];
        }

        double theta = Math.max(0.0, Math.min(1.0, 1.0 - diversity));
        double[] d2 = new double[n];
        java.util.Arrays.fill(d2, 1.0);                  // volume residual
        double[][] C = new double[n][k];                 // fator de Cholesky
        boolean[] avail = new boolean[n];
        java.util.Arrays.fill(avail, true);
        List<Long> chosen = new ArrayList<>(k);

        for (int t = 0; t < k; t++) {
            double best = Double.NEGATIVE_INFINITY;
            int j = -1;
            for (int i = 0; i < n; i++) {
                if (!avail[i]) continue;
                double g = theta * logQ2[i] + (1 - theta) * Math.log(Math.max(d2[i], eps));
                if (g > best) { best = g; j = i; }
            }
            if (j < 0) break;
            // sem volume restante: completa por relevância pura (Cholesky instável)
            if (d2[j] < eps) {
                for (int i = 0; i < n && chosen.size() < k; i++) if (avail[i]) chosen.add(ids.get(i));
                return chosen.subList(0, Math.min(k, chosen.size()));
            }
            chosen.add(ids.get(j));
            avail[j] = false;
            if (chosen.size() == k) break;

            double sj = Math.sqrt(Math.max(d2[j], eps));
            for (int i = 0; i < n; i++) {
                if (!avail[i]) continue;
                double dot = 0;
                for (int d = 0; d < dim; d++) dot += (double) V[j][d] * V[i][d];
                double acc = 0;
                for (int s = 0; s < t; s++) acc += C[j][s] * C[i][s];
                double e = (dot - acc) / sj;
                C[i][t] = e;
                d2[i] -= e * e;
            }
        }
        return chosen;
    }
}
