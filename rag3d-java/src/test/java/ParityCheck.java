import io.rag3d.core.*;
import io.rag3d.Rag3D;
import java.util.*;

/**
 * Paridade tri-linguagem: Java tem que produzir o MESMO holograma que Python e
 * JavaScript para o mesmo texto. Imprime JSON para o comparador cruzado.
 *
 *   javac -d target/classes $(find src -name '*.java')
 *   java -cp target/classes ParityCheck
 */
public class ParityCheck {
    static String jsonList(int[] a, int n) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < n; i++) { if (i > 0) sb.append(","); sb.append(a[i]); }
        return sb.append("]").toString();
    }

    static String jsonBytes(byte[] a, int n) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < n; i++) { if (i > 0) sb.append(","); sb.append(a[i] & 0xFF); }
        return sb.append("]").toString();
    }

    public static void main(String[] args) {
        try {
            Rag3D.validateQuery("😀".repeat(16385));
            throw new AssertionError("oversized UTF-8 query must be rejected");
        } catch (IllegalArgumentException expected) {
            if (!expected.getMessage().contains("65536")) throw expected;
        }

        String[] texts = {
            "O contrato de aluguel vence em 15 de março de 2027.",
            "会议将于星期五上午十点在北京举行。",
            "Lei nº 13.243/2016 e 14.133/2021.",
            "The Artemis rocket launch is scheduled for 12 July 2026."
        };
        Encoders.HashEncoder enc = new Encoders.HashEncoder(1024, 128, 256);
        Holo holo = new Holo(1024, 128);

        StringBuilder out = new StringBuilder("{\"texts\":[");
        for (int i = 0; i < texts.length; i++) {
            Encoders.TriVec v = enc.encode(TextProc.normalize(texts[i]));
            byte[] sig = holo.signDense(v.dense);
            byte[] echo = Holo.quantize(v.dense);
            byte[] konst = holo.signTokens(v.tokens);
            if (i > 0) out.append(",");
            out.append("{\"sig\":").append(jsonBytes(sig, sig.length))
               .append(",\"bands\":").append(jsonList(Holo.bandsOf(sig), 16))
               .append(",\"echo8\":").append(jsonBytes(echo, 8))
               .append(",\"sparse_n\":").append(v.sparse.size())
               .append(",\"const_len\":").append(konst.length)
               .append(",\"const8\":").append(jsonBytes(konst, 8))
               .append("}");
        }
        // fusão: mesmos rankings sintéticos que Python/JS
        Map<String, List<Fusion.Scored>> ch = new LinkedHashMap<>();
        ch.put("a", List.of(new Fusion.Scored(1, 0.9), new Fusion.Scored(2, 0.8), new Fusion.Scored(3, 0.1)));
        ch.put("b", List.of(new Fusion.Scored(1, 0.85), new Fusion.Scored(3, 0.2)));
        ch.put("c", List.of(new Fusion.Scored(1, 0.7), new Fusion.Scored(2, 0.05)));
        Map<String, Double> w = Map.of("a", 1.0, "b", 1.0, "c", 1.0);

        Map<String, List<Fusion.Scored>> nearFlat = Map.of(
            "a", List.of(new Fusion.Scored(2, 1.0), new Fusion.Scored(1, 1.0 - 5e-13))
        );
        List<Fusion.Hit> nearFlatHits = Fusion.quantumFuse(
            nearFlat, Map.of("a", 1.0), 2, 1.0
        );
        if (nearFlatHits.get(0).chunkId != 1 || nearFlatHits.get(1).chunkId != 2) {
            throw new AssertionError("near-flat quantum channel must tie-break by id");
        }

        Map<String, List<Fusion.Scored>> extreme = Map.of(
            "a", List.of(new Fusion.Scored(1, 1e308), new Fusion.Scored(2, -1e308))
        );
        for (Fusion.Hit hit : Fusion.quantumFuse(extreme, Map.of("a", 1.0), 2, 1.0)) {
            if (!Double.isFinite(hit.score)) {
                throw new AssertionError("finite extreme scores must stay finite");
            }
        }

        Map<String, List<Fusion.Scored>> duplicateRrf = new LinkedHashMap<>();
        duplicateRrf.put("a", List.of(
            new Fusion.Scored(1, 0.9),
            new Fusion.Scored(1, 0.8),
            new Fusion.Scored(2, 0.7)
        ));
        duplicateRrf.put("b", List.of(new Fusion.Scored(2, 0.9)));
        List<Fusion.Hit> duplicateRrfHits = Fusion.rrfFuse(
            duplicateRrf, Map.of("a", 1.0, "b", 1.0), 2, 60
        );
        if (duplicateRrfHits.get(0).chunkId != 2
                || duplicateRrfHits.get(1).chunkId != 1
                || duplicateRrfHits.get(1).channels.size() != 1) {
            throw new AssertionError("RRF must count only the first ID occurrence per channel");
        }

        out.append("],\"fusion\":[");
        List<Fusion.Hit> hits = Fusion.quantumFuse(ch, w, 5, 1.0);
        for (int i = 0; i < hits.size(); i++) {
            if (i > 0) out.append(",");
            out.append(String.format(Locale.ROOT, "[%d,%.10f]", hits.get(i).chunkId, hits.get(i).score));
        }
        out.append("]}");
        System.out.println(out);
    }
}
