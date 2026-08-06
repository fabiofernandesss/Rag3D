package io.rag3d;

import io.rag3d.core.*;

import java.sql.SQLException;
import java.util.*;

/**
 * Fachada RAG3D para Java — mesmo índice holográfico de Python e JavaScript.
 *
 * <pre>{@code
 * try (Rag3D rag = Rag3D.connect("jdbc:postgresql://localhost:5432/rag3d", "postgres", "senha")) {
 *     rag.ingest("qualquer texto, em qualquer língua");
 *     Rag3D.Result r = rag.search("pergunta", 8);
 *     r.fused().forEach(h -> System.out.println(h.text()));
 * }
 * }</pre>
 *
 * <p>A busca projeta a pergunta nos três eixos (semântico, léxico, estrutural),
 * funde por interferência quântica e aplica a seleção fermiônica (MAP-DPP) no
 * conjunto final.
 */
public final class Rag3D implements AutoCloseable {

    /** Um resultado hidratado. */
    public record Hit(long id, String text, double score, double interference, List<String> channels) {}

    /** Resultado completo: as três visões por eixo + a fusão. */
    public record Result(String query, List<Hit> fused, Map<String, List<Hit>> views) {}

    private final PgHoloStore store;
    private final Encoders.HashEncoder encoder;

    // configuração (mesmos padrões das outras linguagens)
    public int channelK = 100;
    public int topK = 10;
    public double interferenceStrength = 1.0;
    public double diversity = 0.35;          // seleção fermiônica
    public double[] channelWeights = {1.0, 0.8, 0.9};
    public String fusion = "quantum";        // quantum | rrf
    public int rrfK = 60;
    public int chunkTokens = 400, chunkOverlap = 60, tinyDocTokens = 500;

    private Rag3D(PgHoloStore store, Encoders.HashEncoder encoder) {
        this.store = store;
        this.encoder = encoder;
    }

    public static Rag3D connect(String jdbcUrl, String user, String password) throws SQLException {
        return new Rag3D(new PgHoloStore(jdbcUrl, user, password, 1024, 128),
                new Encoders.HashEncoder(1024, 128, 256));
    }

    public PgHoloStore store() { return store; }

    // ------------------------------------------------------------ ingestão --

    /** Ingere um texto (chunking adaptativo: doc minúsculo entra inteiro). */
    public int ingest(String rawText, String source, String title) throws SQLException {
        String text = TextProc.normalize(rawText);
        if (text.isEmpty()) return 0;
        int nTok = TextProc.estimateTokens(text);
        if (title == null || title.isEmpty()) {
            int[] cp = text.codePoints().toArray();
            int n = Math.min(60, cp.length);
            title = new String(cp, 0, n).replace('\n', ' ') + (cp.length > 60 ? "…" : "");
        }
        long docId = store.addDoc(source, title, nTok);

        List<String> chunks = (nTok <= tinyDocTokens) ? List.of(text) : chunkText(text);
        for (String c : chunks) {
            String ctx = "[" + title + "] " + c;
            store.addChunk(docId, c, ctx, TextProc.estimateTokens(c), encoder.encode(ctx));
        }
        return chunks.size();
    }

    public int ingest(String text) throws SQLException { return ingest(text, "inline", ""); }

    /** Agrupa sentenças até ~chunkTokens, com sobreposição — igual a Python/JS. */
    private List<String> chunkText(String text) {
        List<String> sents = TextProc.splitSentences(text);
        List<String> chunks = new ArrayList<>();
        List<String> cur = new ArrayList<>();
        int curTok = 0;
        for (String s : sents) {
            int st = TextProc.estimateTokens(s);
            if (!cur.isEmpty() && curTok + st > chunkTokens) {
                chunks.add(String.join(" ", cur));
                List<String> keep = new ArrayList<>();
                int kept = 0;
                for (int i = cur.size() - 1; i >= 0; i--) {
                    int pt = TextProc.estimateTokens(cur.get(i));
                    if (kept + pt > chunkOverlap) break;
                    keep.add(0, cur.get(i));
                    kept += pt;
                }
                cur = keep; curTok = kept;
            }
            cur.add(s); curTok += st;
        }
        if (!cur.isEmpty()) chunks.add(String.join(" ", cur));
        chunks.removeIf(String::isBlank);
        return chunks;
    }

    // --------------------------------------------------------------- busca --

    /** Busca tridimensional: 3 eixos -> fusão quântica -> seleção fermiônica. */
    public Result search(String query, int k) throws SQLException {
        int want = k > 0 ? k : topK;
        Encoders.TriVec q = encoder.encode(TextProc.normalize(query));

        List<Fusion.Scored> dense = store.denseSearch(q.dense, channelK);
        List<Fusion.Scored> sparse = store.sparseSearch(q.sparse, channelK);
        Set<Long> pool = new LinkedHashSet<>();
        for (Fusion.Scored s : dense) pool.add(s.id());
        for (Fusion.Scored s : sparse) pool.add(s.id());
        List<Fusion.Scored> struct = store.colbertScores(q.tokens, pool);
        if (struct.size() > channelK) struct = struct.subList(0, channelK);

        Map<String, List<Fusion.Scored>> channels = new LinkedHashMap<>();
        channels.put("semantico", dense);
        channels.put("lexico", sparse);
        channels.put("estrutural", struct);
        Map<String, Double> w = new LinkedHashMap<>();
        w.put("semantico", channelWeights[0]);
        w.put("lexico", channelWeights[1]);
        w.put("estrutural", channelWeights[2]);

        // pool maior quando há seleção fermiônica (ela escolhe o conjunto final)
        int fuseK = diversity > 0 ? Math.max(want, Math.min(40, want * 4)) : want;
        List<Fusion.Hit> hits = "rrf".equals(fusion)
                ? Fusion.rrfFuse(channels, w, fuseK, rrfK)
                : Fusion.quantumFuse(channels, w, fuseK, interferenceStrength);

        List<Long> ids = new ArrayList<>();
        for (Fusion.Hit h : hits) ids.add(h.chunkId);
        Map<Long, PgHoloStore.Chunk> rows = store.getChunks(ids);

        // seleção fermiônica: exclusão de Pauli sobre o conjunto
        if (diversity > 0 && hits.size() > want) {
            List<Fusion.Scored> items = new ArrayList<>();
            Map<Long, float[]> vecs = new LinkedHashMap<>();
            for (Fusion.Hit h : hits) {
                items.add(new Fusion.Scored(h.chunkId, h.score));
                PgHoloStore.Chunk c = rows.get(h.chunkId);
                if (c != null && c.dense() != null) vecs.put(h.chunkId, c.dense());
            }
            List<Long> keep = Fusion.fermionicSelect(items, vecs, want, diversity);
            Map<Long, Fusion.Hit> byId = new LinkedHashMap<>();
            for (Fusion.Hit h : hits) byId.put(h.chunkId, h);
            List<Fusion.Hit> sel = new ArrayList<>();
            for (long id : keep) if (byId.containsKey(id)) sel.add(byId.get(id));
            hits = sel;
        } else if (hits.size() > want) {
            hits = hits.subList(0, want);
        }

        List<Hit> fused = new ArrayList<>();
        for (Fusion.Hit h : hits) {
            PgHoloStore.Chunk c = rows.get(h.chunkId);
            if (c != null) fused.add(new Hit(h.chunkId, c.text(), h.score, h.interference, h.channels));
        }

        Map<String, List<Hit>> views = new LinkedHashMap<>();
        for (Map.Entry<String, List<Fusion.Scored>> e : channels.entrySet()) {
            List<Fusion.Scored> r = e.getValue();
            List<Long> vids = new ArrayList<>();
            for (int i = 0; i < Math.min(want, r.size()); i++) vids.add(r.get(i).id());
            Map<Long, PgHoloStore.Chunk> vrows = store.getChunks(vids);
            List<Hit> vh = new ArrayList<>();
            for (int i = 0; i < vids.size(); i++) {
                PgHoloStore.Chunk c = vrows.get(vids.get(i));
                if (c != null) vh.add(new Hit(c.id(), c.text(), r.get(i).score(), 0, List.of(e.getKey())));
            }
            views.put(e.getKey(), vh);
        }
        return new Result(query, fused, views);
    }

    public Result search(String query) throws SQLException { return search(query, topK); }

    @Override public void close() throws SQLException { store.close(); }
}
