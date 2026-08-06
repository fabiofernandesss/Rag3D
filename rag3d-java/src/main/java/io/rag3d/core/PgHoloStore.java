package io.rag3d.core;

import java.sql.*;
import java.util.*;

/**
 * Backend Postgres PURO (JDBC) — sem pgvector. Espelha rag3d/pgstore.py e
 * rag3d-js/src/pgstore.js e usa o MESMO esquema, então lê e escreve os
 * hologramas das outras linguagens.
 *
 * <p>Só tipos nativos: BIT(1024) para a assinatura (distância de Hamming via
 * bit_count), INT[] para as facetas, BYTEA para eco e constelação.
 */
public final class PgHoloStore implements AutoCloseable {

    private static final String KINDS = "('chunk','turn','summary')";
    private static final String COLS =
            "id,doc_id,parent_id,kind,pos,text,ctx,n_tokens,created,importance,turn_no,accessed_turn";

    private final Connection db;
    private final Holo holo;
    private Integer nCache = null;

    public int bandPrefilterThreshold = 20000;
    public int prefetchFactor = 4;
    public int prefetchMin = 200;

    public PgHoloStore(String jdbcUrl, String user, String password, int denseDim, int colbertDim)
            throws SQLException {
        this.db = DriverManager.getConnection(jdbcUrl, user, password);
        this.db.setAutoCommit(true);
        this.holo = new Holo(denseDim, colbertDim);
        ensureSchema();
    }

    private void ensureSchema() throws SQLException {
        String ddl = """
            CREATE TABLE IF NOT EXISTS holo_meta(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS holo_docs(
              id BIGSERIAL PRIMARY KEY, source TEXT, title TEXT,
              created DOUBLE PRECISION, n_tokens INT, meta TEXT);
            CREATE TABLE IF NOT EXISTS holo_grams(
              id BIGSERIAL PRIMARY KEY, doc_id BIGINT, parent_id BIGINT,
              kind TEXT NOT NULL DEFAULT 'chunk',
              pos INT, text TEXT NOT NULL, ctx TEXT,
              n_tokens INT, created DOUBLE PRECISION,
              importance REAL DEFAULT 0.5, turn_no INT, accessed_turn INT,
              sig BIT(1024), bands INT[], echo BYTEA,
              constellation BYTEA, n_tok INT);
            CREATE INDEX IF NOT EXISTS idx_grams_kind ON holo_grams(kind);
            CREATE INDEX IF NOT EXISTS idx_grams_bands ON holo_grams USING GIN(bands);
            CREATE TABLE IF NOT EXISTS holo_spectrum(
              term BIGINT NOT NULL, gram_id BIGINT NOT NULL, weight REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_spectrum_term ON holo_spectrum(term);
            CREATE INDEX IF NOT EXISTS idx_spectrum_gram ON holo_spectrum(gram_id);
            """;
        try (Statement st = db.createStatement()) { st.execute(ddl); }
    }

    public Holo holo() { return holo; }

    private static double now() { return System.currentTimeMillis() / 1000.0; }

    public long addDoc(String source, String title, int nTokens) throws SQLException {
        try (PreparedStatement ps = db.prepareStatement(
                "INSERT INTO holo_docs(source,title,created,n_tokens,meta) VALUES(?,?,?,?,?) RETURNING id")) {
            ps.setString(1, source); ps.setString(2, title);
            ps.setDouble(3, now()); ps.setInt(4, nTokens); ps.setString(5, "{}");
            try (ResultSet rs = ps.executeQuery()) { rs.next(); return rs.getLong(1); }
        }
    }

    /** Grava o chunk nas três formas — gram + espectro numa query só (CTE). */
    public long addChunk(Long docId, String text, String ctx, int nTokens, Encoders.TriVec vec)
            throws SQLException {
        byte[] sig = holo.signDense(vec.dense);
        String bits = Holo.sigToBitstring(sig);
        Integer[] bands = new Integer[Holo.N_BANDS];
        int[] b = Holo.bandsOf(sig);
        for (int i = 0; i < b.length; i++) bands[i] = b[i];

        Long[] terms = vec.sparse.keySet().toArray(new Long[0]);
        Float[] weights = new Float[terms.length];
        for (int i = 0; i < terms.length; i++) weights[i] = vec.sparse.get(terms[i]);

        String sql = """
            WITH g AS (
              INSERT INTO holo_grams(doc_id,kind,pos,text,ctx,n_tokens,created,importance,
                sig,bands,echo,constellation,n_tok)
              VALUES(?,'chunk',?,?,?,?,?,0.5,?::bit(1024),?,?,?,?) RETURNING id
            ), s AS (
              INSERT INTO holo_spectrum(term, gram_id, weight)
              SELECT u.term, (SELECT id FROM g), u.weight
              FROM unnest(?::bigint[], ?::real[]) AS u(term, weight)
            ) SELECT id FROM g""";
        try (PreparedStatement ps = db.prepareStatement(sql)) {
            int i = 1;
            if (docId == null) ps.setNull(i++, Types.BIGINT); else ps.setLong(i++, docId);
            ps.setInt(i++, 0);
            ps.setString(i++, text); ps.setString(i++, ctx);
            ps.setInt(i++, nTokens); ps.setDouble(i++, now());
            ps.setString(i++, bits);
            ps.setArray(i++, db.createArrayOf("int4", bands));
            ps.setBytes(i++, Holo.quantize(vec.dense));
            ps.setBytes(i++, holo.signTokens(vec.tokens));
            ps.setInt(i++, vec.tokens.size());
            ps.setArray(i++, db.createArrayOf("int8", terms));
            ps.setArray(i++, db.createArrayOf("float4", weights));
            try (ResultSet rs = ps.executeQuery()) { rs.next(); nCache = null; return rs.getLong(1); }
        }
    }

    public int nChunks() throws SQLException {
        if (nCache == null) {
            try (Statement st = db.createStatement();
                 ResultSet rs = st.executeQuery(
                         "SELECT COUNT(*) FROM holo_grams WHERE kind IN " + KINDS)) {
                rs.next(); nCache = rs.getInt(1);
            }
        }
        return nCache;
    }

    // ------------------------------------------------- eixo 1: semântico ---

    public List<Fusion.Scored> denseSearch(float[] qvec, int k) throws SQLException {
        byte[] sig = holo.signDense(qvec);
        String bits = Holo.sigToBitstring(sig);
        int prefetch = Math.max(prefetchFactor * k, prefetchMin);
        boolean big = nChunks() > bandPrefilterThreshold;

        List<Object[]> rows = scan(bits, sig, prefetch, big);
        if (big && rows.size() < prefetch / 2) rows = scan(bits, sig, prefetch, false);
        if (rows.isEmpty()) return List.of();

        // re-pontuação exata pelo eco int8 (cosseno real, não Hamming)
        List<Fusion.Scored> out = new ArrayList<>(rows.size());
        for (Object[] r : rows) {
            float[] echo = Holo.dequantize((byte[]) r[1]);
            double dot = 0;
            for (int d = 0; d < qvec.length && d < echo.length; d++) dot += (double) echo[d] * qvec[d];
            out.add(new Fusion.Scored((Long) r[0], dot));
        }
        out.sort(Comparator.<Fusion.Scored>comparingDouble(s -> -s.score()).thenComparingLong(Fusion.Scored::id));
        return out.subList(0, Math.min(k, out.size()));
    }

    private List<Object[]> scan(String bits, byte[] sig, int prefetch, boolean useBands) throws SQLException {
        String where = "kind IN " + KINDS + " AND sig IS NOT NULL"
                + (useBands ? " AND bands && ?::int[]" : "");
        String sql = "SELECT id, echo, bit_count(sig # ?::bit(1024)) AS ham FROM holo_grams"
                + " WHERE " + where + " ORDER BY ham ASC LIMIT ?";
        try (PreparedStatement ps = db.prepareStatement(sql)) {
            int i = 1;
            ps.setString(i++, bits);
            if (useBands) {
                int[] b = Holo.bandsOf(sig);
                Integer[] bx = new Integer[b.length];
                for (int j = 0; j < b.length; j++) bx[j] = b[j];
                ps.setArray(i++, db.createArrayOf("int4", bx));
            }
            ps.setInt(i, prefetch);
            List<Object[]> rows = new ArrayList<>();
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) rows.add(new Object[]{rs.getLong(1), rs.getBytes(2)});
            }
            return rows;
        }
    }

    // --------------------------------------------------- eixo 2: léxico ----

    public List<Fusion.Scored> sparseSearch(Map<Long, Float> qsparse, int k) throws SQLException {
        if (qsparse.isEmpty()) return List.of();
        int nDocs = Math.max(1, nChunks());
        String sql = "SELECT term, gram_id, weight, COUNT(*) OVER (PARTITION BY term) AS df"
                + " FROM holo_spectrum WHERE term = ANY(?::bigint[])";
        Map<Long, Double> scores = new LinkedHashMap<>();
        try (PreparedStatement ps = db.prepareStatement(sql)) {
            ps.setArray(1, db.createArrayOf("int8", qsparse.keySet().toArray(new Long[0])));
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    long term = rs.getLong(1), gid = rs.getLong(2);
                    double dw = rs.getFloat(3), df = rs.getLong(4);
                    double idf = Math.log(1.0 + (nDocs - df + 0.5) / (df + 0.5));
                    scores.merge(gid, qsparse.get(term) * dw * idf, Double::sum);
                }
            }
        }
        List<Fusion.Scored> out = new ArrayList<>();
        scores.forEach((gid, s) -> out.add(new Fusion.Scored(gid, s)));
        out.sort(Comparator.<Fusion.Scored>comparingDouble(s -> -s.score()).thenComparingLong(Fusion.Scored::id));
        return out.subList(0, Math.min(k, out.size()));
    }

    // ----------------------------------------------- eixo 3: estrutural ----

    public List<Fusion.Scored> colbertScores(List<float[]> qtokens, Collection<Long> candidates)
            throws SQLException {
        if (candidates.isEmpty() || qtokens.isEmpty()) return List.of();
        byte[] qConst = holo.signTokens(qtokens);
        List<Fusion.Scored> out = new ArrayList<>();
        try (PreparedStatement ps = db.prepareStatement(
                "SELECT id, constellation FROM holo_grams WHERE id = ANY(?::bigint[]) AND constellation IS NOT NULL")) {
            ps.setArray(1, db.createArrayOf("int8", candidates.toArray(new Long[0])));
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) out.add(new Fusion.Scored(rs.getLong(1), holo.binaryMaxsim(qConst, rs.getBytes(2))));
            }
        }
        out.sort(Comparator.<Fusion.Scored>comparingDouble(s -> -s.score()).thenComparingLong(Fusion.Scored::id));
        return out;
    }

    // ------------------------------------------------------------ leitura --

    /** Chunk hidratado (texto + metadados + vetor denso do eco). */
    public record Chunk(long id, String text, String ctx, int nTokens, float[] dense) {}

    public Map<Long, Chunk> getChunks(Collection<Long> ids) throws SQLException {
        Map<Long, Chunk> out = new LinkedHashMap<>();
        if (ids.isEmpty()) return out;
        try (PreparedStatement ps = db.prepareStatement(
                "SELECT id, text, ctx, n_tokens, echo FROM holo_grams WHERE id = ANY(?::bigint[])")) {
            ps.setArray(1, db.createArrayOf("int8", ids.toArray(new Long[0])));
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    byte[] echo = rs.getBytes(5);
                    out.put(rs.getLong(1), new Chunk(rs.getLong(1), rs.getString(2), rs.getString(3),
                            rs.getInt(4), echo == null ? null : Holo.dequantize(echo)));
                }
            }
        }
        return out;
    }

    public void reset() throws SQLException {
        try (Statement st = db.createStatement()) {
            st.execute("TRUNCATE holo_grams, holo_spectrum, holo_docs");
        }
        nCache = null;
    }

    @Override public void close() throws SQLException { db.close(); }
}
