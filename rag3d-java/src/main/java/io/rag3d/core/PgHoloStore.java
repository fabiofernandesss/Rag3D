package io.rag3d.core;

import java.sql.*;
import java.util.*;

/**
 * Backend Postgres PURO (JDBC) — sem pgvector. Espelha rag3d/pgstore.py e
 * rag3d-js/src/pgstore.js e usa o mesmo esquema holográfico legado, então lê e
 * escreve índices Hash que ainda não foram certificados pela Retrieval V2.
 * Índices com fingerprint V2 são recusados para evitar corrupção silenciosa.
 *
 * <p>Só tipos nativos: BIT(1024) para a assinatura (distância de Hamming via
 * bit_count), INT[] para as facetas, BYTEA para eco e constelação.
 */
public final class PgHoloStore implements AutoCloseable {

    static final long FINGERPRINT_LOCK_ID = 0x5241473344465032L;
    static final int FINGERPRINT_LOCK_TIMEOUT_MILLIS = 5000;
    static final int SCHEMA_LOCK_TIMEOUT_MILLIS = 5000;
    static final int SCHEMA_STATEMENT_TIMEOUT_MILLIS = 30000;
    private static final String KINDS = "('chunk','turn','summary')";
    private static final String COLS =
            "id,doc_id,parent_id,kind,pos,text,ctx,n_tokens,created,importance,turn_no,accessed_turn";

    private final Connection db;
    private final Holo holo;
    private boolean inBatch;

    public int bandPrefilterThreshold = 20000;
    public int prefetchFactor = 4;
    public int prefetchMin = 200;

    public PgHoloStore(String jdbcUrl, String user, String password, int denseDim, int colbertDim)
            throws SQLException {
        this.holo = createHolo(denseDim, colbertDim);
        this.db = openConnection(jdbcUrl, user, password);
        this.inBatch = false;
        try {
            ensureSchema();
            ensureEncoderFingerprint("hash:" + denseDim + ":" + colbertDim);
        } catch (SQLException | RuntimeException exc) {
            try {
                this.db.close();
            } catch (SQLException closeError) {
                exc.addSuppressed(closeError);
            }
            throw new SQLException("postgres-holo initialization failed");
        }
    }

    private static Holo createHolo(int denseDim, int colbertDim) throws SQLException {
        try {
            return new Holo(denseDim, colbertDim);
        } catch (RuntimeException failure) {
            throw new SQLException("postgres-holo initialization failed");
        }
    }

    private static Connection openConnection(String url, String user, String password)
            throws SQLException {
        Connection connection = null;
        try {
            connection = DriverManager.getConnection(url, user, password);
            connection.setAutoCommit(true);
            return connection;
        } catch (SQLException | RuntimeException failure) {
            if (connection != null) {
                try {
                    connection.close();
                } catch (SQLException ignored) {
                    // Preserve the fixed, secret-safe initialization failure.
                }
            }
            throw new SQLException("postgres-holo initialization failed");
        }
    }

    public void beginBatch() throws SQLException {
        if (inBatch) throw new SQLException("transaction already active");
        db.setAutoCommit(false);
        inBatch = true;
    }

    public void commitBatch() throws SQLException {
        if (!inBatch) throw new SQLException("no active transaction");
        db.commit();
        db.setAutoCommit(true);
        inBatch = false;
    }

    public void rollbackBatch() throws SQLException {
        if (!inBatch) return;
        try {
            db.rollback();
        } finally {
            db.setAutoCommit(true);
            inBatch = false;
        }
    }

    private void ensureSchema() throws SQLException {
        String ddl = """
            CREATE TABLE IF NOT EXISTS holo_meta(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS holo_docs(
              id BIGSERIAL PRIMARY KEY, source TEXT, title TEXT,
              created DOUBLE PRECISION, n_tokens INT, meta TEXT);
            CREATE TABLE IF NOT EXISTS holo_grams(
              id BIGSERIAL PRIMARY KEY,
              doc_id BIGINT CONSTRAINT holo_grams_doc_fk
                REFERENCES holo_docs(id) ON DELETE CASCADE,
              parent_id BIGINT CONSTRAINT holo_grams_parent_fk
                REFERENCES holo_grams(id) ON DELETE SET NULL,
              kind TEXT NOT NULL DEFAULT 'chunk',
              pos INT, text TEXT NOT NULL, ctx TEXT,
              n_tokens INT, created DOUBLE PRECISION,
              importance REAL DEFAULT 0.5, turn_no INT, accessed_turn INT,
              sig BIT(1024), bands INT[], echo BYTEA,
              constellation BYTEA, n_tok INT);
            CREATE INDEX IF NOT EXISTS idx_grams_kind ON holo_grams(kind);
            CREATE INDEX IF NOT EXISTS idx_grams_doc ON holo_grams(doc_id);
            CREATE INDEX IF NOT EXISTS idx_grams_bands ON holo_grams USING GIN(bands);
            CREATE TABLE IF NOT EXISTS holo_spectrum(
              term BIGINT NOT NULL,
              gram_id BIGINT NOT NULL CONSTRAINT holo_spectrum_gram_fk
                REFERENCES holo_grams(id) ON DELETE CASCADE,
              weight REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_spectrum_term ON holo_spectrum(term);
            CREATE INDEX IF NOT EXISTS idx_spectrum_gram ON holo_spectrum(gram_id);
            DO $rag3d$
            BEGIN
              IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conrelid='holo_grams'::regclass AND conname='holo_grams_doc_fk'
              ) THEN
                ALTER TABLE holo_grams ADD CONSTRAINT holo_grams_doc_fk
                  FOREIGN KEY(doc_id) REFERENCES holo_docs(id) ON DELETE CASCADE NOT VALID;
              END IF;
              IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conrelid='holo_grams'::regclass AND conname='holo_grams_parent_fk'
              ) THEN
                ALTER TABLE holo_grams ADD CONSTRAINT holo_grams_parent_fk
                  FOREIGN KEY(parent_id) REFERENCES holo_grams(id) ON DELETE SET NULL NOT VALID;
              END IF;
              IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conrelid='holo_spectrum'::regclass AND conname='holo_spectrum_gram_fk'
              ) THEN
                ALTER TABLE holo_spectrum ADD CONSTRAINT holo_spectrum_gram_fk
                  FOREIGN KEY(gram_id) REFERENCES holo_grams(id) ON DELETE CASCADE NOT VALID;
              END IF;
              IF NOT EXISTS (
                SELECT 1 FROM pg_constraint c
                WHERE c.conrelid='holo_grams'::regclass
                  AND c.conname='holo_grams_doc_fk' AND c.contype='f'
                  AND c.confrelid='holo_docs'::regclass AND c.confdeltype='c'
                  AND c.conkey=ARRAY[(SELECT attnum FROM pg_attribute
                    WHERE attrelid='holo_grams'::regclass AND attname='doc_id')]
                  AND c.confkey=ARRAY[(SELECT attnum FROM pg_attribute
                    WHERE attrelid='holo_docs'::regclass AND attname='id')]
              ) THEN
                RAISE EXCEPTION 'incompatible holo_grams_doc_fk constraint';
              END IF;
              IF NOT EXISTS (
                SELECT 1 FROM pg_constraint c
                WHERE c.conrelid='holo_grams'::regclass
                  AND c.conname='holo_grams_parent_fk' AND c.contype='f'
                  AND c.confrelid='holo_grams'::regclass AND c.confdeltype='n'
                  AND c.conkey=ARRAY[(SELECT attnum FROM pg_attribute
                    WHERE attrelid='holo_grams'::regclass AND attname='parent_id')]
                  AND c.confkey=ARRAY[(SELECT attnum FROM pg_attribute
                    WHERE attrelid='holo_grams'::regclass AND attname='id')]
              ) THEN
                RAISE EXCEPTION 'incompatible holo_grams_parent_fk constraint';
              END IF;
              IF NOT EXISTS (
                SELECT 1 FROM pg_constraint c
                WHERE c.conrelid='holo_spectrum'::regclass
                  AND c.conname='holo_spectrum_gram_fk' AND c.contype='f'
                  AND c.confrelid='holo_grams'::regclass AND c.confdeltype='c'
                  AND c.conkey=ARRAY[(SELECT attnum FROM pg_attribute
                    WHERE attrelid='holo_spectrum'::regclass AND attname='gram_id')]
                  AND c.confkey=ARRAY[(SELECT attnum FROM pg_attribute
                    WHERE attrelid='holo_grams'::regclass AND attname='id')]
              ) THEN
                RAISE EXCEPTION 'incompatible holo_spectrum_gram_fk constraint';
              END IF;
            END
            $rag3d$;
            ALTER TABLE holo_grams VALIDATE CONSTRAINT holo_grams_doc_fk;
            ALTER TABLE holo_grams VALIDATE CONSTRAINT holo_grams_parent_fk;
            ALTER TABLE holo_spectrum VALIDATE CONSTRAINT holo_spectrum_gram_fk;
            """;
        boolean priorAutoCommit = db.getAutoCommit();
        if (priorAutoCommit) db.setAutoCommit(false);
        try {
            try (PreparedStatement timeout = db.prepareStatement(
                    "SELECT set_config('lock_timeout', ?, true), "
                    + "set_config('statement_timeout', ?, true)")) {
                timeout.setString(1, SCHEMA_LOCK_TIMEOUT_MILLIS + "ms");
                timeout.setString(2, SCHEMA_STATEMENT_TIMEOUT_MILLIS + "ms");
                timeout.execute();
            }
            try (Statement st = db.createStatement()) { st.execute(ddl); }
            db.commit();
        } catch (SQLException | RuntimeException failure) {
            try {
                db.rollback();
            } catch (SQLException rollbackError) {
                failure.addSuppressed(rollbackError);
            }
            throw new SQLException("postgres-holo schema initialization failed");
        } finally {
            if (priorAutoCommit) db.setAutoCommit(true);
        }
    }

    static void validateEncoderFingerprint(String stored, String expected) throws SQLException {
        if (stored == null || !stored.equals(expected)) {
            throw new SQLException("incompatible encoder fingerprint");
        }
    }

    static void validateNoRetrievalV2Fingerprint(boolean present) throws SQLException {
        if (present) {
            throw new SQLException(
                "retrieval V2 holographic index requires the Python adapter"
            );
        }
    }

    private static void acquireEncoderFingerprintLock(Connection connection) throws SQLException {
        try {
            try (Statement timeout = connection.createStatement()) {
                timeout.execute(
                    "SET LOCAL lock_timeout = '" + FINGERPRINT_LOCK_TIMEOUT_MILLIS + "ms'"
                );
            }
            try (PreparedStatement lock = connection.prepareStatement(
                    "SELECT pg_advisory_xact_lock(?)")) {
                lock.setLong(1, FINGERPRINT_LOCK_ID);
                lock.execute();
            }
        } catch (SQLException failure) {
            throw new SQLException("encoder fingerprint lock acquisition failed");
        }
    }

    private void ensureEncoderFingerprint(String expected) throws SQLException {
        boolean priorAutoCommit = db.getAutoCommit();
        if (priorAutoCommit) db.setAutoCommit(false);
        try {
            acquireEncoderFingerprintLock(db);

            try (Statement checkV2 = db.createStatement();
                 ResultSet rows = checkV2.executeQuery(
                     "SELECT EXISTS(SELECT 1 FROM holo_meta WHERE key IN "
                     + "('retrieval_v2_fingerprint','retrieval_v2_fingerprint_sha256'))")) {
                rows.next();
                validateNoRetrievalV2Fingerprint(rows.getBoolean(1));
            }

            String stored = null;
            try (PreparedStatement select = db.prepareStatement(
                    "SELECT value FROM holo_meta WHERE key='encoder'")) {
                try (ResultSet rows = select.executeQuery()) {
                    if (rows.next()) stored = rows.getString(1);
                }
            }

            if (stored == null) {
                boolean populated;
                try (Statement count = db.createStatement();
                     ResultSet rows = count.executeQuery(
                         "SELECT EXISTS(SELECT 1 FROM holo_grams LIMIT 1)")) {
                    rows.next();
                    populated = rows.getBoolean(1);
                }
                if (populated) {
                    throw new SQLException(
                        "missing encoder fingerprint on populated holographic index"
                    );
                }
                try (PreparedStatement insert = db.prepareStatement(
                        "INSERT INTO holo_meta(key,value) VALUES('encoder',?) "
                        + "ON CONFLICT(key) DO NOTHING")) {
                    insert.setString(1, expected);
                    insert.executeUpdate();
                }
                try (PreparedStatement select = db.prepareStatement(
                        "SELECT value FROM holo_meta WHERE key='encoder'")) {
                    try (ResultSet rows = select.executeQuery()) {
                        if (rows.next()) stored = rows.getString(1);
                    }
                }
            }

            validateEncoderFingerprint(stored, expected);
            db.commit();
        } catch (SQLException | RuntimeException exc) {
            try {
                db.rollback();
            } catch (SQLException rollbackError) {
                exc.addSuppressed(rollbackError);
            }
            throw exc;
        } finally {
            if (priorAutoCommit) db.setAutoCommit(true);
        }
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
            try (ResultSet rs = ps.executeQuery()) { rs.next(); return rs.getLong(1); }
        }
    }

    public int nChunks() throws SQLException {
        // Other Python/Node/Java processes may write concurrently; a local
        // cache would corrupt ANN threshold selection and sparse IDF.
        try (Statement st = db.createStatement();
             ResultSet rs = st.executeQuery(
                     "SELECT COUNT(*) FROM holo_grams WHERE kind IN " + KINDS)) {
            rs.next();
            return rs.getInt(1);
        }
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

        // re-pontuação aproximada pelo eco int8 quantizado (não Hamming)
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
        Long[] terms = qsparse.keySet().toArray(new Long[0]);
        Double[] weights = new Double[terms.length];
        for (int i = 0; i < terms.length; i++) weights[i] = (double) qsparse.get(terms[i]);
        String sql = """
            WITH query_terms(term,qweight) AS (
              SELECT * FROM unnest(?::bigint[],?::double precision[])
            ), universe AS (
              SELECT GREATEST(COUNT(*),1)::double precision AS n_docs
              FROM holo_grams WHERE kind IN ('chunk','turn','summary')
            ), term_df AS (
              SELECT p.term, COUNT(DISTINCT p.gram_id)::double precision AS df
              FROM holo_spectrum p JOIN holo_grams g ON g.id=p.gram_id
              JOIN query_terms q ON q.term=p.term
              WHERE g.kind IN ('chunk','turn','summary') GROUP BY p.term
            ), scored AS (
              SELECT p.gram_id, SUM(q.qweight::double precision *
                p.weight::double precision * LN(1.0 +
                (u.n_docs - d.df + 0.5) / (d.df + 0.5))) AS score
              FROM holo_spectrum p JOIN holo_grams g ON g.id=p.gram_id
              JOIN query_terms q ON q.term=p.term
              JOIN term_df d ON d.term=p.term CROSS JOIN universe u
              WHERE g.kind IN ('chunk','turn','summary') GROUP BY p.gram_id
            ) SELECT gram_id,score FROM scored
            ORDER BY score DESC,gram_id ASC LIMIT ?
            """;
        List<Fusion.Scored> out = new ArrayList<>();
        try (PreparedStatement ps = db.prepareStatement(sql)) {
            ps.setArray(1, db.createArrayOf("int8", terms));
            ps.setArray(2, db.createArrayOf("float8", weights));
            ps.setInt(3, k);
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    double score = rs.getDouble(2);
                    if (!Double.isFinite(score)) throw new SQLException("sparse scores must be finite");
                    out.add(new Fusion.Scored(rs.getLong(1), score));
                }
            }
        }
        return out;
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
    }

    @Override public void close() throws SQLException { db.close(); }
}
