package io.rag3d.core;

import java.lang.reflect.Method;
import java.sql.SQLException;
import java.util.Set;

/** Pure contract checks that do not require a JDBC driver or live database. */
public final class PgHoloStoreContractCheck {
    private PgHoloStoreContractCheck() {}

    public static void main(String[] args) throws Exception {
        if (PgHoloStore.FINGERPRINT_LOCK_ID != 0x5241473344465032L) {
            throw new AssertionError("fingerprint lock must match Python and JavaScript");
        }
        if (PgHoloStore.FINGERPRINT_LOCK_TIMEOUT_MILLIS != 5000) {
            throw new AssertionError("fingerprint lock wait must be bounded");
        }
        if (PgHoloStore.SCHEMA_LOCK_TIMEOUT_MILLIS != 5000) {
            throw new AssertionError("schema lock wait must be bounded");
        }
        if (PgHoloStore.SCHEMA_STATEMENT_TIMEOUT_MILLIS != 30000) {
            throw new AssertionError("schema validation runtime must be bounded");
        }
        Set<String> methods = new java.util.HashSet<>();
        for (Method method : PgHoloStore.class.getMethods()) {
            methods.add(method.getName());
        }
        if (!methods.containsAll(Set.of("beginBatch", "commitBatch", "rollbackBatch"))) {
            throw new AssertionError("Java store must expose atomic ingest transaction hooks");
        }

        PgHoloStore.validateNoRetrievalV2Fingerprint(false);
        try {
            PgHoloStore.validateNoRetrievalV2Fingerprint(true);
            throw new AssertionError("Java must not mutate a Python V2 index");
        } catch (SQLException expected) {
            if (!expected.getMessage().equals(
                    "retrieval V2 holographic index requires the Python adapter")) {
                throw expected;
            }
        }

        PgHoloStore.validateEncoderFingerprint("hash:1024:128", "hash:1024:128");

        try {
            PgHoloStore.validateEncoderFingerprint("bge-m3:1024:128", "hash:1024:128");
            throw new AssertionError("incompatible encoder fingerprint must fail");
        } catch (SQLException expected) {
            if (!expected.getMessage().contains("incompatible encoder fingerprint")) {
                throw expected;
            }
            if (expected.getMessage().contains("bge-m3")) {
                throw new AssertionError("fingerprint error must not echo stored metadata");
            }
        }
    }
}
