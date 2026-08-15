import { test } from "node:test";
import assert from "node:assert";

import { PgHoloStore } from "../src/pgstore.js";
import { MemStore } from "../src/memstore.js";
import { TriRag } from "../src/engine.js";

class FingerprintDb {
  constructor({ stored = null, populated = false, v2Stored = false, failOn = null } = {}) {
    this.stored = stored;
    this.populated = populated;
    this.v2Stored = v2Stored;
    this.failOn = failOn;
    this.calls = [];
  }

  async query(sql, params = []) {
    this.calls.push([sql, params]);
    if (this.failOn && sql.includes(this.failOn)) {
      throw new Error("postgresql://admin:super-secret@example.invalid/prod");
    }
    if (sql.includes("retrieval_v2_fingerprint")) {
      return { rows: [{ v2_stored: this.v2Stored }] };
    }
    if (sql.includes("SELECT value FROM holo_meta")) {
      return { rows: this.stored === null ? [] : [{ value: this.stored }] };
    }
    if (sql.includes("SELECT EXISTS")) return { rows: [{ populated: this.populated }] };
    if (sql.includes("INSERT INTO holo_meta")) {
      if (this.stored === null) this.stored = params[0];
      return { rows: [] };
    }
    return { rows: [] };
  }
}

test("fingerprint PostgreSQL usa transação, timeout e lock cross-language", async () => {
  const db = new FingerprintDb();
  const store = new PgHoloStore(db, 8, 4);

  await store.ensureEncoderFingerprint("hash:8:4");

  assert.equal(db.stored, "hash:8:4");
  assert.equal(db.calls[0][0], "BEGIN");
  assert.match(db.calls[1][0], /set_config\('lock_timeout'/);
  assert.deepEqual(db.calls[1][1], ["5000ms"]);
  assert.match(db.calls[2][0], /pg_advisory_xact_lock\(\$1::bigint\)/);
  assert.deepEqual(db.calls[2][1], ["5927096870110646322"]);
  assert.equal(db.calls.at(-1)[0], "COMMIT");
});

test("fingerprint PostgreSQL falha fechado sem ecoar segredo", async () => {
  const secret = "super-secret";
  const db = new FingerprintDb({ failOn: "pg_advisory_xact_lock" });
  const store = new PgHoloStore(db, 8, 4);

  await assert.rejects(
    store.ensureEncoderFingerprint("hash:8:4"),
    (error) => {
      assert.equal(error.message, "postgres-holo fingerprint initialization failed");
      assert.ok(!String(error).includes(secret));
      return true;
    },
  );
  assert.equal(db.calls.at(-1)[0], "ROLLBACK");
});

test("fingerprint ausente em índice povoado é incompatível", async () => {
  const db = new FingerprintDb({ populated: true });
  const store = new PgHoloStore(db, 8, 4);

  await assert.rejects(
    store.ensureEncoderFingerprint("hash:8:4"),
    /^Error: missing encoder fingerprint on populated holographic index$/,
  );
  assert.equal(db.calls.at(-1)[0], "ROLLBACK");
});

test("adapter Node não modifica índice Retrieval V2 certificado pelo Python", async () => {
  const db = new FingerprintDb({ stored: "hash:8:4", v2Stored: true });
  const store = new PgHoloStore(db, 8, 4);

  await assert.rejects(
    store.ensureEncoderFingerprint("hash:8:4"),
    /^Error: retrieval V2 holographic index requires the Python adapter$/,
  );
  assert.equal(db.calls.at(-1)[0], "ROLLBACK");
});

test("fingerprint local ausente não certifica índice já povoado", async () => {
  const store = new MemStore(null);
  await store.addChunk(
    null,
    "legacy",
    "legacy",
    1,
    { dense: Float32Array.from([1]), sparse: new Map(), tokens: [] },
  );

  await assert.rejects(
    store.ensureEncoderFingerprint("hash:1:1"),
    /^Error: missing encoder fingerprint on populated local index$/,
  );
  assert.equal(await store.getMeta("encoder"), null);
});

test("contagem PostgreSQL não fica stale entre processos", async () => {
  const counts = [1, 2];
  const db = {
    calls: 0,
    async query() {
      const c = counts[this.calls];
      this.calls += 1;
      return { rows: [{ c }] };
    },
  };
  const store = new PgHoloStore(db, 8, 4);

  assert.equal(await store.nChunks(), 1);
  assert.equal(await store.nChunks(), 2);
  assert.equal(db.calls, 2);
});

test("busca esparsa filtra órfãos/kinds e agrega top-k no SQL", async () => {
  const db = {
    call: null,
    async query(sql, params) {
      this.call = [sql, params];
      return { rows: [{ gram_id: "7", score: "0.75" }] };
    },
  };
  const store = new PgHoloStore(db, 8, 4);

  assert.deepEqual(await store.sparseSearch(new Map([[11, 0.5]]), 3), [[7, 0.75]]);
  assert.match(db.call[0], /JOIN holo_grams g ON g\.id=p\.gram_id/);
  assert.match(db.call[0], /g\.kind IN \('chunk','turn','summary'\)/);
  assert.match(db.call[0], /COUNT\(DISTINCT p\.gram_id\)/);
  assert.match(db.call[0], /ORDER BY score DESC,gram_id ASC LIMIT \$3/);
  assert.deepEqual(db.call[1], [[11], [0.5], 3]);
});

test("bootstrap de schema PostgreSQL usa transação e timeouts locais", async () => {
  const calls = [];
  const client = {
    async connect() {},
    async query(sql, params = []) {
      calls.push([sql, params]);
      return { rows: [] };
    },
    async end() {},
  };

  const store = await PgHoloStore._connectClient(client, 8, 4);

  assert.equal(calls[0][0], "BEGIN");
  assert.match(calls[1][0], /set_config\('lock_timeout'/);
  assert.deepEqual(calls[1][1], ["5000ms"]);
  assert.match(calls[2][0], /set_config\('statement_timeout'/);
  assert.deepEqual(calls[2][1], ["30000ms"]);
  assert.match(calls[3][0], /CREATE TABLE IF NOT EXISTS holo_meta/);
  assert.equal(calls.at(-1)[0], "COMMIT");
  await store.close();
});

test("falha de schema PostgreSQL fecha o client e mascara driver", async () => {
  let ended = 0;
  const client = {
    async connect() {},
    async query() { throw new Error("postgresql://admin:super-secret@db/prod"); },
    async end() { ended += 1; },
  };

  await assert.rejects(
    PgHoloStore._connectClient(client, 8, 4),
    /^Error: postgres-holo initialization failed$/,
  );
  assert.equal(ended, 1);
});

test("facade fecha store quando fingerprint falha", async () => {
  const original = PgHoloStore.connect;
  let closeCalls = 0;
  const store = {
    async ensureEncoderFingerprint() { throw new Error("incompatible encoder fingerprint"); },
    async close() { closeCalls += 1; },
  };
  PgHoloStore.connect = async () => store;
  try {
    await assert.rejects(
      TriRag.create({ pgDsn: "postgresql://example.invalid/rag3d_test" }),
      /incompatible encoder fingerprint/,
    );
    assert.equal(closeCalls, 1);
  } finally {
    PgHoloStore.connect = original;
  }
});
