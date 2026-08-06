package io.rag3d.core;

import java.nio.charset.StandardCharsets;
import java.util.zip.CRC32;

/**
 * Núcleo determinístico portável — bit a bit igual a rag3d/portable.py e
 * rag3d-js/src/portable.js.
 *
 * <p>Mesmos hiperplanos LSH, mesma assinatura, mesmo crc32 das outras
 * linguagens. Um Holograma Textual salvo por Python/JS é idêntico ao salvo por
 * Java: retrieval cruzado no MESMO Postgres funciona. É o que faz o RAG3D valer
 * para todas as linguagens de verdade.
 *
 * <p>splitmix64 em long (wraparound de 64 bits é nativo em Java);
 * Box-Muller para normais; CRC-32 IEEE via java.util.zip.
 */
public final class Portable {

    private static final long GOLDEN = 0x9E3779B97F4A7C15L;
    private static final long C1 = 0xBF58476D1CE4E5B9L;
    private static final long C2 = 0x94D049BB133111EBL;

    private Portable() {}

    /** splitmix64 counter-based: idêntico ao sequencial (estado_k = seed + k*GOLDEN). */
    private static long splitmix64(long seed, long k) {
        long z = seed + k * GOLDEN;
        z = (z ^ (z >>> 30)) * C1;
        z = (z ^ (z >>> 27)) * C2;
        return z ^ (z >>> 31);
    }

    /** n doubles em [0,1) — top 53 bits (igual a Python e JS). */
    public static double[] uniforms(long seed, int n) {
        double[] out = new double[n];
        double denom = (double) (1L << 53);
        for (int i = 0; i < n; i++) {
            out[i] = (double) (splitmix64(seed, i + 1L) >>> 11) / denom;
        }
        return out;
    }

    /** n normais padrão via Box-Muller (ramo do cosseno), idêntico às outras linguagens. */
    public static float[] normals(long seed, int n) {
        double[] u = uniforms(seed, 2 * n);
        float[] out = new float[n];
        for (int j = 0; j < n; j++) {
            double u1 = Math.max(u[2 * j], 1e-12);
            double u2 = u[2 * j + 1];
            out[j] = (float) (Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2));
        }
        return out;
    }

    /** Matriz (rows x cols) achatada em ordem C (row-major) — mesma de numpy. */
    public static float[] hyperplanes(long seed, int rows, int cols) {
        return normals(seed, rows * cols); // índice i*cols + j
    }

    /** CRC-32 IEEE dos bytes UTF-8 — mesmo resultado de zlib.crc32 (Python) e crc32str (JS). */
    public static long crc32(String s) {
        CRC32 c = new CRC32();
        c.update(s.getBytes(StandardCharsets.UTF_8));
        return c.getValue(); // 0 .. 2^32-1
    }
}
