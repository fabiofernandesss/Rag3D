package io.rag3d.core;

import java.util.List;

/**
 * Holograma Textual — espelha rag3d/holo.py e rag3d-js/src/holo.js.
 *
 * <p>Projeta o TriVec em assinatura LSH (BIT), facetas (INT[]), eco int8
 * (BYTEA) e constelação de tokens (BYTEA). Os hiperplanos vêm de
 * {@link Portable}, idênticos aos de Python/JS — então as assinaturas batem
 * bit a bit entre as três linguagens.
 */
public final class Holo {

    public static final int HOLO_BITS = 1024;
    public static final int TOKEN_BITS = 128;
    public static final int N_BANDS = 16;
    private static final long SEED = 7742L;

    private final int denseDim;
    private final int colbertDim;
    private final float[] planes;     // [d*HOLO_BITS + bit]
    private final float[] tokPlanes;  // [d*TOKEN_BITS + bit]

    public Holo(int denseDim, int colbertDim) {
        this.denseDim = denseDim;
        this.colbertDim = colbertDim;
        this.planes = Portable.hyperplanes(SEED, denseDim, HOLO_BITS);
        this.tokPlanes = Portable.hyperplanes(SEED ^ 0x5DL, colbertDim, TOKEN_BITS);
    }

    /** Bits do projetor (dot >= 0), acumulando em double como o JS. */
    private static byte[] project(float[] vec, float[] planes, int dim, int out) {
        double[] acc = new double[out];
        for (int d = 0; d < dim; d++) {
            float v = vec[d];
            if (v == 0.0f) continue;                 // vetor esparso: pula
            int base = d * out;
            for (int b = 0; b < out; b++) acc[b] += (double) v * planes[base + b];
        }
        byte[] bits = new byte[out];
        for (int b = 0; b < out; b++) bits[b] = (byte) (acc[b] >= 0 ? 1 : 0);
        return bits;
    }

    /** Empacota bits em bytes, big-endian (bit 0 -> MSB), igual np.packbits. */
    private static byte[] packbits(byte[] bits) {
        byte[] out = new byte[(bits.length + 7) / 8];
        for (int i = 0; i < bits.length; i++) {
            if (bits[i] != 0) out[i >> 3] |= (byte) (1 << (7 - (i & 7)));
        }
        return out;
    }

    /** Vetor denso -> 128 bytes (1024 bits de LSH por hiperplano). */
    public byte[] signDense(float[] vec) {
        return packbits(project(vec, planes, denseDim, HOLO_BITS));
    }

    /** 128 bytes -> "0101..." (1024) para coluna BIT(1024) do Postgres. */
    public static String sigToBitstring(byte[] sig) {
        StringBuilder sb = new StringBuilder(sig.length * 8);
        for (byte b : sig) {
            for (int i = 7; i >= 0; i--) sb.append(((b >> i) & 1) != 0 ? '1' : '0');
        }
        return sb.toString();
    }

    /** Facetas: banda i = (i<<8)|byte_i (o índice entra no valor, sem colisão). */
    public static int[] bandsOf(byte[] sig) {
        int[] out = new int[N_BANDS];
        for (int i = 0; i < N_BANDS; i++) out[i] = (i << 8) | (sig[i] & 0xFF);
        return out;
    }

    /** Vetor normalizado -> int8 (eco para re-pontuação aproximada). */
    public static byte[] quantize(float[] vec) {
        byte[] out = new byte[vec.length];
        for (int i = 0; i < vec.length; i++) {
            double q = Math.rint((double) vec[i] * 127.0);   // round-half-to-even (= np.round)
            out[i] = (byte) Math.max(-127, Math.min(127, (int) q));
        }
        return out;
    }

    public static float[] dequantize(byte[] echo) {
        float[] out = new float[echo.length];
        for (int i = 0; i < echo.length; i++) out[i] = echo[i] / 127.0f;
        return out;
    }

    /** Tokens -> T assinaturas de 128 bits, concatenadas (T*16 bytes). */
    public byte[] signTokens(List<float[]> tokens) {
        int nb = TOKEN_BITS / 8;
        byte[] out = new byte[tokens.size() * nb];
        for (int t = 0; t < tokens.size(); t++) {
            byte[] packed = packbits(project(tokens.get(t), tokPlanes, colbertDim, TOKEN_BITS));
            System.arraycopy(packed, 0, out, t * nb, nb);
        }
        return out;
    }

    /** MaxSim binário: para cada token da consulta, menor Hamming no doc. */
    public double binaryMaxsim(byte[] qConst, byte[] dConst) {
        int nb = TOKEN_BITS / 8;
        int tq = qConst.length / nb, td = dConst.length / nb;
        if (tq == 0 || td == 0) return 0.0;
        double sum = 0;
        for (int i = 0; i < tq; i++) {
            int best = TOKEN_BITS;
            for (int j = 0; j < td; j++) {
                int ham = 0;
                for (int k = 0; k < nb; k++) {
                    ham += Integer.bitCount((qConst[i * nb + k] ^ dConst[j * nb + k]) & 0xFF);
                }
                if (ham < best) best = ham;
            }
            sum += 1.0 - (2.0 * best) / TOKEN_BITS;
        }
        return sum / tq;
    }

    public static int hamming(byte[] a, byte[] b) {
        int h = 0;
        for (int i = 0; i < a.length; i++) h += Integer.bitCount((a[i] ^ b[i]) & 0xFF);
        return h;
    }
}
