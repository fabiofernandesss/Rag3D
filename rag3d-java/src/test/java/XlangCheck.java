import io.rag3d.Rag3D;
import java.util.*;

/**
 * Prova tri-linguagem: o Java encontra, no MESMO Postgres, os documentos que o
 * Python (ou o JS) ingeriu — e recomputa a assinatura com Hamming 0/1024.
 *
 *   python3 tests/xlang_ingest.py            # Python escreve
 *   java -cp target/classes:lib/postgresql.jar XlangCheck   # Java lê
 */
public class XlangCheck {
    public static void main(String[] args) throws Exception {
        String url = System.getenv().getOrDefault("RAG3D_JDBC",
                "jdbc:postgresql://localhost:5434/rag3d");
        String user = System.getenv().getOrDefault("RAG3D_USER", "postgres");
        String pass = System.getenv().getOrDefault("RAG3D_PASS", "rag3d");

        String[][] probes = {
            {"contrato de aluguel prazo", "contrato"},
            {"rocket launch date", "rocket"},
            {"会议在哪里举行", "北京"},
            {"receita bolo de fubá", "fubá"},
            {"código do cofre", "7742"},
            {"tension artérielle régime", "artérielle"},
        };

        try (Rag3D rag = Rag3D.connect(url, user, pass)) {
            System.out.println("chunks no índice: " + rag.store().nChunks());
            int found = 0;
            for (String[] p : probes) {
                Rag3D.Result r = rag.search(p[0], 5);
                boolean hit = r.fused().stream().anyMatch(h -> h.text().contains(p[1]));
                System.out.printf("  %-32s -> %s%n", "\"" + p[0] + "\"", hit ? "achou" : "NÃO achou");
                if (hit) found++;
                if (!r.fused().isEmpty() && r.views().size() != 3) {
                    throw new IllegalStateException("esperava 3 visões por eixo");
                }
            }
            System.out.printf("JAVA achou %d/%d docs ingeridos pelo Python%n", found, probes.length);
            if (found < probes.length) System.exit(1);
        }
    }
}
