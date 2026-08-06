package io.rag3d.spring;

import io.rag3d.Rag3D;

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.sql.SQLException;
import java.util.List;
import java.util.Map;

/**
 * Controller REST opcional — expõe ingestão e busca tridimensional.
 *
 * <p>Ative com {@code rag3d.web.enabled=true} no application.yml. Serve como
 * exemplo pronto: em produção, prefira injetar o bean {@link Rag3D} no seu
 * próprio serviço e controlar autenticação/autorização por lá.
 *
 * <pre>
 * POST /rag3d/ingest   {"text": "...", "title": "contrato.pdf"}
 * GET  /rag3d/search?q=prazo&amp;k=8
 * </pre>
 */
@RestController
@RequestMapping("/rag3d")
@ConditionalOnProperty(prefix = "rag3d.web", name = "enabled", havingValue = "true")
public class Rag3DController {

    private final Rag3D rag;

    public Rag3DController(Rag3D rag) {
        this.rag = rag;
    }

    public record IngestRequest(String text, String title) {}

    @PostMapping("/ingest")
    public ResponseEntity<?> ingest(@RequestBody IngestRequest req) throws SQLException {
        if (req == null || req.text() == null || req.text().isBlank()) {
            return ResponseEntity.badRequest().body(Map.of("error", "texto vazio"));
        }
        int chunks = rag.ingest(req.text(), "api", req.title() == null ? "" : req.title());
        return ResponseEntity.ok(Map.of("chunks", chunks));
    }

    @GetMapping("/search")
    public ResponseEntity<?> search(@RequestParam("q") String q,
                                    @RequestParam(value = "k", defaultValue = "8") int k)
            throws SQLException {
        if (q == null || q.isBlank()) {
            return ResponseEntity.badRequest().body(Map.of("error", "consulta vazia"));
        }
        Rag3D.Result r = rag.search(q, k);
        List<Map<String, Object>> fused = r.fused().stream()
                .map(h -> Map.<String, Object>of(
                        "id", h.id(), "text", h.text(),
                        "score", h.score(), "interference", h.interference(),
                        "channels", h.channels()))
                .toList();
        return ResponseEntity.ok(Map.of("query", r.query(), "fused", fused,
                "views", r.views().keySet()));
    }

    @GetMapping("/stats")
    public ResponseEntity<?> stats() throws SQLException {
        return ResponseEntity.ok(Map.of(
                "chunks", rag.store().nChunks(),
                "backend", "postgres-holo (sem pgvector)",
                "fusion", rag.fusion,
                "diversity", rag.diversity));
    }
}
