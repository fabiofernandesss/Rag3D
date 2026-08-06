package io.rag3d.spring;

import io.rag3d.Rag3D;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.sql.SQLException;

/**
 * Auto-configuração Spring Boot do RAG3D.
 *
 * <p>Basta ter a dependência no classpath e configurar o datasource no
 * {@code application.yml} — o bean {@link Rag3D} fica disponível para injeção:
 *
 * <pre>{@code
 * rag3d:
 *   jdbc-url: jdbc:postgresql://localhost:5432/rag3d   # Postgres COMUM, sem pgvector
 *   username: postgres
 *   password: ${DB_PASSWORD}
 *   top-k: 10
 *   diversity: 0.35        # seleção fermiônica (0 = ranking puro)
 *   fusion: quantum        # quantum | rrf
 * }</pre>
 *
 * <pre>{@code
 * @Service
 * public class MeuServico {
 *     private final Rag3D rag;
 *     public MeuServico(Rag3D rag) { this.rag = rag; }   // injetado
 * }
 * }</pre>
 */
@Configuration
@ConditionalOnProperty(prefix = "rag3d", name = "jdbc-url")
public class Rag3DAutoConfiguration {

    @Value("${rag3d.jdbc-url}") private String jdbcUrl;
    @Value("${rag3d.username:postgres}") private String username;
    @Value("${rag3d.password:}") private String password;
    @Value("${rag3d.top-k:10}") private int topK;
    @Value("${rag3d.channel-k:100}") private int channelK;
    @Value("${rag3d.diversity:0.35}") private double diversity;
    @Value("${rag3d.fusion:quantum}") private String fusion;
    @Value("${rag3d.interference-strength:1.0}") private double interferenceStrength;

    @Bean(destroyMethod = "close")
    @ConditionalOnMissingBean
    public Rag3D rag3D() throws SQLException {
        Rag3D rag = Rag3D.connect(jdbcUrl, username, password);
        rag.topK = topK;
        rag.channelK = channelK;
        rag.diversity = diversity;
        rag.fusion = fusion;
        rag.interferenceStrength = interferenceStrength;
        return rag;
    }
}
