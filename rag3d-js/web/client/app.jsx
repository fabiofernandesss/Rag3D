import React, { useState, useEffect, useRef } from "react";
import { createRoot } from "react-dom/client";
import { marked } from "marked";

marked.setOptions({ breaks: true, gfm: true });

const C = {
  bg: "#0d1117", panel: "#161b22", border: "#30363d", text: "#e6edf3",
  dim: "#8b949e", accent: "#7c5cff", accent2: "#2ea043", user: "#1f6feb",
};

// renderiza markdown (negrito, listas, títulos, código) da resposta da IA
function Markdown({ text }) {
  const html = marked.parse(text || "");
  return <div className="md" dangerouslySetInnerHTML={{ __html: html }} />;
}

function CopyBtn({ text }) {
  const [ok, setOk] = useState(false);
  return (
    <button
      onClick={async () => { try { await navigator.clipboard.writeText(text); setOk(true); setTimeout(() => setOk(false), 1500); } catch {} }}
      style={{ fontSize: 11, padding: "2px 8px", borderRadius: 6, cursor: "pointer",
        background: "transparent", color: ok ? C.accent2 : C.dim, border: `1px solid ${C.border}` }}
      title="Copiar resposta">
      {ok ? "✓ copiado" : "⧉ copiar"}
    </button>
  );
}

function useStats() {
  const [stats, setStats] = useState(null);
  const refresh = () => fetch("/api/stats").then((r) => r.json()).then(setStats).catch(() => {});
  useEffect(() => { refresh(); }, []);
  return [stats, refresh];
}

function Uploader({ onDone, chunks }) {
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState(false);
  const [log, setLog] = useState([]);
  const inputRef = useRef();

  async function send(files) {
    if (!files.length) return;
    setBusy(true);
    const fd = new FormData();
    for (const f of files) fd.append("files", f);
    try {
      const r = await fetch("/api/upload", { method: "POST", body: fd });
      const data = await r.json();
      setLog((L) => [...data.results, ...L].slice(0, 20));
      onDone?.(data.stats);
    } catch (e) {
      setLog((L) => [{ file: "erro", ok: false, error: e.message }, ...L]);
    } finally { setBusy(false); }
  }

  async function reset() {
    if (!confirm("Limpar todos os documentos do índice?")) return;
    setBusy(true);
    try {
      await fetch("/api/reset", { method: "POST" });
      setLog([]);
      onDone?.();
    } finally { setBusy(false); }
  }

  return (
    <div style={{ padding: 16 }}>
      <div
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); send([...e.dataTransfer.files]); }}
        onClick={() => inputRef.current.click()}
        style={{
          border: `2px dashed ${drag ? C.accent : C.border}`, borderRadius: 12, padding: 28,
          textAlign: "center", cursor: "pointer", background: drag ? "#1c2333" : C.panel,
          transition: "all .15s",
        }}
      >
        <input ref={inputRef} type="file" multiple accept=".pdf,.txt,.md"
          style={{ display: "none" }} onChange={(e) => send([...e.target.files])} />
        <div style={{ fontSize: 32, marginBottom: 8 }}>📄</div>
        <div style={{ fontWeight: 600 }}>{busy ? "Processando…" : "Solte PDF/TXT aqui ou clique"}</div>
        <div style={{ color: C.dim, fontSize: 13, marginTop: 4 }}>
          cada arquivo vira hologramas nos 3 eixos
        </div>
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 10 }}>
        <span style={{ color: C.dim, fontSize: 12 }}>{chunks ?? 0} chunks no índice</span>
        {chunks > 0 && (
          <button onClick={reset} disabled={busy} style={{
            fontSize: 12, padding: "4px 10px", borderRadius: 6, cursor: "pointer",
            background: "transparent", color: "#f85149", border: `1px solid ${C.border}`,
          }}>🗑 Limpar índice</button>
        )}
      </div>
      <div style={{ marginTop: 8 }}>
        {log.map((r, i) => (
          <div key={i} style={{ fontSize: 13, padding: "4px 8px", color: r.ok ? C.text : "#f85149" }}>
            {r.ok ? "✓" : "✗"} <b>{r.file}</b>{" "}
            {r.ok ? <span style={{ color: C.dim }}>{r.chunks} chunks · ~{r.tokens} tokens</span>
              : <span style={{ color: "#f85149" }}>{r.error}</span>}
          </div>
        ))}
      </div>
    </div>
  );
}

function Sources({ sources }) {
  if (!sources?.length) return null;
  return (
    <div style={{ marginTop: 8, borderTop: `1px solid ${C.border}`, paddingTop: 8 }}>
      <div style={{ color: C.dim, fontSize: 12, marginBottom: 4 }}>evidências (fusão dos 3 eixos):</div>
      {sources.map((s, i) => (
        <div key={i} style={{ fontSize: 12, color: C.dim, marginBottom: 4 }}>
          <span style={{ color: C.accent }}>[{i + 1}]</span>{" "}
          {s.channels?.length ? <span title="eixos que acharam">({s.channels.join("·")}) </span> : null}
          {s.text}
        </div>
      ))}
    </div>
  );
}

function Chat() {
  const [msgs, setMsgs] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const endRef = useRef();
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [msgs, busy]);

  // stream via SSE: tokens chegam ao vivo e vão sendo anexados à mensagem
  async function send() {
    const q = input.trim();
    if (!q || busy) return;
    setInput("");
    setMsgs((M) => [...M, { role: "user", text: q }, { role: "ia", text: "", streaming: true }]);
    setBusy(true);
    const patchLast = (patch) => setMsgs((M) => {
      const c = [...M]; const i = c.length - 1;
      c[i] = { ...c[i], ...(typeof patch === "function" ? patch(c[i]) : patch) };
      return c;
    });
    try {
      const r = await fetch("/api/chat/stream", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: q }),
      });
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = "", acc = "", lastFlush = 0;
      // agrupa tokens: atualiza a tela no máx a cada 60ms (evita 100s de re-renders)
      const flush = (force) => {
        const now = performance.now();
        if (acc && (force || now - lastFlush > 60)) {
          const chunk = acc; acc = ""; lastFlush = now;
          patchLast((m) => ({ text: m.text + chunk }));
        }
      };
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split("\n\n");
        buf = lines.pop();
        for (const line of lines) {
          const t = line.trim();
          if (!t.startsWith("data:")) continue;
          const ev = JSON.parse(t.slice(5).trim());
          if (ev.token) { acc += ev.token; flush(false); }
          else if (ev.done) { flush(true); patchLast({ streaming: false, sources: ev.sources, sub: ev.subAnswers, readMode: ev.readMode, note: ev.note }); }
          else if (ev.error) { flush(true); patchLast({ streaming: false, error: ev.error }); }
        }
      }
      flush(true);
      patchLast({ streaming: false });
    } catch (e) {
      patchLast((m) => ({ streaming: false, text: m.text || "erro: " + e.message }));
    } finally { setBusy(false); }
  }

  const suggestions = [
    "Resuma o documento em 3 linhas",
    "Quais são as informações mais importantes?",
    "Liste as regras ou requisitos principais",
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ flex: 1, overflowY: "auto" }}>
        <div style={{ maxWidth: 820, margin: "0 auto", padding: "24px 20px 8px" }}>
          {msgs.length === 0 && (
            <div className="fade-in" style={{ textAlign: "center", marginTop: "14vh", color: C.dim }}>
              <div style={{ fontSize: 46, marginBottom: 12 }}>🔺</div>
              <div style={{ fontSize: 22, fontWeight: 700, color: C.text }}>Pergunte ao RAG3D</div>
              <div style={{ marginTop: 6 }}>Suba um PDF/TXT e a IA responde com base nele.</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 10, justifyContent: "center", marginTop: 22 }}>
                {suggestions.map((s) => (
                  <button key={s} className="chip" onClick={() => { setInput(s); }}>{s}</button>
                ))}
              </div>
            </div>
          )}
          {msgs.map((m, i) => <MsgRow key={i} m={m} />)}
          <div ref={endRef} />
        </div>
      </div>
      <div style={{ borderTop: `1px solid ${C.border}`, background: "rgba(13,17,23,0.85)", backdropFilter: "blur(6px)" }}>
        <div style={{ maxWidth: 820, margin: "0 auto", padding: "14px 20px 10px" }}>
          <div className={"composer" + (busy ? " busy" : "")}>
            <textarea value={input} onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
              rows={1} placeholder="Pergunte sobre os arquivos…  (Enter envia, Shift+Enter quebra linha)"
              style={{ flex: 1, resize: "none", maxHeight: 160, background: "transparent", color: C.text, border: 0, outline: "none", fontSize: 15, lineHeight: 1.5, fontFamily: "inherit" }} />
            <button onClick={send} disabled={busy || !input.trim()} className="send-btn" title="Enviar">
              {busy ? <span className="spinner" /> : "↑"}
            </button>
          </div>
          <div style={{ textAlign: "center", fontSize: 11, color: C.dim, marginTop: 8 }}>
            RAG3D lê os 3 eixos e a IA sintetiza · respostas baseadas só no documento
          </div>
        </div>
      </div>
    </div>
  );
}

// avatar circular (usuário / IA)
function Avatar({ ia }) {
  return (
    <div className="avatar" style={{
      background: ia ? "linear-gradient(135deg,#7c5cff,#4b32c3)" : "#1f6feb",
    }}>{ia ? "🔺" : "🙂"}</div>
  );
}

// pontos animados "pensando"
function Thinking() {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, color: C.dim, fontSize: 14 }}>
      <span className="dots"><i></i><i></i><i></i></span> lendo os 3 eixos…
    </div>
  );
}

// uma linha de mensagem estilo ChatGPT (avatar + conteúdo)
function MsgRow({ m }) {
  const isUser = m.role === "user";
  return (
    <div className="msg-row fade-in">
      <Avatar ia={!isUser} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 12.5, fontWeight: 700, color: isUser ? C.text : C.accent, marginBottom: 3 }}>
          {isUser ? "Você" : `IA${m.readMode ? " · " + m.readMode : ""}`}
        </div>
        {isUser
          ? <div style={{ whiteSpace: "pre-wrap", fontSize: 15, lineHeight: 1.55 }}>{m.text}</div>
          : (m.text
            ? <div style={{ position: "relative" }}>
                <Markdown text={m.text} />
                {m.streaming && <span className="cursor">▍</span>}
              </div>
            : (!m.error && <Thinking />))}
        {m.error && <div style={{ marginTop: m.text ? 8 : 0, color: "#f85149", fontSize: 13 }}>⚠ erro: {m.error}</div>}
        {m.sub && (
          <details className="axes">
            <summary>▸ 3 leitores por eixo</summary>
            {Object.entries(m.sub).map(([ax, ans]) => (
              <div key={ax} style={{ fontSize: 12.5, color: C.dim, marginTop: 6 }}>
                <b style={{ color: C.accent }}>{ax}:</b> {ans}
              </div>
            ))}
          </details>
        )}
        <Sources sources={m.sources} />
        {!isUser && m.text && !m.streaming && (
          <div style={{ marginTop: 8 }}><CopyBtn text={m.text} /></div>
        )}
      </div>
    </div>
  );
}

function App() {
  const [stats, refresh] = useStats();
  return (
    <div style={{ display: "flex", height: "100vh", background: C.bg, color: C.text, fontFamily: "system-ui, sans-serif" }}>
      <div style={{ width: 380, borderRight: `1px solid ${C.border}`, display: "flex", flexDirection: "column" }}>
        <div style={{ padding: 16, borderBottom: `1px solid ${C.border}` }}>
          <div style={{ fontSize: 22, fontWeight: 800 }}>
            🔺 RAG<span style={{ color: C.accent }}>3D</span>
          </div>
          <div style={{ color: C.dim, fontSize: 13 }}>RAG tridimensional · fusão quântica · Postgres sem pgvector</div>
        </div>
        <Uploader onDone={() => refresh()} chunks={stats?.chunks} />
        {stats && (
          <div style={{ marginTop: "auto", padding: 16, borderTop: `1px solid ${C.border}`, fontSize: 12, color: C.dim }}>
            <div>backend: <b style={{ color: C.text }}>{stats.backend}</b></div>
            <div>llm: <b style={{ color: C.text }}>{stats.llm}</b></div>
            <div>chunks no índice: <b style={{ color: C.text }}>{stats.chunks}</b> · fusão {stats.fusao}</div>
          </div>
        )}
      </div>
      <div style={{ flex: 1 }}><Chat /></div>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
