// Adaptador de LLM agnóstico de provedor — espelha trirag/llm.py.
// Usa fetch nativo do Node 18+. Anthropic, OpenAI-compatível, Ollama, callable.

async function postJson(url, headers, payload, timeout = 120000) {
  let lastErr;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), timeout);
      const resp = await fetch(url, {
        method: "POST", headers, body: JSON.stringify(payload), signal: ctrl.signal,
      });
      clearTimeout(t);
      if (!resp.ok) {
        const detail = (await resp.text()).slice(0, 500);
        if ([429, 500, 502, 503, 529].includes(resp.status) && attempt < 2) {
          await sleep(2000 * (attempt + 1)); continue;
        }
        throw new Error(`LLM HTTP ${resp.status}: ${detail}`);
      }
      return await resp.json();
    } catch (e) {
      lastErr = e;
      if (attempt < 2) { await sleep(1000 * (attempt + 1)); continue; }
      throw lastErr;
    }
  }
  throw lastErr;
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export class NoLLM {
  constructor() { this.provider = "none"; this.model = ""; }
  available() { return false; }
  async complete() { throw new Error("Nenhum LLM configurado (modo só-retrieval)."); }
  async completeStream() { throw new Error("Nenhum LLM configurado (modo só-retrieval)."); }
}

export class AnthropicLLM {
  constructor(model = "") {
    this.provider = "anthropic";
    this.key = process.env.ANTHROPIC_API_KEY || "";
    this.model = model || "claude-sonnet-5";
  }
  available() { return !!this.key; }
  // fallback: Anthropic usa outro formato de stream; emite tudo de uma vez
  async completeStream(system, messages, maxTokens, onToken) {
    const full = await this.complete(system, messages, maxTokens);
    onToken(full);
    return full;
  }
  async complete(system, messages, maxTokens = 1500) {
    const data = await postJson("https://api.anthropic.com/v1/messages", {
      "x-api-key": this.key, "anthropic-version": "2023-06-01", "content-type": "application/json",
    }, { model: this.model, max_tokens: maxTokens, system, messages });
    return (data.content || []).map((b) => b.text || "").join("");
  }
}

export class OpenAICompatLLM {
  constructor(model = "", baseUrl = "", apiKey = "") {
    this.provider = "openai";
    this.baseUrl = (baseUrl || process.env.OPENAI_BASE_URL || "https://api.openai.com/v1").replace(/\/$/, "");
    this.key = apiKey || process.env.OPENAI_API_KEY || "";
    this.model = model || process.env.TRIRAG_LLM_MODEL || "gpt-4o-mini";
  }
  available() { return !!this.key || /localhost|127\.0\.0\.1/.test(this.baseUrl); }
  async complete(system, messages, maxTokens = 1500) {
    const data = await postJson(`${this.baseUrl}/chat/completions`, {
      Authorization: `Bearer ${this.key || "none"}`, "Content-Type": "application/json",
    }, { model: this.model, max_tokens: maxTokens, messages: [{ role: "system", content: system }, ...messages] });
    return data.choices[0].message.content || "";
  }

  // streaming token a token (SSE do OpenAI-compat). onToken(delta) por chunk.
  // Retenta a CONEXÃO se falhar antes do primeiro token ("fetch failed"
  // transitório sob concorrência); uma vez começado o stream, não retenta.
  async completeStream(system, messages, maxTokens, onToken) {
    let lastErr;
    for (let attempt = 0; attempt < 3; attempt++) {
      let resp;
      try {
        resp = await fetch(`${this.baseUrl}/chat/completions`, {
          method: "POST",
          headers: { Authorization: `Bearer ${this.key || "none"}`, "Content-Type": "application/json" },
          body: JSON.stringify({
            model: this.model, max_tokens: maxTokens, stream: true,
            messages: [{ role: "system", content: system }, ...messages],
          }),
        });
      } catch (e) {
        lastErr = new Error(`stream conexão falhou: ${e.cause?.code || e.cause?.message || e.message}`);
        await sleep(800 * (attempt + 1));
        continue;
      }
      if (!resp.ok) {
        const detail = (await resp.text()).slice(0, 300);
        if ([429, 500, 502, 503, 529].includes(resp.status) && attempt < 2) {
          await sleep(1000 * (attempt + 1));
          lastErr = new Error(`LLM HTTP ${resp.status}: ${detail}`);
          continue;
        }
        throw new Error(`LLM HTTP ${resp.status}: ${detail}`);
      }
      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let buf = "", full = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop();
        for (const line of lines) {
          const t = line.trim();
          if (!t.startsWith("data:")) continue;
          const data = t.slice(5).trim();
          if (data === "[DONE]") continue;
          try {
            const d = JSON.parse(data).choices?.[0]?.delta?.content;
            if (d) { full += d; onToken(d); }
          } catch { /* keep-alive/parcial */ }
        }
      }
      return full;
    }
    throw lastErr || new Error("stream falhou");
  }
}

export class CallableLLM {
  constructor(fn, name = "callable") { this.provider = "callable"; this.model = name; this.fn = fn; }
  available() { return true; }
  async complete(system, messages, maxTokens = 1500) { return this.fn(system, messages, maxTokens); }
  async completeStream(system, messages, maxTokens, onToken) {
    const full = await this.fn(system, messages, maxTokens);
    onToken(full);
    return full;
  }
}

async function ollamaUp() {
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 1000);
    const r = await fetch("http://localhost:11434/api/tags", { signal: ctrl.signal });
    clearTimeout(t);
    return r.ok;
  } catch { return false; }
}

export async function makeLLM(provider = "auto", model = "") {
  if (provider === "anthropic") return new AnthropicLLM(model);
  if (provider === "openai") return new OpenAICompatLLM(model);
  if (provider === "ollama") return new OpenAICompatLLM(model || "llama3.1", "http://localhost:11434/v1", "ollama");
  if (provider === "none") return new NoLLM();
  // auto
  const a = new AnthropicLLM(model);
  if (a.available()) return a;
  if (process.env.OPENAI_API_KEY) return new OpenAICompatLLM(model);
  if (await ollamaUp()) return new OpenAICompatLLM(model || "llama3.1", "http://localhost:11434/v1", "ollama");
  return new NoLLM();
}
