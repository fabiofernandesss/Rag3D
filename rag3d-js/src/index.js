// API pública do RAG3D (JS).
export { TriRag, TriRag as Rag3D, NoLLM, CallableLLM } from "./engine.js";
export { defaultConfig } from "./config.js";
export { fuse, quantumFuse, rrfFuse } from "./fusion.js";
export { Holographer, HOLO_BITS } from "./holo.js";
export { HashEncoder, makeEncoder } from "./encoders.js";
export * as portable from "./portable.js";
