"""RAG3D — Retrieval-Augmented Generation tridimensional.

Três eixos de representação para cada texto:
  1. Semântico  (vetor denso)         -> significado
  2. Léxico     (vetor esparso)       -> termos exatos, nomes, códigos
  3. Estrutural (late-interaction)    -> correspondência fina token a token

Os três eixos atuam na ingestão (todo texto é salvo nas três formas) e na
busca (a consulta é projetada nos três eixos). A fusão dos três rankings usa
interferência quântica: cada canal contribui uma amplitude complexa e a
concordância entre canais gera interferência construtiva.

Funciona em qualquer língua (encoder multilíngue ou fallback agnóstico de
língua) e com qualquer LLM (Anthropic, OpenAI-compatível, Ollama, callable).
"""

__version__ = "0.1.0"
__author__ = "Fabio Fernandes Jesus <fabiofernandesjesus@hotmail.com>"

from .config import TriRagConfig
from .engine import TriRag

# nomes oficiais do projeto (aliases limpos)
Rag3D = TriRag
Rag3DConfig = TriRagConfig

__all__ = ["Rag3D", "Rag3DConfig", "TriRag", "TriRagConfig", "__version__"]
