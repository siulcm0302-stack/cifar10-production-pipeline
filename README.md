# AI Scratchpad / Production Agentic RAG Pipeline

Pipeline RAG avanzado con enrutamiento agéntico (`LangGraph`), vector DB (`Qdrant` con autodetección dinámica de módulos), reranking semántico (`BGE Cross-Encoder`), interfaz interactiva en `Gradio` y arnés de evaluación cuantitativa.

---

## 🏗️ Arquitectura
* **Orquestador / Agente**: LangGraph (`StateGraph` con nodos de ruteo, recuperación y generación)
* **Vector DB**: Qdrant (local/Docker)
* **Embedding Model**: `bge-small-en-v1.5`
* **Reranker**: `BAAI/bge-reranker-large`
* **UI**: Gradio (`app.py`)

---

## 🔄 Flujo de Enrutamiento Agéntico
```text
[User Input] --> (Router Node) --[decision: retrieve]--> [Retrieve Node (Qdrant + BGE)] --> [Generate Node] --> [Output]
                               --[decision: direct]  --> -------------------------------> [Generate Node] --> [Output]