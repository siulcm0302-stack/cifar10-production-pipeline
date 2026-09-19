import gradio as gr
from agent import agent_app
from typing import List, Tuple

def chat_interface(user_message: str, history: List[Tuple[str, str]]) -> Tuple[str, str]:
    state_input = {"query": user_message, "context": [], "response": "", "decision": ""}
    result = agent_app.invoke(state_input)
    
    response_text = result.get("response", "Sin respuesta")
    contexts = result.get("context", [])
    
    sources_text = "\n".join([f"• [{i+1}] {ctx}" for i, ctx in enumerate(contexts)]) if contexts else "No se requirieron fuentes (respuesta directa)."
    
    return response_text, sources_text

with gr.Blocks(title="AI Scratchpad - Agentic RAG") as demo:
    gr.Markdown("# 🚀 Production Agentic RAG (LangGraph + Qdrant + Reranker)")
    gr.Markdown("Pipeline inteligente con enrutamiento de estado y recuperación vectorial.")
    
    with gr.Row():
        with gr.Column(scale=2):
            msg = gr.Textbox(label="Haz una pregunta a tus documentos", placeholder="Ej. ¿Qué permite Hugging Face Spaces?")
            submit_btn = gr.Button("Submit", variant="primary")
            clear_btn = gr.Button("Clear")
            
        with gr.Column(scale=3):
            out_answer = gr.Textbox(label="💡 Respuesta del Agente:", lines=4, interactive=False)
            out_sources = gr.Textbox(label="🔍 Fuentes / Contexto evaluado:", lines=5, interactive=False)

    submit_btn.click(fn=chat_interface, inputs=[msg], outputs=[out_answer, out_sources])
    msg.submit(fn=chat_interface, inputs=[msg], outputs=[out_answer, out_sources])
    clear_btn.click(lambda: ("", "", ""), outputs=[msg, out_answer, out_sources])

if __name__ == "__main__":
    demo.launch()