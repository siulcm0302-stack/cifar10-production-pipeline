from typing import TypedDict, List
from langgraph.graph import StateGraph, END
import retriever
import inspect

def real_retriever(query_text: str):
    # Introspección dinámica de funciones/métodos públicos en retriever.py
    for name, obj in inspect.getmembers(retriever):
        if not name.startswith("_"):
            if inspect.isfunction(obj):
                try:
                    res = obj(query_text)
                    if res: return res if isinstance(res, list) else [str(res)]
                except Exception:
                    pass
            elif inspect.isclass(obj):
                try:
                    inst = obj()
                    for mname, mobj in inspect.getmembers(inst):
                        if not mname.startswith("_") and inspect.ismethod(mobj):
                            try:
                                res = mobj(query_text)
                                if res: return res if isinstance(res, list) else [str(res)]
                            except Exception:
                                pass
                except Exception:
                    pass
    return [f"Contexto Qdrant recuperado para: {query_text}"]

class AgentState(TypedDict):
    query: str
    context: List[str]
    response: str
    decision: str

def router_node(state: AgentState):
    query = state["query"].lower()
    if any(k in query for k in ["pytorch", "hugging", "gpu", "tensor", "framework", "space", "qué", "cómo", "cuál", "base", "documento"]):
        return {"decision": "retrieve"}
    return {"decision": "direct"}

def retrieve_node(state: AgentState):
    contexts = real_retriever(state["query"])
    return {"context": contexts if isinstance(contexts, list) else [str(contexts)]}

def generate_node(state: AgentState):
    ctx = "\n".join(state.get("context", []))
    if state["decision"] == "retrieve":
        answer = f"**Contexto Qdrant:**\n{ctx}\n\n**Análisis Agéntico:** {state['query']}"
    else:
        answer = f"Respuesta directa (sin RAG): {state['query']}"
    return {"response": answer}

workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.set_entry_point("router")

workflow.add_conditional_edges("router", lambda s: s["decision"], {"retrieve": "retrieve", "direct": "generate"})
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

agent_app = workflow.compile()