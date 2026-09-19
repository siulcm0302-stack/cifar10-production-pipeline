import torch
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def hybrid_retrieve(query: str, collection_name: str = "prod_knowledge_base", top_k=20, top_n=5):
    # 1. Retrieval vectorial local (Top-K=20)
    print(f"🔍 Buscando Top-{top_k} candidatos en Qdrant...")
    client = QdrantClient(path="./qdrant_storage")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings)
    docs = vector_store.similarity_search(query, k=top_k)
    
    if not docs:
        print("⚠️ No se encontraron documentos para la query.")
        return []

    # 2. Cross-Encoder Reranker (Top-N=5)
    print("🔄 Aplicando Cross-Encoder Reranking (BGE-reranker-large)...")
    reranker_id = "BAAI/bge-reranker-large"
    tokenizer = AutoTokenizer.from_pretrained(reranker_id)
    model = AutoModelForSequenceClassification.from_pretrained(reranker_id)
    model.eval()
    
    pairs = [[query, d.page_content] for d in docs]
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze()
        if logits.ndim == 0:
            scores = [logits.item()]
        else:
            scores = logits.tolist()
            
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_n]]

if __name__ == "__main__":
    # Prueba rápida de recuperación
    q = "¿Qué permite Hugging Face Spaces?"
    results = hybrid_retrieve(q)
    print(f"\n✨ Top-{len(results)} fragmentos re-rankeados para la query '{q}':")
    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] Contenido: {doc.page_content}")