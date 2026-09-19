import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore

def run_ingestion(file_path: str, collection_name: str = "prod_knowledge_base"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")

    # 1. Carga flexible
    print(f"📄 Cargando {file_path}...")
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()

    # 2. Chunking estructurado
    print("✂️ Aplicando recursive text splitting...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        add_start_index=True
    )
    splits = splitter.split_documents(docs)
    print(f"   -> Total de chunks: {len(splits)}")

    # 3. Embedding model
    print("🧠 Inicializando embeddings (BGE)...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # 4. Configurar cliente local y asegurar que exista la colección (dim=384 para bge-small)
    print("💾 Configurando Qdrant local...")
    client = QdrantClient(path="./qdrant_storage")
    
    existing_collections = [c.name for c in client.get_collections().collections]
    if collection_name not in existing_collections:
        print(f"🛠️ Creando colección '{collection_name}' (vector size: 384)...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    vector_store.add_documents(splits)
    print(f"✅ ¡Ingesta exitosa en colección '{collection_name}' (guardada en ./qdrant_storage)!")

if __name__ == "__main__":
    test_file = "test_doc.txt"
    if not os.path.exists(test_file):
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("PyTorch es un framework de Deep Learning basado en tensores con ejecución dinámica. Hugging Face Spaces permite desplegar apps con ZeroGPU.")
    
    run_ingestion(test_file)