import json
import pandas as pd
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Cargar dataset
with open("eval_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Cargar cross-encoder de reranking que ya usas en tu pipeline
reranker = CrossEncoder("BAAI/bge-reranker-large")

results = []
for item in data:
    # 1. Answer Relevance / Ground truth similarity via cross-encoder
    score_reference = reranker.predict([(item["reference"], item["response"])])
    
    # 2. Context Relevance (score prompt vs retrieved context)
    score_context = reranker.predict([(item["user_input"], item["retrieved_contexts"][0])])
    
    results.append({
        "user_input": item["user_input"],
        "answer_vs_reference_score": float(score_reference[0]),
        "context_relevance_score": float(score_context[0])
    })

df = pd.DataFrame(results)
print("\n--- Custom Evaluation Report ---")
print(df)
df.to_json("ragas_report.json", orient="records", indent=2)
print("\n✅ Reporte guardado en ragas_report.json")