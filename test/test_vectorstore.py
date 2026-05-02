import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import pandas as pd
from rag.embedder import Embedder
from rag.vector_store import VectorStore

# Load the cleaned dataset
df = pd.read_csv("dataset/storage/clean_data.csv")
df = df.rename(columns={"query": "input", "response": "output"})

embedder = Embedder()
vs = VectorStore(embedder)

# Build the FAISS index
vs.build_index(df)

# Save index and mapping for reuse
vs.save_index("dataset/storage/faiss_index.index", "dataset/storage/mapping.pkl")

# Run a test search
query = "I need help resetting my password"
results = vs.search(query, top_k=3)

print(f"\n🔎 Top matches for query: '{query}'")
for i, (q, a, score) in enumerate(results, 1):
    print(f"\nResult #{i}:")
    print("Query   :", q)
    print("Response:", a)
    print(f"Score (distance): {score}")

