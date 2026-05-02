import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.retriever import Retriever

retriever = Retriever()

query = "How can I cancel my subscription?"
results = retriever.search(query, top_k=3)

print(f"\n🔎 Top matches for: '{query}'")
for i, (q, a, score) in enumerate(results, 1):
    print(f"\nResult #{i}:")
    print("Query   :", q)
    print("Response:", a)
    print(f"Score (distance): {score}")

