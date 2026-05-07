from datasets import load_dataset
import pandas as pd
import os

#  ADD THESE IMPORTS
from rag.embedder import Embedder
from rag.vector_store import VectorStore


def load_and_clean_dataset():
    ds = load_dataset("MohammadOthman/mo-customer-support-tweets-945k")
    
    df = pd.DataFrame(ds["train"])
    
    df = df[["input", "output"]].rename(columns={"input": "query", "output": "response"})
    
    df = df.dropna().drop_duplicates()

    return df


if __name__ == "__main__":
    df = load_and_clean_dataset()
    
    os.makedirs("dataset/storage", exist_ok=True)

    # Save cleaned data
    df.to_csv("dataset/storage/clean_data.csv", index=False)
    print("✅ Cleaned dataset saved to storage/clean_data.csv")

    # 🚀 BUILD FAISS INDEX
    print("🔧 Building FAISS index...")

    embedder = Embedder()
    vector_store = VectorStore(embedder)

    # IMPORTANT: column names expected by VectorStore
    df = df.rename(columns={"query": "input", "response": "output"})

    vector_store.build_index(df)
    vector_store.save_index()

    print("✅ FAISS index built and saved successfully!")