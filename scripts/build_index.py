import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from rag.embedder import Embedder
from rag.vector_store import VectorStore

CSV_PATH = "dataset/storage/clean_data.csv"

# ── Step 1: Download and save CSV if it doesn't exist ──────────────────────
if not os.path.exists(CSV_PATH):
    print("📥 clean_data.csv not found — downloading from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset("MohammadOthman/mo-customer-support-tweets-945k")
    df_raw = pd.DataFrame(ds["train"])

    print("🔎 Columns:", df_raw.columns.tolist())
    print("🔎 First row:", df_raw.iloc[0].to_dict())

    # Rename to standard input/output — raw dataset uses these names
    df_raw = df_raw.rename(columns={"input": "input", "output": "output"})

    # Keep only the two relevant columns, drop nulls and duplicates
    df_raw = df_raw[["input", "output"]].dropna().drop_duplicates()
    df_raw = df_raw.head(50000)

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    df_raw.to_csv(CSV_PATH, index=False)
    print(f"✅ Saved {len(df_raw)} rows to {CSV_PATH}")
else:
    print(f"✅ {CSV_PATH} already exists — skipping download")

# ── Step 2: Load CSV ────────────────────────────────────────────────────────
print("📦 Loading CSV...")
df = pd.read_csv(CSV_PATH)

print("🔎 Columns in CSV:", df.columns.tolist())
print(f"📊 Total rows loaded: {len(df)}")

# Normalise column names to what VectorStore.build_index() expects
if "query" in df.columns and "response" in df.columns:
    df = df.rename(columns={"query": "input", "response": "output"})

df = df[["input", "output"]].dropna().head(50000)
print(f"📊 Rows after cleaning: {len(df)}")

# ── Step 3: Build and save FAISS index ─────────────────────────────────────
print("🔧 Initialising embedder...")
embedder = Embedder()

print("🏗  Building FAISS index (this may take a few minutes)...")
vector_store = VectorStore(embedder)
vector_store.build_index(df)

print("💾 Saving index and mapping...")
vector_store.save_index(
    index_path="dataset/storage/faiss_index.index",
    mapping_path="dataset/storage/mapping.pkl"
)
print("🎉 Done! Index is ready.")
