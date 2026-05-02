import pandas as pd
import faiss
import numpy as np
import pickle
from .embedder import Embedder


class VectorStore:
    def __init__(self, embedder, embedding_dim=384):  # Adjust dim based on the embedder used
        self.embedder = embedder
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(self.embedding_dim)  # 👈 This must be initialized
        self.data_mapping = []

    def build_index(self, dataframe):
        dataframe = dataframe.dropna(subset=["input", "output"])
        queries = dataframe["input"].astype(str).tolist()
        responses = dataframe["output"].astype(str).tolist()

        self.data_mapping = list(zip(queries, responses))

        embeddings = self.embedder.encode(queries)
        embeddings = np.array(embeddings).astype("float32")

        self.index.add(embeddings)
        print("✅ FAISS index built with", len(embeddings), "entries.")

    def load_index(self, index_path="dataset/storage/faiss_index.index", mapping_path="dataset/storage/mapping.pkl"):
        try:
            # Try loading the FAISS index
            self.index = faiss.read_index(index_path)
            with open(mapping_path, "rb") as f:
                self.data_mapping = pickle.load(f)
            print(f"✅ Loaded index from {index_path} and mapping from {mapping_path}")
        except Exception as e:
            print(f"⚠️ Error loading index and mapping: {e}")
            print("Attempting to rebuild the index...")
            # You can either try to rebuild the index or raise an error if needed
            raise FileNotFoundError(f"Index not found at {index_path}")


    def save_index(self, index_path="dataset/storage/faiss_index.index", mapping_path="dataset/storage/mapping.pkl"):
        faiss.write_index(self.index, index_path)
        with open(mapping_path, "wb") as f:
            pickle.dump(self.data_mapping, f)
        print(f"✅ Index and mapping saved to {index_path} and {mapping_path}")

    def search(self, query, top_k=3):
        query_vec = self.embedder.encode([query])
        query_vec = np.array(query_vec).astype("float32")

        D, I = self.index.search(query_vec, top_k)

        print("Type of I:", type(I))
        print("I =", I)

        results = []

        if isinstance(I, np.ndarray) and len(I.shape) > 1:
            for idx, dist in zip(I[0], D[0]):
                if idx == -1:
                    continue
                try:
                    query_text, response_text = self.data_mapping[idx]
                    similarity = round(1 - (float(dist) / 2), 4)
                    results.append((query_text, response_text, similarity))
                except Exception as e:
                    print(f"Skipping index {idx} due to error: {e}")
                    continue
        else:
            print("FAISS returned unexpected index format.")

        return results

