from .vector_store import VectorStore
from .embedder import Embedder

class Retriever:
    def __init__(self):
        embedder = Embedder()
        self.vector_store = VectorStore(embedder)
        print("Attempting to load index from dataset/storage/faiss_index.index")
        print("Attempting to load mapping from dataset/storage/mapping.pkl")
        try:
            self.vector_store.load_index(
                index_path="dataset/storage/faiss_index.index",
                mapping_path="dataset/storage/mapping.pkl"
            )
            print("✅ Loaded index from storage/faiss_index.index and mapping from dataset/storage/mapping.pkl")
        except FileNotFoundError:
            print("⚠️ Index not found. You may need to build it first.")

    def search(self, query, top_k=3):
        return self.vector_store.search(query, top_k=top_k)  # <- Make sure this returns 3 items
