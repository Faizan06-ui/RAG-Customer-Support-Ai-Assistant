# RAG AI Support Assistant

> RAG-powered customer support AI with explainability, multi-turn conversation, and confidence guardrails.

---

## 🎯 Overview

This is a Retrieval-Augmented Generation (RAG) system that answers customer support queries by retrieving semantically similar historical Q&A pairs and generating context-aware responses via a large language model. The system is trained on the [MohammadOthman/mo-customer-support-tweets-945k](https://huggingface.co/datasets/MohammadOthman/mo-customer-support-tweets-945k) dataset, with 50,000 rows indexed using FAISS for fast similarity search. The stack combines FastAPI, sentence-transformers, FAISS, and Mistral (via OpenRouter) with a custom vanilla JS frontend. ---

## 🏗️ Architecture

**RAG Pipeline Flow:**

```
User Query → Embedder → FAISS Retrieval → LLM Generator → Response
                              ↓
                        Retrieved Sources (shown in UI)
```

1. User submits a query via the HTML frontend
2. Query is embedded using `sentence-transformers` (`all-MiniLM-L6-v2`)
3. FAISS vector store retrieves the top-3 most similar historical Q&A pairs using cosine similarity
4. Retrieved context + conversation history + current query are sent to Mistral via the OpenRouter API
5. Response, sources, and confidence level are returned to the frontend
6. All interactions are logged to `logs/conversations.csv` for auditability

---

## ✨ Features

- [x] Split-panel UI: chat on left, retrieved sources on right
- [x] Real-time similarity scores for each retrieved source
- [x] Multi-turn conversation memory (chat history passed to LLM)
- [x] Confidence guardrail: refuses off-topic queries gracefully
- [x] Markdown rendering in chat responses
- [x] Interaction logging to CSV for auditability
- [x] `/health` endpoint for system status monitoring
- [x] Light themed responsive frontend (no external CSS frameworks)

---

## 🛠️ Tech Stack

| Component    | Technology                                          |
|--------------|-----------------------------------------------------|
| Frontend     | Vanilla HTML / CSS / JS                             |
| Backend      | FastAPI + Uvicorn                                   |
| Embeddings   | sentence-transformers (`all-MiniLM-L6-v2`)          |
| Vector Store | FAISS (Facebook AI Similarity Search)               |
| LLM          | Mistral via OpenRouter API                          |
| Dataset      | MohammadOthman/mo-customer-support-tweets-945k      |
| Logging      | CSV (`logs/conversations.csv`)                      |

---

## 📁 Project Structure

```
├── app.py                        # FastAPI app, API endpoints, serves frontend
├── pipeline.py                   # Orchestrates retrieval + generation + logging
├── main.py                       # Gradio UI alternative interface
├── rag/
│   ├── embedder.py               # sentence-transformer embedding wrapper
│   ├── retriever.py              # Loads FAISS index, runs similarity search
│   ├── vector_store.py           # FAISS index build / load / search logic
│   └── generator.py              # OpenRouter API call, prompt construction
├── static/
│   └── index.html                # Split-panel frontend UI
├── scripts/
│   └── build_index.py            # Downloads dataset, builds FAISS index
├── dataset/
│   ├── dataset_preprocessor.py   # Cleans and saves the raw dataset
│   └── storage/                  # Stores clean_data.csv, faiss_index, mapping.pkl
├── logs/                         # CSV interaction logs
└── test/                         # Unit tests for each module
```

---

## 🚀 How to Run Locally

### Prerequisites

- Python 3.9+
- OpenRouter API key (free at [openrouter.ai](https://openrouter.ai))

### Installation

**Step 1:** Clone the repo
```bash
git clone https://github.com/Faizan06-ui/RAG-Customer-Support-Ai-Assistant.git
cd RAG-Customer-Support-Ai-Assistant
```

**Step 2:** Install dependencies
```bash
pip install -r requirements.txt
```

**Step 3:** Create a `.env` file in the project root
```
OPENROUTER_API_KEY=your_key_here
```

**Step 4:** Build the FAISS index (downloads dataset, ~5 min first run)
```bash
python scripts/build_index.py
```

**Step 5:** Start the server
```bash
uvicorn app:app --reload --port 8000
```

**Step 6:** Open your browser at [http://localhost:8000](http://localhost:8000)

---

## 🧪 Test Queries

| # | Query | What it tests |
|---|-------|---------------|
| 1 | "I ordered a laptop, but it arrived with a broken screen. What should I do?" | Damage/return flow retrieval |
| 2 | "I need help resetting my password." → "I didn't receive the reset link." | Multi-turn conversation memory |
| 3 | "My cat chewed my phone charger. Is this covered under warranty?" | Edge case warranty retrieval |
| 4 | "Why did you suggest contacting support?" | Explainability — check the sources panel |

---

## 📊 Explainability

Every response surfaces the top-3 retrieved sources with their cosine similarity scores (0–1 scale), colour-coded in the UI: **green** for scores above 70%, **orange** for 40–70%, and **red** below 40%. A confidence guardrail blocks generation entirely when the top score falls below 0.50, returning a graceful refusal instead of a hallucinated answer. All interactions — including the full retrieved context — are logged to `logs/conversations.csv`, providing a complete audit trail.

---

## 🔮 Future Improvements

- Hybrid retrieval (BM25 + dense embeddings) for better keyword matching
- Cross-encoder re-ranking for improved precision
- RAGAS-based automated evaluation pipeline
- LangGraph agentic extension with tools (order lookup, ticket creation)
- Move to managed vector DB (Weaviate / Pinecone) for production scale

---

## 🔗 API Endpoints

| Method | Endpoint             | Description                                                   |
|--------|----------------------|---------------------------------------------------------------|
| POST   | `/generate_response` | Accepts `user_query` + `history`, returns `response` + `sources` + `confidence` |
| GET    | `/health`            | Returns system status and model info                          |
| GET    | `/`                  | Serves the frontend UI                                        |

---

## 👤 Author

Built by **Faizan Khurshid**
