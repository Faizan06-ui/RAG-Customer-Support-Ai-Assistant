# Customer Support AI Assistant

## 🔍 Overview
This project is a **Retrieval-Augmented Generation (RAG)** based AI assistant designed to answer customer support queries. It retrieves the most relevant historical Q&A data and generates contextual, human-like responses using the **`mistralai/mistral-7b-instruct`** model via **OpenRouter.ai** API.

The app is now **fully deployed** with optimized performance and dual interfaces — powered by **FastAPI** and a more interactive **Gradio** UI for quick testing.

---

## ✅ Submission Checklist
- [x] Clean, modular GitHub repository ✅  
- [x] Fully deployed model integration with OpenRouter ✅  
- [x] FastAPI and Gradio interfaces ✅  
- [x] Optimized codebase for performance ✅  
- [x] README and PPT with approach, RAG details, and explainability ✅  

---

## 🛠️ Tech Stack
- Python  
- FastAPI  
- Gradio  
- FAISS (Facebook AI Similarity Search)  
- Hugging Face SentenceTransformers (`all-MiniLM-L6-v2`)  
- OpenRouter API (`mistralai/mistral-7b-instruct`)  

---

## 🧠 Approach Taken

1. **Data Loading & Cleaning:**
   - Loaded customer support Q&A dataset from Hugging Face: `MohammadOthman/mo-customer-support-tweets-945k`
   - Removed nulls and duplicates, saved as `clean_data.csv`.

2. **Embedding & Indexing:**
   - Used `all-MiniLM-L6-v2` for semantic embedding.
   - Built FAISS index for fast vector similarity search.

3. **RAG Pipeline:**
   - **Retrieval:** Top 3 most semantically similar Q&As are selected.
   - **Augmentation:** Prompt is formed using retrieved context and the user query.
   - **Generation:** Prompt sent to `mistralai/mistral-7b-instruct` via OpenRouter API to generate a helpful, context-aware answer.

4. **Serving the App:**
   - **FastAPI** backend with `/generate` endpoint.
   - **Gradio** front-end for real-time interaction (preferred for usability).

---

## 🔄 RAG Implementation
**Retrieval**  
- Embeds and indexes Q&A pairs using `all-MiniLM-L6-v2` + FAISS  
- Uses L2 similarity to retrieve relevant examples

**Augmentation**  
- Combines top 3 retrieved Q&As with the user query to build the context-rich prompt

**Generation**  
- Prompt passed to `mistralai/mistral-7b-instruct` via OpenRouter  
- Output is cleaned and post-processed for delivery

---

## 📢 Explainability Techniques
- Matching Q&As and similarity scores are logged for traceability.
- Debugging and transparency supported with optional logs.
- *(Planned)* LIME/SHAP integration for deeper model understanding.

---

## 📦 Directory Structure
```
customer-support-ai/
├── app.py                            # FastAPI app entrypoint
├── pipeline.py                       # RAG pipeline (retriever + generator)
├── requirements.txt                  # Dependencies
├── README.md                         # Project overview & how to run
├── .gitignore                        # Ignore unwanted files (e.g., __pycache__)
│
├── rag/                              # Core logic of RAG
│   ├── __init__.py
│   ├── embedder.py                   # SentenceTransformer embedder
│   ├── vector_store.py               # FAISS vector DB logic
│   ├── retriever.py                  # Retrieval logic
│   ├── generator.py                  # LLM-based response generator
│
├── test/                             # Test scripts for each module
│   ├── __init__.py
│   ├── test_embedder.py              # (optional, can test embeddings)
│   ├── test_vector_store.py
│   ├── test_retriever.py
│   ├── test_generator.py
│   ├── test_pipeline.py
│
├── dataset/                          # Dataset preprocessing & raw/clean files
│   ├── dataset_preprocessor.py
│   ├── raw/                          # (optional) raw downloaded files
│   └── storage/                      # Cleaned CSV, FAISS index, mapping.pkl
│       ├── clean_data.csv
│       ├── faiss_index.index
│       ├── mapping.pkl
│
├── logs/                             # Logs for interaction history
│   └── conversations.csv


```

---

## 🚀 How to Run Locally
```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run FastAPI server
uvicorn app.main:app --reload
```

---

## 🚀 How to Run Locally
```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run FastAPI server
uvicorn app:app --reload

# (Optional) Run Gradio UI
python gradio_ui.py
```
---

## 🚀 Notes on security
```bash
API keys (OpenRouter) are stored in .env and never exposed.

python-dotenv handles secure loading of environment variables.
```
---

## 🔗 Live Demo
```bash
https://costumer-support-ai-asistant-1.onrender.com/docs
```
----

## 🤝 Credits
```bash
Built by Aadil Maqbool for the Customer Support AI Assistant Challenge.

