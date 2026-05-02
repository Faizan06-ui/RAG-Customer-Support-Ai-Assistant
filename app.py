from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pipeline import Pipeline

# Load the RAG pipeline once
rag_pipeline = Pipeline()

# Create FastAPI app instance
app = FastAPI()

# Define the input and output models
class QueryRequest(BaseModel):
    user_query: str
    history: list = []

class QueryResponse(BaseModel):
    response: str
    sources: list = []
    confidence: str = "high"

@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "mistralai/ministral-14b-2410",
        "index_loaded": True,
        "version": "1.0.0",
    }

@app.post("/generate_response", response_model=QueryResponse)
def generate_response(query: QueryRequest):
    result = rag_pipeline.run(query.user_query, history=query.history)
    return QueryResponse(response=result["response"], sources=result["sources"], confidence=result["confidence"])

app.mount("/static", StaticFiles(directory="static"), name="static")
