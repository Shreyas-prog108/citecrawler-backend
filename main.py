from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone
from typing import List
import os
import requests
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Citecrawler API", version="1.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

_pc = None
_index = None

def get_pinecone():
    global _pc, _index
    if _pc is None:
        _pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        _index = _pc.Index(os.getenv("INDEX_NAME", "papers-index"))
    return _pc, _index

def generate_embeddings(text: str) -> List[float]:
    api_key = os.getenv('COHERE_API_KEY')
    if not api_key:
        raise HTTPException(status_code=503, detail="COHERE_API_KEY not set")
    
    url = "https://api.cohere.ai/v1/embed"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {"texts": [text], "model": "embed-english-light-v3.0", "input_type": "search_query"}
    
    response = requests.post(url, headers=headers, json=data, timeout=10)
    response.raise_for_status()
    return response.json()["embeddings"][0]

@app.get("/")
def root():
    return {"message": "CiteCrawler API", "version": "1.0.0"}

@app.get("/search")
def search(q: str = Query(...), top_k: int = Query(5, ge=1, le=50), page: int = Query(1, ge=1)):
    pc, index = get_pinecone()
    query_embedding = generate_embeddings(q)
    start_idx = (page - 1) * top_k
    res = index.query(vector=query_embedding, top_k=min(100, start_idx + top_k), include_metadata=True)
    
    results = [{"id": m["id"], "title": m["metadata"].get("title", ""), "link": m["metadata"].get("link", ""), "abstract": m["metadata"].get("abstract", ""), "score": round(m["score"], 4)} for m in res["matches"]]
    
    return {"results": results[start_idx:start_idx + top_k], "total": len(results), "page": page}

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)