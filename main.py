import os
import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import faiss
import json
from sentence_transformers import SentenceTransformer

# === Logging Configuration ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log"),
    ],
)
logger = logging.getLogger(__name__)

# === Paths & Environment Variables ===
MODEL_PATH = os.getenv("MODEL_PATH", "./Preprocessing/all-MiniLM-L6-v2")
INDEX_PATH = os.getenv("INDEX_PATH", "./Preprocessing/books.index")
META_PATH = os.getenv("META_PATH", "./Preprocessing/books_meta.json")

# === App State ===
class AppState:
    model: Optional[SentenceTransformer] = None
    index: Optional[faiss.IndexFlatL2] = None
    books: Optional[List[Dict[str, Any]]] = None
    cache: Dict[str, List[Any]] = {}

app_state = AppState()

# === FastAPI App ===
app = FastAPI(
    title="Book Summary + Page Search API",
    description="Semantic search over book summaries and internal pages",
    version="1.0.0"
)

# === CORS Middleware ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Middleware: Measure Request Time ===
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    process_time = time.time() - start
    response.headers["X-Process-Time"] = str(round(process_time, 4))
    return response

# === Global Exception Handler ===
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

# === Models ===
class QueryRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(5, ge=1, le=50)
    max_score: float = Field(1.0, ge=0.0, le=2.0)
    page: int = Field(1, ge=1)
    page_size: int = Field(10, ge=1, le=100)

class PageSearchRequest(BaseModel):
    book_id: str = Field(..., description="Book ID or title")
    query: str = Field(..., description="Search query")
    context_window: int = Field(1, ge=0, le=5)

class BookResult(BaseModel):
    title: str
    summary: str
    score: float
    page_number: Optional[int] = None
    page_content: Optional[str] = None

class PageSearchResponse(BaseModel):
    book_title: str
    query: str
    page_number: Optional[int]
    page_content: Optional[str]
    score: Optional[float]
    took_ms: float

class SearchResponse(BaseModel):
    query: str
    top_k: int
    max_score: float
    results: List[BookResult]
    took_ms: float
    total_results: int
    page: int
    page_size: int
    total_pages: int

# === Utility: Load Resources ===
async def get_resources():
    if app_state.model is None:
        try:
            logger.info(f"🔍 Loading model from: {MODEL_PATH}")
            app_state.model = SentenceTransformer(MODEL_PATH)

            logger.info(f"📦 Loading FAISS index from: {INDEX_PATH}")
            app_state.index = faiss.read_index(INDEX_PATH)

            logger.info(f"📖 Loading metadata from: {META_PATH}")
            with open(META_PATH, "r", encoding="utf-8") as f:
                app_state.books = json.load(f)

            logger.info(f"✅ Loaded {len(app_state.books)} books")
        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to load resources")
    return {
        "model": app_state.model,
        "index": app_state.index,
        "books": app_state.books
    }

# === Utility: Page-level Matching ===
def find_most_relevant_page(query: str, pages: List[Dict], model: SentenceTransformer, context_window=1) -> Optional[Dict]:
    if not pages:
        return None
    try:
        filtered = [(i, p.get("content", "").strip()) for i, p in enumerate(pages) if p.get("content", "").strip()]
        if not filtered:
            return None
        indices, contents = zip(*filtered)

        contents_with_context = []
        for i, content in enumerate(contents):
            context = content
            if context_window > 0:
                if i > 0:
                    context = " ".join(contents[max(0, i - context_window):i]) + " " + context
                if i < len(contents) - 1:
                    context = context + " " + " ".join(contents[i + 1:min(len(contents), i + 1 + context_window)])
            contents_with_context.append(context)

        query_emb = model.encode([query], convert_to_numpy=True)
        page_embs = model.encode(contents_with_context, convert_to_numpy=True)
        query_emb = query_emb / np.linalg.norm(query_emb)
        page_embs = page_embs / np.linalg.norm(page_embs, axis=1, keepdims=True)
        sims = np.dot(page_embs, query_emb.T).squeeze()

        best_idx = int(np.argmax(sims))
        best_score = sims[best_idx]
        if best_score > 0.5:
            return {
                "number": pages[indices[best_idx]].get("number", indices[best_idx] + 1),
                "content": contents[best_idx],
                "score": float(best_score)
            }
    except Exception as e:
        logger.error(f"Page match failed: {e}", exc_info=True)
    return None

# === /search Endpoint ===
@app.post("/search", response_model=SearchResponse)
async def search_books(req: QueryRequest, resources: Dict[str, Any] = Depends(get_resources)):
    start_time = time.time()
    model, index, books = resources["model"], resources["index"], resources["books"]

    try:
        cache_key = f"{req.query}_{req.top_k}_{req.max_score}"
        if cache_key in app_state.cache:
            all_results = app_state.cache[cache_key]
        else:
            query_emb = model.encode([req.query], convert_to_numpy=True)
            distances, indices = index.search(query_emb, max(100, req.top_k * 2))

            all_results = []
            for dist, idx in zip(distances[0], indices[0]):
                if 0 <= idx < len(books) and dist <= req.max_score:
                    book = books[idx]
                    matched = find_most_relevant_page(req.query, book.get("pages", []), model)
                    all_results.append(BookResult(
                        title=book["title"],
                        summary=book["summary"],
                        score=round(float(dist), 4),
                        page_number=matched["number"] if matched else None,
                        page_content=matched["content"] if matched else None,
                    ))

            if len(app_state.cache) > 100:
                app_state.cache.pop(next(iter(app_state.cache)))
            app_state.cache[cache_key] = all_results

        total = len(all_results)
        total_pages = (total + req.page_size - 1) // req.page_size
        page = min(req.page, total_pages or 1)
        start_idx = (page - 1) * req.page_size
        end_idx = start_idx + req.page_size
        paginated = all_results[start_idx:end_idx]

        return SearchResponse(
            query=req.query,
            top_k=req.top_k,
            max_score=req.max_score,
            results=paginated,
            took_ms=round((time.time() - start_time) * 1000, 2),
            total_results=total,
            page=page,
            page_size=req.page_size,
            total_pages=total_pages
        )
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Search error")

# === /search-page Endpoint ===
@app.post("/search-page", response_model=PageSearchResponse)
async def search_page(req: PageSearchRequest, resources: Dict[str, Any] = Depends(get_resources)):
    start_time = time.time()
    model, books = resources["model"], resources["books"]

    book = next((b for b in books if b.get("id") == req.book_id or b.get("title") == req.book_id), None)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    matched = find_most_relevant_page(req.query, book.get("pages", []), model, req.context_window)

    return PageSearchResponse(
        book_title=book["title"],
        query=req.query,
        page_number=matched["number"] if matched else None,
        page_content=matched["content"] if matched else None,
        score=matched["score"] if matched else None,
        took_ms=round((time.time() - start_time) * 1000, 2)
    )

# === Health Check ===
@app.get("/health")
async def health():
    return {"status": "ok", "uptime": time.time()}

# === Startup Hook ===
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 FastAPI server starting...")
    await get_resources()
