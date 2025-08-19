from __future__ import annotations
import logging
from io import BytesIO
from typing import List, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pathlib import Path
from pypdf import PdfReader
from uuid import uuid4

from .schemas import SearchResponse, AskResponse
from .chunker import split_into_word_chunks
from . import retrieval

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("rag")

app = FastAPI(title=f"RAG Minimal (PDF+TXT, bge-m3 + {retrieval._get_llm_model()})")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

UI_PATH = Path(__file__).parent / "ui" / "index.html"

# Простое in-memory хранилище прогресса индексации (переживает жизнь процесса)
INDEX_PROGRESS: Dict[str, Dict[str, object]] = {}

@app.get("/", response_class=HTMLResponse)
def index():
    html = UI_PATH.read_text(encoding="utf-8")
    model = retrieval._get_llm_model()
    html = html.replace("{{MODEL}}", model)
    return html

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/about")
def about():
    return {"model": retrieval._get_llm_model()}

@app.get("/documents")
def documents():
    return retrieval.list_documents()

def _read_txt(data: bytes) -> str:
    return data.decode("utf-8", errors="ignore")

def _read_pdf(data: bytes) -> str:
    reader = PdfReader(BytesIO(data))
    texts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        texts.append(t)
    return "\n".join(texts)

def _index_in_background(task_id: str, filename: str, chunks: List[str]):
    """Фоновая индексация чанков с обновлением прогресса."""
    from . import retrieval
    state = INDEX_PROGRESS.get(task_id) or {}
    state.update({"filename": filename, "processed": 0, "total": len(chunks), "done": False})
    INDEX_PROGRESS[task_id] = state
    def on_progress(done: int, total: int):
        st = INDEX_PROGRESS.get(task_id)
        if st is not None:
            st["processed"] = int(done)
            st["total"] = int(total)
            INDEX_PROGRESS[task_id] = st
    try:
        info = retrieval.upsert_chunks(filename, chunks, progress_callback=on_progress)
        st = INDEX_PROGRESS.get(task_id) or {}
        st.update({"done": True, "ok": True, "chunks": int(info.get("chunks", 0))})
        INDEX_PROGRESS[task_id] = st
    except Exception as e:
        st = INDEX_PROGRESS.get(task_id) or {}
        st.update({"done": True, "ok": False, "error": str(e)})
        INDEX_PROGRESS[task_id] = st

@app.post("/upload")
async def upload(
    files: List[UploadFile] = File(...),
    chunk_words: int = Query(220, ge=50, le=1000),
    chunk_overlap: int = Query(40, ge=0, le=500),
):
    if not files:
        raise HTTPException(400, "Нет файлов")
    total_chunks = 0
    for f in files:
        name = (f.filename or "upload").strip()
        raw_bytes = await f.read()
        ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
        if ext == "txt":
            raw_text = _read_txt(raw_bytes)
        elif ext == "pdf":
            try:
                raw_text = _read_pdf(raw_bytes)
            except Exception as e:
                raise HTTPException(415, f"Не удалось прочитать PDF: {e}")
        else:
            raise HTTPException(415, "Поддерживаются только .pdf и .txt")
        raw_text = raw_text.strip()
        if not raw_text:
            raise HTTPException(400, f"Файл {name} не содержит извлекаемого текста")
        chunks = split_into_word_chunks(raw_text, chunk_words, chunk_overlap)
        try:
            info = retrieval.upsert_chunks(name, chunks)
        except Exception as e:
            raise HTTPException(500, f"Ошибка индексации в Qdrant: {e}")
        total_chunks += info.get("chunks", 0)
        log.info("Uploaded %s: %s chunks", name, info.get("chunks", 0))
    return {"ok": True, "chunks": total_chunks}

@app.post("/upload_async")
async def upload_async(
    background: BackgroundTasks,
    files: List[UploadFile] = File(...),
    chunk_words: int = Query(220, ge=50, le=1000),
    chunk_overlap: int = Query(40, ge=0, le=500),
):
    if not files:
        raise HTTPException(400, "Нет файлов")
    tasks = []
    for f in files:
        name = (f.filename or "upload").strip()
        raw_bytes = await f.read()
        ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
        if ext == "txt":
            raw_text = _read_txt(raw_bytes)
        elif ext == "pdf":
            try:
                raw_text = _read_pdf(raw_bytes)
            except Exception as e:
                raise HTTPException(415, f"Не удалось прочитать PDF: {e}")
        else:
            raise HTTPException(415, "Поддерживаются только .pdf и .txt")
        raw_text = raw_text.strip()
        if not raw_text:
            raise HTTPException(400, f"Файл {name} не содержит извлекаемого текста")
        chunks = split_into_word_chunks(raw_text, chunk_words, chunk_overlap)
        task_id = str(uuid4())
        INDEX_PROGRESS[task_id] = {"filename": name, "processed": 0, "total": len(chunks), "done": False}
        background.add_task(_index_in_background, task_id, name, chunks)
        tasks.append({"task_id": task_id, "filename": name, "total": len(chunks)})
        log.info("Scheduled background indexing %s: %s chunks", name, len(chunks))
    return {"ok": True, "tasks": tasks}

@app.get("/progress")
def progress(task_id: str):
    st = INDEX_PROGRESS.get(task_id)
    if not st:
        raise HTTPException(404, "task_id not found")
    return st

@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=2),
    top_k: int = Query(5, ge=1, le=50),
    fetch_k: int = Query(50, ge=1, le=200),
    ef: int = Query(512, ge=64, le=4096),
    exact: bool = Query(False),
):
    items, di = retrieval.search(q, top_k=top_k, fetch_k=fetch_k, ef=ef, exact=exact)
    return SearchResponse(items=items, diagnostics=di)

@app.get("/ask", response_model=AskResponse)
def ask(
    q: str = Query(..., min_length=2),
    top_k: int = Query(5, ge=1, le=10),
):
    try:
        answer, sources, di = retrieval.answer_with_sources(q, top_k=top_k)
        return AskResponse(answer=answer, sources=sources, diagnostics=di)
    except RuntimeError as e:
        # Например, отсутствует OPENAI_API_KEY
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Ask failed: {e}")

@app.post("/reindex")
def reindex(force: bool = False):
    if not force:
        raise HTTPException(400, "Set force=true to recreate collection")
    try:
        from .retrieval import _qd, COLLECTION
        _qd.delete_collection(COLLECTION)
    except Exception as e:
        log.warning("Delete collection failed or not exists: %s", e)
    return {"ok": True}

