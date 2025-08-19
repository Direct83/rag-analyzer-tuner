import os
import uuid
import logging
from typing import List, Tuple, Dict, Optional, Callable
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, SearchParams
from openai import OpenAI
from .schemas import SearchItem

log = logging.getLogger(__name__)

QDRANT_URL = "http://localhost:6333"
COLLECTION = "documents"
LLM_MODEL = "gpt-5"  # дефолтная модель; может быть переопределена через OPENAI_MODEL

_model: Optional[SentenceTransformer] = None
_qd = QdrantClient(QDRANT_URL)
# Клиент OpenAI создаём лениво внутри функции, чтобы не требовать ключ при импорте

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        # Отключаем любые progress bars, чтобы не рушить вывод в Windows/git-bash
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TQDM_DISABLE", "1")
        # Подхватить переменные из .env (если есть)
        load_dotenv(override=False)
        _model = SentenceTransformer("BAAI/bge-m3")
    return _model

def _embed_batch(texts: List[str]) -> List[List[float]]:
    model = _get_model()
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vecs]

def embed_one(text: str) -> List[float]:
    return _embed_batch([text])[0]

def _get_llm_model() -> str:
    # Подхватываем модель из окружения/файла .env при каждом вызове
    load_dotenv(override=False)
    return os.getenv("OPENAI_MODEL") or LLM_MODEL

def ensure_collection() -> None:
    dim = _get_model().get_sentence_embedding_dimension()
    try:
        _qd.get_collection(COLLECTION)
        return
    except Exception:
        pass
    _qd.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    log.info("Created collection %s (dim=%s, cosine)", COLLECTION, dim)

def upsert_chunks(
    filename: str,
    chunks: List[str],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, int]:
    if not chunks:
        return {"chunks": 0}
    ensure_collection()
    BATCH = 128
    total = 0
    i = 0
    total_chunks = len(chunks)
    while i < total_chunks:
        batch = chunks[i:i+BATCH]
        vecs = _embed_batch(batch)
        points = []
        for j, (t, vec) in enumerate(zip(batch, vecs), start=i):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={"text": t, "filename": filename, "idx": j}
            ))
        _qd.upsert(collection_name=COLLECTION, points=points)
        total += len(points)
        i += len(batch)
        if progress_callback is not None:
            try:
                progress_callback(min(i, total_chunks), total_chunks)
            except Exception:
                # прогресс необязателен, ошибки колбэка игнорируем
                pass
    return {"chunks": total}

def list_documents(limit: int = 1000) -> Dict[str, int]:
    # Если коллекция отсутствует (после очистки) — вернуть пустой список без ошибки
    try:
        _qd.get_collection(COLLECTION)
    except Exception:
        return {}
    filenames: Dict[str, int] = {}
    next_offset = None
    try:
        while True:
            points, next_offset = _qd.scroll(
                collection_name=COLLECTION,
                with_payload=True,
                limit=256,
                offset=next_offset
            )
            if not points:
                break
            for p in points:
                fn = (p.payload or {}).get("filename")
                if fn:
                    filenames[fn] = filenames.get(fn, 0) + 1
            if next_offset is None:
                break
    except Exception:
        # На случай гонки при одновременной очистке — вернуть пусто
        return {}
    return filenames

def search(q: str, top_k: int = 5, fetch_k: int = 50, ef: int = 512, exact: bool = False) -> Tuple[List[SearchItem], Dict[str, object]]:
    qvec = embed_one(q)
    params = SearchParams(hnsw_ef=ef, exact=exact)
    hits = _qd.search(
        collection_name=COLLECTION,
        query_vector=qvec,
        limit=fetch_k,
        with_payload=True,
        search_params=params
    )
    items = []
    for h in hits[:top_k]:
        payload = h.payload or {}
        items.append(SearchItem(
            score=float(h.score),
            text=str(payload.get("text","")),
            filename=payload.get("filename"),
            idx=payload.get("idx")
        ))
    di = {"ef": ef, "exact": exact, "fetch_k": fetch_k, "top_k": top_k, "count": len(hits)}
    return items, di

def answer_with_sources(q: str, top_k: int = 5) -> Tuple[str, List[SearchItem], Dict[str, object]]:
    load_dotenv(override=False)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Export it before calling /ask.")
    items, di = search(q, top_k=top_k, fetch_k=50, ef=512, exact=False)
    context_lines = []
    for i, it in enumerate(items, start=1):
        mark = f"[{i}] {it.filename or 'doc'}:{it.idx or 0}"
        context_lines.append(f"{mark}\n{it.text}")
    context = "\n\n".join(context_lines) if context_lines else "(пусто)"
    system = (
        "Ты отвечаешь на русском строго по источникам. "
        "Если в источниках нет ответа — скажи, что информации недостаточно."
    )
    prompt = f"Вопрос: {q}\n\nИсточники:\n{context}"
    oa = OpenAI(api_key=api_key)
    resp = oa.chat.completions.create(
        model=_get_llm_model(),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
    )
    answer = (resp.choices[0].message.content or "").strip()
    return answer, items, di

