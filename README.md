# RAG Minimal (PDF/TXT · bge-m3 · Qdrant · GPT-5)

👉 Загрузи свои PDF/TXT, настрой параметры разбиения и поиска. Система не только выдаёт ответ от LLM, но и показывает топ‑чанк, из которого он получен.

Минимальный локальный RAG: эмбеддинги bge‑m3 + Qdrant для поиска по чанкам и LLM для формулирования ответа. 
UI — лёгкий и современный: drag&drop загрузка, фоновой прогресс, параметры чанкинга/поиска, модальное окно с чанками.

## Что умеет
- Загрузка .pdf/.txt (drag&drop/диалог) → фоновая индексация с прогрессом.
- Чанкинг на стороне сервера: `chunk_words` (размер) и `chunk_overlap` (перекрытие) задаются в UI.
- Поиск: «Найти чанки» открывает модалку с результатами и диагностикой (`top_k`, `fetch_k`, `ef`, `exact`).
- Ответ: «Задать вопрос» выводит только текст ответа. Чанки можно посмотреть отдельно через «Найти чанки».
- Очистка базы (кнопка «Очистить базу», сервер: `POST /reindex?force=true`).

## Быстрый старт
1) Установить зависимости
   pip install -r requirements.txt
2) Запустить Qdrant локально (Docker)
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
3) Ключ и модель OpenAI (нужны только для `/ask`)
   - Создай файл `.env` (или используй переменные окружения)
   - Переменные: `OPENAI_API_KEY`, опционально `OPENAI_MODEL` (по умолчанию `gpt-5`)
   Пример `.env`:
   OPENAI_API_KEY=sk-...
   OPENAI_MODEL=gpt-4o-mini
4) Запустить сервер
   uvicorn app.main:app --reload
5) Открыть UI
   http://localhost:8000/

## API (кратко)
- GET `/` — UI
- GET `/health` — healthcheck
- GET `/documents` — { filename → count }
- POST `/upload` — синхронная загрузка
  - query: `chunk_words` (default 220), `chunk_overlap` (default 40)
  - form: `files` (.pdf/.txt)
- POST `/upload_async` — фоновая индексация
  - query: `chunk_words`, `chunk_overlap`
  - form: `files`
  - resp: `{ ok, tasks:[{ task_id, filename, total }] }`
- GET `/progress?task_id=...` — `{ processed, total, done, ok?, error? }`
- GET `/search` — `q`, `top_k`, `fetch_k`, `ef`, `exact` → чанки и диагностика
- GET `/ask` — `q`, `top_k` → текст ответа (без вывода чанков на странице)
- POST `/reindex?force=true` — очистка коллекции

## Технические детали
- Эмбеддинги: `BAAI/bge-m3` (normalize_embeddings=True, cosine)
- Векторное хранилище: Qdrant (HNSW); управляем поиском через `ef`, `exact`, `fetch_k`.
- Чанкинг: разбиение по словам с параметрами `chunk_words`/`chunk_overlap`.
- UI: результаты поиска — в модальном окне; ответы — в основном блоке.

## Ограничения/подсказки
- Для `/ask` обязателен `OPENAI_API_KEY`. Без него доступны загрузка и поиск.
- Модель для чата берётся из `OPENAI_MODEL` (если не задана — используется `gpt-5`).
- Прогресс индексации хранится в памяти процесса и сбрасывается при рестарте.


