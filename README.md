# Banchan Assistant — Hybrid RAG for a Restaurant

A conversational assistant for a Korean restaurant, built as a **LangGraph agent that
routes between semantic retrieval and structured database queries** depending on what
the question actually needs.

The central idea is that vector search is the wrong tool for a lot of restaurant
questions. "What's under £6" is a numeric predicate. "I'm allergic to sesame" has to
be exact — a near miss in embedding space is a safety problem, not a ranking problem.
So the agent has both paths and chooses.

| Question | Route | Why |
|---|---|---|
| "How much is bibimbap?" | MongoDB query | Exact value |
| "What's under £6?" | MongoDB `$lt` | Numeric predicate |
| "I'm allergic to sesame" | MongoDB `$nin` | Must be exact, never approximate |
| "Something warm and spicy" | FAISS | Fuzzy intent, no keyword overlap |
| "Japchae vs gimbap?" | FAISS | Comparative, needs prose |
| "Do you deliver to UB7?" | FAISS over knowledge base | Not in the database |

---

## Architecture

```
chat-ui (React 19)
   │  POST /chat/stream    Server-Sent Events
   ▼
FastAPI  ──  LangGraph agent  ──▶  ChatGroq (openai/gpt-oss-120b)
                    │
       ┌────────────┼──────────────┬──────────────────┐
       ▼            ▼              ▼                  ▼
  search_menu   filter_menu   search_knowledge   wikipedia / arxiv
  list_categories             (FAISS + embeddings)
  restaurant_info
       │                           │
       ▼                           ▼
  MongoDB Atlas              FAISS index over
  (menu, categories,         menu passages +
   restaurant)               knowledge markdown
```

### Retrieval

The index covers two sources: every menu item rendered as a prose passage
(name, description, category, price, dietary labels, allergens), and a hand-written
knowledge base in `server/data/knowledge/` covering the allergen policy, a Korean
food glossary, delivery terms, restaurant details, and an FAQ.

Vectors are L2-normalised in a FAISS `IndexFlatIP`, making the score exact cosine
similarity.

### Pluggable embeddings

| `EMBEDDING_PROVIDER` | Model | Dim | Notes |
|---|---|---|---|
| `jina` *(default)* | `jina-embeddings-v3` | 1024 | Free tier, no card, usable rate limit |
| `voyage` | `voyage-3-lite` | 512 | MongoDB's own provider; free tier is 3 req/min without a card |
| `openai` | `text-embedding-3-small` | 1536 | Paid |
| `local` | `all-MiniLM-L6-v2` | 384 | Offline, but installs torch (~2 GB) — not viable on serverless or small instances |

Both hosted providers encode passages and queries **asymmetrically** — Jina via
`task=retrieval.passage` / `retrieval.query`, Voyage via `input_type`. They share a
REST base class, so adding another is a request-body method.

The index records the provider, model and dimension it was built with. Loading it
under a different configuration raises a clear "rebuild the index" error instead of
crashing inside FAISS — or worse, silently returning confident nonsense when the two
models happen to share a dimension.

---

## Running it

### Backend

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r server/requirements.txt

cp server/.env.example server/.env # then fill it in

cd server
python -m scripts.build_index      # embed the corpus
uvicorn app.main:app --reload
```

`python -m scripts.build_index --dry-run` reports the corpus without embedding it,
which is the quickest way to check the database connection and chunking.

For `EMBEDDING_PROVIDER=local`, also `pip install sentence-transformers`.

### Frontend

```bash
cd chat-ui
npm install
npm start
```

The API URL comes from `REACT_APP_API_URL` (`chat-ui/.env.development`,
`chat-ui/.env.production`).

### Tests

```bash
cd server && pytest        # 115 tests
cd chat-ui && npm test
```

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Liveness |
| `GET` | `/health` | Readiness — reports Mongo, index size, embedding provider, tools |
| `POST` | `/chat` | Buffered reply, one JSON body |
| `POST` | `/chat/stream` | Server-Sent Events |

### SSE event protocol

```
event: token       data: {"content": "..."}       incremental output
event: tool_start  data: {"tool": "search_menu"}  a tool began
event: tool_end    data: {"tool": "search_menu"}  a tool returned
event: done        data: {"content": "..."}       full answer, terminal
event: error       data: {"message": "..."}       failure, terminal
```

`error` matters: the HTTP status is committed the moment the stream opens, so a
failure after that point cannot become a 500. Without an in-stream error event the
client waits forever for a `done` that never comes.

---

## Data

MongoDB Atlas, shared with the [Banchan Korean Restaurant](https://github.com/meheru273)
ordering app — `menuitems`, `categories` and `restaurants`. The assistant connects
with a **read-only** user; it never writes.

Menu items carry structured `allergens` (the 14 UK-regulated values) and `dietary`
flags, which is what makes exact allergen filtering possible.

---

## Layout

```
server/
  app/
    api/routes/     chat.py (SSE), health.py
    agent/          graph.py, prompts.py, llm.py, state.py
    tools/          menu.py, knowledge.py, external.py
    rag/            embeddings.py, vectorstore.py, documents.py, chunking.py, ingest.py
    db/             mongo.py, models.py, repository.py
    config.py       all environment variables, declared once
  data/knowledge/   the corpus, as markdown
  scripts/          build_index.py
  tests/
chat-ui/            React frontend
docs/               plan.md, architecture.md
```

---

## Deployment

Backend on **Render** (`render.yaml`), frontend on **Vercel**. The FAISS index is
built during the Render build step, not at request time.

Vercel's Python runtime is not suitable for the backend: the 250 MB function limit
and the absence of a warm process mean the index and connection pool would be rebuilt
on every cold start.

---

## Tech

LangGraph · LangChain · Groq (`openai/gpt-oss-120b`) · FAISS · Jina / Voyage /
OpenAI / sentence-transformers embeddings · FastAPI · sse-starlette ·
MongoDB (Motor) · React 19 · pytest
