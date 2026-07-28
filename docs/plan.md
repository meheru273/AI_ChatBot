# Integration Plan — Restaurant RAG Assistant

**Status:** phases 1–9 implemented; phase 0 (security) outstanding — see §10
**Date:** 2026-07-28
**See also:** [architecture.md](architecture.md) for the decision records
**Goal:** turn the existing tool-calling chatbot into a real hybrid RAG assistant with a modular
codebase, streaming API, and an integration test suite.

---

## 1. Where the project stands today

| Component | Current state |
|---|---|
| Agent | LangGraph `StateGraph` + `ToolNode` + `tools_condition` — works, keep the pattern |
| LLM | Groq `Gemma2-9b-It` |
| Tools | Wikipedia; a Postgres menu lookup (`ILIKE '%term%'`) |
| Retrieval | **None.** No embeddings, no vector store, no chunking |
| API | FastAPI, single `POST /chat` returning one JSON blob — no streaming |
| Frontend | React 19 (CRA), `fetch`, API URL hardcoded to `localhost:8000` |
| Tests | None (CRA default `App.test.js` only, and it would fail) |
| Layout | Flat — every Python module at repo root |
| Secrets | `.env` committed since the first commit; live key hardcoded in `console_test.py` |

### Target capability set

1. Real RAG — chunking, embeddings, FAISS, a retriever exposed as an agent tool
2. Pluggable embedding providers — OpenAI **and** local sentence-transformers
3. MongoDB as the structured data source (replacing Postgres)
4. SSE streaming end to end, token by token
5. Wikipedia + arXiv tools restored alongside the domain tools
6. Modular package layout (`server/` mirroring `chat-ui/`)
7. Integration tests, including tests against the streaming endpoint

---

## 2. Key architectural decisions

### 2.1 MongoDB replaces Postgres

The Postgres `Restaurant1` table holds `id, ProductName, Price, amount` — four columns, no prose,
no attributes. There is nothing meaningful to embed.

The Banchan Korean Restaurant Atlas cluster (`services/menu-service`) already holds:

- **MenuItem** — `name`, `description` (prose), `price`, `category` ref, `dietary`
  (vegetarian/vegan/glutenFree/halal), `allergens` (14-value UK enum), `calories`,
  `preparationTime`, `isAvailable`, `isPopular`
- **Category** — 6 categories with slugs and descriptions
- **Restaurant** — address, postcode, per-day opening hours, delivery radius, minimum order,
  delivery fee, geo coordinates
- **21 seeded Korean dishes** with real descriptions

**Decision:** connect to this cluster read-only. It supplies both the structured-query surface and
the text corpus for embedding.

**Required:** a dedicated read-only Atlas user (`rag_readonly`) scoped to the menu database. The
assistant must never hold write credentials to live restaurant data.

### 2.2 Hybrid retrieval — vector search does not replace structured queries

Routing each question to the right mechanism is the core design idea:

| Question type | Mechanism | Why |
|---|---|---|
| "How much is bibimbap?" | Mongo structured query | Exact values; embeddings would approximate a fact |
| "What's under £6?" | Mongo `$lt` filter | Numeric predicate — vectors cannot do this |
| "I'm allergic to sesame" | Mongo `$nin` on `allergens` | Safety-critical; must be exact, not similar |
| "Something warm and spicy" | FAISS semantic search | Fuzzy intent, no keyword overlap |
| "Difference between japchae and gimbap?" | FAISS over descriptions | Comparative, needs prose |
| "Do you deliver to UB7?" | FAISS over knowledge docs | Not in the menu collection |

The agent picks; the LLM sees both tools and routes.

> **Allergen safety note:** allergen filtering goes through the structured tool only, never through
> semantic retrieval. A near-miss vector hit on an allergy question is a real-world harm, not just a
> wrong answer. The system prompt states this explicitly, and the tool returns allergen data as
> authoritative structured fields.

### 2.3 Pluggable embeddings

Both providers, selected by config — no code change to switch.

| Provider | Model | Dim | Notes |
|---|---|---|---|
| `openai` | `text-embedding-3-small` | 1536 | Matches CV wording; needs `OPENAI_API_KEY`; ~cents at this scale |
| `local` | `all-MiniLM-L6-v2` | 384 | Free, offline, no key; pulls in `torch` (~2 GB) |

A shared `EmbeddingProvider` protocol with two implementations behind a factory.

**Dimension mismatch guard:** the FAISS index is written with a `meta.json` sidecar recording
provider, model name, and dimension. Loading an index built by a different provider raises a clear
error telling you to rebuild rather than failing deep inside FAISS with a shape error.

**Deploy consequence:** `local` cannot run on Vercel serverless (size limit). See §7.

### 2.4 FAISS, not Atlas Vector Search

Atlas has native vector search and the data already lives there, so it is the obvious question.
Choosing FAISS anyway because: it matches the CV, it keeps retrieval working against the local
knowledge corpus without a cluster round-trip, and it stays free on the M0 tier. Documented here so
the tradeoff is a deliberate choice rather than an oversight.

Index size is ~40 documents — trivially in-memory, rebuilt in seconds.

### 2.5 Knowledge corpus

Mongo covers dishes. It does not cover policy, provenance, or explanation. Authored markdown in
`server/data/knowledge/`:

| File | Contents |
|---|---|
| `allergens-and-dietary.md` | The 14 UK allergens, cross-contamination policy, how to order safely |
| `korean-food-glossary.md` | Gochujang, banchan, tteok, jjigae, doenjang, kimchi varieties |
| `ordering-and-delivery.md` | Delivery radius, minimum order, fees, timings, tracking |
| `restaurant-info.md` | Location, opening hours, contact, accessibility |
| `faq.md` | Payment, cancellation, refunds, group orders, catering |

Menu items are **also** embedded, as synthesized passages combining name + description + category +
dietary + allergens, so semantic queries can reach dish content.

---

## 3. Target file structure

```
RAG_Application/
├── docs/
│   └── plan.md                       ← this file
├── server/                           ← all backend code (mirrors chat-ui/)
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                   # create_app(), CORS, lifespan
│   │   ├── config.py                 # pydantic-settings Settings
│   │   ├── api/
│   │   │   ├── schemas.py            # request/response models
│   │   │   ├── deps.py               # DI providers
│   │   │   └── routes/
│   │   │       ├── health.py         # GET /, GET /health
│   │   │       └── chat.py           # POST /chat, POST /chat/stream
│   │   ├── agent/
│   │   │   ├── state.py              # AgentState TypedDict
│   │   │   ├── prompts.py            # system prompt
│   │   │   ├── llm.py                # Groq factory
│   │   │   └── graph.py              # StateGraph assembly
│   │   ├── tools/
│   │   │   ├── __init__.py           # get_tools() registry
│   │   │   ├── menu.py               # structured Mongo queries
│   │   │   ├── knowledge.py          # FAISS retriever tool
│   │   │   ├── wikipedia.py
│   │   │   └── arxiv.py
│   │   ├── rag/
│   │   │   ├── embeddings.py         # provider protocol + factory
│   │   │   ├── vectorstore.py        # FAISS build/save/load/search
│   │   │   ├── documents.py          # Mongo + markdown → Documents
│   │   │   ├── chunking.py           # splitter config
│   │   │   └── ingest.py             # end-to-end pipeline
│   │   └── db/
│   │       ├── mongo.py              # Motor async client
│   │       ├── models.py             # Pydantic mirrors of the Mongoose schemas
│   │       └── repository.py         # query methods
│   ├── data/
│   │   ├── knowledge/*.md            # authored corpus (committed)
│   │   └── index/                    # FAISS artifacts (gitignored)
│   ├── scripts/
│   │   └── build_index.py            # CLI: rebuild the vector index
│   ├── tests/
│   │   ├── conftest.py               # fixtures, fakes
│   │   ├── test_health.py
│   │   ├── test_chat.py
│   │   ├── test_chat_stream.py       # SSE integration tests
│   │   ├── test_tools.py
│   │   └── test_rag.py
│   ├── requirements.txt
│   ├── .env.example
│   └── pyproject.toml                # pytest/ruff config
├── chat-ui/                          ← unchanged location
├── .gitignore                        ← rewritten
└── README.md                         ← rewritten
```

### Files removed

| File | Reason |
|---|---|
| `main.py` | Superseded prototype; executes a query at import, references undefined `app` |
| `console_test.py` | Contains a hardcoded API key; replaced by the test suite |
| `DBConnection.py`, `db_tool.py`, `graph.py`, `server.py` | Moved into `server/app/` |
| `__pycache__/` | Should never have been committed |
| `chat-ui/build/` | Build artifact; should never have been committed |

---

## 4. Step-by-step integration

### Phase 0 — Security remediation (do first, blocks everything)

The repo is public and linked from a CV. Credentials are live.

1. **Rotate the Groq key** at console.groq.com — the current one is in git history and in
   `console_test.py:18`
2. **Rotate the Postgres `DATABASE_URL`** credentials (or decommission the DB — we are leaving it)
3. **Create the read-only Atlas user** `rag_readonly`; do not reuse `feastfleet_admin`
4. `git rm --cached .env` and add to `.gitignore`
5. Rewrite `.gitignore` for Python + Node (currently the CRA one, which ignores neither `.env`,
   `__pycache__/`, nor a venv)
6. **Purge from history** with `git filter-repo --path .env --invert-paths`, then force-push
   — rotation alone leaves the old values readable in every clone
7. Untrack `__pycache__/` and `chat-ui/build/`

> Steps 1–3 and 6 need your credentials and your GitHub account — you run those, I prepare
> everything else.

### Phase 1 — Restructure

8. Create the `server/` tree from §3 with package `__init__.py` files
9. Move existing logic into place (no behaviour change yet), splitting `graph.py` into
   `agent/{state,prompts,llm,graph}.py`
10. Add `config.py` — a `pydantic-settings` `Settings` object; every env var read in exactly one
    place instead of scattered `os.environ.get` calls
11. Add `.env.example` documenting all variables with placeholders
12. Update `Procfile` / deployment config for the new module path

**Checkpoint:** existing `/chat` still answers correctly against the new layout.

### Phase 2 — MongoDB data layer

13. `db/mongo.py` — Motor client, pooled, opened/closed in the FastAPI lifespan
14. `db/models.py` — Pydantic models mirroring the Mongoose schemas (`MenuItem`, `Category`,
    `Restaurant`), including the `allergens` enum
15. `db/repository.py` — `search_by_name`, `get_by_id`, `list_by_category`,
    `filter_by_dietary(exclude_allergens=[...], vegan=True, ...)`, `get_restaurant_info`
16. Delete the Postgres layer

**Checkpoint:** a script prints all 21 dishes with allergens from Atlas.

### Phase 3 — RAG layer

17. `rag/embeddings.py` — `EmbeddingProvider` protocol; `OpenAIEmbeddings` and `LocalEmbeddings`
    implementations; `get_embedding_provider()` factory reading `EMBEDDING_PROVIDER`
18. `rag/documents.py` — build `Document`s from (a) Mongo menu items as synthesized passages,
    (b) the markdown knowledge corpus; attach metadata (`source`, `type`, `item_id`, `category`)
19. `rag/chunking.py` — `RecursiveCharacterTextSplitter`, ~500 chars / 50 overlap; menu passages
    stay whole (they are already short and self-contained)
20. `rag/vectorstore.py` — FAISS build/save/load, plus the `meta.json` provider guard from §2.3
21. `rag/ingest.py` + `scripts/build_index.py` — `python -m scripts.build_index [--provider ...]`
22. Author the five knowledge markdown files

**Checkpoint:** `build_index.py` produces an index; a query script returns sensible hits under both
providers.

### Phase 4 — Tools

23. `tools/menu.py` — structured tool over `repository.py`, with a **typed args schema**
    (`pydantic` `args_schema`) rather than the current free-text string input. This fixes the
    existing bug where the system prompt documents JSON operations the tool cannot parse.
24. `tools/knowledge.py` — retriever tool: embed query → FAISS top-k → return passages with sources
25. `tools/wikipedia.py`, `tools/arxiv.py` — restore both
26. `tools/__init__.py` — `get_tools()` registry
27. Rewrite the system prompt in `agent/prompts.py` to describe the tools that **actually exist**,
    with explicit routing guidance and the allergen-safety rule from §2.2

### Phase 5 — Agent

28. Rebuild `agent/graph.py` against the new tool registry
29. Add conversation history support — thread the full message list through instead of only the
    last user message (currently dropped, so the bot has no memory)
30. Add a recursion cap so tool loops cannot run away

### Phase 6 — SSE streaming

31. `POST /chat/stream` returning `StreamingResponse` with `text/event-stream`
32. Use LangGraph `astream_events` to emit typed events:
    - `token` — incremental LLM output
    - `tool_start` / `tool_end` — so the UI can show "searching the menu…"
    - `done` — terminal
    - `error` — failure inside an already-200 stream
33. Keep `POST /chat` non-streaming for tests and simple clients
34. Correct SSE framing (`data: {...}\n\n`), heartbeats, and `X-Accel-Buffering: no`

### Phase 7 — Frontend

35. Replace hardcoded `localhost:8000` with `REACT_APP_API_URL` (`.env.development` /
    `.env.production`) — this is why the deployed frontend cannot reach the deployed backend
36. Consume SSE, appending tokens as they arrive
37. Render tool-activity indicators from `tool_start` / `tool_end`
38. Keep the existing abort/cancel wiring; make it abort the stream

### Phase 8 — Tests

39. `pytest` + `pytest-asyncio` + `httpx.AsyncClient`
40. `conftest.py` — fake embedding provider (deterministic vectors, no API calls), in-memory FAISS
    fixture, mocked Mongo, stubbed LLM
41. `test_chat_stream.py` — the integration tests the CV claims:
    - stream returns `text/event-stream`
    - events are well-formed SSE frames
    - token events concatenate to the final message
    - tool events are emitted when a tool fires
    - stream terminates with `done`
    - client disconnect mid-stream is handled
    - upstream error surfaces as an `error` event, not a hang
42. `test_rag.py` — chunking, retrieval ordering, dimension-mismatch guard
43. `test_tools.py` — menu tool arg validation, allergen filtering correctness
44. Delete CRA's default `App.test.js`; add a real render test
45. GitHub Actions workflow running both suites

### Phase 9 — Docs & deploy

46. Rewrite `README.md` to describe what the system actually does
47. `docs/architecture.md` — diagram, retrieval routing table, decision record
48. Deployment config per §7
49. Update the CV bullet to match the delivered system

---

## 5. Configuration

```bash
# LLM
GROQ_API_KEY=
GROQ_MODEL=Gemma2-9b-It

# Embeddings — "openai" or "local"
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=                       # required when provider=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2

# MongoDB (read-only user)
MONGO_URI=
MONGO_DB=banchan

# Vector index
FAISS_INDEX_PATH=server/data/index
RETRIEVAL_TOP_K=4

# Server
PORT=8000
CORS_ORIGINS=http://localhost:3000
```

---

## 6. Dependencies to add

```
# RAG
faiss-cpu
langchain-openai            # provider=openai
sentence-transformers       # provider=local (heavy)

# Mongo
motor
pymongo

# Config
pydantic-settings

# Tests
pytest
pytest-asyncio
httpx

# Restored
arxiv
```

Removed: `asyncpg`. `sentence-transformers` is optional — imported lazily so an OpenAI-only
deployment never pays the `torch` cost.

---

## 7. Deployment

Vercel's Python runtime cannot host this well: a 250 MB unzipped function limit that
`sentence-transformers` blows past on its own, plus no warm process to hold the FAISS index or the
Mongo pool, so every cold start rebuilds both.

**Recommendation:** move the backend to **Render** — you already run Banchan's services there, so
it is one dashboard, and a persistent process fits the index and connection pool properly. Frontend
stays on Vercel.

The index is built at deploy time by a build command, not at request time.

---

## 8. Sequencing

| Phase | Depends on |
|---|---|
| 0 Security | — (do first) |
| 1 Restructure | 0 |
| 2 Mongo | 1 |
| 3 RAG | 2 (needs menu docs to embed) |
| 4 Tools | 2, 3 |
| 5 Agent | 4 |
| 6 SSE | 5 |
| 7 Frontend | 6 |
| 8 Tests | 6 (7 in parallel) |
| 9 Docs | all |

Phases 1–6 are one continuous backend build. 7 and 8 can proceed in parallel afterwards.

---

## 9. Honest CV wording once delivered

> **Restaurant AI Assistant — Hybrid RAG System**
> - Built a retrieval-augmented assistant on **FastAPI** with **SSE token streaming**: a
>   **LangGraph** agent routes between **FAISS** semantic search (pluggable OpenAI / local
>   sentence-transformer embeddings) and structured **MongoDB** queries, so factual lookups
>   (price, allergens) stay exact while open-ended questions use vector retrieval.
> - Developed the **React** streaming chat client and an integration test suite covering the SSE
>   event protocol; deployed frontend on Vercel and backend on Render.

Every claim traceable to code.

---

## 10. Open items

All of these need your credentials or your accounts; none can be done from the codebase.

1. **Rotate the Groq API key.** The old one is in git history and was hardcoded in
   `console_test.py`. The file is deleted, but history still has it.
2. **Rotate or decommission the Postgres credentials** in the old `.env`. The Postgres
   layer is gone from the code; the credentials are still exposed in history.
3. **Create the read-only Atlas user** (`rag_readonly`), scoped to the menu database,
   and put its URI in `server/.env`.
4. **Purge `.env` from git history** — `git filter-repo --path .env --invert-paths`,
   then force-push. `.env` is untracked as of this work, but rotation alone leaves the
   old values readable in every existing clone.
5. **Confirm Render** for the backend, and set `CORS_ORIGINS` to the deployed Vercel
   origin plus `REACT_APP_API_URL` on Vercel to the Render URL.

### Delivered

Phases 1–9. `server/` package with `api/agent/tools/rag/db` modules, MongoDB layer,
FAISS retrieval with pluggable OpenAI/local embeddings and a provider guard, six
agent tools, SSE streaming with an in-band error event, streaming React client, 71
backend tests and 6 frontend tests, CI workflow, Render blueprint.

Not yet verified end-to-end against the live Atlas cluster — that needs item 3.
