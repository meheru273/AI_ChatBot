# Deployment

Backend on **Render**, frontend on **Vercel**.

The split is not arbitrary. The backend holds a FAISS index and a MongoDB connection
pool in memory, both of which want a warm process; Vercel's Python functions give you
neither, and its 250 MB unzipped limit is uncomfortably close to what FAISS (62 MB on
its own) plus numpy plus the LangChain stack consume. The frontend is static files,
which is exactly what Vercel is best at.

---

## 0. Prerequisites

### Get a Voyage API key (free, no card)

1. Sign up at <https://dashboard.voyageai.com>
2. Create an API key — it starts with `pa-`
3. Free tier is 200M tokens. This project uses roughly 20k tokens to build the index
   and ~10 tokens per question, so you will not approach the limit.

### Add it locally and build the index

Add to your root `.env`:

```bash
VOYAGE_API_KEY=pa-your_key_here
EMBEDDING_PROVIDER=voyage
MONGO_DB=menu
```

Then verify end to end:

```bash
cd server
python -m scripts.build_index --dry-run   # no API calls; checks Mongo + chunking
python -m scripts.build_index             # real build, ~20k tokens
uvicorn app.main:app --reload
```

`GET /health` should report `index_loaded: true` and a chunk count of about 61.

> **Not yet verified:** the Voyage request/response handling is covered by unit tests
> with a stubbed transport, but no real Voyage call has been made. The first real
> `build_index` run is what confirms it. If `voyage-3-lite` reports an unexpected
> dimension, the build fails loudly with a clear message rather than corrupting the
> index.

---

## 1. Push to GitHub

History has been rewritten to purge the leaked Groq key, so this needs `--force`:

```bash
git push --force -u origin main
```

A backup of the pre-rewrite history is at `../RAG_Application-backup-20260729.bundle`.
Restore with `git clone RAG_Application-backup-20260729.bundle` if anything goes wrong.

Because history changed, any existing clone of this repo is now incompatible and
should be re-cloned rather than pulled.

---

## 2. Backend on Render

### Option A — Blueprint (uses `render.yaml`)

1. <https://dashboard.render.com> → **New** → **Blueprint**
2. Connect the `AI_ChatBot` repository
3. Render reads `render.yaml` and proposes the service
4. Fill in the four secrets it marks as required:

| Variable | Value |
|---|---|
| `GROQ_API_KEY` | your rotated Groq key |
| `VOYAGE_API_KEY` | the `pa-...` key from step 0 |
| `MONGODB_URI` | your Atlas connection string |
| `CORS_ORIGINS` | leave as `http://localhost:3000` for now; corrected in step 4 |

5. **Apply**

### Option B — Manual

**New** → **Web Service** → connect the repo, then:

| Setting | Value |
|---|---|
| Root Directory | `server` |
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt && python -m scripts.build_index` |
| Start Command | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| Health Check Path | `/health` |

Add the environment variables from `render.yaml` by hand.

### Verify

The build takes 3–5 minutes; the index build runs at the end and should log
`Saved index of 61 chunks`. Then:

```bash
curl https://banchan-rag-assistant.onrender.com/health
```

Expect `mongo_connected: true`, `index_loaded: true`, `embedding_provider:
voyage/voyage-3-lite (...)`.

Note the service URL — you need it next.

---

## 3. Frontend on Vercel

1. <https://vercel.com/new> → import the `AI_ChatBot` repository
2. **Root Directory**: `chat-ui` — this matters, the repo root is not the app
3. Framework preset: Create React App (auto-detected via `chat-ui/vercel.json`)
4. Add an environment variable **before** the first deploy:

   | Name | Value |
   |---|---|
   | `REACT_APP_API_URL` | `https://banchan-rag-assistant.onrender.com` |

   No trailing slash. CRA inlines this at **build** time, so changing it later
   requires a redeploy, not just a restart.

5. **Deploy**

---

## 4. Connect the two

Back in Render → your service → **Environment**, set:

```
CORS_ORIGINS=https://your-app.vercel.app
```

Use the exact Vercel origin, no trailing slash. Multiple origins are comma-separated:

```
CORS_ORIGINS=https://your-app.vercel.app,http://localhost:3000
```

Save — Render restarts automatically. Without this the browser blocks every request
with a CORS error even though the backend is healthy.

---

## 5. The free-tier cold start

Render's free plan sleeps a service after 15 minutes of inactivity, and waking takes
**about 50 seconds**. For a demo link on a CV that is a bad first impression — a
recruiter clicks, sees a spinner, and leaves.

Three options:

1. **Keep it warm.** A free cron pinging `/health` every 10 minutes — e.g.
   [cron-job.org](https://cron-job.org) or a GitHub Action on a schedule. Cheapest
   fix, works well.
2. **Render Starter** ($7/month) — no sleeping.
3. **Accept it** and note "first load may take ~50s" next to the demo link. Honest,
   but you lose people.

I'd do option 1.

---

## Troubleshooting

**CORS error in the browser console.** `CORS_ORIGINS` on Render does not exactly match
the Vercel origin. Check for a trailing slash or `http` vs `https`.

**`index_loaded: false`.** The build-time index step failed — check the Render build
log for the `build_index` output. Usually a bad `VOYAGE_API_KEY` or an unreachable
Atlas cluster.

**`mongo_connected: false`.** Atlas Network Access is probably blocking Render. Add
`0.0.0.0/0` to the IP allowlist, or Render's static outbound IPs on a paid plan.

**Frontend shows "Disconnected".** `REACT_APP_API_URL` was missing or wrong at build
time. Fix it in Vercel and **redeploy** — a restart is not enough.

**Streaming shows nothing, then everything at once.** A proxy is buffering. The app
sends `X-Accel-Buffering: no`; if you put anything else in front of Render, it needs
to respect that.

---

## Still outstanding

1. **Create a read-only Atlas user.** The assistant currently connects with whatever
   credentials are in your `.env`. A chatbot should not hold write access to live
   restaurant data — create a user with `read` on the `menu` database only, and use
   that URI in Render.
2. **Decommission the old Postgres database.** `DATABASE_URL` is still in your local
   `.env` and its credentials were exposed in git history. Nothing in the code uses
   it anymore.
