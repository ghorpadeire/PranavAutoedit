# PranavAutoedit — Filler Remover for Premiere Pro

Automatically removes filler words, dead air, and false starts from Adobe Premiere Pro videos.
A rule-based detector (O(n)) feeds into a Claude AI review, returning precise time ranges to cut.

---

## Architecture

```
CLIENT SIDE (editor's machine)
┌──────────────────────────────────────┐
│  Premiere Pro                        │
│  └── UXP Plugin Panel  (plugin/)     │
│       • API key input (saved secure) │
│       • "Get Transcript" button      │
│       • "Analyze Fillers" button     │
│       • Cut list + Copy JSON         │
└────────────────┬─────────────────────┘
                 │  HTTPS
                 │  X-API-Key: <product key>
                 │  X-Claude-Key: <editor's own Anthropic key>
                 ▼
YOUR SIDE (local dev or Railway/Render)
┌──────────────────────────────────────┐
│  FastAPI  (api.py)                   │
│  POST /v1/analyze      multipart     │
│  POST /v1/analyze-raw  raw JSON      │
│  GET  /health                        │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Pipeline  (pipeline.py)             │
│  Step 1+2: analyze_transcript.py     │  O(n) rule-based
│  Step 3:   ai_review.py             │  Claude API review
└──────────────────────────────────────┘
```

**What the editor installs:** Premiere UXP plugin (load via UXP Developer Tool)
**What the editor configures:** Their own Anthropic API key (entered once in the plugin)
**What you manage:** The backend server — push code → Railway/Render → live instantly

---

## What it detects

| Type | Method | Complexity |
|---|---|---|
| **Disfluency tags** | Premiere's own `"disfluency"` labels (um/uh sounds) | O(1) per word |
| **Text fillers** | `um/uh/hmm` always removed; `okay/so/well` only at sentence start | O(1) per word |
| **Long silences** | Gap between words > 0.8 s | O(1) per word |
| **False starts** | Repeated sentences — Jaccard similarity + bigram overlap | O(k) per sentence |
| **AI review** | Claude confirms, rejects, or flags every candidate cut | O(1) API call |

Full pipeline runs in **O(n + m log m)** — under 15 ms for a 1-hour transcript (excluding Claude API network time).

---

## Quick start — local dev (no Docker)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Start the API
uvicorn api:app --reload
# → http://localhost:8000
# → http://localhost:8000/docs   (interactive OpenAPI docs)

# 3. Test with curl
curl -X POST http://localhost:8000/v1/analyze-raw \
  -H "X-API-Key: devkey" \
  -H "X-Claude-Key: sk-ant-YOUR-KEY" \
  -H "Content-Type: application/json" \
  -d @transcript.json
```

---

## Quick start — Docker / Podman

> ⚠️ **MNC note:** Docker Desktop requires a paid licence for companies > 250 employees.
> Replace `docker` / `docker-compose` with `podman` / `podman-compose` — all commands are identical.
> Install Podman: https://podman.io/docs/installation

```bash
# Start API + Redis
docker-compose up

# Run tests inside the container
docker-compose run api python -m unittest discover -s tests -v
```

---

## Load the Premiere Pro plugin

1. Open Premiere Pro
2. **Window → Extensions → UXP Developer Tool**
3. Click **Add Plugin** → navigate to the `plugin/` folder → **Load**
4. **"Filler Remover"** panel appears in the Window menu

### Plugin UI

| Section | What it does |
|---|---|
| 🔑 **API Key** | Paste your Anthropic key once → stored in OS keychain, never sent anywhere else |
| 📂 **Get Transcript** | Opens a file picker — select the `.json` exported from Premiere |
| ▶ **Analyze** | Sends transcript to the backend, waits ~10–15 s, shows results |
| 📊 **Results** | Cut list sorted by timestamp — blue = remove, amber = review manually |
| ⚙ **Advanced** | Change backend URL (for cloud) and product key |

**How to export the transcript from Premiere:**
`File → Export → Transcript…` → save as `.json` → pick it in the plugin.

---

## API reference

All `/v1/*` endpoints require two headers:

| Header | Value |
|---|---|
| `X-API-Key` | Product key (`devkey` by default locally, set `API_KEYS` env var) |
| `X-Claude-Key` | Your Anthropic key — used for this request only, never stored |

| Endpoint | Body | Returns |
|---|---|---|
| `GET /health` | — | `{status, model, version}` |
| `POST /v1/analyze` | multipart file upload | `{elapsed, final, flagged, word_count, …}` |
| `POST /v1/analyze-raw` | raw JSON body | same as above |

Rate limit: **10 requests/minute** per product key.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `API_KEYS` | `devkey` | Comma-separated product keys |
| `REDIS_URL` | `redis://localhost:6379` | Redis for docker-compose setup |
| `CLAUDE_MODEL` | `claude-sonnet-4-5` | Anthropic model to use |
| `GAP_THRESHOLD` | `0.8` | Seconds of silence to flag as long gap |
| `OVERLAP_THRESHOLD` | `0.6` | Jaccard threshold for false-start detection |
| `BIGRAM_THRESHOLD` | `0.5` | Bigram overlap threshold (secondary signal) |
| `MAX_FILE_SIZE_MB` | `50` | Max transcript file size |

All thresholds are in `config.py` — override via env var, no code changes needed.

---

## Project structure

```
PranavAutoedit/
├── analyze_transcript.py   Step 1+2 — rule-based filler detection (O(n))
├── ai_review.py            Step 3  — Claude API review (pure functions)
├── pipeline.py             Orchestrator — chains Steps 1→2→3
├── api.py                  FastAPI REST service
├── config.py               All constants (env-var overridable)
├── export_transcripts.py   BEFORE/AFTER text export for manual review
│
├── plugin/
│   ├── manifest.json       UXP plugin manifest (Premiere Pro 22.0+)
│   └── index.html          Plugin panel UI + logic (zero build step)
│
├── tests/
│   ├── test_analyze.py     Rule-based detector tests
│   ├── test_ai_review.py   Claude review engine tests (mocked)
│   ├── test_pipeline.py    End-to-end pipeline tests (mocked)
│   ├── test_api.py         FastAPI endpoint tests (mocked)
│   ├── test_config.py      Config + env override tests
│   └── test_export.py      Export tests
│
├── Dockerfile              python:3.11-slim, starts uvicorn
├── docker-compose.yml      API + Redis, hot-reload volume
└── requirements.txt        anthropic, fastapi, uvicorn, slowapi, redis
```

---

## Run tests

```bash
# Native (fast)
python -m unittest discover -s tests -v

# Inside Docker
docker run pranavautoedit python -m unittest discover -s tests -v
```

**153 tests** — all mocked, no real API key needed to run.

---

## Roadmap

| PR | Status | What |
|---|---|---|
| #1 | ✅ merged | Rule-based detector + BEFORE/AFTER export |
| #2 | ✅ merged | Centralized config + bigram false-start detection |
| #3 | ✅ merged | `ai_review.py` — Claude API review engine |
| #4 | ✅ merged | `pipeline.py` — full orchestrator |
| #5 | ✅ merged | Docker + Compose + expanded CI |
| #6 | ✅ ready  | `api.py` — FastAPI REST service |
| #7 | ✅ ready  | UXP Premiere plugin |
| #8 | 🔜 next   | Async jobs + Redis + cloud deploy (Railway/Render) |
| #9 | 🔜 future | Apply cuts directly to Premiere timeline |
