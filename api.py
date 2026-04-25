"""
api.py — FastAPI REST service wrapping the filler-removal pipeline.

Endpoints
---------
GET  /health              → {status, model, version}
POST /v1/analyze          → {elapsed, final, flagged, word_count, ...}  (multipart)
POST /v1/analyze-raw      → same, but accepts raw JSON body              (UXP plugin)

Auth
----
Every /v1/* request requires two headers:
  X-API-Key    : product key (set API_KEYS env var, comma-separated, default 'devkey')
  X-Claude-Key : caller's own Anthropic key — used for THIS request only, never stored

Design note: synchronous for now (PR #6/#7).
Async + Redis job queue added in PR #8 after manual Premiere Pro testing confirms
the pipeline produces correct results end-to-end.

Rate limiting: 10 requests/minute per product key (NOT per IP — MNC offices
route many editors through a single IP address).
"""

import json
import logging
import os
import tempfile
import time

from fastapi import Depends, FastAPI, Header, HTTPException, Request, UploadFile
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

import config
import pipeline

# ---------------------------------------------------------------------------
# Logging — JSON lines so Railway/Render dashboard can filter by field
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Auth — product key
# ---------------------------------------------------------------------------
_API_KEYS: set[str] = set(
    k for k in os.environ.get('API_KEYS', 'devkey').split(',') if k.strip()
)


def require_api_key(x_api_key: str = Header(...)) -> str:
    """FastAPI dependency — validates X-API-Key header."""
    if x_api_key not in _API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key


# ---------------------------------------------------------------------------
# Rate limiting — keyed on product key, NOT on IP address
# ---------------------------------------------------------------------------
def _key_from_header(request: Request) -> str:
    """Return the product key (or client IP as fallback) for rate-limit bucketing."""
    return request.headers.get('x-api-key', request.client.host)


limiter = Limiter(key_func=_key_from_header)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PranavAutoedit API",
    description=(
        "Filler-removal pipeline for Adobe Premiere Pro transcripts. "
        "Upload a transcript JSON, receive confirmed cut ranges."
    ),
    version="1.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get('/health', tags=['System'])
def health() -> dict:
    """Uptime probe for Railway / Render. Returns 200 when the service is ready."""
    return {
        'status': 'ok',
        'model':  config.CLAUDE_MODEL,
        'version': '1.0.0',
    }


# ---------------------------------------------------------------------------
# Shared validation + pipeline execution
# (used by both /v1/analyze and /v1/analyze-raw)
# ---------------------------------------------------------------------------

async def _run_pipeline(content: bytes, claude_key: str) -> dict:
    """
    Validate *content*, write to a temp file, run the pipeline, clean up.
    Raises HTTPException on any error — never leaks temp files.
    """
    max_bytes = config.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Content too large — max {config.MAX_FILE_SIZE_MB} MB",
        )

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="Body is not valid JSON")

    if 'segments' not in data:
        raise HTTPException(
            status_code=422,
            detail="JSON missing required key 'segments'",
        )

    tmp = tempfile.NamedTemporaryFile(mode='wb', suffix='.json', delete=False)
    tmp.write(content)
    tmp.close()

    t0 = time.time()
    try:
        result = pipeline.run(tmp.name, claude_api_key=claude_key)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Internal error: temp file missing")
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"AI response error: {exc}")
    except Exception as exc:
        log.error(f"pipeline_error={exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    elapsed = round(time.time() - t0, 2)
    log.info(
        f"status=done words={result['word_count']} "
        f"cuts={result['final_count']} elapsed={elapsed}s"
    )
    return {'elapsed': elapsed, **result}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post('/v1/analyze', tags=['Pipeline'])
@limiter.limit('10/minute')
async def analyze(
    request: Request,
    file: UploadFile,
    x_claude_key: str = Header(...,
        description="Your personal Anthropic API key (sk-ant-...). "
                    "Used for this request only — never stored."),
    _key: str = Depends(require_api_key),
) -> dict:
    """
    Analyze a Premiere Pro transcript JSON and return filler cut ranges.

    - **file**: Premiere transcript JSON (multipart upload)
    - **X-Claude-Key**: Your Anthropic API key
    - **X-API-Key**: Product key

    Returns confirmed cuts (`final`), uncertain cuts (`flagged`), and stats.
    The call blocks ~5–15 s while the AI reviews candidate cuts.
    """
    content = await file.read()
    return await _run_pipeline(content, x_claude_key)


@app.post('/v1/analyze-raw', tags=['Pipeline'])
@limiter.limit('10/minute')
async def analyze_raw(
    request: Request,
    x_claude_key: str = Header(...,
        description="Your personal Anthropic API key (sk-ant-...). "
                    "Used for this request only — never stored."),
    _key: str = Depends(require_api_key),
) -> dict:
    """
    Same as `/v1/analyze` but accepts a **raw JSON body** instead of multipart.

    Used by the Premiere Pro UXP plugin (avoids multipart encoding in UXP fetch).

    - **Body**: Premiere transcript JSON (Content-Type: application/json)
    - **X-Claude-Key**: Your Anthropic API key
    - **X-API-Key**: Product key
    """
    content = await request.body()
    return await _run_pipeline(content, x_claude_key)
