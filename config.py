"""
config.py — single source of truth for all thresholds and settings.

Every constant can be overridden via environment variable so the MNC
deployment team never has to touch Python source code.
"""
import os


# ---------------------------------------------------------------------------
# Processing thresholds
# ---------------------------------------------------------------------------
GAP_THRESHOLD     = float(os.environ.get('GAP_THRESHOLD',     '0.8'))
OVERLAP_THRESHOLD = float(os.environ.get('OVERLAP_THRESHOLD', '0.6'))
BIGRAM_THRESHOLD  = float(os.environ.get('BIGRAM_THRESHOLD',  '0.5'))

# ---------------------------------------------------------------------------
# Filler word lists
# ---------------------------------------------------------------------------
STRONG_FILLERS = set(
    os.environ.get(
        'STRONG_FILLERS', 'um,uh,ah,er,hmm,hm,mm,mhm'
    ).split(',')
)

SENTENCE_START_FILLERS = set(
    os.environ.get(
        'SENTENCE_FILLERS', 'okay,so,right,well,alright,anyway'
    ).split(',')
)

# ---------------------------------------------------------------------------
# Claude API settings
# ---------------------------------------------------------------------------
CLAUDE_MODEL     = os.environ.get('CLAUDE_MODEL', 'claude-sonnet-4-5')
MAX_TOKENS       = int(os.environ.get('MAX_TOKENS',       '4096'))
MAX_FILE_SIZE_MB = int(os.environ.get('MAX_FILE_SIZE_MB', '50'))


# ---------------------------------------------------------------------------
# API key loader (local CLI use only — API layer supplies key per-request)
# ---------------------------------------------------------------------------
def get_local_api_key(path: str = 'claudekey.txt') -> str | None:
    """
    Returns the Anthropic API key for local script use.
    Priority: env var ANTHROPIC_API_KEY → claudekey.txt → None
    """
    if key := os.environ.get('ANTHROPIC_API_KEY'):
        return key
    try:
        with open(path) as f:
            return f.read().strip() or None
    except FileNotFoundError:
        return None
