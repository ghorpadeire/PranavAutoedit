"""
ai_review.py — Step 3: Claude API review of candidate cuts.

All functions are PURE (no file I/O, no side effects).
Importable by pipeline.py, api.py, and test files
without spinning up any external services.

Complexity:
  build_transcript_text  → O(n)
  enrich_removals        → O(n + m)  two-pointer
  call_claude            → O(1) our code (network call)
  parse_decisions        → O(m)
"""

import json
from typing import Optional

import config


def _fmt(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ---------------------------------------------------------------------------
# 1. Transcript formatting  — O(n)
# ---------------------------------------------------------------------------

def build_transcript_text(words: list) -> str:
    """
    Format word list as readable timestamped sentences.
    Sent to Claude with cache_control=ephemeral — billed once per session.
    """
    lines, current, ts = [], [], None

    for w in words:
        if not w['text']:
            continue
        if ts is None:
            ts = w['start']
        current.append(w['text'])
        if w.get('eos'):
            lines.append(f"[{_fmt(ts)}] {' '.join(current)}")
            current, ts = [], None

    # Flush any trailing words not ended by eos
    if current and ts is not None:
        lines.append(f"[{_fmt(ts)}] {' '.join(current)}")

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Context enrichment  — O(n + m)  two-pointer
# ---------------------------------------------------------------------------

def enrich_removals(removals: list, words: list) -> list:
    """
    Add 'before' and 'after' context words to each removal entry.

    Both lists are sorted by start time.
    ptr advances forward only — never resets → O(n + m) total, NOT O(n × m).
    """
    real_words = [w for w in words if w['text']]   # O(n) filter once
    enriched   = []
    ptr        = 0

    for i, r in enumerate(removals):
        # Advance ptr to the first word that starts at or after removal end
        while ptr < len(real_words) and real_words[ptr]['start'] < r['end']:
            ptr += 1

        # prev = last word whose end is at or before the removal start
        prev_candidate = real_words[ptr - 1] if ptr > 0 else None
        prev = (prev_candidate
                if prev_candidate and prev_candidate.get('end', 0) <= r['start']
                else None)
        nxt  = real_words[ptr] if ptr < len(real_words) else None

        enriched.append({
            'id':     i + 1,
            **r,
            'before': prev['text'] if prev else '[START]',
            'after':  nxt['text']  if nxt  else '[END]',
        })

    return enriched


# ---------------------------------------------------------------------------
# 3. Claude API call  — O(1) our code
# ---------------------------------------------------------------------------

def _build_removal_lines(enriched: list) -> str:
    lines = []
    for r in enriched:
        reason = r['reason']
        if reason == 'long_gap':
            detail = f"({r.get('gap_seconds')}s silence)"
        elif reason == 'text_filler':
            detail = f"word='{r.get('word', '')}'"
        elif reason == 'false_start':
            detail = f'"{r.get("text", "")}"'
        else:
            detail = ''
        lines.append(
            f"{r['id']}. [{r['start']:.2f}s-{r['end']:.2f}s] "
            f"{reason} {detail}  |  ...{r['before']} __ {r['after']}..."
        )
    return '\n'.join(lines)


def parse_ai_response(raw: str) -> dict:
    """
    Robustly parse Claude's JSON response.
    Strips markdown code fences (```json or ```) if present.
    Raises ValueError on parse failure so the caller can retry.
    """
    cleaned = raw.strip()
    for fence in ['```json', '```']:
        cleaned = cleaned.removeprefix(fence).removesuffix('```').strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Claude returned invalid JSON: {cleaned[:300]}"
        ) from exc


def call_claude(
    client,
    transcript_text: str,
    enriched: list,
    retry: bool = True,
) -> dict:
    """
    Send transcript + candidate removals to Claude for review.

    The transcript is sent with cache_control=ephemeral so it is billed
    only on the first call; subsequent calls within the session are cheap.

    Returns the parsed decision dict:
      {"decisions": [{"id": int, "action": str, "reason": str}],
       "missed_cuts": [...]}
    """
    removal_lines = _build_removal_lines(enriched)

    def _messages(strict: bool = False) -> list:
        prefix = (
            "Respond with ONLY raw JSON. No markdown. No explanation.\n\n"
            if strict else ""
        )
        return [{
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': f"Video transcript:\n\n{transcript_text}\n",
                    'cache_control': {'type': 'ephemeral'},
                },
                {
                    'type': 'text',
                    'text': (
                        f"{prefix}"
                        "A rule-based system flagged these time ranges for removal:\n\n"
                        f"{removal_lines}\n\n"
                        "For each, decide:\n"
                        "- REMOVE  → definitely cut it (filler, dead air, false start)\n"
                        "- KEEP    → leave it in (dramatic pause, natural rhythm)\n"
                        "- FLAG    → unsure — human should review\n\n"
                        "Rules of thumb:\n"
                        "- Long gaps < 2 s before a new topic: KEEP\n"
                        "- Long gaps > 3 s with no purpose: REMOVE\n"
                        "- um / uh / hmm: always REMOVE\n"
                        '- "okay"/"so" at sentence start with nothing to connect: REMOVE\n'
                        '- "okay"/"so" as a real transition to the next idea: KEEP\n\n'
                        "Respond ONLY with valid JSON — no markdown, no explanation:\n"
                        '{"decisions":[{"id":1,"action":"REMOVE","reason":"..."},...], '
                        '"missed_cuts":[]}'
                    ),
                },
            ],
        }]

    response = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=config.MAX_TOKENS,
        messages=_messages(strict=False),
    )
    raw = response.content[0].text

    try:
        return parse_ai_response(raw)
    except ValueError:
        if not retry:
            raise
        # One retry with a stricter prefix
        response2 = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=config.MAX_TOKENS,
            messages=_messages(strict=True),
        )
        return parse_ai_response(response2.content[0].text)


# ---------------------------------------------------------------------------
# 4. Decision parsing  — O(m)
# ---------------------------------------------------------------------------

def validate_cut(cut: dict) -> bool:
    """Guard: reject Claude suggestions that aren't valid time ranges."""
    return (
        isinstance(cut.get('start'), (int, float)) and
        isinstance(cut.get('end'),   (int, float)) and
        cut['end'] > cut['start']
    )


def parse_decisions(result: dict, enriched: list) -> tuple:
    """
    Split AI decisions into final (REMOVE) and flagged (FLAG) lists.
    KEEP decisions are silently discarded.
    Internal fields (id, before, after) are stripped from output.
    O(m) single pass.
    """
    id_map = {r['id']: r for r in enriched}
    final, flagged = [], []

    for d in result.get('decisions', []):
        rid = d.get('id')
        if rid not in id_map:
            continue
        entry = {
            k: v for k, v in id_map[rid].items()
            if k not in ('id', 'before', 'after')
        }
        entry['ai_reason'] = d.get('reason', '')

        action = d.get('action', '').upper()
        if action == 'REMOVE':
            final.append(entry)
        elif action == 'FLAG':
            flagged.append(entry)
        # KEEP → discarded

    # AI-detected cuts not found by the rule-based system
    for mc in result.get('missed_cuts', []):
        if validate_cut(mc):
            final.append({**mc, 'reason': 'ai_detected'})

    return final, flagged


# ---------------------------------------------------------------------------
# Standalone script entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    from anthropic import Anthropic
    from analyze_transcript import load_words

    TRANSCRIPT = 'Postpartum care what every mom should know.mp4.json'
    REMOVALS   = 'removals.json'

    transcript_path = sys.argv[1] if len(sys.argv) > 1 else TRANSCRIPT
    removals_path   = sys.argv[2] if len(sys.argv) > 2 else REMOVALS

    api_key = config.get_local_api_key()
    if not api_key:
        print("ERROR: No API key found. Set ANTHROPIC_API_KEY or create claudekey.txt")
        sys.exit(1)

    client   = Anthropic(api_key=api_key)
    words    = load_words(transcript_path)
    removals = json.load(open(removals_path))

    transcript_text = build_transcript_text(words)
    enriched        = enrich_removals(removals, words)

    print(f"Sending {len(enriched)} candidate cuts to Claude ({config.CLAUDE_MODEL})...")
    result = call_claude(client, transcript_text, enriched)

    final, flagged = parse_decisions(result, enriched)

    json.dump(final,   open('final_removals.json',  'w'), indent=2)
    json.dump(flagged, open('flagged_removals.json', 'w'), indent=2)

    keep_count = sum(
        1 for d in result.get('decisions', [])
        if d.get('action', '').upper() == 'KEEP'
    )
    print(f"REMOVE : {len(final)}")
    print(f"KEEP   : {keep_count}")
    print(f"FLAG   : {len(flagged)}")
    print("\nWritten: final_removals.json, flagged_removals.json")
