"""
pipeline.py — Full pipeline orchestrator.

Chains Steps 1 → 2 → 3 in a single call.
Returns a plain dict (no file I/O, no side effects).

Usage:
    from anthropic import Anthropic
    from pipeline import run

    result = run("video.mp4.json", claude_api_key="sk-ant-...")
    # result["final"]   → confirmed cuts for the video editor
    # result["flagged"] → cuts needing human review

The Anthropic client is constructed INSIDE run() using the
caller-supplied key — never a module-level singleton, so each
request is isolated (important for the multi-tenant API layer).
"""

from anthropic import Anthropic

from analyze_transcript import load_words, detect_removals, merge_overlaps
from ai_review import (
    build_transcript_text,
    enrich_removals,
    call_claude,
    parse_decisions,
)


def run(transcript_path: str, claude_api_key: str) -> dict:
    """
    Execute the full filler-removal pipeline.

    Args:
        transcript_path: Path to the Premiere Pro transcript JSON file.
        claude_api_key:  Caller's Anthropic API key (never stored).

    Returns:
        {
          "final":          list of confirmed cut ranges (REMOVE decisions),
          "flagged":        list of uncertain cuts (FLAG decisions),
          "word_count":     total words in transcript,
          "candidate_count": candidate cuts from rule-based system,
          "final_count":    number of confirmed cuts,
          "keep_count":     number of cuts Claude chose to keep,
          "flagged_count":  number of flagged cuts,
        }
    """
    # Client created per-request — never a global singleton
    client = Anthropic(api_key=claude_api_key)

    # Step 1 — Read transcript
    words = load_words(transcript_path)

    # Step 2 — Rule-based detection
    candidates = detect_removals(words)
    merged     = merge_overlaps(candidates)

    # Step 3 — AI review
    transcript_text = build_transcript_text(words)
    enriched        = enrich_removals(merged, words)
    ai_result       = call_claude(client, transcript_text, enriched)
    final, flagged  = parse_decisions(ai_result, enriched)

    keep_count = sum(
        1 for d in ai_result.get('decisions', [])
        if d.get('action', '').upper() == 'KEEP'
    )

    return {
        'final':           final,
        'flagged':         flagged,
        'word_count':      len(words),
        'candidate_count': len(merged),
        'final_count':     len(final),
        'keep_count':      keep_count,
        'flagged_count':   len(flagged),
    }


if __name__ == '__main__':
    import sys
    import json
    import config

    transcript = sys.argv[1] if len(sys.argv) > 1 else (
        'Postpartum care what every mom should know.mp4.json'
    )
    api_key = config.get_local_api_key()
    if not api_key:
        print("ERROR: No API key. Set ANTHROPIC_API_KEY or create claudekey.txt")
        sys.exit(1)

    print(f"Running full pipeline on: {transcript}")
    result = run(transcript, api_key)

    json.dump(result['final'],   open('final_removals.json',  'w'), indent=2)
    json.dump(result['flagged'], open('flagged_removals.json', 'w'), indent=2)

    print(f"Words processed : {result['word_count']}")
    print(f"Candidates      : {result['candidate_count']}")
    print(f"REMOVE          : {result['final_count']}")
    print(f"KEEP            : {result['keep_count']}")
    print(f"FLAG            : {result['flagged_count']}")
    print("\nWritten: final_removals.json, flagged_removals.json")
