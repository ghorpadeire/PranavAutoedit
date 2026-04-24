# PranavAutoedit

A Python pipeline that reads Adobe Premiere Pro transcript JSON files and automatically detects filler words, long silences, and false starts — producing a clean list of time ranges to cut from your video.

## Pipeline

```
Premiere JSON transcript
        ↓
  analyze_transcript.py   →  removals.json  (time ranges to cut)
        ↓
  export_transcripts.py   →  _BEFORE.txt / _AFTER.txt  (review the diff)
        ↓
  [Step 3 — AI review]    coming soon
        ↓
  [Step 4 — Video cut]    coming soon
```

## What it detects

| Type | Method |
|---|---|
| **Disfluency tags** | Premiere's own `"disfluency"` labels (um, uh sounds) |
| **Text fillers** | Hardcoded tiers — always remove `um/uh/hmm`, sentence-start only for `okay/so/well` |
| **Long silences** | Gap between consecutive words > 0.8 s |
| **False starts** | Repeated sentences detected via Jaccard similarity + hash map |

All detection runs in a single O(n) pass — under 12 ms for a 1-hour transcript.

## Usage

```bash
# 1. Detect removals (reads transcript JSON, writes removals.json)
python analyze_transcript.py "your-video.mp4.json"

# 2. Export before/after transcripts for review
python export_transcripts.py "your-video.mp4.json"
```

Your Premiere transcript JSON is the file exported from the **Captions** panel in Premiere Pro (right-click sequence → Export → JSON).

## Output

- `removals.json` — sorted list of `{start, end, reason}` time ranges
- `*_BEFORE.txt` — full transcript with timestamps
- `*_AFTER.txt`  — clean transcript with fillers removed (for review before cutting)

## Tuning

Edit the constants at the top of `analyze_transcript.py`:

```python
GAP_THRESHOLD = 0.8          # seconds — raise to keep more pauses
STRONG_FILLERS = {'um', 'uh', 'ah', 'er', 'hmm', ...}
SENTENCE_START_FILLERS = {'okay', 'so', 'right', 'well', ...}
```
