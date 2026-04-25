import json
import sys
import time
from collections import deque

import config

INPUT_FILE = 'Postpartum care what every mom should know.mp4.json'
OUTPUT_FILE = 'removals.json'

# All thresholds and word lists now live in config.py
GAP_THRESHOLD          = config.GAP_THRESHOLD
OVERLAP_THRESHOLD      = config.OVERLAP_THRESHOLD
BIGRAM_THRESHOLD       = config.BIGRAM_THRESHOLD
STRONG_FILLERS         = config.STRONG_FILLERS
SENTENCE_START_FILLERS = config.SENTENCE_START_FILLERS

def ngrams(word_list: list, n: int = 2) -> set:
    """
    Return ordered n-gram tuples from a word list.
    Used as a secondary false-start signal: catches stammers like
    "I — I think" where Jaccard (unordered) would miss the repetition.
    O(k) where k = sentence length.
    """
    tokens = [w['text'].lower().strip('.,!?') for w in word_list if w['text']]
    return set(zip(tokens, tokens[1:])) if len(tokens) >= n else set()


def load_words(path):
    with open(path) as f:
        data = json.load(f)
    words = []
    for segment in data['segments']:
        for word in segment['words']:
            words.append({**word, 'end': round(word['start'] + word['duration'], 4)})
    return words

def detect_removals(words, gap_threshold=None):
    """
    Rule-based filler detection.  Returns candidates sorted by start time.

    Three candidate types — disfluency, text_filler, long_gap — are appended
    to ``removals`` in transcript (time) order: their start timestamps are
    monotonically non-decreasing by construction.

    False-start candidates are different: they point to the *previous*
    sentence's start time, which may be earlier than removals already in the
    list.  Collecting them in a separate ``false_starts`` buffer and
    merge-sorting at the end preserves the O(n) guarantee:

        O(f log f)  — sort the tiny false_starts list  (f << n, typically 0–3)
        O(n + f)    — two-pointer merge into removals

    Total: O(n).  compare to the old approach: O(m log m) sort in
    merge_overlaps, where m could approach n in the worst case.

    Args:
        words:         Flat list of word dicts from load_words().
        gap_threshold: Seconds of silence to flag as a long gap.
                       Overrides config.GAP_THRESHOLD when provided.
    """
    threshold     = gap_threshold if gap_threshold is not None else GAP_THRESHOLD
    removals      = []   # disfluency / text_filler / long_gap — appended in time order
    false_starts  = []   # false_start entries — may point to earlier timestamps
    seen_sentences = {}
    sentences      = []
    current_sentence = deque()
    prev_word      = None

    for word in words:
        token = word['text'].lower().strip('.,!?"\' ')

        # 2b: Long gap — checked FIRST so its start (prev_word.end) is appended
        #     before the current word's filler start (word.start ≥ prev_word.end).
        #     Reversing this order would break the monotone-sorted invariant when a
        #     filler word immediately follows a silence.
        if prev_word is not None:
            gap = word['start'] - prev_word['end']
            if gap > threshold:
                removals.append({
                    'start': prev_word['end'],
                    'end':   word['start'],
                    'reason': 'long_gap',
                    'gap_seconds': round(gap, 2),
                })

        # 2a: Disfluency tag — O(1) check
        if 'disfluency' in word['tags']:
            removals.append({'start': word['start'], 'end': word['end'], 'reason': 'disfluency'})

        # 2a+: Text-pattern fillers (catches untagged um/uh and sentence-start fillers)
        elif token:
            at_sentence_start = prev_word is None or prev_word.get('eos', False)
            if token in STRONG_FILLERS or (token in SENTENCE_START_FILLERS and at_sentence_start):
                removals.append({'start': word['start'], 'end': word['end'],
                                  'reason': 'text_filler', 'word': token})

        # 2c: False start — rolling hash fingerprint + hash map
        if word['text']:
            current_sentence.append(word)

        if word.get('eos') and current_sentence:
            sent_words = list(current_sentence)
            sentences.append(sent_words)
            current_sentence.clear()

            tokens = frozenset(w['text'].lower().strip('.,!?') for w in sent_words)
            if len(tokens) >= 3:
                fp = hash(tokens)
                if fp in seen_sentences:
                    prev_sent   = sentences[seen_sentences[fp]]
                    prev_tokens = frozenset(
                        w['text'].lower().strip('.,!?') for w in prev_sent
                    )
                    jaccard = len(tokens & prev_tokens) / len(tokens | prev_tokens)

                    # Bigram secondary signal — ordered pairs catch stammers
                    # like "I — I think" where unordered Jaccard would miss them
                    bg_curr    = ngrams(sent_words)
                    bg_prev    = ngrams(prev_sent)
                    bigram_sim = (
                        len(bg_curr & bg_prev) / max(len(bg_curr | bg_prev), 1)
                    )

                    if jaccard > OVERLAP_THRESHOLD or bigram_sim > BIGRAM_THRESHOLD:
                        # prev_sent started in the past → buffer separately
                        false_starts.append({
                            'start':  prev_sent[0]['start'],
                            'end':    prev_sent[-1]['end'],
                            'reason': 'false_start',
                            'text':   ' '.join(w['text'] for w in prev_sent),
                        })
                seen_sentences[fp] = len(sentences) - 1

        if word['text']:
            prev_word = word

    # ── Merge false_starts into removals while preserving sort order ──────
    # Common case: no false starts at all — return immediately, O(0) extra work.
    if not false_starts:
        return removals

    # Sort the tiny false_starts buffer: O(f log f), f typically 0–3.
    false_starts.sort(key=lambda r: r['start'])

    # Two-pointer linear merge: O(n + f).
    result = []
    i = j  = 0
    while i < len(removals) and j < len(false_starts):
        if removals[i]['start'] <= false_starts[j]['start']:
            result.append(removals[i]);      i += 1
        else:
            result.append(false_starts[j]);  j += 1
    result.extend(removals[i:])
    result.extend(false_starts[j:])
    return result


def merge_overlaps(removals):
    """
    Collapse abutting or overlapping removal ranges into minimal segments.

    Precondition: *removals* is sorted by start time — guaranteed by
    detect_removals() via the two-pointer merge above.  No sort needed here.

    Time:  O(m)  where m = len(removals).
    Space: O(m).
    """
    merged = []
    for r in removals:
        if merged and r['start'] <= merged[-1]['end']:
            merged[-1]['end'] = max(merged[-1]['end'], r['end'])
        else:
            merged.append(dict(r))
    return merged

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE
    t0 = time.perf_counter()

    words = load_words(path)
    removals = detect_removals(words)
    merged = merge_overlaps(removals)

    elapsed = time.perf_counter() - t0

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(merged, f, indent=2)

    counts = {}
    for r in merged:
        counts[r['reason']] = counts.get(r['reason'], 0) + 1

    print(f"Words processed : {len(words)}")
    print(f"Time elapsed    : {elapsed*1000:.1f} ms")
    print(f"Total removals  : {len(merged)}")
    for reason, n in sorted(counts.items()):
        print(f"  {reason:<20}: {n}")
    print(f"\nOutput -> {OUTPUT_FILE}\n")
    for r in merged:
        label = r['reason']
        extra = f"  ({r.get('gap_seconds')}s gap)" if label == 'long_gap' else \
                f"  \"{r.get('text') or r.get('word', '')}\"" if label in ('false_start', 'text_filler') else ''
        print(f"  [{r['start']:7.2f}s -> {r['end']:7.2f}s] {label}{extra}")

if __name__ == '__main__':
    main()
