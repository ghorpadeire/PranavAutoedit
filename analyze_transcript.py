import json
import sys
import time
from collections import deque

INPUT_FILE = 'Postpartum care what every mom should know.mp4.json'
OUTPUT_FILE = 'removals.json'

GAP_THRESHOLD = 0.8      # seconds of silence = long gap
OVERLAP_THRESHOLD = 0.6  # Jaccard similarity for false-start detection

# Tier 1: Always a filler — pure hesitation sounds, no legitimate use
STRONG_FILLERS = {'um', 'uh', 'ah', 'er', 'hmm', 'hm', 'mm', 'mhm'}

# Tier 2: Filler only at sentence start (after eos=True on previous word)
# Mid-sentence these words are often legitimate ("feel okay", "turn right")
SENTENCE_START_FILLERS = {'okay', 'so', 'right', 'well', 'alright', 'anyway'}

def load_words(path):
    with open(path) as f:
        data = json.load(f)
    words = []
    for segment in data['segments']:
        for word in segment['words']:
            words.append({**word, 'end': round(word['start'] + word['duration'], 4)})
    return words

def detect_removals(words):
    removals = []
    seen_sentences = {}
    sentences = []
    current_sentence = deque()
    prev_word = None

    for word in words:
        token = word['text'].lower().strip('.,!?"\' ')

        # 2a: Disfluency tag — O(1) check
        if 'disfluency' in word['tags']:
            removals.append({'start': word['start'], 'end': word['end'], 'reason': 'disfluency'})

        # 2a+: Text-pattern fillers (catches untagged um/uh and sentence-start fillers)
        elif token:
            at_sentence_start = prev_word is None or prev_word.get('eos', False)
            if token in STRONG_FILLERS or (token in SENTENCE_START_FILLERS and at_sentence_start):
                removals.append({'start': word['start'], 'end': word['end'],
                                  'reason': 'text_filler', 'word': token})

        # 2b: Long gap — O(1) pointer compare
        if prev_word is not None:
            gap = word['start'] - prev_word['end']
            if gap > GAP_THRESHOLD:
                removals.append({
                    'start': prev_word['end'],
                    'end': word['start'],
                    'reason': 'long_gap',
                    'gap_seconds': round(gap, 2)
                })

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
                    prev_sent = sentences[seen_sentences[fp]]
                    prev_tokens = frozenset(w['text'].lower().strip('.,!?') for w in prev_sent)
                    jaccard = len(tokens & prev_tokens) / len(tokens | prev_tokens)
                    if jaccard > OVERLAP_THRESHOLD:
                        removals.append({
                            'start': prev_sent[0]['start'],
                            'end': prev_sent[-1]['end'],
                            'reason': 'false_start',
                            'text': ' '.join(w['text'] for w in prev_sent)
                        })
                seen_sentences[fp] = len(sentences) - 1

        if word['text']:
            prev_word = word

    return removals

def merge_overlaps(removals):
    removals.sort(key=lambda r: r['start'])
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
