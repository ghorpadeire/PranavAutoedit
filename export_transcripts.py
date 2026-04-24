import json
import sys

TRANSCRIPT_FILE = 'Postpartum care what every mom should know.mp4.json'

# Must match analyze_transcript.py
STRONG_FILLERS          = {'um', 'uh', 'ah', 'er', 'hmm', 'hm', 'mm', 'mhm'}
SENTENCE_START_FILLERS  = {'okay', 'so', 'right', 'well', 'alright', 'anyway'}

def load_words(path):
    with open(path) as f:
        data = json.load(f)
    words = []
    for segment in data['segments']:
        for word in segment['words']:
            words.append({**word, 'end': round(word['start'] + word['duration'], 4)})
    return words

def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02}:{s:02}" if h else f"{m}:{s:02}"

def is_filler(word, prev_word):
    """Return True if this word should be removed from clean transcript."""
    if 'disfluency' in word['tags']:
        return True
    token = word['text'].lower().strip('.,!?"\' ')
    if token in STRONG_FILLERS:
        return True
    at_sentence_start = prev_word is None or prev_word.get('eos', False)
    if token in SENTENCE_START_FILLERS and at_sentence_start:
        return True
    return False

def build_lines(words, clean=False):
    """
    Returns list of (timestamp_str, line_text) per sentence.
    clean=True  → skip filler words (AFTER version)
    clean=False → include everything (BEFORE version)
    """
    lines = []
    current = []
    sentence_start = None
    prev_word = None

    for word in words:
        if not word['text']:               # empty disfluency entries
            prev_word = word
            continue

        skip = clean and is_filler(word, prev_word)

        if not skip:
            if sentence_start is None:
                sentence_start = word['start']
            current.append(word['text'])

        if word.get('eos'):
            if current:
                lines.append((format_time(sentence_start), ' '.join(current)))
            current = []
            sentence_start = None

        prev_word = word

    if current:
        lines.append((format_time(sentence_start), ' '.join(current)))

    return lines

def write_file(path, lines, title):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*60}\n  {title}\n{'='*60}\n\n")
        for ts, text in lines:
            f.write(f"[{ts}]  {text}\n")
    print(f"Written: {path}  ({len(lines)} sentences)")

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else TRANSCRIPT_FILE
    stem = path.rsplit('.json', 1)[0]

    words = load_words(path)

    before = build_lines(words, clean=False)
    after  = build_lines(words, clean=True)

    write_file(f"{stem}_BEFORE.txt", before, f"BEFORE  —  {path}")
    write_file(f"{stem}_AFTER.txt",  after,  f"AFTER   —  {path}")

    print(f"Sentences: {len(before)} before, {len(after)} after")

if __name__ == '__main__':
    main()
