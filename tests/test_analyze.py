import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from analyze_transcript import detect_removals, merge_overlaps


def word(text, start, duration, tags=None, eos=False):
    return {
        'text': text,
        'start': start,
        'duration': duration,
        'end': round(start + duration, 4),
        'tags': tags or [],
        'eos': eos,
        'confidence': 1.0,
        'type': 'word',
    }


class TestDisfluencyDetection(unittest.TestCase):
    def test_disfluency_tag_flagged(self):
        words = [
            word('Hello', 0.0, 0.5),
            word('', 0.5, 0.3, tags=['disfluency']),
            word('world.', 0.8, 0.4, eos=True),
        ]
        removals = [r for r in detect_removals(words) if r['reason'] == 'disfluency']
        self.assertEqual(len(removals), 1)
        self.assertAlmostEqual(removals[0]['start'], 0.5)
        self.assertAlmostEqual(removals[0]['end'], 0.8)

    def test_no_disfluency_when_no_tag(self):
        words = [word('Hello', 0.0, 0.5), word('world.', 0.5, 0.4, eos=True)]
        removals = [r for r in detect_removals(words) if r['reason'] == 'disfluency']
        self.assertEqual(len(removals), 0)


class TestGapDetection(unittest.TestCase):
    def test_gap_above_threshold_flagged(self):
        words = [
            word('Hello.', 0.0, 0.5, eos=True),
            word('World.', 2.0, 0.4, eos=True),   # 1.5 s gap
        ]
        gaps = [r for r in detect_removals(words) if r['reason'] == 'long_gap']
        self.assertEqual(len(gaps), 1)
        self.assertAlmostEqual(gaps[0]['start'], 0.5)
        self.assertAlmostEqual(gaps[0]['end'], 2.0)
        self.assertAlmostEqual(gaps[0]['gap_seconds'], 1.5)

    def test_gap_below_threshold_ignored(self):
        words = [
            word('Hello.', 0.0, 0.5, eos=True),
            word('World.', 1.0, 0.4, eos=True),   # 0.5 s gap — under 0.8 s
        ]
        gaps = [r for r in detect_removals(words) if r['reason'] == 'long_gap']
        self.assertEqual(len(gaps), 0)

    def test_gap_exactly_at_threshold_ignored(self):
        words = [
            word('Hello.', 0.0, 0.5, eos=True),
            word('World.', 1.3, 0.4, eos=True),   # exactly 0.8 s — not > threshold
        ]
        gaps = [r for r in detect_removals(words) if r['reason'] == 'long_gap']
        self.assertEqual(len(gaps), 0)


class TestTextFillerDetection(unittest.TestCase):
    def test_strong_filler_always_removed(self):
        for filler in ('um', 'uh', 'hmm', 'ah', 'er'):
            with self.subTest(filler=filler):
                words = [
                    word('I', 0.0, 0.2),           # non-filler first word
                    word(filler, 0.2, 0.2),
                    word('yeah.', 0.4, 0.3, eos=True),
                ]
                fillers = [r for r in detect_removals(words) if r['reason'] == 'text_filler']
                self.assertEqual(len(fillers), 1, f"'{filler}' mid-sentence should be removed")

    def test_sentence_start_filler_removed(self):
        words = [
            word('Done.', 0.0, 0.4, eos=True),
            word('So', 0.4, 0.2),          # sentence-start filler
            word('anyway.', 0.6, 0.4, eos=True),
        ]
        fillers = [r for r in detect_removals(words) if r['reason'] == 'text_filler']
        self.assertEqual(len(fillers), 1)
        self.assertEqual(fillers[0]['word'], 'so')

    def test_sentence_start_filler_mid_sentence_kept(self):
        words = [
            word('Turn', 0.0, 0.3),
            word('right', 0.3, 0.3),       # mid-sentence — must NOT be removed
            word('now.', 0.6, 0.3, eos=True),
        ]
        fillers = [r for r in detect_removals(words) if r['reason'] == 'text_filler']
        self.assertEqual(len(fillers), 0)

    def test_filler_punctuation_stripped(self):
        words = [
            word('Okay,', 0.0, 0.3),       # 'okay,' at very start (prev_word is None)
            word('let us go.', 0.3, 0.5, eos=True),
        ]
        fillers = [r for r in detect_removals(words) if r['reason'] == 'text_filler']
        self.assertEqual(len(fillers), 1)


class TestFalseStartDetection(unittest.TestCase):
    def test_repeated_sentence_flagged(self):
        words = [
            word('I', 0.0, 0.2),
            word('was', 0.2, 0.2),
            word('going', 0.4, 0.3),
            word('there.', 0.7, 0.3, eos=True),
            word('I', 2.0, 0.2),
            word('was', 2.2, 0.2),
            word('going', 2.4, 0.3),
            word('there.', 2.7, 0.3, eos=True),
        ]
        false_starts = [r for r in detect_removals(words) if r['reason'] == 'false_start']
        self.assertEqual(len(false_starts), 1)
        self.assertAlmostEqual(false_starts[0]['start'], 0.0)

    def test_unique_sentences_not_flagged(self):
        words = [
            word('I', 0.0, 0.2),
            word('like', 0.2, 0.2),
            word('coffee.', 0.4, 0.4, eos=True),
            word('She', 2.0, 0.2),
            word('likes', 2.2, 0.2),
            word('tea.', 2.4, 0.4, eos=True),
        ]
        false_starts = [r for r in detect_removals(words) if r['reason'] == 'false_start']
        self.assertEqual(len(false_starts), 0)

    def test_short_sentence_not_flagged(self):
        # < 3 unique tokens — should be ignored
        words = [
            word('Yes.', 0.0, 0.3, eos=True),
            word('Yes.', 1.0, 0.3, eos=True),
        ]
        false_starts = [r for r in detect_removals(words) if r['reason'] == 'false_start']
        self.assertEqual(len(false_starts), 0)


class TestMergeOverlaps(unittest.TestCase):
    def test_overlapping_ranges_merged(self):
        removals = [
            {'start': 1.0, 'end': 2.0, 'reason': 'disfluency'},
            {'start': 1.5, 'end': 3.0, 'reason': 'long_gap'},
        ]
        merged = merge_overlaps(removals)
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0]['start'], 1.0)
        self.assertAlmostEqual(merged[0]['end'], 3.0)

    def test_non_overlapping_ranges_unchanged(self):
        removals = [
            {'start': 1.0, 'end': 2.0, 'reason': 'disfluency'},
            {'start': 3.0, 'end': 4.0, 'reason': 'long_gap'},
        ]
        self.assertEqual(len(merge_overlaps(removals)), 2)

    def test_adjacent_ranges_merged(self):
        removals = [
            {'start': 1.0, 'end': 2.0, 'reason': 'disfluency'},
            {'start': 2.0, 'end': 3.0, 'reason': 'long_gap'},
        ]
        self.assertEqual(len(merge_overlaps(removals)), 1)

    def test_unsorted_input_sorted_in_output(self):
        removals = [
            {'start': 5.0, 'end': 6.0, 'reason': 'disfluency'},
            {'start': 1.0, 'end': 2.0, 'reason': 'long_gap'},
        ]
        merged = merge_overlaps(removals)
        self.assertEqual(merged[0]['start'], 1.0)
        self.assertEqual(merged[1]['start'], 5.0)

    def test_empty_input(self):
        self.assertEqual(merge_overlaps([]), [])

    def test_single_item(self):
        removals = [{'start': 1.0, 'end': 2.0, 'reason': 'disfluency'}]
        self.assertEqual(len(merge_overlaps(removals)), 1)


if __name__ == '__main__':
    unittest.main()
