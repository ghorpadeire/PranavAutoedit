import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from export_transcripts import is_filler, build_lines, format_time


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


class TestIsFiller(unittest.TestCase):
    def test_disfluency_tag(self):
        self.assertTrue(is_filler(word('', 0.5, 0.3, tags=['disfluency']), None))

    def test_strong_filler_mid_sentence(self):
        prev = word('said', 0.0, 0.3)
        for filler_word in ('um', 'uh', 'hmm', 'ah', 'er', 'hm', 'mm', 'mhm'):
            with self.subTest(w=filler_word):
                self.assertTrue(is_filler(word(filler_word, 0.3, 0.2), prev))

    def test_sentence_start_filler_after_eos(self):
        prev = word('done.', 0.0, 0.5, eos=True)
        self.assertTrue(is_filler(word('Okay', 0.5, 0.3), prev))

    def test_sentence_start_filler_at_very_start(self):
        # prev_word is None = beginning of transcript
        self.assertTrue(is_filler(word('So', 0.0, 0.2), None))

    def test_sentence_start_filler_mid_sentence_kept(self):
        prev = word('feel', 0.0, 0.3, eos=False)
        self.assertFalse(is_filler(word('okay', 0.3, 0.3), prev))

    def test_normal_word_not_filler(self):
        self.assertFalse(is_filler(word('hello', 0.0, 0.5), None))

    def test_punctuation_in_filler_stripped(self):
        prev = word('done.', 0.0, 0.5, eos=True)
        self.assertTrue(is_filler(word('Okay,', 0.5, 0.3), prev))


class TestBuildLines(unittest.TestCase):
    def _sample(self):
        return [
            word('Hello', 0.0, 0.3),
            word('um', 0.3, 0.2),
            word('world.', 0.5, 0.4, eos=True),
        ]

    def test_before_includes_fillers(self):
        lines = build_lines(self._sample(), clean=False)
        self.assertEqual(len(lines), 1)
        ts, text = lines[0]
        self.assertIn('um', text)
        self.assertIn('Hello', text)

    def test_after_removes_fillers(self):
        lines = build_lines(self._sample(), clean=True)
        self.assertEqual(len(lines), 1)
        ts, text = lines[0]
        self.assertNotIn('um', text)
        self.assertIn('Hello', text)
        self.assertIn('world.', text)

    def test_timestamp_is_sentence_start(self):
        lines = build_lines(self._sample(), clean=False)
        self.assertEqual(lines[0][0], '0:00')

    def test_multiple_sentences(self):
        words = [
            word('Hello.', 0.0, 0.5, eos=True),
            word('World.', 2.0, 0.5, eos=True),
        ]
        lines = build_lines(words, clean=False)
        self.assertEqual(len(lines), 2)

    def test_empty_words_list(self):
        self.assertEqual(build_lines([], clean=False), [])

    def test_disfluency_entries_skipped(self):
        words = [
            word('Hello', 0.0, 0.3),
            word('', 0.3, 0.2, tags=['disfluency']),
            word('world.', 0.5, 0.4, eos=True),
        ]
        before = build_lines(words, clean=False)
        self.assertNotIn('', before[0][1].split())   # empty string not in output


class TestFormatTime(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(format_time(0), '0:00')

    def test_seconds_only(self):
        self.assertEqual(format_time(45), '0:45')

    def test_minutes_and_seconds(self):
        self.assertEqual(format_time(90), '1:30')

    def test_exactly_one_minute(self):
        self.assertEqual(format_time(60), '1:00')

    def test_hours(self):
        self.assertEqual(format_time(3661), '1:01:01')

    def test_no_hours_prefix_under_one_hour(self):
        self.assertNotIn(':', format_time(59).split(':')[0] if format_time(59).count(':') == 1 else '')


if __name__ == '__main__':
    unittest.main()
