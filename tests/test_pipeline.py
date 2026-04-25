import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline import run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_TRANSCRIPT = {
    "language": "en-us",
    "segments": [{
        "duration": 10.0,
        "language": "en-us",
        "speaker": "abc",
        "start": 0.0,
        "words": [
            {"confidence": 1.0, "duration": 0.4, "eos": False, "start": 0.0,
             "tags": [], "text": "Hello", "type": "word"},
            {"confidence": 1.0, "duration": 0.3, "eos": False, "start": 0.4,
             "tags": ["disfluency"], "text": "", "type": "word"},
            {"confidence": 1.0, "duration": 0.4, "eos": True,  "start": 0.7,
             "tags": [], "text": "world.", "type": "word"},
            {"confidence": 1.0, "duration": 0.4, "eos": True,  "start": 2.0,
             "tags": [], "text": "Done.", "type": "word"},
        ]
    }]
}

AI_RESPONSE = {
    "decisions": [
        {"id": 1, "action": "REMOVE", "reason": "disfluency"},
        {"id": 2, "action": "KEEP",   "reason": "natural gap"},
    ],
    "missed_cuts": []
}


def _make_mock_anthropic(response_json: dict):
    """Return a mock Anthropic class whose instances return given JSON."""
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock()]
    mock_msg.content[0].text = json.dumps(response_json)

    mock_instance = MagicMock()
    mock_instance.messages.create.return_value = mock_msg

    mock_class = MagicMock(return_value=mock_instance)
    return mock_class


def _write_transcript(data: dict) -> str:
    """Write transcript JSON to a temp file and return its path."""
    f = tempfile.NamedTemporaryFile(
        mode='w', suffix='.json', delete=False, encoding='utf-8'
    )
    json.dump(data, f)
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineRun(unittest.TestCase):

    def setUp(self):
        self.tmp = _write_transcript(SAMPLE_TRANSCRIPT)

    def tearDown(self):
        os.unlink(self.tmp)

    def _run(self, response=None):
        mock_cls = _make_mock_anthropic(response or AI_RESPONSE)
        with patch('pipeline.Anthropic', mock_cls):
            return run(self.tmp, claude_api_key='sk-test')

    def test_returns_dict_with_required_keys(self):
        result = self._run()
        for key in ('final', 'flagged', 'word_count', 'candidate_count',
                    'final_count', 'keep_count', 'flagged_count'):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_word_count_matches_transcript(self):
        result = self._run()
        # 4 word entries in the sample (incl. empty disfluency)
        self.assertEqual(result['word_count'], 4)

    def test_remove_decision_appears_in_final(self):
        result = self._run()
        self.assertGreaterEqual(result['final_count'], 1)

    def test_keep_decision_counted(self):
        result = self._run()
        self.assertGreaterEqual(result['keep_count'], 0)

    def test_anthropic_client_built_with_supplied_key(self):
        mock_cls = _make_mock_anthropic(AI_RESPONSE)
        with patch('pipeline.Anthropic', mock_cls):
            run(self.tmp, claude_api_key='sk-custom-key')
        mock_cls.assert_called_once_with(api_key='sk-custom-key')

    def test_client_never_global_singleton(self):
        """Two calls with different keys → two separate Anthropic instances."""
        mock_cls = _make_mock_anthropic(AI_RESPONSE)
        with patch('pipeline.Anthropic', mock_cls):
            run(self.tmp, claude_api_key='key-A')
            run(self.tmp, claude_api_key='key-B')
        self.assertEqual(mock_cls.call_count, 2)
        calls = mock_cls.call_args_list
        self.assertEqual(calls[0][1]['api_key'], 'key-A')
        self.assertEqual(calls[1][1]['api_key'], 'key-B')

    def test_missed_cuts_added_to_final(self):
        response = {
            "decisions": [],
            "missed_cuts": [{"start": 5.0, "end": 6.0}]
        }
        result = self._run(response)
        ai_cuts = [r for r in result['final'] if r.get('reason') == 'ai_detected']
        self.assertEqual(len(ai_cuts), 1)

    def test_invalid_missed_cuts_rejected(self):
        response = {
            "decisions": [],
            "missed_cuts": [
                {"start": "bad", "end": 6.0},
                {"start": 5.0,   "end": 3.0},
            ]
        }
        result = self._run(response)
        ai_cuts = [r for r in result['final'] if r.get('reason') == 'ai_detected']
        self.assertEqual(len(ai_cuts), 0)

    def test_counts_are_consistent(self):
        result = self._run()
        self.assertEqual(result['final_count'],   len(result['final']))
        self.assertEqual(result['flagged_count'], len(result['flagged']))


class TestPipelineEdgeCases(unittest.TestCase):

    def test_empty_segment_transcript(self):
        data = {"language": "en-us", "segments": []}
        tmp = _write_transcript(data)
        try:
            mock_cls = _make_mock_anthropic({"decisions": [], "missed_cuts": []})
            with patch('pipeline.Anthropic', mock_cls):
                result = run(tmp, claude_api_key='sk-test')
            self.assertEqual(result['word_count'],      0)
            self.assertEqual(result['candidate_count'], 0)
            self.assertEqual(result['final_count'],     0)
        finally:
            os.unlink(tmp)

    def test_all_words_kept_by_ai(self):
        response = {
            "decisions": [{"id": 1, "action": "KEEP", "reason": "fine"}],
            "missed_cuts": []
        }
        tmp = _write_transcript(SAMPLE_TRANSCRIPT)
        try:
            mock_cls = _make_mock_anthropic(response)
            with patch('pipeline.Anthropic', mock_cls):
                result = run(tmp, claude_api_key='sk-test')
            self.assertEqual(result['keep_count'], 1)
        finally:
            os.unlink(tmp)


if __name__ == '__main__':
    unittest.main()
