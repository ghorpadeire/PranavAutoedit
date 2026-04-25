import json
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ai_review import (
    build_transcript_text,
    enrich_removals,
    parse_ai_response,
    call_claude,
    parse_decisions,
    validate_cut,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def word(text, start, duration, eos=False):
    return {
        'text': text, 'start': start,
        'duration': duration, 'end': round(start + duration, 4),
        'tags': [], 'eos': eos,
    }


def removal(start, end, reason='disfluency', **extra):
    return {'start': start, 'end': end, 'reason': reason, **extra}


def make_mock_client(response_json: dict):
    """Mock Anthropic client that returns given JSON on every call."""
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock()]
    mock_msg.content[0].text = json.dumps(response_json)
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_msg
    return mock_client


MOCK_RESPONSE = {
    'decisions': [
        {'id': 1, 'action': 'REMOVE', 'reason': 'clear filler'},
        {'id': 2, 'action': 'KEEP',   'reason': 'dramatic pause'},
        {'id': 3, 'action': 'FLAG',   'reason': 'uncertain context'},
    ],
    'missed_cuts': [],
}


# ---------------------------------------------------------------------------
# build_transcript_text
# ---------------------------------------------------------------------------

class TestBuildTranscriptText(unittest.TestCase):

    def test_basic_format(self):
        words = [word('Hello', 0.0, 0.3), word('world.', 0.3, 0.4, eos=True)]
        text = build_transcript_text(words)
        self.assertIn('[0:00]', text)
        self.assertIn('Hello world.', text)

    def test_skips_empty_disfluency_entries(self):
        words = [word('Hello', 0.0, 0.3), word('', 0.3, 0.2), word('world.', 0.5, 0.4, eos=True)]
        text = build_transcript_text(words)
        # Should not have double space from empty word
        self.assertNotIn('Hello  world', text)

    def test_multiple_sentences_produce_multiple_lines(self):
        words = [
            word('First.', 0.0, 0.5, eos=True),
            word('Second.', 60.0, 0.5, eos=True),
        ]
        lines = build_transcript_text(words).split('\n')
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[1].startswith('[1:00]'))

    def test_timestamp_over_one_hour(self):
        words = [word('Late.', 3661.0, 0.5, eos=True)]
        text = build_transcript_text(words)
        self.assertIn('[1:01:01]', text)

    def test_trailing_words_without_eos_flushed(self):
        words = [word('No', 0.0, 0.3), word('eos', 0.3, 0.3)]  # no eos=True
        text = build_transcript_text(words)
        self.assertIn('No eos', text)

    def test_empty_word_list_returns_empty_string(self):
        self.assertEqual(build_transcript_text([]), '')

    def test_all_empty_words_returns_empty_string(self):
        words = [word('', 0.0, 0.3), word('', 0.3, 0.3)]
        self.assertEqual(build_transcript_text(words), '')


# ---------------------------------------------------------------------------
# enrich_removals
# ---------------------------------------------------------------------------

class TestEnrichRemovals(unittest.TestCase):

    def _words(self):
        return [
            word('Hello', 0.0, 0.5),
            word('world.', 0.5, 0.5, eos=True),
            word('Next', 2.0, 0.3),
            word('sentence.', 2.3, 0.5, eos=True),
        ]

    def test_before_and_after_context(self):
        # Gap starts at world.'s END (1.0), not its start — matches real pipeline output
        removals = [removal(1.0, 2.0, 'long_gap', gap_seconds=1.0)]
        enriched = enrich_removals(removals, self._words())
        self.assertEqual(enriched[0]['before'], 'world.')
        self.assertEqual(enriched[0]['after'],  'Next')

    def test_removal_at_very_start(self):
        removals = [removal(0.0, 0.3, 'disfluency')]
        enriched = enrich_removals(removals, self._words())
        self.assertEqual(enriched[0]['before'], '[START]')

    def test_removal_at_very_end(self):
        removals = [removal(2.8, 4.0, 'long_gap', gap_seconds=1.2)]
        enriched = enrich_removals(removals, self._words())
        self.assertEqual(enriched[0]['after'], '[END]')

    def test_ids_are_sequential_from_one(self):
        removals = [
            removal(0.5, 1.0, 'disfluency'),
            removal(1.5, 2.0, 'long_gap', gap_seconds=0.5),
        ]
        enriched = enrich_removals(removals, self._words())
        self.assertEqual([r['id'] for r in enriched], [1, 2])

    def test_original_fields_preserved(self):
        removals = [removal(0.5, 2.0, 'long_gap', gap_seconds=1.5)]
        enriched = enrich_removals(removals, self._words())
        self.assertEqual(enriched[0]['gap_seconds'], 1.5)
        self.assertEqual(enriched[0]['reason'], 'long_gap')

    def test_empty_removals_returns_empty(self):
        self.assertEqual(enrich_removals([], self._words()), [])

    def test_empty_words_returns_start_end_labels(self):
        removals = [removal(0.0, 1.0)]
        enriched = enrich_removals(removals, [])
        self.assertEqual(enriched[0]['before'], '[START]')
        self.assertEqual(enriched[0]['after'],  '[END]')

    def test_two_pointer_never_skips_context(self):
        """Stress: many removals in sequence — ptr must not overshoot."""
        words = [word(f'w{i}', i * 1.0, 0.5) for i in range(10)]
        removals = [removal(i * 1.0 + 0.5, (i + 1) * 1.0, 'long_gap', gap_seconds=0.5)
                    for i in range(9)]
        enriched = enrich_removals(removals, words)
        self.assertEqual(len(enriched), 9)
        for e in enriched:
            self.assertIn('before', e)
            self.assertIn('after', e)


# ---------------------------------------------------------------------------
# parse_ai_response
# ---------------------------------------------------------------------------

class TestParseAiResponse(unittest.TestCase):

    def _data(self):
        return {'decisions': [], 'missed_cuts': []}

    def test_clean_json(self):
        self.assertEqual(parse_ai_response(json.dumps(self._data())), self._data())

    def test_strips_json_fence(self):
        raw = f"```json\n{json.dumps(self._data())}\n```"
        self.assertEqual(parse_ai_response(raw), self._data())

    def test_strips_bare_fence(self):
        raw = f"```\n{json.dumps(self._data())}\n```"
        self.assertEqual(parse_ai_response(raw), self._data())

    def test_strips_leading_trailing_whitespace(self):
        raw = f"\n\n  {json.dumps(self._data())}  \n\n"
        self.assertEqual(parse_ai_response(raw), self._data())

    def test_invalid_json_raises_value_error(self):
        with self.assertRaises(ValueError):
            parse_ai_response("not json at all")

    def test_empty_string_raises_value_error(self):
        with self.assertRaises(ValueError):
            parse_ai_response("")


# ---------------------------------------------------------------------------
# call_claude  (mocked)
# ---------------------------------------------------------------------------

class TestCallClaude(unittest.TestCase):

    def _enriched(self):
        return [
            {'id': 1, 'start': 1.0, 'end': 1.5, 'reason': 'disfluency',
             'before': 'hello', 'after': 'world'},
        ]

    def test_returns_parsed_dict(self):
        client   = make_mock_client(MOCK_RESPONSE)
        result   = call_claude(client, "transcript text", self._enriched())
        self.assertIn('decisions', result)
        self.assertIn('missed_cuts', result)

    def test_client_messages_create_called(self):
        client = make_mock_client(MOCK_RESPONSE)
        call_claude(client, "transcript text", self._enriched())
        self.assertTrue(client.messages.create.called)

    def test_retry_on_bad_json(self):
        """If first response is invalid JSON, retry once with strict prompt."""
        bad_msg  = MagicMock()
        bad_msg.content = [MagicMock()]
        bad_msg.content[0].text = "not json"

        good_msg = MagicMock()
        good_msg.content = [MagicMock()]
        good_msg.content[0].text = json.dumps(MOCK_RESPONSE)

        client = MagicMock()
        client.messages.create.side_effect = [bad_msg, good_msg]

        result = call_claude(client, "transcript", self._enriched(), retry=True)
        self.assertIn('decisions', result)
        self.assertEqual(client.messages.create.call_count, 2)

    def test_no_retry_raises_on_bad_json(self):
        bad_msg = MagicMock()
        bad_msg.content = [MagicMock()]
        bad_msg.content[0].text = "not json"
        client = MagicMock()
        client.messages.create.return_value = bad_msg
        with self.assertRaises(ValueError):
            call_claude(client, "transcript", self._enriched(), retry=False)

    def test_transcript_included_in_message(self):
        client = make_mock_client(MOCK_RESPONSE)
        call_claude(client, "MY UNIQUE TRANSCRIPT", self._enriched())
        call_args = client.messages.create.call_args
        messages  = call_args[1].get('messages') or call_args[0][2]
        content   = messages[0]['content']
        texts     = [c['text'] for c in content if 'text' in c]
        self.assertTrue(any('MY UNIQUE TRANSCRIPT' in t for t in texts))


# ---------------------------------------------------------------------------
# parse_decisions
# ---------------------------------------------------------------------------

class TestParseDecisions(unittest.TestCase):

    def _enriched(self):
        return [
            {'id': 1, 'start': 1.0, 'end': 1.5, 'reason': 'disfluency',
             'before': 'a', 'after': 'b'},
            {'id': 2, 'start': 2.0, 'end': 3.5, 'reason': 'long_gap',
             'gap_seconds': 1.5, 'before': 'c', 'after': 'd'},
            {'id': 3, 'start': 4.0, 'end': 4.3, 'reason': 'text_filler',
             'word': 'um', 'before': 'e', 'after': 'f'},
        ]

    def test_remove_goes_to_final(self):
        result = {'decisions': [{'id': 1, 'action': 'REMOVE', 'reason': 'filler'}], 'missed_cuts': []}
        final, flagged = parse_decisions(result, self._enriched())
        self.assertEqual(len(final), 1)
        self.assertEqual(len(flagged), 0)

    def test_keep_is_discarded(self):
        result = {'decisions': [{'id': 2, 'action': 'KEEP', 'reason': 'pause'}], 'missed_cuts': []}
        final, flagged = parse_decisions(result, self._enriched())
        self.assertEqual(len(final), 0)
        self.assertEqual(len(flagged), 0)

    def test_flag_goes_to_flagged(self):
        result = {'decisions': [{'id': 3, 'action': 'FLAG', 'reason': 'unsure'}], 'missed_cuts': []}
        final, flagged = parse_decisions(result, self._enriched())
        self.assertEqual(len(flagged), 1)
        self.assertEqual(flagged[0]['ai_reason'], 'unsure')

    def test_ai_reason_attached(self):
        result = {'decisions': [{'id': 1, 'action': 'REMOVE', 'reason': 'clear um'}], 'missed_cuts': []}
        final, _ = parse_decisions(result, self._enriched())
        self.assertEqual(final[0]['ai_reason'], 'clear um')

    def test_internal_fields_stripped(self):
        result = {'decisions': [{'id': 1, 'action': 'REMOVE', 'reason': 'x'}], 'missed_cuts': []}
        final, _ = parse_decisions(result, self._enriched())
        self.assertNotIn('id',     final[0])
        self.assertNotIn('before', final[0])
        self.assertNotIn('after',  final[0])

    def test_case_insensitive_action(self):
        for action in ('remove', 'Remove', 'REMOVE'):
            result = {'decisions': [{'id': 1, 'action': action, 'reason': 'x'}], 'missed_cuts': []}
            final, _ = parse_decisions(result, self._enriched())
            self.assertEqual(len(final), 1, f"action={action} should produce 1 final entry")

    def test_unknown_id_ignored(self):
        result = {'decisions': [{'id': 99, 'action': 'REMOVE', 'reason': 'x'}], 'missed_cuts': []}
        final, _ = parse_decisions(result, self._enriched())
        self.assertEqual(len(final), 0)

    def test_valid_missed_cut_added_to_final(self):
        result = {'decisions': [], 'missed_cuts': [{'start': 10.0, 'end': 11.0}]}
        final, _ = parse_decisions(result, self._enriched())
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0]['reason'], 'ai_detected')

    def test_invalid_missed_cuts_rejected(self):
        result = {
            'decisions': [],
            'missed_cuts': [
                {'start': 'hello', 'end': 2.0},    # non-numeric
                {'start': 5.0,     'end': 3.0},    # end before start
                {'start': 10.0,    'end': 11.0},   # valid
            ],
        }
        final, _ = parse_decisions(result, self._enriched())
        self.assertEqual(len(final), 1)

    def test_empty_result_returns_empty_lists(self):
        final, flagged = parse_decisions({'decisions': [], 'missed_cuts': []}, self._enriched())
        self.assertEqual(final,   [])
        self.assertEqual(flagged, [])


# ---------------------------------------------------------------------------
# validate_cut
# ---------------------------------------------------------------------------

class TestValidateCut(unittest.TestCase):

    def test_valid_float(self):
        self.assertTrue(validate_cut({'start': 1.0, 'end': 2.0}))

    def test_valid_int(self):
        self.assertTrue(validate_cut({'start': 1, 'end': 2}))

    def test_end_before_start(self):
        self.assertFalse(validate_cut({'start': 2.0, 'end': 1.0}))

    def test_equal_start_end(self):
        self.assertFalse(validate_cut({'start': 1.0, 'end': 1.0}))

    def test_non_numeric_start(self):
        self.assertFalse(validate_cut({'start': 'hello', 'end': 2.0}))

    def test_non_numeric_end(self):
        self.assertFalse(validate_cut({'start': 1.0, 'end': None}))

    def test_missing_start(self):
        self.assertFalse(validate_cut({'end': 2.0}))

    def test_missing_end(self):
        self.assertFalse(validate_cut({'start': 1.0}))

    def test_empty_dict(self):
        self.assertFalse(validate_cut({}))


if __name__ == '__main__':
    unittest.main()
