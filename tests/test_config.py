import os
import sys
import unittest
import importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def reload_config(env_overrides: dict = None):
    """Reload config module with custom env vars, then restore originals."""
    original = {}
    if env_overrides:
        for k, v in env_overrides.items():
            original[k] = os.environ.get(k)
            os.environ[k] = v
    import config
    importlib.reload(config)
    # Restore
    for k, orig_v in original.items():
        if orig_v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = orig_v
    return config


class TestConfigDefaults(unittest.TestCase):
    def setUp(self):
        # Remove any env overrides so we test defaults
        for key in ['GAP_THRESHOLD', 'OVERLAP_THRESHOLD', 'BIGRAM_THRESHOLD',
                    'STRONG_FILLERS', 'SENTENCE_FILLERS',
                    'CLAUDE_MODEL', 'MAX_TOKENS', 'MAX_FILE_SIZE_MB']:
            os.environ.pop(key, None)
        import config
        importlib.reload(config)
        import config as cfg
        self.cfg = cfg

    def test_gap_threshold_default(self):
        self.assertAlmostEqual(self.cfg.GAP_THRESHOLD, 0.8)

    def test_overlap_threshold_default(self):
        self.assertAlmostEqual(self.cfg.OVERLAP_THRESHOLD, 0.6)

    def test_bigram_threshold_default(self):
        self.assertAlmostEqual(self.cfg.BIGRAM_THRESHOLD, 0.5)

    def test_strong_fillers_is_set(self):
        self.assertIsInstance(self.cfg.STRONG_FILLERS, set)
        self.assertIn('um', self.cfg.STRONG_FILLERS)
        self.assertIn('uh', self.cfg.STRONG_FILLERS)

    def test_sentence_start_fillers_is_set(self):
        self.assertIsInstance(self.cfg.SENTENCE_START_FILLERS, set)
        self.assertIn('okay', self.cfg.SENTENCE_START_FILLERS)
        self.assertIn('so',   self.cfg.SENTENCE_START_FILLERS)

    def test_claude_model_default(self):
        self.assertIn('claude', self.cfg.CLAUDE_MODEL.lower())

    def test_max_tokens_default(self):
        self.assertGreater(self.cfg.MAX_TOKENS, 0)

    def test_max_file_size_default(self):
        self.assertGreater(self.cfg.MAX_FILE_SIZE_MB, 0)


class TestConfigEnvOverride(unittest.TestCase):
    def test_gap_threshold_override(self):
        cfg = reload_config({'GAP_THRESHOLD': '1.5'})
        self.assertAlmostEqual(cfg.GAP_THRESHOLD, 1.5)

    def test_overlap_threshold_override(self):
        cfg = reload_config({'OVERLAP_THRESHOLD': '0.8'})
        self.assertAlmostEqual(cfg.OVERLAP_THRESHOLD, 0.8)

    def test_strong_fillers_override(self):
        cfg = reload_config({'STRONG_FILLERS': 'um,uh,err'})
        self.assertEqual(cfg.STRONG_FILLERS, {'um', 'uh', 'err'})

    def test_sentence_fillers_override(self):
        cfg = reload_config({'SENTENCE_FILLERS': 'so,okay'})
        self.assertEqual(cfg.SENTENCE_START_FILLERS, {'so', 'okay'})

    def test_claude_model_override(self):
        cfg = reload_config({'CLAUDE_MODEL': 'claude-opus-4-5'})
        self.assertEqual(cfg.CLAUDE_MODEL, 'claude-opus-4-5')

    def test_max_tokens_override(self):
        cfg = reload_config({'MAX_TOKENS': '8192'})
        self.assertEqual(cfg.MAX_TOKENS, 8192)


class TestGetLocalApiKey(unittest.TestCase):
    def setUp(self):
        os.environ.pop('ANTHROPIC_API_KEY', None)
        import config
        importlib.reload(config)
        import config as cfg
        self.cfg = cfg

    def test_env_var_takes_priority(self):
        os.environ['ANTHROPIC_API_KEY'] = 'sk-test-env'
        key = self.cfg.get_local_api_key()
        self.assertEqual(key, 'sk-test-env')
        del os.environ['ANTHROPIC_API_KEY']

    def test_missing_file_returns_none(self):
        key = self.cfg.get_local_api_key(path='nonexistent_key_file.txt')
        self.assertIsNone(key)

    def test_file_read(self):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('sk-ant-test123\n')
            tmp_path = f.name
        try:
            key = self.cfg.get_local_api_key(path=tmp_path)
            self.assertEqual(key, 'sk-ant-test123')
        finally:
            os.unlink(tmp_path)

    def test_empty_file_returns_none(self):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('   \n')
            tmp_path = f.name
        try:
            key = self.cfg.get_local_api_key(path=tmp_path)
            self.assertIsNone(key)
        finally:
            os.unlink(tmp_path)


class TestNgramsFunction(unittest.TestCase):
    """Tests for the ngrams() helper added to analyze_transcript.py."""

    def setUp(self):
        import analyze_transcript
        self.ngrams = analyze_transcript.ngrams

    def _words(self, texts):
        return [{'text': t, 'start': i * 0.5, 'end': i * 0.5 + 0.4}
                for i, t in enumerate(texts)]

    def test_basic_bigrams(self):
        result = self.ngrams(self._words(['I', 'am', 'here']))
        self.assertEqual(result, {('i', 'am'), ('am', 'here')})

    def test_single_word_returns_empty(self):
        self.assertEqual(self.ngrams(self._words(['Hello'])), set())

    def test_empty_list_returns_empty(self):
        self.assertEqual(self.ngrams([]), set())

    def test_punctuation_stripped(self):
        result = self.ngrams(self._words(['Hello,', 'world.']))
        self.assertIn(('hello', 'world'), result)

    def test_skips_empty_text(self):
        words = self._words(['I', '', 'think'])
        words[1]['text'] = ''
        result = self.ngrams(words)
        self.assertIn(('i', 'think'), result)


if __name__ == '__main__':
    unittest.main()
