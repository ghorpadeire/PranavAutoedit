"""
tests/test_api.py — FastAPI endpoint tests.

All tests use FastAPI's TestClient (synchronous WSGI wrapper).
pipeline.run is mocked throughout — no real Anthropic key needed.
No Redis to mock — api.py v1 is synchronous with no job queue.
"""

import io
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Set API_KEYS env var BEFORE importing api so _API_KEYS is populated correctly
os.environ.setdefault('API_KEYS', 'testkey,otherkey')

from fastapi.testclient import TestClient
from api import app, limiter

CLIENT = TestClient(app, raise_server_exceptions=False)


def _reset_rate_limiter():
    """Clear the in-memory rate-limit storage between tests to avoid 429s."""
    try:
        limiter._storage.reset()
    except Exception:
        pass   # defensive — if storage API changes, tests still run

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

VALID_TRANSCRIPT = json.dumps({
    "language": "en-us",
    "segments": [{
        "duration": 5.0,
        "language": "en-us",
        "speaker": "abc",
        "start": 0.0,
        "words": [
            {"confidence": 1.0, "duration": 0.4, "eos": False,
             "start": 0.0, "tags": [], "text": "Hello", "type": "word"},
            {"confidence": 1.0, "duration": 0.4, "eos": True,
             "start": 0.4, "tags": [], "text": "world.", "type": "word"},
        ]
    }]
}).encode()

FAKE_RESULT = {
    'final':           [{'start': 1.0, 'end': 2.0, 'reason': 'disfluency'}],
    'flagged':         [],
    'word_count':      10,
    'candidate_count': 2,
    'final_count':     1,
    'keep_count':      1,
    'flagged_count':   0,
}

GOOD_HEADERS = {
    'X-API-Key':    'testkey',
    'X-Claude-Key': 'sk-ant-fake',
}


def _upload(content: bytes = VALID_TRANSCRIPT, headers: dict = None) -> object:
    """POST /v1/analyze with a file upload."""
    hdrs = headers if headers is not None else GOOD_HEADERS
    return CLIENT.post(
        '/v1/analyze',
        files={'file': ('transcript.json', io.BytesIO(content), 'application/json')},
        headers=hdrs,
    )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth(unittest.TestCase):

    def test_health_returns_200(self):
        r = CLIENT.get('/health')
        self.assertEqual(r.status_code, 200)

    def test_health_has_status_ok(self):
        r = CLIENT.get('/health')
        self.assertEqual(r.json()['status'], 'ok')

    def test_health_has_model_key(self):
        r = CLIENT.get('/health')
        self.assertIn('model', r.json())

    def test_health_has_version_key(self):
        r = CLIENT.get('/health')
        self.assertIn('version', r.json())


# ---------------------------------------------------------------------------
# Auth — X-API-Key
# ---------------------------------------------------------------------------

class TestAuth(unittest.TestCase):

    def setUp(self):
        _reset_rate_limiter()

    def test_missing_api_key_returns_422(self):
        """FastAPI returns 422 when a required Header is absent."""
        r = CLIENT.post(
            '/v1/analyze',
            files={'file': ('t.json', io.BytesIO(VALID_TRANSCRIPT), 'application/json')},
            headers={'X-Claude-Key': 'sk-ant-fake'},   # no X-API-Key
        )
        self.assertEqual(r.status_code, 422)

    def test_wrong_api_key_returns_401(self):
        r = _upload(headers={'X-API-Key': 'wrong', 'X-Claude-Key': 'sk-ant-fake'})
        self.assertEqual(r.status_code, 401)

    def test_valid_api_key_does_not_401(self):
        with patch('api.pipeline.run', return_value=FAKE_RESULT):
            r = _upload()
        self.assertNotEqual(r.status_code, 401)

    def test_second_valid_key_also_accepted(self):
        with patch('api.pipeline.run', return_value=FAKE_RESULT):
            r = _upload(headers={'X-API-Key': 'otherkey', 'X-Claude-Key': 'sk-ant-fake'})
        self.assertNotEqual(r.status_code, 401)


# ---------------------------------------------------------------------------
# Auth — X-Claude-Key
# ---------------------------------------------------------------------------

class TestClaudeKeyHeader(unittest.TestCase):

    def setUp(self):
        _reset_rate_limiter()

    def test_missing_claude_key_returns_422(self):
        r = CLIENT.post(
            '/v1/analyze',
            files={'file': ('t.json', io.BytesIO(VALID_TRANSCRIPT), 'application/json')},
            headers={'X-API-Key': 'testkey'},   # no X-Claude-Key
        )
        self.assertEqual(r.status_code, 422)

    def test_claude_key_forwarded_to_pipeline(self):
        with patch('api.pipeline.run', return_value=FAKE_RESULT) as mock_run:
            _upload(headers={'X-API-Key': 'testkey', 'X-Claude-Key': 'sk-ant-mykey'})
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        self.assertEqual(kwargs['claude_api_key'], 'sk-ant-mykey')


# ---------------------------------------------------------------------------
# File validation
# ---------------------------------------------------------------------------

class TestFileValidation(unittest.TestCase):

    def setUp(self):
        _reset_rate_limiter()

    def test_too_large_returns_413(self):
        """Simulate a file just over the MAX_FILE_SIZE_MB limit."""
        import api as api_module
        original = api_module.config.MAX_FILE_SIZE_MB
        api_module.config.MAX_FILE_SIZE_MB = 0   # 0 MB → everything too large
        try:
            r = _upload()
            self.assertEqual(r.status_code, 413)
        finally:
            api_module.config.MAX_FILE_SIZE_MB = original

    def test_invalid_json_returns_422(self):
        r = _upload(content=b'not valid json at all')
        self.assertEqual(r.status_code, 422)

    def test_missing_segments_key_returns_422(self):
        bad = json.dumps({"language": "en-us"}).encode()
        r = _upload(content=bad)
        self.assertEqual(r.status_code, 422)

    def test_empty_segments_list_is_valid(self):
        """Empty segments list is structurally valid — pipeline handles it."""
        content = json.dumps({"language": "en-us", "segments": []}).encode()
        with patch('api.pipeline.run', return_value={**FAKE_RESULT, 'word_count': 0}):
            r = _upload(content=content)
        self.assertEqual(r.status_code, 200)


# ---------------------------------------------------------------------------
# POST /v1/analyze — happy path
# ---------------------------------------------------------------------------

class TestAnalyzeHappyPath(unittest.TestCase):

    def setUp(self):
        _reset_rate_limiter()
        self.patcher = patch('api.pipeline.run', return_value=FAKE_RESULT)
        self.mock_run = self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_returns_200(self):
        r = _upload()
        self.assertEqual(r.status_code, 200)

    def test_response_contains_final(self):
        r = _upload()
        self.assertIn('final', r.json())

    def test_response_contains_flagged(self):
        r = _upload()
        self.assertIn('flagged', r.json())

    def test_response_contains_word_count(self):
        r = _upload()
        self.assertIn('word_count', r.json())

    def test_response_contains_elapsed(self):
        r = _upload()
        self.assertIn('elapsed', r.json())

    def test_final_matches_fake_result(self):
        r = _upload()
        self.assertEqual(r.json()['final'], FAKE_RESULT['final'])

    def test_pipeline_called_once(self):
        _upload()
        self.mock_run.assert_called_once()

    def test_pipeline_receives_a_file_path(self):
        _upload()
        args, _ = self.mock_run.call_args
        path = args[0]
        # path should be a string (temp file path)
        self.assertIsInstance(path, str)
        self.assertTrue(path.endswith('.json'))


# ---------------------------------------------------------------------------
# POST /v1/analyze — pipeline errors
# ---------------------------------------------------------------------------

class TestAnalyzePipelineErrors(unittest.TestCase):

    def setUp(self):
        _reset_rate_limiter()

    def test_value_error_returns_502(self):
        with patch('api.pipeline.run', side_effect=ValueError("bad JSON")):
            r = _upload()
        self.assertEqual(r.status_code, 502)

    def test_generic_exception_returns_500(self):
        with patch('api.pipeline.run', side_effect=RuntimeError("boom")):
            r = _upload()
        self.assertEqual(r.status_code, 500)

    def test_file_not_found_returns_500(self):
        with patch('api.pipeline.run', side_effect=FileNotFoundError):
            r = _upload()
        self.assertEqual(r.status_code, 500)


# ---------------------------------------------------------------------------
# Temp file cleanup
# ---------------------------------------------------------------------------

class TestTempFileCleanup(unittest.TestCase):

    def setUp(self):
        _reset_rate_limiter()

    def _get_tmp_path(self, mock_run_side_effect=None):
        """Run analyze, capture the temp path passed to pipeline.run."""
        captured = {}

        def fake_run(path, claude_api_key):
            captured['path'] = path
            if mock_run_side_effect:
                raise mock_run_side_effect
            return FAKE_RESULT

        with patch('api.pipeline.run', side_effect=fake_run):
            _upload()
        return captured.get('path')

    def test_tmp_file_deleted_on_success(self):
        path = self._get_tmp_path()
        self.assertIsNotNone(path)
        self.assertFalse(os.path.exists(path), "Temp file should be deleted after success")

    def test_tmp_file_deleted_on_error(self):
        path = self._get_tmp_path(mock_run_side_effect=RuntimeError("boom"))
        self.assertIsNotNone(path)
        self.assertFalse(os.path.exists(path), "Temp file should be deleted after error")


if __name__ == '__main__':
    unittest.main()
