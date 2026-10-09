import hashlib
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import app


class SaveTests(unittest.TestCase):
    def test_transient_windows_lock_is_retried(self):
        path = app.RESULTS / 'test-save-lock.json'
        original = Path.replace
        calls = []
        def replace(source, destination):
            calls.append(source)
            if len(calls) < 3:
                raise PermissionError('Simulated Windows sharing lock')
            return original(source, destination)
        try:
            with patch.object(Path, 'replace', replace), patch.object(app.time, 'sleep'):
                app.save_json(path, {'saved': True})
            self.assertEqual(json.loads(path.read_text()), {'saved': True})
            self.assertEqual(len(calls), 3)
            self.assertFalse(calls[0].exists())
        finally:
            path.unlink(missing_ok=True)

    def test_permanent_lock_preserves_previous_and_recovery_copy(self):
        path = app.RESULTS / 'test-save-persistent.json'
        app.save_json(path, {'previous': True})
        try:
            with patch.object(Path, 'replace', side_effect=PermissionError('locked')), patch.object(app.time, 'sleep'):
                with self.assertRaisesRegex(ValueError, 'recovery file'):
                    app.save_json(path, {'new': True})
            self.assertEqual(json.loads(path.read_text()), {'previous': True})
            pending = list(app.RESULTS.glob('test-save-persistent-*.tmp'))
            self.assertEqual(len(pending), 1)
            self.assertEqual(json.loads(pending[0].read_text()), {'new': True})
        finally:
            path.unlink(missing_ok=True)
            for pending in app.RESULTS.glob('test-save-persistent-*.tmp'):
                pending.unlink()

    def test_cache_recovers_more_complete_pending_record(self):
        inputs = {'test': 'recovery-fixture'}
        record = dict(inputs=inputs, model='fixture', compression={'compressed_context': 'retained'}, original_answer={'text': 'answer'})
        path = app.RESULTS / 'pair-test-recovery.tmp'
        try:
            path.write_text(json.dumps(record))
            self.assertEqual(app.cached_record(inputs, 'fixture', 'retained'), record)
            self.assertIsNone(app.cached_record(inputs, 'fixture', 'changed'))
        finally:
            path.unlink(missing_ok=True)


class CompressionTests(unittest.TestCase):
    def test_dataset_integrity(self):
        cases = app.load_dataset()
        self.assertEqual(len({c['id'] for c in cases}), 40)
        self.assertEqual(Counter(c['category'] for c in cases), dict(conversation=10, document=10, code=10, data=10))
        self.assertEqual(sum(c['split'] == 'reserved' for c in cases), 10)
        for c in cases:
            self.assertTrue(all(f in c['context'] for f in c['required_facts']), c['id'])
        path = app.ROOT / 'data' / 'benchmark.json'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), path.with_suffix('.sha256').read_text().strip())

    def test_transitive_code_dependencies(self):
        c = next(c for c in app.load_dataset() if c['id'] == 'dependency_chain')
        result = app.compress(**app.case_inputs(c))
        self.assertTrue(all(f in result['compressed_context'] for f in c['required_facts']))
        self.assertNotIn('unrelated_', result['compressed_context'])
        self.assertEqual(result['system_prompt'], c['system_prompt'])
        self.assertEqual(result['query'], c['query'])

    def test_linked_tables_exceed_soft_budget(self):
        c = next(c for c in app.load_dataset() if c['id'] == 'table_join')
        r = app.compress(**app.case_inputs(c))
        self.assertIn('O7,C4', r['compressed_context'])
        self.assertIn('C4,Leena', r['compressed_context'])
        self.assertFalse(r['budget_met'])
        self.assertTrue(r['warnings'])

    def test_fallback_and_empty_context(self):
        for context, query in [('One fact.\n\nAnother fact.', 'Summarize everything.'),
                               ('Warranty expires in May.', 'Car coverage?'),
                               ('```python\nx = 1', 'x?')]:
            r = app.compress('', context, query, 1)
            self.assertEqual(r['compressed_context'], context)
            self.assertTrue(r['warnings'])
        r = app.compress('', '', 'Anything?')
        self.assertEqual(r['original_tokens'], 0)
        self.assertEqual(r['compressed_tokens'], 0)

    def test_duplicates_and_recent_protection(self):
        r = app.compress('', 'Alpha.\n\nAlpha.\n\nBeta.', 'Alpha?', 1, 1)
        self.assertEqual(r['duplicates_removed'], 1)
        self.assertEqual(r['compressed_context'], 'Alpha.\n\nBeta.')

    def test_validation(self):
        for kwargs in [dict(query=''), dict(target_tokens=True), dict(target_tokens=0),
                       dict(recent_blocks=21), dict(context='x' * 500001)]:
            values = dict(system_prompt='', context='test', query='test?')
            values.update(kwargs)
            with self.assertRaises(ValueError):
                app.compress(**values)


class GeminiTests(unittest.TestCase):
    def test_automatic_selection_falls_back_and_remembers_working_model(self):
        candidates = [{'name': 'models/gemini-a'}, {'name': 'models/gemini-b'}]
        with patch.object(app, 'MODEL_COOLDOWNS', {}), patch.object(app, 'PREFERRED_MODEL', None), \
             patch.object(app, 'automatic_models', return_value=candidates), \
             patch.object(app, 'compare', side_effect=[app.GeminiError(429, 'Quota'), {'model': 'gemini-b'}, {'model': 'gemini-b'}]) as compare:
            result = app.compare_automatically(app.load_dataset()[0])
            self.assertEqual(result['model'], 'gemini-b')
            self.assertEqual(result['fallback_attempts'], [{'model': 'models/gemini-a', 'status': 429}])
            app.compare_automatically(app.load_dataset()[0])
            self.assertEqual([call.args[1] for call in compare.call_args_list],
                             ['models/gemini-a', 'models/gemini-b', 'models/gemini-b'])

    def test_automatic_selection_does_not_retry_invalid_key(self):
        with patch.object(app, 'MODEL_COOLDOWNS', {}), patch.object(app, 'compare', side_effect=app.GeminiError(403, 'Access denied')) as compare:
            with self.assertRaises(app.GeminiError):
                app.compare_automatically(app.load_dataset()[0], [{'name': 'models/gemini-a'}, {'name': 'models/gemini-b'}])
            self.assertEqual(compare.call_count, 1)

    def test_automatic_discovery_excludes_media_models(self):
        names = ['gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemini-image', 'gemini-tts', 'gemini-deep-research', 'gemma-3']
        with patch.object(app, 'list_models', return_value=[{'name': 'models/' + n, 'displayName': n} for n in names]):
            self.assertEqual([m['name'] for m in app.automatic_models()], ['models/gemini-2.5-flash-lite', 'models/gemini-2.5-flash'])

    def test_resume_caches_completed_stages(self):
        case = app.load_dataset()[0]
        cache_before = set(app.RESULTS.glob('pair-*.json')) if app.RESULTS.exists() else set()
        app.RESULTS.mkdir(exist_ok=True)
        with patch.object(app.time, 'sleep'), \
             patch.object(app, 'actual_tokens', side_effect=[100, 50]) as counts, \
             patch.object(app, 'generate_answer', side_effect=[{'text': 'original'}, ValueError('quota'), {'text': 'compressed'}]) as answers:
            try:
                with self.assertRaisesRegex(ValueError, 'quota'):
                    app.compare(case, 'models/test-model')
                r = app.compare(case, 'test-model')
                self.assertEqual(r['actual_reduction_pct'], 50)
                self.assertEqual(counts.call_count, 2)
                self.assertEqual(answers.call_count, 3)
                app.compare(case, 'test-model')
                self.assertEqual(answers.call_count, 3)
            finally:
                for path in set(app.RESULTS.glob('pair-*.json')) - cache_before:
                    path.unlink(missing_ok=True)

    def test_model_path_validation(self):
        for name in ['../bad', '', None, 'model?key=oops']:
            with self.assertRaises(ValueError):
                app.validate_model(name)

    def test_request_shape(self):
        response = {'candidates': [{'content': {'parts': [{'text': 'private thought', 'thought': True}, {'text': 'Answer'}]}, 'finishReason': 'STOP'}]}
        with patch.object(app, 'gemini_request', return_value=response) as request:
            result = app.generate_answer('test-model', 'System rules', 'Reference', 'Question')
        self.assertEqual(result['text'], 'Answer')
        path, payload = request.call_args.args
        self.assertEqual(path, 'models/test-model:generateContent')
        self.assertEqual(payload['systemInstruction']['parts'][0]['text'], 'System rules')
        self.assertIn('Reference', payload['contents'][0]['parts'][0]['text'])


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = app.ThreadingHTTPServer(('127.0.0.1', 0), app.Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = 'http://127.0.0.1:' + str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_compress_endpoint(self):
        body = app.case_inputs(app.load_dataset()[0])
        req = urllib.request.Request(self.base + '/api/compress', json.dumps(body).encode(),
            headers={'Content-Type': 'application/json', 'X-Context-Surgeon': 'local'})
        with urllib.request.urlopen(req) as response:
            result = json.load(response)
        self.assertIn('15 October', result['compressed_context'])

    def test_reject_cross_origin_and_private_files(self):
        req = urllib.request.Request(self.base + '/api/compress', b'{}', headers={'Content-Type': 'application/json'})
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(req)
        self.assertEqual(error.exception.code, 403)
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(self.base + '/.env')
        self.assertEqual(error.exception.code, 404)

    def test_ui_served(self):
        with urllib.request.urlopen(self.base + '/') as response:
            self.assertIn(b'Context Surgeon', response.read())


if __name__ == '__main__':
    unittest.main()
