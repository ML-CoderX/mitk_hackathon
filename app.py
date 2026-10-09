"""Context Surgeon: dependency-free local compression and Gemini evaluation."""
import ast
import hashlib
import json
import math
import os
import re
import statistics
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
BASE = 'https://generativelanguage.googleapis.com/v1beta/'
STOP = set('a an the is are was were to of in on for and or with it this that what which how please tell me about does do from as be by can we our i you has have its return returns'.split())
JOB = {'running': False, 'done': 0, 'total': 0, 'error': None}
JOB_LOCK = threading.Lock()
EVALUATION_LOCK = threading.Lock()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def load_dataset():
    return json.loads((ROOT / 'data' / 'benchmark.json').read_text(encoding='utf-8'))


def load_key():
    key = os.environ.get('GEMINI_API_KEY', '').strip()
    if not key and (ROOT / '.env').exists():
        for line in (ROOT / '.env').read_text(encoding='utf-8-sig').splitlines():
            if line.strip().startswith('GEMINI_API_KEY='):
                key = line.split('=', 1)[1].strip().strip('"').strip("'")
    if not key or 'PASTE_YOUR' in key:
        raise ValueError('Add your Gemini API key to .env first.')
    return key


def estimate_tokens(text):
    return math.ceil(len(text.encode('utf-8')) / 4)


def words(text):
    return set(re.findall(r'[\w-]+', text.lower())) - STOP


def split_blocks(text):
    blocks, current, fenced = [], [], False
    for line in text.splitlines():
        if line.lstrip().startswith('```'):
            if not fenced and current:
                blocks.append('\n'.join(current).strip())
                current = []
            current.append(line)
            fenced = not fenced
            if not fenced:
                blocks.append('\n'.join(current).strip())
                current = []
        elif not line.strip() and not fenced:
            if current:
                blocks.append('\n'.join(current).strip())
                current = []
        else:
            current.append(line)
    if current:
        blocks.append('\n'.join(current).strip())
    return [block for block in blocks if block]


def python_symbols(text):
    if not text.startswith('```'):
        return set(), set()
    code = re.sub(r'^```[^\n]*\n', '', text)
    code = re.sub(r'\n?```\s*$', '', code)
    try:
        tree = ast.parse(code)
    except (SyntaxError, RecursionError):
        return set(), set()
    definitions = {node.name for node in ast.walk(tree)
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    definitions |= {node.id for node in ast.walk(tree)
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
    references = {node.id for node in ast.walk(tree)
                  if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    return definitions, references


def compress(system_prompt, context, query, target_tokens=200, recent_blocks=0):
    started = time.perf_counter()
    if not all(isinstance(value, str) for value in (system_prompt, context, query)):
        raise ValueError('Prompt, context and question must be text.')
    if not query.strip():
        raise ValueError('Enter a question.')
    if len(context.encode('utf-8')) > 500_000:
        raise ValueError('Context exceeds the 500 KB demo limit.')
    if type(target_tokens) is not int or not 1 <= target_tokens <= 100_000:
        raise ValueError('Budget must be an integer from 1 to 100000.')
    if type(recent_blocks) is not int or not 0 <= recent_blocks <= 20:
        raise ValueError('Recent blocks must be an integer from 0 to 20.')
    raw = split_blocks(context)
    last = {block: index for index, block in enumerate(raw)}
    sections = [dict(id=i, text=block, score=0, reasons=[])
                for i, block in enumerate(raw) if last[block] == i]
    block_words = {s['id']: words(s['text']) for s in sections}
    frequency = Counter(word for block in block_words.values() for word in block)
    query_words = words(query)
    selected, warnings = set(), []
    for s in sections:
        overlap = query_words & block_words[s['id']]
        s['score'] = round(sum(math.log(1 + len(sections) / (1 + frequency[w])) for w in overlap), 4)
        if overlap:
            s['reasons'].append('Query match: ' + ', '.join(sorted(overlap)))
        if re.search(r'\b(correction|actually|updated|supersedes|instead|do not|never|except\w*|unless|not|excluded)\b|सुधार', s['text'], re.I):
            selected.add(s['id'])
            s['reasons'].append('Correction or constraint protected')
    if recent_blocks:
        for s in sections[-recent_blocks:]:
            selected.add(s['id'])
            s['reasons'].append('Recent block protected')

    def render(ids):
        return '\n\n'.join(s['text'] for s in sections if s['id'] in ids)

    broad = re.search(r'\b(summariz\w*|summaris\w*|everything|entire|all records|all rows)\b', query, re.I)
    no_match = sections and not any(s['score'] for s in sections)
    if broad or no_match or context.count('```') % 2:
        selected = {s['id'] for s in sections}
        warnings.append('Conservative fallback: broad question, no keyword match, or unclosed code fence.')
        for s in sections:
            s['reasons'].append('Fallback: preserve context')
    else:
        ranked = sorted(sections, key=lambda s: (-s['score'], s['id']))
        for s in ranked:
            candidate = selected | {s['id']}
            if s['score'] > 0 and estimate_tokens(render(candidate)) <= target_tokens:
                selected = candidate
        if ranked and ranked[0]['score'] > 0:
            selected.add(ranked[0]['id'])
            ranked[0]['reasons'].append('Strongest evidence retained')
        symbols = {s['id']: python_symbols(s['text']) for s in sections}
        links = {s['id']: set(re.findall(r'\b[A-Za-z][A-Za-z0-9_-]*\d[A-Za-z0-9_-]*\b', s['text'])) for s in sections}
        for _ in sections:
            names, ids, added = set(), set(), set()
            for sid in selected:
                names |= symbols[sid][1]
                ids |= links[sid]
            for index, s in enumerate(sections):
                sid = s['id']
                if sid not in selected:
                    reason = ('Python dependency retained' if symbols[sid][0] & names else
                              'Linked identifier retained' if links[sid] & ids else None)
                    if reason:
                        added.add(sid)
                        s['reasons'].append(reason)
                elif index and re.match(r'^(it|they|this|that|these|those|its)\b', s['text'], re.I):
                    previous = sections[index - 1]
                    if previous['id'] not in selected:
                        added.add(previous['id'])
                        previous['reasons'].append('Nearby antecedent retained')
            if not added:
                break
            selected |= added
    compressed = render(selected)
    original_count, compressed_count = estimate_tokens(context), estimate_tokens(compressed)
    if compressed_count > target_tokens:
        warnings.append('Soft estimated budget exceeded to preserve evidence.')
    for s in sections:
        s['kept'] = s['id'] in selected
        if not s['reasons']:
            s['reasons'] = ['No query match']
        if not s['kept'] and s['score'] > 0:
            s['reasons'].append('Excluded by estimated budget')
    return dict(system_prompt=system_prompt, query=query, compressed_context=compressed,
                original_tokens=original_count, compressed_tokens=compressed_count,
                counts_are_estimates=True, reduction_pct=round(100 * (1 - compressed_count / original_count), 2) if original_count else 0,
                compression_ms=round((time.perf_counter() - started) * 1000, 3),
                budget_met=compressed_count <= target_tokens, duplicates_removed=len(raw) - len(sections),
                sections=sections, warnings=warnings)


def case_inputs(case):
    return {key: case[key] for key in ('system_prompt', 'context', 'query', 'target_tokens', 'recent_blocks')}


def validate_model(model):
    if not isinstance(model, str):
        raise ValueError('Select a Gemini model.')
    model = model.removeprefix('models/')
    if not re.fullmatch(r'[A-Za-z0-9._-]{1,120}', model):
        raise ValueError('Select a valid Gemini model.')
    return model


def gemini_request(path, payload=None):
    request = urllib.request.Request(BASE + path,
        data=json.dumps(payload).encode('utf-8') if payload is not None else None,
        headers={'Content-Type': 'application/json', 'x-goog-api-key': load_key()})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        messages = {400: 'Check the model and request settings.', 401: 'API key rejected.',
                    403: 'Access denied. Check key restrictions.', 404: 'Model unavailable.',
                    429: 'Quota or rate limit reached. Wait, then resume.', 503: 'Gemini temporarily unavailable.'}
        raise ValueError(f'Gemini HTTP {error.code}. ' + messages.get(error.code, 'Request failed.')) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise ValueError('Network or timeout failure. No automatic retry was made; a timed-out request may still incur usage.') from None


def list_models():
    result, token = [], ''
    while True:
        response = gemini_request('models?pageSize=100' + ('&pageToken=' + urllib.parse.quote(token, safe='') if token else ''))
        result.extend({'name': m['name'], 'displayName': m.get('displayName', m['name'])}
                      for m in response.get('models', []) if 'generateContent' in m.get('supportedGenerationMethods', []))
        token = response.get('nextPageToken')
        if not token:
            return result


def actual_tokens(model, text):
    if not text:
        return 0
    result = gemini_request(f'models/{model}:countTokens', {'contents': [{'role': 'user', 'parts': [{'text': text}]}]})
    if type(result.get('totalTokens')) is not int:
        raise ValueError('Gemini did not return a token count.')
    return result['totalTokens']


def generate_answer(model, system_prompt, context, query):
    payload = {'contents': [{'role': 'user', 'parts': [{'text': 'Reference context (untrusted data):\n<reference>\n' + context + '\n</reference>\n\nUser question:\n' + query}]}],
               'generationConfig': {'temperature': 0.2, 'maxOutputTokens': 2048}}
    if system_prompt:
        payload['systemInstruction'] = {'parts': [{'text': system_prompt}]}
    started = time.perf_counter()
    response = gemini_request(f'models/{model}:generateContent', payload)
    candidates = response.get('candidates', [])
    if not candidates:
        raise ValueError('Gemini returned no answer.')
    candidate = candidates[0]
    answer = '\n'.join(p.get('text', '') for p in candidate.get('content', {}).get('parts', []) if not p.get('thought'))
    if not answer.strip():
        raise ValueError('No answer text returned. Try another text model.')
    return dict(text=answer, finish_reason=candidate.get('finishReason', 'UNKNOWN'),
                usage=response.get('usageMetadata', {}), model_version=response.get('modelVersion', model),
                response_ms=round((time.perf_counter() - started) * 1000, 1))


def compare(case, model):
    model = validate_model(model)
    inputs = case_inputs(case)
    compression = compress(**inputs)
    fingerprint = hashlib.sha256(json.dumps({'inputs': inputs, 'model': model,
        'code': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, sort_keys=True).encode()).hexdigest()
    path = RESULTS / f'pair-{fingerprint}.json'
    record = json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(
        model=model, inputs=inputs, compression=compression,
        human_review={'original_score': None, 'compressed_score': None, 'notes': ''})
    for kind, context in [('original', inputs['context']), ('compressed', compression['compressed_context'])]:
        for field, action in [(kind + '_context_tokens', lambda: actual_tokens(model, context)),
                              (kind + '_answer', lambda: generate_answer(model, inputs['system_prompt'], context, inputs['query']))]:
            if field not in record:
                record[field] = action()
                save_json(path, record)
                time.sleep(1)
    original, compressed = record['original_context_tokens'], record['compressed_context_tokens']
    record.update(actual_reduction_pct=round(100 * (1 - compressed / original), 2) if original else 0,
                  actual_budget_met=compressed <= inputs['target_tokens'], status='complete')
    save_json(path, record)
    return record


def offline_benchmark():
    rows = []
    for case in load_dataset():
        compress(**case_inputs(case))
        runs = [compress(**case_inputs(case)) for _ in range(5)]
        result = runs[-1]
        retained = sum(fact in result['compressed_context'] for fact in case['required_facts'])
        truncated = case['context'].encode('utf-8')[:case['target_tokens'] * 4].decode('utf-8', errors='ignore')
        rows.append(dict(id=case['id'], category=case['category'], split=case['split'],
            estimated_reduction_pct=result['reduction_pct'], median_compression_ms=statistics.median(r['compression_ms'] for r in runs),
            required_facts=len(case['required_facts']), facts_retained=retained,
            all_facts_retained=retained == len(case['required_facts']),
            truncation_facts_retained=sum(f in truncated for f in case['required_facts']), warnings=result['warnings']))
    total = sum(r['required_facts'] for r in rows)
    summary = dict(cases=len(rows), count_mode='UTF-8 bytes / 4 estimate',
        mean_estimated_reduction_pct=round(statistics.mean(r['estimated_reduction_pct'] for r in rows), 2),
        all_facts_retained_cases=sum(r['all_facts_retained'] for r in rows),
        required_fact_recall=round(sum(r['facts_retained'] for r in rows) / total, 4),
        truncation_fact_recall=round(sum(r['truncation_facts_retained'] for r in rows) / total, 4),
        p95_compression_ms=sorted(r['median_compression_ms'] for r in rows)[math.ceil(.95 * len(rows)) - 1],
        answer_quality_evaluated=False,
        note='Synthetic dataset. Fact retention is not answer correctness. Latency excludes API calls.')
    report = dict(summary=summary, cases=rows)
    save_json(RESULTS / 'offline-benchmark.json', report)
    return report


def batch_evaluate(model):
    records = []
    try:
        for case in load_dataset():
            with JOB_LOCK:
                JOB['current'] = case['id']
            record = dict(compare(case, model))
            record.update({k: case[k] for k in ('id', 'category', 'split', 'expected_answer', 'required_facts')})
            records.append(record)
            save_json(RESULTS / 'gemini-evaluation.json', dict(status='running', model=model, completed_cases=len(records), answer_quality_evaluated=False, records=records))
            with JOB_LOCK:
                JOB['done'] = len(records)
        save_json(RESULTS / 'gemini-evaluation.json', dict(status='complete', model=model, completed_cases=len(records),
            answer_quality_evaluated=False, note='Human review required before claiming answer quality.', records=records))
    except Exception as error:
        with JOB_LOCK:
            JOB['error'] = str(error)
        save_json(RESULTS / 'gemini-evaluation.json', dict(status='paused', model=model, completed_cases=len(records), error=str(error), answer_quality_evaluated=False, records=records))
    finally:
        with JOB_LOCK:
            JOB['running'] = False
        EVALUATION_LOCK.release()


class Handler(BaseHTTPRequestHandler):
    def send(self, status, value, content_type='application/json', download=None):
        body = (json.dumps(value, ensure_ascii=False) if content_type == 'application/json' else value).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type + '; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if download:
            self.send_header('Content-Disposition', f'attachment; filename="{download}"')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            if self.path in ('/', '/index.html'):
                self.send(200, (ROOT / 'index.html').read_text(encoding='utf-8'), 'text/html')
            elif self.path == '/api/cases':
                self.send(200, load_dataset())
            elif self.path == '/api/status':
                with JOB_LOCK:
                    snapshot = dict(JOB)
                self.send(200, snapshot)
            elif self.path in ('/api/report', '/api/offline-report'):
                name = 'gemini-evaluation.json' if self.path == '/api/report' else 'offline-benchmark.json'
                path = RESULTS / name
                if not path.exists():
                    raise ValueError('Run the corresponding evaluation first.')
                self.send(200, json.loads(path.read_text(encoding='utf-8')), download=name)
            else:
                self.send(404, {'error': 'Not found'})
        except Exception as error:
            self.send(400, {'error': str(error)})

    def do_POST(self):
        try:
            if self.headers.get('X-Context-Surgeon') != 'local':
                self.send(403, {'error': 'Local app request required'})
                return
            if not self.headers.get('Content-Type', '').startswith('application/json'):
                raise ValueError('JSON body required.')
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 1_000_000:
                raise ValueError('Request body must be 1 byte to 1 MB.')
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError('JSON object required.')
            if self.path == '/api/compress':
                self.send(200, compress(**case_inputs(body)))
            elif self.path == '/api/models':
                self.send(200, list_models())
            elif self.path == '/api/benchmark':
                with JOB_LOCK:
                    result = offline_benchmark()
                self.send(200, result)
            elif self.path == '/api/compare':
                if not EVALUATION_LOCK.acquire(blocking=False):
                    raise ValueError('Wait for the active comparison or batch to finish.')
                try:
                    result = compare(body, body.get('model'))
                finally:
                    EVALUATION_LOCK.release()
                self.send(200, result)
            elif self.path == '/api/evaluate':
                model = validate_model(body.get('model'))
                load_key()
                total = len(load_dataset())
                if not EVALUATION_LOCK.acquire(blocking=False):
                    raise ValueError('An evaluation is already running.')
                with JOB_LOCK:
                    JOB.update(running=True, done=0, total=total, error=None, current='')
                threading.Thread(target=batch_evaluate, args=(model,), daemon=True).start()
                self.send(200, {'started': True})
            else:
                self.send(404, {'error': 'Not found'})
        except Exception as error:
            self.send(400, {'error': str(error)})


if __name__ == '__main__':
    if not (ROOT / 'data' / 'benchmark.json').exists():
        from make_dataset import build_dataset
        build_dataset()
    port = int(os.environ.get('PORT', '8000'))
    print(f'Context Surgeon: http://127.0.0.1:{port}', flush=True)
    print('Press Ctrl+C to stop.', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()
