# Context Surgeon

Local explainable context-compression MVP for PS-02. Python 3.10+; no third-party packages or build step required.

## Run

```powershell
python make_dataset.py
python app.py
```

Open http://127.0.0.1:8000. Use `py` if Windows does not recognize `python`.
The server generates the dataset on first start if missing. To change ports, set
`$env:PORT = '8001'` before starting. Press Ctrl+C to stop.

## Demo

1. Select `order_delay` and click **Compress locally**.
2. Inspect section decisions; try `constant_dependency`, `dependency_chain`, and `table_join`.
3. Run the offline benchmark and download its JSON report.
4. For Gemini evaluation, set `GEMINI_API_KEY=your_key` in `.env`; see `.env.example`. Environment variables take priority.
5. Click **Compare this example**. The server discovers compatible Gemini text models and automatically uses one that can complete your request. There is no model picker. If a model is unavailable or quota-limited, the server tries another; both answers in a comparison always use the same model.
6. Run/resume all 40 cases. Selection is automatic for each case, and each record identifies the model actually used. Every successful API stage is cached under its model in `results/`. Models that fail are temporarily skipped; working models are preferred for later cases.

Gemini comparisons send the system prompt, reference context and question to Google.
An uncached full batch uses 80 successful generations and up to 80 countTokens calls for the selected models. Model fallback can add requests. API quota
and charges may apply. Local compression and offline benchmarking make no API calls.
Keep the server running during a batch. After interruption, rerun to reuse matching cached stages. A batch can contain
different models, so compare quality per model when reviewing the report. Never commit or submit `.env`.

## Method and limitations

Splits text into blank-line-delimited blocks, preserves code fences, removes exact
duplicates, and scores query-word overlap with inverse block frequency. Protects
corrections, constraints and optional recent blocks. Selected blocks expand to
referenced Python definitions, linked identifiers and nearby pronoun antecedents.
Broad queries, no lexical match and unclosed fences conservatively retain context.
The system prompt and question remain separate and unchanged.

Budgets are soft: preserving evidence can exceed them. Local counts estimate UTF-8
bytes / 4, excluding the system prompt, question and API wrappers. Gemini provides
context-only actual counts plus complete generation usage metadata. This heuristic
can miss relevant evidence or retain unnecessary blocks. Tables remain whole blocks,
so joins on large tables may exceed the budget. This is a local demo, not a production server.

## Evaluation

The generator creates 40 synthetic cases (10 per category), a dataset card and SHA256.
30 development and 10 reserved cases are not an independently authored hidden test.
Required facts are source evidence, not computed answers, and are validated against
each context. Offline reports measure exact evidence-substring recall, estimated
reduction, a truncation baseline and median/p95 local compression latency.

Fact retention is **not answer correctness**. Manually compare generated answers
against expected answers and fill the exported human-review fields before making
quality claims. The dataset is small, mostly short and English-dominant.

- `app.py`: compressor, server, Gemini comparison, cache and batch runner.
- `index.html`: responsive UI and downloadable reports.
- `make_dataset.py`: reproducible 40-case generator.
- `data/`: dataset, SHA256 and dataset card.
- `results/offline-benchmark.json`: offline report.
- `results/gemini-evaluation.json`: batch answers and actual counts.
- `test_app.py`: compression, dataset, HTTP and mocked Gemini cache tests.

```powershell
python -m unittest -v
```
