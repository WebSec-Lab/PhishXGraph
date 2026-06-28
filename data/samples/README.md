# Sample bundle

- `input/sample_urls.csv` — 30 URL/label pairs
- `features/<hash>/instrumentation.json` — 30 redacted
  instrumentation payloads (stack traces and cookie value
  previews removed)

To score with the reference pipeline:

```
python -m phishxgraph.cli test \
    --input    data/samples/input/sample_urls.csv \
    --features-dir data/samples/features \
    --model-dir <path-to-your-trained-model> \
    --output   predictions.csv
```

Note: `page.html` is intentionally excluded. Structural and
behavioral features that depend on the DOM cannot be computed
without it; for a full re-run, regenerate the cache with
`phishxgraph.cli collect`.
