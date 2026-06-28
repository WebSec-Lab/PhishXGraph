# PhishXGraph — open-science reference release

Reference implementation for *"PhishXGraph: Robust Phishing Website
Detection using Cross-Layer Interaction Graphs"*.

This directory contains a self-contained release of the core pipeline.
The implementation intentionally mirrors the paper's terminology and
defaults so reviewers can trace every claim to a specific file.

```
PhishXGraph/
├── phishxgraph/           # library
│   ├── graph.py           #   §4.1-4.2 cross-layer graph
│   ├── features.py        #   §4.3 — 101 features
│   ├── model.py           #   §4.4 — XGBoost + ablation classifiers
│   ├── collect.py         #   §4.2 — Playwright+CDP, 30s timeout
│   ├── ablation.py        #   §5.5 Table 7 — category × classifier
│   ├── blacklists.py      #   §5.6 — OpenPhish / PhishTank / GSB
│   ├── realtime.py        #   §5.6 — CertStream worker + blacklist lead
│   ├── realtime_stats.py  #   §5.6 summary (mean lead, confirmed, …)
│   ├── rescan.py          #   §5.6 deferred blacklist re-scan
│   └── cli.py             #   collect / train / test
├── evasion/               # §5.4 Table 6
│   ├── attacks/           #   10 attacks × 6 variants (URL/HTML/Logo)
│   ├── generate.py        #   Phase 1: single-surface cache generator
│   ├── generate_combos.py #   Phase 2: URL+HTML/URL+Logo/… combos
│   ├── score.py           #   score PhishXGraph on adversarial caches
│   └── aggregate.py       #   surface + combo ASR roll-up
├── baselines/             # §5.3 Table 5 — the 5 paper baselines
│   ├── urlnet/  lbp/  stackmodel/  bpe/          # CPU
│   ├── phishpedia/  phishintention/              # GPU
│   └── (each: Dockerfile + run.py + optional train.py)
├── docker/                # Dockerfile for the PhishXGraph container
├── docker-compose.yml     # collect → train → test → ablation → evasion → realtime + baselines
├── requirements.txt
├── scripts/
│   ├── smoke_test.sh           # end-to-end 30-URL smoke check
│   ├── download_pretrained.sh  # Phishpedia/PhishIntention weights from Google Drive
│   └── stage_samples.py        # curate safe-to-share sample bundle
├── data/samples/          # tiny bundled URL lists + redacted instrumentation
├── docs/
│   ├── PAPER_MAPPING.md   # paper section → file:function
│   ├── FEATURES.md        # full 101-feature list with definitions
│   ├── BASELINES.md       # how to run Table 5 / Table 6 with the baselines
│   └── DATA.md            # what's included and what isn't
└── tests/
    ├── test_features.py                        # 101-feature contract
    ├── test_ablation_evasion_blacklist.py      # Table 6/7 + §5.6 unit
    └── test_blacklists_live.py                 # opt-in live net tests
```

## Quick start

### Option A — Docker (recommended)

```bash
cd PhishXGraph
docker compose build
# 1) instrument a URL list (Playwright + CDP, 30-second timeout per page)
docker compose run --rm collect
# 2) train XGBoost classifier on the cached features
docker compose run --rm train
# 3) score a held-out URL list
docker compose run --rm test
# 4) ablation study (§5.5, Table 7): C/S/B × XGBoost/RF/SVM/MLP
docker compose run --rm ablation
# 5) robustness (§5.4, Table 6):
#    - generate adversarial caches (10 attacks × 6 variants)
#    - score PhishXGraph on them
#    - collapse into Table 6's surface/combo ASR
docker compose run --rm evasion-generate
docker compose run --rm evasion-generate-combos   # optional
docker compose run --rm evasion-score
docker compose run --rm evasion-aggregate
# 6) realtime (§5.6): CertStream + OpenPhish/PhishTank/GSB lead comparison
docker compose up realtime
# 7) summarize realtime lead times
docker compose run --rm realtime-stats
```

Set `GOOGLE_API_KEY` in the environment (or a `.env` file) before step 6
if you want GSB comparison included; PhishTank + OpenPhish are free.

Expected inputs under `./data/`:

- `data/input/urls.csv`       — training set (columns: `url,label` with
  label ∈ `phish|benign`)
- `data/input/test_urls.csv`  — test set (same schema)

Outputs:

- `data/features/<hash>/{page.html, instrumentation.json}` — cached
  instrumentation (shared with other baselines)
- `data/models/phishxgraph/{model.pkl, feature_names.pkl, classifier.txt}`
- `data/results/test_predictions.csv`
- `data/results/realtime_hits.csv`

### Option B — local Python

```bash
cd PhishXGraph
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium

python -m phishxgraph.cli collect  --input data/samples/input/smoke_train.csv \
    --features-dir data/smoke/features
python -m phishxgraph.cli train    --input data/samples/input/smoke_train.csv \
    --features-dir data/smoke/features --model-dir data/smoke/models
python -m phishxgraph.cli test     --input data/samples/input/smoke_test.csv \
    --features-dir data/smoke/features --model-dir data/smoke/models \
    --output data/smoke/results/predictions.csv
```

Or use the bundled script: `bash scripts/smoke_test.sh`.

## Paper claims at a glance

| Paper § | Claim | Code |
|---|---|---|
| §4.1, Table 2 | 4 node types × 4 edge types | `phishxgraph/graph.py` |
| §4.2 | Playwright + CDP, 30 s timeout | `phishxgraph/collect.py` |
| §4.3, Tables 9–11 | **101** features (42 C + 43 S + 16 B) | `phishxgraph/features.py` |
| §4.4 | XGBoost classifier | `phishxgraph/model.py` |
| §5.4, Table 6 | Adversarial robustness (10 attacks, 4 combos) | `evasion/score.py`, `evasion/aggregate.py` |
| §5.5, Table 7 | Category ablation (C/S/B…) × classifier (XGB/RF/SVM/MLP) | `phishxgraph/ablation.py` |
| §5.6 | Real-time CertStream pipeline | `phishxgraph/realtime.py` |
| §5.6 | Lead time vs OpenPhish / PhishTank / GSB | `phishxgraph/blacklists.py`, `realtime_stats.py` |

See [docs/PAPER_MAPPING.md](docs/PAPER_MAPPING.md) for the full mapping.

## Reproducibility notes

- **Random seeds**: `random_state=42` throughout (model.py).
- **Feature ordering**: `FEATURE_NAMES` in `phishxgraph/features.py` is
  the canonical list; `PhishXGraphClassifier` serialises it alongside the
  model so train-time and test-time column orders match.
- **101-feature contract**: asserted at module import time
  (`assert len(FEATURE_NAMES) == 101`) and again in
  `tests/test_features.py`.
- **Timeout**: 30 s per page matches §5.1; pages that time out are
  excluded from the dataset, not silently scored.

## Scope

This repository is self-contained: everything the paper claims can be
reproduced without cloning anything else.

- §4 — graph construction, 101-feature extraction, XGBoost classifier
- §5.2 — detection performance
- §5.3 — **Table 5 baselines** (URLNet, LBP, StackModel, BPE, Phishpedia,
  PhishIntention) live in [`baselines/`](baselines/); see
  [docs/BASELINES.md](docs/BASELINES.md)
- §5.4 — **Table 6** adversarial robustness (10 attacks × 6 variants)
- §5.5 — **Table 7** ablation (feature category × classifier)
- §5.6 — real-time detection vs OpenPhish / PhishTank / GSB

The only external dependencies are (a) Docker, (b) ~1 GB pretrained
weights from Google Drive for the two vision baselines
(`bash scripts/download_pretrained.sh all`), and (c) the phishing URL
feeds (PhishTank / OpenPhish) you're already using.

## Citation

```bibtex
@inproceedings{phishxgraph,
  title     = {PhishXGraph: Robust Phishing Website Detection using
               Cross-Layer Interaction Graphs},
  author    = {Bae, Jaeho and Kim, Junghoon and Kim, Doowon and Wi, Seongil},
  booktitle = {European Symposium on Research in Computer Security (ESORICS)},
  year      = {2026}
}
```
