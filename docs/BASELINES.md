# Baselines bundled with this release

The paper compares PhishXGraph against five baselines (Table 5, §5.3).
All five are **included** in this release under [`baselines/`](../baselines/)
along with a one-shot pretrained-weight download script for the two
vision baselines.

| Baseline | Path | Surface | Hardware |
|---|---|---|---|
| URLNet | [`baselines/urlnet/`](../baselines/urlnet/) | URL strings | CPU |
| LBP | [`baselines/lbp/`](../baselines/lbp/) | URL + hosting graph | CPU |
| StackModel | [`baselines/stackmodel/`](../baselines/stackmodel/) | URL + HTML | CPU |
| BPE | [`baselines/bpe/`](../baselines/bpe/) | URL tokens + hosting | CPU |
| Phishpedia | [`baselines/phishpedia/`](../baselines/phishpedia/) | Logo + brand | **GPU** |
| PhishIntention | [`baselines/phishintention/`](../baselines/phishintention/) | Logo + layout + CRP | **GPU** |

Each baseline is a thin wrapper that fetches the upstream author's code
at build time (see the comment in each `Dockerfile`) and plugs it into
the shared `data/features/` cache so one crawl feeds every model.

## Shared cache layout

Every crawled page is keyed by `sha256(url)[:16]`:

```
data/features/<hash>/
├── page.html               # rendered DOM (StackModel, PhishXGraph, BPE, LBP)
├── instrumentation.json    # requests + redirects + storage_ops (PhishXGraph)
├── screenshot.png          # full-page screenshot (Phishpedia, PhishIntention)
└── crp_result.json         # CRP classifier output (PhishIntention)
```

`phishxgraph.cli collect` writes `page.html` + `instrumentation.json`;
the vision baselines add their own artifacts on first use.

## Reproducing Table 5 (§5.3)

```bash
# 1) One-time: pretrained weights for the vision baselines (~1 GB from
#    Google Drive; CPU baselines train from scratch and need no weights).
bash scripts/download_pretrained.sh all

# 2) Train every CPU baseline on the shared feature cache.
for b in urlnet lbp stackmodel bpe; do
    docker compose run --rm -e MODE=train "$b"
done

# 3) Test every baseline (CPU + GPU) and PhishXGraph itself.
for b in urlnet lbp stackmodel bpe phishpedia phishintention; do
    docker compose run --rm -e MODE=test "$b"
done
docker compose run --rm test       # PhishXGraph
```

Each service writes `data/results/<baseline>/results.csv`.

## Reproducing Table 6 (§5.4) — adversarial robustness

Every baseline reads `FEATURES_DIR` from the environment, so the same
`docker compose run` command also scores adversarial caches by just
overriding that env var:

```bash
# 1) Generate adversarial caches (10 attacks × 6 variants).
docker compose run --rm evasion-generate

# 2) Re-score each baseline against each attack/variant cache.
for attack in A1_url_shortener A2_url_combosquat A3_url_tld_sub \
              A4_html_invisible_link A5_html_object_ratio \
              A6_html_dom_expand A7_html_input_obfuscate \
              A8_logo_masking A9_logo_manipulation A10_logo_font_subst; do
    for variant in v1 v2 v3 v4 v5 v6; do
        for b in urlnet lbp stackmodel bpe phishpedia phishintention; do
            docker compose run --rm \
                -e MODE=test \
                -e FEATURES_DIR=/data/evasion_features/single/$attack/$variant \
                -e OUTPUT_DIR=/data/results/evasion/$attack/$variant/$b \
                "$b"
        done
    done
done

# 3) PhishXGraph scores via its dedicated (faster) path:
docker compose run --rm evasion-score
docker compose run --rm evasion-aggregate
```

## Phishpedia / PhishIntention notes

- Built with **Detectron2 + siamese** stacks; require an NVIDIA GPU.
- `scripts/download_pretrained.sh` pulls the authors' weights from
  Google Drive into `models/{phishpedia,phishintention}/pretrained/`,
  which `docker-compose.yml` bind-mounts read-only into each container.
- Supports both **pretrained-only** (the paper's Table 5 default) and
  **retrained-overlay** modes. The symlink-swap is in
  [`baselines/phishpedia/run.py`](../baselines/phishpedia/run.py)'s
  `_prepare_models_dir`: trained weights in `MODEL_DIR` take precedence
  over the mounted pretrained tree when `MODE=test`.
