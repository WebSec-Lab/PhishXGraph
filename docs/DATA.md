# Data Release Notes

## What we include

| Artifact | Path | Size | Why |
|---|---|---|---|
| URL samples | `data/samples/input/smoke_{train,test}.csv` | < 10 KB | Smoke test inputs |
| Feature-group schema | `phishxgraph/features.py` | 25 KB | Canonical 101-feature list |
| Paper→code mapping | `docs/PAPER_MAPPING.md` | 4 KB | Review aid |

## What we do **not** include, and why

| Not included | Why | How to obtain |
|---|---|---|
| Full 53 K phishing + 53 K benign training set | Phishing pages contain live credential-harvesting payloads; sharing is unsafe | Use the URL-list release (URLs only, no DOM) from PhishTank / OpenPhish |
| Trained model checkpoints (.pkl) | Embeds characteristics of the above corpus | Rebuild with `docker compose run --rm train` |
| Adversarial perturbed HTML / screenshots | Same privacy concern as raw phishing data | Regenerate with `python -m evasion.generate` |
| Pre-trained weights for Phishpedia / PhishIntention | Third-party weights (~1 GB on Google Drive) | Run `bash scripts/download_pretrained.sh all` |
| Full 15-day CertStream capture | Contains hostnames that were later taken down | Re-run `phishxgraph.realtime` with `--duration` |

## Reproducing paper-scale experiments

1. Collect phishing URLs from PhishTank ([phishtank.org](https://www.phishtank.com/))
   and OpenPhish ([openphish.com](https://openphish.com/)) (§5.1).
2. Collect benign URLs from [Tranco](https://tranco-list.eu/).
3. Materialise `data/input/urls.csv` with columns `url,label`.
4. Run `docker compose run --rm collect`.
5. Train (`docker compose run --rm train`) and test (`docker compose run --rm test`) as usual.