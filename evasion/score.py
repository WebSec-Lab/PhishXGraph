"""
Score a trained PhishXGraph model against adversarial feature caches.

Consumes the ``{attack}/{variant}/{hash}/`` layout produced by
:mod:`evasion.generate` (and ``{scenario}/<hash>/`` from
:mod:`evasion.generate_combos`):

    data/evasion_features/single/A1_url_shortener/v1/<hash>/page.html
                                                 /v1/<hash>/instrumentation.json
    data/evasion_features/single/A1_url_shortener/v2/...
    ...
    data/evasion_features/combo/URL_HTML/<hash>/...

Emits a long-format CSV with one row per (attack, variant, url), suitable
for :mod:`evasion.aggregate`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys

# Re-use the library path the same way cli.py does.
if __name__ == "__main__":  # pragma: no cover
    sys.path.insert(0, os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..")))

from phishxgraph.features import extract_all_features  # noqa: E402
from phishxgraph.graph import build_graph, extract_domain  # noqa: E402
from phishxgraph.model import PhishXGraphClassifier  # noqa: E402

log = logging.getLogger("phishxgraph.evasion.score")


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _load(features_dir: str, url_hash: str) -> dict | None:
    d = os.path.join(features_dir, url_hash)
    html_path = os.path.join(d, "page.html")
    meta_path = os.path.join(d, "instrumentation.json")
    if not os.path.isfile(html_path):
        return None
    try:
        html = open(html_path, errors="replace").read()
    except Exception:
        return None
    meta = {"requests": [], "redirects": [], "storage_ops": []}
    if os.path.isfile(meta_path):
        try:
            meta = json.load(open(meta_path))
        except Exception:
            pass
    return {
        "html": html,
        "requests": meta.get("requests", []),
        "redirects": meta.get("redirects", []),
        "storage_ops": meta.get("storage_ops", []),
    }


def _score_one(model: PhishXGraphClassifier, url: str, instr: dict) -> float:
    domain = extract_domain(url)
    G, soup = build_graph(url, instr, domain)
    feats = extract_all_features(G, url, domain, soup)
    _, proba = model.predict([feats])
    return float(proba[0])


def score_attack_dir(model, attack_dir: str, test_rows: list[tuple[str, str]],
                     attack_name: str):
    """Yield one result row per (variant, url) under attack_dir."""
    variants = sorted(
        d for d in os.listdir(attack_dir)
        if os.path.isdir(os.path.join(attack_dir, d))
    )
    for variant in variants:
        vdir = os.path.join(attack_dir, variant)
        for url, label in test_rows:
            h = _url_hash(url)
            instr = _load(vdir, h)
            if instr is None:
                yield {
                    "attack": attack_name, "variant": variant, "url": url,
                    "label": label, "score": "", "prediction": "",
                    "status": "missing",
                }
                continue
            try:
                score = _score_one(model, url, instr)
            except Exception as exc:
                log.warning("score failed for %s/%s/%s: %s",
                            attack_name, variant, url, exc)
                yield {
                    "attack": attack_name, "variant": variant, "url": url,
                    "label": label, "score": "", "prediction": "",
                    "status": f"error:{type(exc).__name__}",
                }
                continue
            pred = "phish" if score >= 0.5 else "benign"
            yield {
                "attack": attack_name, "variant": variant, "url": url,
                "label": label, "score": f"{score:.4f}",
                "prediction": pred, "status": "ok",
            }


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [evasion.score] %(levelname)s: %(message)s",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test", required=True,
                   help="CSV (url,label) of the test set (true labels)")
    p.add_argument("--attacks-dir", required=True,
                   help="Root of {attack}/{variant}/{hash}/ adversarial cache")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--attacks", default="",
                   help="Comma-separated attack folder names to restrict to; "
                        "empty means all subdirs of --attacks-dir")
    args = p.parse_args(argv)

    with open(args.test) as f:
        rows = [(r["url"], r.get("label", "phish"))
                for r in csv.DictReader(f)]
    model = PhishXGraphClassifier.load(args.model_dir)

    subset = set(x.strip() for x in args.attacks.split(",") if x.strip())
    attacks = sorted(
        d for d in os.listdir(args.attacks_dir)
        if os.path.isdir(os.path.join(args.attacks_dir, d))
    )
    if subset:
        attacks = [a for a in attacks if a in subset]
    log.info("scoring %d attacks × ~6 variants × %d urls", len(attacks), len(rows))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        fieldnames = ["attack", "variant", "url", "label", "score",
                      "prediction", "status"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for attack in attacks:
            log.info("attack: %s", attack)
            for row in score_attack_dir(
                model, os.path.join(args.attacks_dir, attack), rows, attack
            ):
                w.writerow(row)
    log.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
