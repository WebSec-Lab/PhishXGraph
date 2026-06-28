"""
Command-line entry points.

Examples
--------
    # Crawl 10 URLs into /data/features/
    python -m phishxgraph.cli collect --input urls.csv --features-dir ./features

    # Train XGBoost classifier on cached features
    python -m phishxgraph.cli train --input urls.csv \\
        --features-dir ./features --model-dir ./model

    # Score URLs with a trained model
    python -m phishxgraph.cli test --input test_urls.csv \\
        --features-dir ./features --model-dir ./model --output results.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys

from .collect import (
    DEFAULT_TIMEOUT_SEC,
    collect_urls,
    load_cached_instrumentation,
)
from .features import extract_all_features
from .graph import build_graph, extract_domain
from .model import PhishXGraphClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [phishxgraph] %(levelname)s: %(message)s",
)
log = logging.getLogger("phishxgraph.cli")


def read_urls(csv_path: str):
    urls, labels = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        if "url" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} must have a 'url' column")
        for row in reader:
            urls.append(row["url"])
            labels.append(row.get("label", "unknown"))
    return urls, labels


def build_feature_dicts(urls, features_dir: str):
    feats = []
    cached, missing = 0, 0
    for i, url in enumerate(urls):
        instr = load_cached_instrumentation(features_dir, url)
        if instr is None:
            instr = {"html": "", "requests": [],
                     "redirects": [], "storage_ops": []}
            missing += 1
        else:
            cached += 1
        domain = extract_domain(url)
        G, soup = build_graph(url, instr, domain)
        feats.append(extract_all_features(G, url, domain, soup))
        if (i + 1) % 25 == 0:
            log.info("features: %d/%d", i + 1, len(urls))
    log.info("features: %d cached, %d missing", cached, missing)
    return feats


def cmd_collect(args):
    urls, _ = read_urls(args.input)
    if args.limit:
        urls = urls[: args.limit]
    stats = collect_urls(
        urls, features_dir=args.features_dir,
        timeout_sec=args.timeout,
        headless=not args.headful,
    )
    log.info("collect done: %s", stats)


def cmd_train(args):
    urls, labels = read_urls(args.input)
    feats = build_feature_dicts(urls, args.features_dir)
    clf = PhishXGraphClassifier(classifier=args.classifier)
    metrics = clf.fit(feats, labels)
    clf.save(args.model_dir)
    log.info("train metrics: %s", metrics)
    log.info("model saved to %s", args.model_dir)


def cmd_test(args):
    urls, labels = read_urls(args.input)
    feats = build_feature_dicts(urls, args.features_dir)
    clf = PhishXGraphClassifier.load(args.model_dir)
    y_pred, proba = clf.predict(feats)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["url", "label", "prediction", "score"])
        for url, lbl, p, s in zip(urls, labels, y_pred, proba):
            w.writerow([url, lbl, "phish" if p == 1 else "benign",
                        f"{float(s):.4f}"])
    log.info("scored %d URLs → %s", len(urls), args.output)

    # Quick confusion summary if labels given
    known = [(l, p) for l, p in zip(labels, y_pred)
             if str(l).lower() in ("phish", "benign", "0", "1")]
    if known:
        tp = sum(1 for l, p in known if str(l).lower() == "phish" and p == 1)
        fp = sum(1 for l, p in known if str(l).lower() == "benign" and p == 1)
        tn = sum(1 for l, p in known if str(l).lower() == "benign" and p == 0)
        fn = sum(1 for l, p in known if str(l).lower() == "phish" and p == 0)
        acc = (tp + tn) / max(len(known), 1)
        log.info("quick eval: tp=%d fp=%d tn=%d fn=%d acc=%.4f",
                 tp, fp, tn, fn, acc)


def build_parser():
    p = argparse.ArgumentParser(
        prog="phishxgraph",
        description="PhishXGraph reference CLI (paper §4).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="Playwright+CDP instrumentation")
    c.add_argument("--input", required=True,
                   help="CSV with a 'url' column")
    c.add_argument("--features-dir", required=True)
    c.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SEC,
                   help="Per-page timeout in seconds (paper default: 30)")
    c.add_argument("--headful", action="store_true",
                   help="Run browser in headful mode (debugging)")
    c.add_argument("--limit", type=int, default=0)
    c.set_defaults(func=cmd_collect)

    t = sub.add_parser("train", help="Train classifier from cached features")
    t.add_argument("--input", required=True)
    t.add_argument("--features-dir", required=True)
    t.add_argument("--model-dir", required=True)
    t.add_argument("--classifier", default="xgboost",
                   choices=["xgboost", "random_forest", "svm", "mlp"],
                   help="Paper default: xgboost (§4.4)")
    t.set_defaults(func=cmd_train)

    s = sub.add_parser("test", help="Score URLs with a trained model")
    s.add_argument("--input", required=True)
    s.add_argument("--features-dir", required=True)
    s.add_argument("--model-dir", required=True)
    s.add_argument("--output", required=True)
    s.set_defaults(func=cmd_test)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
