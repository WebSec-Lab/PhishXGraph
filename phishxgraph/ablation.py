"""
Reproduce the §5.5 Ablation study (Table 7).

Two axes:
  1. Feature category ablation: C-only, S-only, B-only, C+S, C+B, S+B, C+S+B
  2. Classifier ablation: XGBoost, Random Forest, SVM (RBF), MLP

Both axes share the same cached feature cache and train/test split.

CLI:
    python -m phishxgraph.ablation \\
        --train    data/input/train.csv \\
        --test     data/input/test.csv \\
        --features-dir data/features \\
        --output   data/results/ablation.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import os

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

from .cli import build_feature_dicts, read_urls
from .features import FEATURE_GROUPS, FEATURE_NAMES
from .model import PhishXGraphClassifier, _make_classifier

log = logging.getLogger("phishxgraph.ablation")


# Paper §4.3 / Table 3 groupings.
CATEGORY_MAP = {
    "C": [  # Content (42)
        "content_url_lexical", "content_url_entropy_token",
        "content_html_phishing_cue", "content_html_content",
    ],
    "S": [  # Structural (43)
        "structural_graph_topology", "structural_dom_shape",
        "structural_centrality_hierarchy",
        "structural_node_type_distribution",
        "structural_edge_type_distribution",
    ],
    "B": [  # Behavioral (16)
        "behavioral_subgraph_path", "behavioral_edge_interaction",
        "behavioral_data_flow",
    ],
}


def _features_for_categories(categories: list[str]) -> list[str]:
    """Return the subset of FEATURE_NAMES covered by the given categories."""
    names: list[str] = []
    for cat in categories:
        for group_key in CATEGORY_MAP[cat]:
            names.extend(FEATURE_GROUPS[group_key])
    return names


def _matrix(feature_dicts, names):
    X = np.array(
        [[fd.get(k, 0) for k in names] for fd in feature_dicts], dtype=float
    )
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _labels_to_y(labels):
    return np.array([
        1 if str(l).lower() in ("phish", "1", "true", "phishing") else 0
        for l in labels
    ])


def evaluate(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }


def run_category_ablation(feats_train, y_train, feats_test, y_test,
                          classifier: str = "xgboost"):
    """Paper Table 7 top half — C/S/B combinations."""
    combos = [
        ("C",     ["C"]),
        ("S",     ["S"]),
        ("B",     ["B"]),
        ("C+S",   ["C", "S"]),
        ("C+B",   ["C", "B"]),
        ("S+B",   ["S", "B"]),
        ("C+S+B", ["C", "S", "B"]),
    ]
    rows = []
    for label, cats in combos:
        names = _features_for_categories(cats)
        X_train = _matrix(feats_train, names)
        X_test = _matrix(feats_test, names)
        clf = _make_classifier(classifier)
        metrics = evaluate(clf, X_train, y_train, X_test, y_test)
        rows.append({
            "axis": "feature_category",
            "config": label,
            "n_features": len(names),
            "classifier": classifier,
            **metrics,
        })
        log.info(
            "FEAT %-6s (%3d feats, %s): acc=%.4f p=%.4f r=%.4f f1=%.4f",
            label, len(names), classifier,
            metrics["accuracy"], metrics["precision"],
            metrics["recall"], metrics["f1"],
        )
    return rows


def run_classifier_ablation(feats_train, y_train, feats_test, y_test):
    """Paper Table 7 bottom half — classifier comparison on all 101 features."""
    names = FEATURE_NAMES
    X_train = _matrix(feats_train, names)
    X_test = _matrix(feats_test, names)
    rows = []
    for clf_name in ["random_forest", "svm", "mlp", "xgboost"]:
        try:
            clf = _make_classifier(clf_name)
        except RuntimeError as exc:
            log.warning("skipping %s: %s", clf_name, exc)
            continue
        metrics = evaluate(clf, X_train, y_train, X_test, y_test)
        rows.append({
            "axis": "classifier",
            "config": clf_name,
            "n_features": len(names),
            "classifier": clf_name,
            **metrics,
        })
        log.info(
            "CLF  %-15s: acc=%.4f p=%.4f r=%.4f f1=%.4f",
            clf_name,
            metrics["accuracy"], metrics["precision"],
            metrics["recall"], metrics["f1"],
        )
    return rows


def write_csv(rows, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        fieldnames = ["axis", "config", "n_features", "classifier",
                      "accuracy", "precision", "recall", "f1"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})
    log.info("wrote %s (%d rows)", path, len(rows))


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [ablation] %(levelname)s: %(message)s",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True, help="training CSV (url,label)")
    p.add_argument("--test", required=True, help="test CSV (url,label)")
    p.add_argument("--features-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--classifier", default="xgboost",
        choices=["xgboost", "random_forest", "svm", "mlp"],
        help="Classifier for the feature-category axis (paper: xgboost)",
    )
    p.add_argument(
        "--skip-category", action="store_true",
        help="Skip the C/S/B feature-category sweep",
    )
    p.add_argument(
        "--skip-classifier", action="store_true",
        help="Skip the classifier sweep",
    )
    args = p.parse_args(argv)

    urls_train, labels_train = read_urls(args.train)
    urls_test, labels_test = read_urls(args.test)
    log.info("train=%d test=%d", len(urls_train), len(urls_test))

    feats_train = build_feature_dicts(urls_train, args.features_dir)
    feats_test = build_feature_dicts(urls_test, args.features_dir)
    y_train = _labels_to_y(labels_train)
    y_test = _labels_to_y(labels_test)

    rows = []
    if not args.skip_category:
        rows += run_category_ablation(
            feats_train, y_train, feats_test, y_test,
            classifier=args.classifier,
        )
    if not args.skip_classifier:
        rows += run_classifier_ablation(
            feats_train, y_train, feats_test, y_test,
        )

    write_csv(rows, args.output)


if __name__ == "__main__":
    main()
