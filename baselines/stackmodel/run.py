#!/usr/bin/env python
"""
StackModel wrapper: reads phishinglist.csv, fetches HTML, extracts features.
Supports MODE=train (train XGBoost model) and MODE=test (inference).
"""
import csv
import os
import sys
import subprocess
import tempfile
import time
import shutil

INPUT_CSV = os.environ.get("INPUT_CSV", "/data/input/phishinglist.csv")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/data/results/stackmodel")
MODEL_DIR = os.environ.get("MODEL_DIR", "/data/models/stackmodel")
FEATURES_DIR = os.environ.get("FEATURES_DIR", "/data/features")
MODE = os.environ.get("MODE", "test")
STACKMODEL_DIR = "/app/repo/StackModel"
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(MODEL_DIR, "model.pkl"))

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("[StackModel] Missing dependencies", file=sys.stderr)
    sys.exit(1)


def read_urls(csv_path):
    """Read URLs and labels from phishinglist.csv"""
    urls, labels = [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            urls.append(row["url"])
            labels.append(row.get("label", "unknown"))
    return urls, labels


def fetch_html(url, timeout=10):
    """Fetch HTML content from a URL"""
    try:
        resp = requests.get(url, timeout=timeout, verify=False,
                           headers={"User-Agent": "Mozilla/5.0"})
        return resp.text
    except Exception as e:
        print("[StackModel] Failed to fetch {}: {}".format(url, e))
        return ""


import hashlib as _hashlib

HTML_CACHE_DIR = os.path.join(MODEL_DIR, "html_cache")


def _url_hash(url):
    """Hash matching feature_collector's convention."""
    return _hashlib.sha256(url.encode()).hexdigest()[:16]


def _load_from_shared(url, filename):
    """Try loading from shared feature cache."""
    h = _url_hash(url)
    shared_path = os.path.join(FEATURES_DIR, h, filename)
    if os.path.isfile(shared_path):
        return shared_path
    return None


def _append_collected_url(url):
    """Append a single URL to collected_urls.csv after caching."""
    import csv as _csv_mod
    from datetime import datetime as _dt
    log_path = os.path.join(MODEL_DIR, "collected_urls.csv")
    write_header = not os.path.isfile(log_path)
    with open(log_path, "a", newline="") as f:
        w = _csv_mod.writer(f)
        if write_header:
            w.writerow(["url", "cached_at", "status", "cache_path"])
        cache_file = _url_cache_path(url)
        status = "ok" if os.path.isfile(cache_file) else "fail"
        ts = _dt.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        w.writerow([url, ts, status, cache_file if status == "ok" else ""])


def _url_cache_path(url):
    h = _hashlib.sha256(url.encode()).hexdigest()[:16]
    return os.path.join(HTML_CACHE_DIR, "{}.html".format(h))


def _load_cached_html(url):
    # Check shared features directory first
    shared_path = _load_from_shared(url, "page.html")
    if shared_path is not None:
        try:
            with open(shared_path, "r", errors="replace") as f:
                return f.read()
        except Exception:
            pass

    # Fall back to legacy model-specific cache
    path = _url_cache_path(url)
    if os.path.isfile(path):
        try:
            with open(path, "r", errors="replace") as f:
                return f.read()
        except Exception:
            pass
    return None


def _save_cached_html(url, html):
    os.makedirs(HTML_CACHE_DIR, exist_ok=True)
    dest = _url_cache_path(url)
    tmp = dest + ".tmp"
    with open(tmp, "w") as f:
        f.write(html)
    os.replace(tmp, dest)


def _count_ok():
    """Count successful entries in collected_urls.csv."""
    import csv as _csv_mod
    log_path = os.path.join(MODEL_DIR, "collected_urls.csv")
    if not os.path.isfile(log_path):
        return 0
    count = 0
    with open(log_path, "r") as f:
        for row in _csv_mod.DictReader(f):
            if row.get("status") == "ok":
                count += 1
    return count


def collect_html(urls):
    """Fetch and cache HTML for URLs (MODE=collect)."""
    max_collect = int(os.environ.get("MAX_COLLECT", "0"))
    ok_count = _count_ok()
    if max_collect > 0 and ok_count >= max_collect:
        print("[StackModel] Collection cap reached ({}/{})".format(ok_count, max_collect))
        return

    to_fetch = [u for u in urls if _load_cached_html(u) is None]
    cached = len(urls) - len(to_fetch)
    if cached > 0:
        print("[StackModel] Already cached: {}".format(cached))
    if not to_fetch:
        print("[StackModel] All URLs already cached.")
        return

    if max_collect > 0:
        to_fetch = to_fetch[:max_collect - ok_count]

    print("[StackModel] Fetching HTML for {} new URLs...".format(len(to_fetch)))
    for i, url in enumerate(to_fetch):
        html = fetch_html(url)
        _save_cached_html(url, html)
        _append_collected_url(url)
        if (i + 1) % 50 == 0:
            print("[StackModel] Cached {}/{}".format(i + 1, len(to_fetch)))
    print("[StackModel] Cached {} HTML files".format(len(to_fetch)))


def prepare_data_folder(urls, labels, data_dir):
    """Create folder structure expected by StackModel test.py/train.py.
    Uses only pre-collected data; no live fetching in train/test mode."""
    cached_count = 0
    skipped_count = 0
    for i, (url, label) in enumerate(zip(urls, labels)):
        site_dir = os.path.join(data_dir, "site_{}".format(i))
        os.makedirs(site_dir, exist_ok=True)

        with open(os.path.join(site_dir, "info.txt"), "w") as f:
            f.write(url + "\n")

        html = _load_cached_html(url)
        if html is not None:
            cached_count += 1
        else:
            html = ""
            skipped_count += 1
        with open(os.path.join(site_dir, "html.txt"), "w") as f:
            f.write(html)

        with open(os.path.join(site_dir, "label.txt"), "w") as f:
            f.write(label + "\n")
    print("[StackModel] Used {} cached HTML files, {} skipped (no cache)".format(
        cached_count, skipped_count))


def train_stackmodel(data_dir, model_dir):
    """Train StackModel: extract features via test.py, then train XGBoost.

    Original train.py expects hardcoded pickle files and is not usable.
    Instead, we use test.py's feature extraction pipeline + our own XGBoost fit.
    """
    os.makedirs(model_dir, exist_ok=True)

    # Step 1: Extract features using test.py's feature extraction (no prediction)
    feature_dir = os.path.join(model_dir, "_train_features")
    os.makedirs(feature_dir, exist_ok=True)
    cmd = [
        sys.executable, os.path.join(STACKMODEL_DIR, "test.py"),
        "-f", data_dir,
        "-o", feature_dir,
    ]
    # Run without -md flag so it extracts features but skips prediction
    print("[StackModel] Extracting features: {}".format(" ".join(cmd)))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=STACKMODEL_DIR)
    print(result.stdout.decode('utf-8', errors='replace'))

    # Step 2: Load features and labels, train XGBoost
    feature_csv = os.path.join(feature_dir, "feature.csv")
    if not os.path.isfile(feature_csv):
        print("[StackModel] Feature extraction failed - no feature.csv produced")
        if result.returncode != 0:
            print("[StackModel] STDERR: {}".format(result.stderr.decode('utf-8', errors='replace')), file=sys.stderr)
        return 1

    try:
        import pandas as pd
        import pickle as _pkl

        df = pd.read_csv(feature_csv, index_col=0)
        print("[StackModel] Features: {} samples x {} features".format(*df.shape))

        # Build URL→label map from site folders
        url_to_label = {}
        for site in os.listdir(data_dir):
            site_path = os.path.join(data_dir, site)
            if not os.path.isdir(site_path):
                continue
            info_file = os.path.join(site_path, "info.txt")
            label_file = os.path.join(site_path, "label.txt")
            if os.path.isfile(info_file) and os.path.isfile(label_file):
                with open(info_file) as f:
                    site_url = f.read().strip()
                with open(label_file) as f:
                    lab = f.read().strip()
                url_to_label[site_url] = 1 if lab == "phish" else 0

        # Match feature.csv index (may have trailing \n) to labels
        labels = []
        matched = 0
        for idx in df.index:
            clean_idx = str(idx).strip()
            if clean_idx in url_to_label:
                labels.append(url_to_label[clean_idx])
                matched += 1
            else:
                labels.append(0)
        print("[StackModel] Label matching: {}/{} URLs matched".format(
            matched, len(df)))

        y = pd.Series(labels, index=df.index)
        X = df.fillna(0)

        # Train XGBoost
        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            use_label_encoder=False, eval_metric='logloss')
        model.fit(X, y)
        acc = (model.predict(X) == y).mean()
        print("[StackModel] Train accuracy: {:.4f}".format(acc))

        # Save model
        with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
            _pkl.dump(model, f)
        print("[StackModel] Model saved to {}".format(model_dir))
        return 0

    except Exception as e:
        print("[StackModel] Training error: {}".format(e), file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def run_stackmodel(data_dir, output_dir):
    """Run StackModel test.py"""
    cmd = [
        sys.executable, os.path.join(STACKMODEL_DIR, "test.py"),
        "-f", data_dir,
        "-o", output_dir,
    ]

    # Add model directory if exists (test.py -md expects directory, not file)
    if os.path.exists(MODEL_DIR):
        cmd.extend(["-md", MODEL_DIR])

    print("[StackModel] Running: {}".format(" ".join(cmd)))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=STACKMODEL_DIR)
    print(result.stdout.decode('utf-8', errors='replace'))
    if result.returncode != 0:
        print("[StackModel] STDERR: {}".format(result.stderr.decode('utf-8', errors='replace')), file=sys.stderr)
    return result.returncode


def write_output(predictions, output_path):
    """Write standardized results CSV"""
    os.makedirs(output_path, exist_ok=True)
    csv_path = os.path.join(output_path, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "prediction", "score"])
        for url, pred, score in predictions:
            writer.writerow([url, pred, score])
    print("[StackModel] Results written to {}".format(csv_path))


def main():
    print("[StackModel] Starting (mode={})...".format(MODE))

    if MODE == "collect":
        print("[StackModel] Collect mode: feature-collector service handles extraction. Sleeping.")
        while True:
            time.sleep(3600)

    if not os.path.exists(INPUT_CSV):
        print("[StackModel] ERROR: Input file not found: {}".format(INPUT_CSV))
        sys.exit(1)

    urls, labels = read_urls(INPUT_CSV)
    print("[StackModel] Loaded {} URLs".format(len(urls)))

    if MODE == "train":
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "sites")
            os.makedirs(data_dir, exist_ok=True)
            print("[StackModel] Preparing HTML for {} URLs...".format(len(urls)))
            prepare_data_folder(urls, labels, data_dir)
            ret = train_stackmodel(data_dir, MODEL_DIR)
            if ret == 0:
                print("[StackModel] Training complete. Model saved to {}".format(MODEL_DIR))
            else:
                print("[StackModel] Training failed.")
                sys.exit(1)
        return

    # Test mode — extract features with test.py (no -md), predict with our model
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "sites")
        feature_dir = os.path.join(tmpdir, "features")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(feature_dir, exist_ok=True)

        print("[StackModel] Preparing HTML for {} URLs...".format(len(urls)))
        prepare_data_folder(urls, labels, data_dir)

        # Step 1: Extract features only (no -md flag → no prediction)
        cmd = [
            sys.executable, os.path.join(STACKMODEL_DIR, "test.py"),
            "-f", data_dir,
            "-o", feature_dir,
        ]
        print("[StackModel] Extracting features: {}".format(" ".join(cmd)))
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                cwd=STACKMODEL_DIR)
        print(result.stdout.decode('utf-8', errors='replace'))

        # Step 2: Load features and predict with our trained model
        feature_csv = os.path.join(feature_dir, "feature.csv")
        model_path = os.path.join(MODEL_DIR, "model.pkl")
        predictions = []

        if os.path.isfile(feature_csv) and os.path.isfile(model_path):
            try:
                import pandas as pd
                import pickle as _pkl

                df = pd.read_csv(feature_csv, index_col=0)
                X = df.fillna(0)
                print("[StackModel] Test features: {} samples x {} features".format(*X.shape))

                with open(model_path, "rb") as f:
                    model = _pkl.load(f)

                y_pred = model.predict(X)
                y_prob = model.predict_proba(X)[:, 1]

                for url_idx, pred_val, prob_val in zip(X.index, y_pred, y_prob):
                    pred = "phish" if pred_val == 1 else "benign"
                    predictions.append((str(url_idx).strip(), pred, float(prob_val)))

                n_pos = sum(1 for p in y_pred if p == 1)
                print("[StackModel] Predicted: {} phish, {} benign".format(
                    n_pos, len(y_pred) - n_pos))

            except Exception as e:
                print("[StackModel] Prediction error: {}".format(e), file=sys.stderr)
                import traceback
                traceback.print_exc()
                predictions = [(url, "unknown", 0.0) for url in urls]
        else:
            if not os.path.isfile(feature_csv):
                print("[StackModel] Feature extraction failed - no feature.csv")
            if not os.path.isfile(model_path):
                print("[StackModel] No trained model at {}".format(model_path))
            predictions = [(url, "unknown", 0.0) for url in urls]

    write_output(predictions, OUTPUT_DIR)
    print("[StackModel] Done.")


if __name__ == "__main__":
    main()
