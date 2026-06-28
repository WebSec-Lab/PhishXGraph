#!/usr/bin/env python3
"""
PhishIntention wrapper: reads phishinglist.csv, captures screenshots, runs PhishIntention.
"""
import csv
import json
import os
import sys
import subprocess
import time

sys.path.insert(0, "/app")

INPUT_CSV = os.environ.get("INPUT_CSV", "/data/input/phishinglist.csv")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/data/results/phishintention")
MODEL_DIR = os.environ.get("MODEL_DIR", "/data/models/phishintention")
FEATURES_DIR = os.environ.get("FEATURES_DIR", "/data/features")
MODE = os.environ.get("MODE", "test")
PHISHINTENTION_DIR = "/app/phishintention"


import hashlib as _hashlib

SCREENSHOT_CACHE_DIR = os.path.join(MODEL_DIR, "screenshot_cache")


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


def read_urls(csv_path):
    """Read URLs and labels from phishinglist.csv"""
    urls, labels = [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            urls.append(row["url"])
            labels.append(row.get("label", "unknown"))
    return urls, labels


def _url_cache_path(url):
    h = _hashlib.sha256(url.encode()).hexdigest()[:16]
    return os.path.join(SCREENSHOT_CACHE_DIR, "{}.png".format(h))


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


def _read_collected_status():
    """Read collected_urls.csv and return (attempted set, ok count)."""
    import csv as _csv_mod
    log_path = os.path.join(MODEL_DIR, "collected_urls.csv")
    attempted = set()
    ok_count = 0
    if not os.path.isfile(log_path):
        return attempted, ok_count
    with open(log_path, "r") as f:
        for row in _csv_mod.DictReader(f):
            attempted.add(row.get("url", ""))
            if row.get("status") == "ok":
                ok_count += 1
    return attempted, ok_count


def _reset_stale_failures():
    """Remove fail entries from collected_urls.csv so failed URLs can be retried."""
    import csv as _csv_mod
    log_path = os.path.join(MODEL_DIR, "collected_urls.csv")
    if not os.path.isfile(log_path):
        return 0
    rows = []
    removed = 0
    with open(log_path, "r") as f:
        for row in _csv_mod.DictReader(f):
            if row.get("status") == "ok":
                rows.append(row)
            else:
                removed += 1
    if removed > 0:
        with open(log_path, "w", newline="") as f:
            w = _csv_mod.DictWriter(f, fieldnames=["url", "cached_at", "status", "cache_path"])
            w.writeheader()
            w.writerows(rows)
    return removed


def collect_screenshots(urls):
    """Capture and cache screenshots for URLs (MODE=collect)."""
    from screenshot import capture_screenshot

    os.makedirs(SCREENSHOT_CACHE_DIR, exist_ok=True)
    max_collect = int(os.environ.get("MAX_COLLECT", "0"))

    attempted, ok_count = _read_collected_status()
    if max_collect > 0 and ok_count >= max_collect:
        print("[PhishIntention] Collection cap reached ({}/{})".format(ok_count, max_collect))
        return

    to_capture = [u for u in urls if u not in attempted]
    to_capture.reverse()

    if max_collect > 0:
        to_capture = to_capture[:max_collect - ok_count]

    cached = len(attempted)
    if cached > 0:
        print("[PhishIntention] Already attempted: {} ({} ok)".format(cached, ok_count))
    if not to_capture:
        print("[PhishIntention] No new URLs to capture.")
        return
    print("[PhishIntention] Capturing screenshots for {} new URLs...".format(len(to_capture)))
    for i, url in enumerate(to_capture):
        capture_screenshot(url, _url_cache_path(url))
        _append_collected_url(url)
        if (i + 1) % 10 == 0:
            print("[PhishIntention] Captured {}/{}".format(i + 1, len(to_capture)))
    print("[PhishIntention] Cached {} screenshots".format(len(to_capture)))


def prepare_test_folder(urls, test_dir):
    """Create folder structure expected by PhishIntention.
    Uses only pre-collected data from shared feature cache or model screenshot cache.
    URLs without cached data are skipped (will produce 'unknown' predictions)."""
    import shutil

    cached_count = 0
    skipped_count = 0
    for i, url in enumerate(urls):
        site_dir = os.path.join(test_dir, "site_{}".format(i))
        os.makedirs(site_dir, exist_ok=True)

        with open(os.path.join(site_dir, "info.txt"), "w") as f:
            f.write(url + "\n")

        # Screenshot — only from pre-collected cache, no live capture
        shot_path = os.path.join(site_dir, "shot.png")
        shared_path = _load_from_shared(url, "screenshot.png")
        cache_path = _url_cache_path(url)
        if shared_path is not None:
            shutil.copy2(shared_path, shot_path)
            cached_count += 1
        elif os.path.isfile(cache_path):
            shutil.copy2(cache_path, shot_path)
            cached_count += 1
        else:
            skipped_count += 1

        # HTML — only from pre-collected cache, no live fetch
        html_path = os.path.join(site_dir, "html.txt")
        shared_html = _load_from_shared(url, "page.html")
        if shared_html is not None:
            shutil.copy2(shared_html, html_path)
        elif not os.path.isfile(html_path):
            with open(html_path, "w") as f:
                f.write("")

        # CRP screenshot — cached from feature collection
        crp_shot = _load_from_shared(url, "crp_screenshot.png")
        crp_result = _load_from_shared(url, "crp_result.json")
        if crp_shot is not None:
            shutil.copy2(crp_shot, os.path.join(site_dir, "crp_shot.png"))
        if crp_result is not None:
            shutil.copy2(crp_result, os.path.join(site_dir, "crp_result.json"))

    print("[PhishIntention] Used {} cached screenshots, {} skipped (no cache)".format(
        cached_count, skipped_count))


def run_phishintention(test_dir, output_file):
    """Run PhishIntention detection."""
    cmd = [
        "python3", os.path.join(PHISHINTENTION_DIR, "phishintention.py"),
        "--folder", test_dir,
        "--output_fn", output_file,
    ]

    print("[PhishIntention] Running: {}".format(" ".join(cmd)))
    result = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=PHISHINTENTION_DIR)

    print(result.stdout)
    if result.returncode != 0:
        print("[PhishIntention] STDERR: {}".format(result.stderr), file=sys.stderr)
    return result.returncode


def parse_phishintention_output(output_file, urls):
    """Parse PhishIntention JSON output.

    Output format is a list of dicts: [{folder, url, phish, ...}, ...]
    or a dict keyed by folder name (older format).
    """
    predictions = []

    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                results = json.load(f)

            # Build folder→entry map from either list or dict format
            parsed = {}
            if isinstance(results, list):
                for entry in results:
                    folder = entry.get("folder", "")
                    parsed[folder] = entry
            elif isinstance(results, dict):
                parsed = results

            for i, url in enumerate(urls):
                folder = "site_{}".format(i)
                if folder in parsed:
                    entry = parsed[folder]
                    # Support both formats: phish_category (string) and phish (int)
                    cat = entry.get("phish_category", "")
                    if cat:
                        pred = "phish" if cat not in ("benign", "legitimate") else "benign"
                    else:
                        pred = "phish" if entry.get("phish", 0) == 1 else "benign"
                    score = entry.get("siamese_conf", entry.get("score", entry.get("confidence", 0.0)))
                    if score is None:
                        score = 0.0
                    predictions.append((url, pred, float(score)))
                else:
                    predictions.append((url, "unknown", 0.0))
        except (json.JSONDecodeError, KeyError) as e:
            print("[PhishIntention] Error parsing output: {}".format(e))
            predictions = [(url, "unknown", 0.0) for url in urls]
    else:
        predictions = [(url, "unknown", 0.0) for url in urls]

    return predictions


def write_output(predictions, output_path):
    """Write standardized results CSV."""
    os.makedirs(output_path, exist_ok=True)
    csv_path = os.path.join(output_path, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "prediction", "score"])
        for url, pred, score in predictions:
            writer.writerow([url, pred, score])
    print("[PhishIntention] Results written to {}".format(csv_path))


def train_phishintention(urls, labels):
    """Train: fine-tune PhishIntention with labeled screenshots.

    Training components (paper: USENIX Security 2022):
    1. Shared: Logo Detector + Siamese Network (same as Phishpedia)
    2. AWL Layout Classifier - credential-requiring page detection
    3. CRP Transition Classifier - page transition analysis

    Uses only pre-collected data; URLs without cached data are skipped.
    """
    import random
    import shutil
    import tempfile

    # Subsample to avoid OOM during logo detection
    MAX_PER_LABEL = int(os.environ.get("PHISHINTENTION_MAX_PER_LABEL", "5000"))
    if len(urls) > 2 * MAX_PER_LABEL:
        rng = random.Random(42)
        phish_idx = [i for i, l in enumerate(labels) if l == "phish"]
        benign_idx = [i for i, l in enumerate(labels) if l == "benign"]
        rng.shuffle(phish_idx)
        rng.shuffle(benign_idx)
        keep = set(phish_idx[:MAX_PER_LABEL] + benign_idx[:MAX_PER_LABEL])
        urls = [u for i, u in enumerate(urls) if i in keep]
        labels = [l for i, l in enumerate(labels) if i in keep]
        print("[PhishIntention] Subsampled to {} URLs ({} per label)".format(
            len(urls), MAX_PER_LABEL))

    print("[PhishIntention] Training with {} labeled URLs...".format(len(urls)))

    with tempfile.TemporaryDirectory() as tmpdir:
        train_dir = os.path.join(tmpdir, "train_sites")
        os.makedirs(train_dir, exist_ok=True)

        cached_count = 0
        skipped_count = 0
        for i, (url, label) in enumerate(zip(urls, labels)):
            site_dir = os.path.join(train_dir, "site_{}".format(i))
            os.makedirs(site_dir, exist_ok=True)
            with open(os.path.join(site_dir, "info.txt"), "w") as f:
                f.write(url + "\n")
            with open(os.path.join(site_dir, "label.txt"), "w") as f:
                f.write(label + "\n")
            shot_path = os.path.join(site_dir, "shot.png")
            shared_path = _load_from_shared(url, "screenshot.png")
            cache_path = _url_cache_path(url)
            if shared_path is not None:
                shutil.copy2(shared_path, shot_path)
                cached_count += 1
            elif os.path.isfile(cache_path):
                shutil.copy2(cache_path, shot_path)
                cached_count += 1
            else:
                skipped_count += 1

            # HTML — only from pre-collected cache
            html_path = os.path.join(site_dir, "html.txt")
            shared_html = _load_from_shared(url, "page.html")
            if shared_html is not None:
                shutil.copy2(shared_html, html_path)
            elif not os.path.isfile(html_path):
                with open(html_path, "w") as f:
                    f.write("")

            # CRP data — from pre-collected cache
            crp_shot_src = _load_from_shared(url, "crp_screenshot.png")
            crp_result_src = _load_from_shared(url, "crp_result.json")
            if crp_shot_src is not None:
                shutil.copy2(crp_shot_src, os.path.join(site_dir, "crp_shot.png"))
            if crp_result_src is not None:
                shutil.copy2(crp_result_src, os.path.join(site_dir, "crp_result.json"))

        print("[PhishIntention] Used {} cached, {} skipped (no cache)".format(
            cached_count, skipped_count))

        # Run PhishIntention training (shared + AWL + CRP)
        train_script = os.path.join(PHISHINTENTION_DIR, "train.py")
        if os.path.exists(train_script):
            cmd = [
                "python3", train_script,
                "--folder", train_dir,
                "--output", MODEL_DIR,
            ]
            print("[PhishIntention] Running training: {}".format(" ".join(cmd)))
            sys.stdout.flush()
            # Stream output live so crashes (OOM etc.) are visible immediately
            result = subprocess.run(cmd, cwd=PHISHINTENTION_DIR)
            if result.returncode != 0:
                print("[PhishIntention] Training script failed (returncode={}), saving reference data only".format(result.returncode))
        else:
            print("[PhishIntention] No train.py found, saving reference data for model")

        # Save labeled data as model reference (fallback / backup)
        os.makedirs(MODEL_DIR, exist_ok=True)
        ref_dir = os.path.join(MODEL_DIR, "reference_data")
        if os.path.exists(ref_dir):
            shutil.rmtree(ref_dir)
        shutil.copytree(train_dir, ref_dir)
        print("[PhishIntention] Training data saved to {}".format(MODEL_DIR))


def _epoch_reset():
    """Clear collected_urls.csv and screenshot cache for a fresh epoch."""
    import shutil
    log_path = os.path.join(MODEL_DIR, "collected_urls.csv")
    removed = 0
    if os.path.isfile(log_path):
        os.remove(log_path)
        removed += 1
    if os.path.isdir(SCREENSHOT_CACHE_DIR):
        count = sum(len(f) for _, _, f in os.walk(SCREENSHOT_CACHE_DIR))
        shutil.rmtree(SCREENSHOT_CACHE_DIR)
        removed += count
    return removed


def _prepare_models_dir():
    """Populate /app/phishintention/models from pretrained source, then
    overlay trained weights from MODEL_DIR when running in test mode.

    Mirrors the Phishpedia wrapper: the pretrained tree is mounted read-only
    at /app/phishintention/_pretrained_src and we symlink entries into
    /app/phishintention/models (which configs/configs.yaml references via
    relative paths). In test mode, trained weights from MODEL_DIR override
    the corresponding symlinks.
    """
    import shutil

    pretrained_src = "/app/phishintention/_pretrained_src"
    models_dst = os.path.join(PHISHINTENTION_DIR, "models")

    if not os.path.isdir(pretrained_src):
        print("[PhishIntention] WARNING: pretrained source not found at {}".format(
            pretrained_src))
        return

    if os.path.islink(models_dst) or os.path.isfile(models_dst):
        os.remove(models_dst)
    elif os.path.isdir(models_dst):
        shutil.rmtree(models_dst)
    os.makedirs(models_dst, exist_ok=True)

    for entry in os.listdir(pretrained_src):
        src = os.path.join(pretrained_src, entry)
        dst = os.path.join(models_dst, entry)
        os.symlink(src, dst)

    # Clear stale logo feature cache so siamese changes take effect
    for cache_name in ("LOGO_FEATS.npy", "LOGO_FILES.npy"):
        cache_path = os.path.join(PHISHINTENTION_DIR, cache_name)
        if os.path.isfile(cache_path):
            os.remove(cache_path)

    if MODE != "test" or not MODEL_DIR or not os.path.isdir(MODEL_DIR):
        print("[PhishIntention] Using pretrained models only")
        return

    overrides = {
        "awl_detector.pth": "layout_detector.pth",
        "crp_classifier.pth": "crp_classifier.pth.tar",
        "siamese_pedia.pth": "ocr_siamese.pth.tar",
    }
    applied = []
    for trained_name, pretrained_name in overrides.items():
        trained_path = os.path.join(MODEL_DIR, trained_name)
        if not os.path.isfile(trained_path):
            continue
        dst = os.path.join(models_dst, pretrained_name)
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(trained_path, dst)
        applied.append(trained_name)

    if applied:
        print("[PhishIntention] Using trained weights from {}: {}".format(
            MODEL_DIR, ", ".join(applied)))
    else:
        print("[PhishIntention] No trained weights found in {}, using pretrained".format(
            MODEL_DIR))


def main():
    print("[PhishIntention] Starting (mode={})...".format(MODE))

    if MODE == "collect":
        print("[PhishIntention] Collect mode: feature-collector service handles extraction. Sleeping.")
        while True:
            time.sleep(3600)

    if not os.path.exists(INPUT_CSV):
        print("[PhishIntention] ERROR: Input file not found: {}".format(INPUT_CSV))
        sys.exit(1)

    _prepare_models_dir()

    urls, labels = read_urls(INPUT_CSV)
    print("[PhishIntention] Loaded {} URLs".format(len(urls)))

    if MODE == "train":
        train_phishintention(urls, labels)
        print("[PhishIntention] Training complete.")
        return

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = os.path.join(tmpdir, "test_sites")
        os.makedirs(test_dir, exist_ok=True)

        prepare_test_folder(urls, test_dir)

        output_file = os.path.join(tmpdir, "phishintention_output.json")
        ret = run_phishintention(test_dir, output_file)

        if os.path.isfile(output_file):
            print("[PhishIntention] Output file: {} ({} bytes)".format(
                output_file, os.path.getsize(output_file)))
        else:
            print("[PhishIntention] WARNING: Output file not found: {}".format(output_file))
        predictions = parse_phishintention_output(output_file, urls)
        n_unk = sum(1 for _, p, _ in predictions if p == "unknown")
        print("[PhishIntention] Parsed: {} total, {} unknown".format(len(predictions), n_unk))

    write_output(predictions, OUTPUT_DIR)
    print("[PhishIntention] Done.")


if __name__ == "__main__":
    main()
