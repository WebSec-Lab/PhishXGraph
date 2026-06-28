#!/usr/bin/env python3
"""
Phishpedia wrapper: reads phishinglist.csv, captures screenshots, runs Phishpedia.
"""
import csv
import os
import sys
import subprocess
import time

sys.path.insert(0, "/app")

INPUT_CSV = os.environ.get("INPUT_CSV", "/data/input/phishinglist.csv")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/data/results/phishpedia")
MODEL_DIR = os.environ.get("MODEL_DIR", "/data/models/phishpedia")
FEATURES_DIR = os.environ.get("FEATURES_DIR", "/data/features")
MODE = os.environ.get("MODE", "test")
PHISHPEDIA_DIR = "/app/phishpedia"


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
        print("[Phishpedia] Collection cap reached ({}/{})".format(ok_count, max_collect))
        return

    # Skip already-attempted URLs (both ok and fail); process newest first
    to_capture = [u for u in urls if u not in attempted]
    to_capture.reverse()

    if max_collect > 0:
        to_capture = to_capture[:max_collect - ok_count]

    cached = len(attempted)
    if cached > 0:
        print("[Phishpedia] Already attempted: {} ({} ok)".format(cached, ok_count))
    if not to_capture:
        print("[Phishpedia] No new URLs to capture.")
        return
    print("[Phishpedia] Capturing screenshots for {} new URLs...".format(len(to_capture)))
    for i, url in enumerate(to_capture):
        cache_path = _url_cache_path(url)
        capture_screenshot(url, cache_path)
        _append_collected_url(url)
        if (i + 1) % 10 == 0:
            print("[Phishpedia] Captured {}/{}".format(i + 1, len(to_capture)))
    print("[Phishpedia] Cached {} screenshots".format(len(to_capture)))


def prepare_test_folder(urls, test_dir):
    """Create folder structure expected by Phishpedia.
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

    print("[Phishpedia] Used {} cached screenshots, {} skipped (no cache)".format(
        cached_count, skipped_count))


def run_phishpedia(test_dir, output_file):
    """Run Phishpedia detection."""
    cmd = [
        "python3", os.path.join(PHISHPEDIA_DIR, "phishpedia.py"),
        "--folder", test_dir,
        "--output_txt", output_file,
    ]

    # Try pixi run first, fallback to direct python
    print("[Phishpedia] Running: {}".format(" ".join(cmd)))
    result = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=PHISHPEDIA_DIR)

    print(result.stdout)
    if result.returncode != 0:
        print("[Phishpedia] STDERR: {}".format(result.stderr), file=sys.stderr)
    return result.returncode


def parse_phishpedia_output(output_file, urls):
    """Parse Phishpedia output.

    Phishpedia writes two lines per site:
      Line 1: folder_name\\tURL
      Line 2: \\tpred_label\\tbrand\\tdomains\\tscore\\ttime1\\ttime2
    Line 2 starts with a tab character.
    """
    predictions = []
    parsed = {}

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            lines = f.readlines()
        current_folder = None
        for line in lines:
            if line.startswith("\t"):
                # Continuation line with prediction details
                if current_folder is not None:
                    parts = line.strip().split("\t")
                    try:
                        pred_label = int(parts[0]) if parts[0].isdigit() else 0
                        score = float(parts[3]) if len(parts) > 3 and parts[3] != "None" else 0.0
                    except (ValueError, IndexError):
                        pred_label = 0
                        score = 0.0
                    parsed[current_folder] = ("phish" if pred_label == 1 else "benign", score)
                    current_folder = None
            else:
                # Folder line: folder_name\tURL
                parts = line.strip().split("\t")
                if parts[0].startswith("site_"):
                    current_folder = parts[0]

    for i, url in enumerate(urls):
        folder = "site_{}".format(i)
        if folder in parsed:
            predictions.append((url, parsed[folder][0], parsed[folder][1]))
        else:
            predictions.append((url, "unknown", 0.0))

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
    print("[Phishpedia] Results written to {}".format(csv_path))


def train_phishpedia(urls, labels):
    """Train: fine-tune Phishpedia's logo detector and Siamese network.

    Training components (paper: USENIX Security 2021):
    1. Logo Detector (Detectron2 Faster R-CNN) fine-tuning with pseudo-labels
    2. Siamese Network (ResNet50) contrastive learning for logo matching
    3. Reference logo database construction from benign samples

    Uses only pre-collected data; URLs without cached data are skipped.
    """
    import random
    import shutil
    import tempfile

    # Subsample to avoid OOM during logo detection (holds PIL crops for all
    # sites in memory). Cap is per-label so the 1:1 balance is preserved.
    MAX_PER_LABEL = int(os.environ.get("PHISHPEDIA_MAX_PER_LABEL", "5000"))
    if len(urls) > 2 * MAX_PER_LABEL:
        rng = random.Random(42)
        phish_idx = [i for i, l in enumerate(labels) if l == "phish"]
        benign_idx = [i for i, l in enumerate(labels) if l == "benign"]
        rng.shuffle(phish_idx)
        rng.shuffle(benign_idx)
        keep = set(phish_idx[:MAX_PER_LABEL] + benign_idx[:MAX_PER_LABEL])
        urls = [u for i, u in enumerate(urls) if i in keep]
        labels = [l for i, l in enumerate(labels) if i in keep]
        print("[Phishpedia] Subsampled to {} URLs ({} per label)".format(
            len(urls), MAX_PER_LABEL))

    print("[Phishpedia] Training with {} labeled URLs...".format(len(urls)))

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

        print("[Phishpedia] Used {} cached, {} skipped (no cache)".format(
            cached_count, skipped_count))

        # Run Phishpedia training (logo detector + Siamese + reference list)
        train_script = os.path.join(PHISHPEDIA_DIR, "train.py")
        if os.path.exists(train_script):
            cmd = [
                "python3", train_script,
                "--folder", train_dir,
                "--output", MODEL_DIR,
            ]
            print("[Phishpedia] Running training: {}".format(" ".join(cmd)))
            sys.stdout.flush()
            # Stream output live so crashes (OOM etc.) are visible immediately
            result = subprocess.run(cmd, cwd=PHISHPEDIA_DIR)
            if result.returncode != 0:
                print("[Phishpedia] Training script failed (returncode={}), saving reference data only".format(result.returncode))
        else:
            print("[Phishpedia] No train.py found, saving reference data for model")

        # Save labeled data as model reference (fallback / backup)
        os.makedirs(MODEL_DIR, exist_ok=True)
        ref_dir = os.path.join(MODEL_DIR, "reference_data")
        if os.path.exists(ref_dir):
            shutil.rmtree(ref_dir)
        shutil.copytree(train_dir, ref_dir)
        print("[Phishpedia] Training data saved to {}".format(MODEL_DIR))


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
    """Populate /app/phishpedia/models from pretrained source, then overlay
    trained weights from MODEL_DIR when running in test mode.

    The docker-compose mount now puts the read-only pretrained tree at
    /app/phishpedia/_pretrained_src. We symlink each entry into
    /app/phishpedia/models so phishpedia.py (which hardcodes relative paths)
    sees a working models dir. In test mode, if trained weights exist in
    MODEL_DIR we swap the relevant symlinks so the trained models are used.
    """
    import shutil

    pretrained_src = "/app/phishpedia/_pretrained_src"
    models_dst = os.path.join(PHISHPEDIA_DIR, "models")

    if not os.path.isdir(pretrained_src):
        print("[Phishpedia] WARNING: pretrained source not found at {}".format(
            pretrained_src))
        return

    if os.path.islink(models_dst) or os.path.isfile(models_dst):
        os.remove(models_dst)
    elif os.path.isdir(models_dst):
        shutil.rmtree(models_dst)
    os.makedirs(models_dst, exist_ok=True)

    # Symlink every entry from pretrained_src into models/
    for entry in os.listdir(pretrained_src):
        src = os.path.join(pretrained_src, entry)
        dst = os.path.join(models_dst, entry)
        os.symlink(src, dst)

    # Clear stale logo feature cache; must regenerate if siamese weights change.
    for cache_name in ("LOGO_FEATS.npy", "LOGO_FILES.npy"):
        cache_path = os.path.join(PHISHPEDIA_DIR, cache_name)
        if os.path.isfile(cache_path):
            os.remove(cache_path)

    # In test mode, overlay trained weights on top of pretrained
    if MODE != "test" or not MODEL_DIR or not os.path.isdir(MODEL_DIR):
        print("[Phishpedia] Using pretrained models only")
        return

    # Map trained filename -> pretrained filename expected by configs.yaml
    overrides = {
        "rcnn_logo.pth": "rcnn_bet365.pth",
        "siamese_pedia.pth": "resnetv2_rgb_new.pth.tar",
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
        print("[Phishpedia] Using trained weights from {}: {}".format(
            MODEL_DIR, ", ".join(applied)))
    else:
        print("[Phishpedia] No trained weights found in {}, using pretrained".format(
            MODEL_DIR))


def main():
    print("[Phishpedia] Starting (mode={})...".format(MODE))

    if MODE == "collect":
        print("[Phishpedia] Collect mode: feature-collector service handles extraction. Sleeping.")
        while True:
            time.sleep(3600)

    if not os.path.exists(INPUT_CSV):
        print("[Phishpedia] ERROR: Input file not found: {}".format(INPUT_CSV))
        sys.exit(1)

    _prepare_models_dir()

    urls, labels = read_urls(INPUT_CSV)
    print("[Phishpedia] Loaded {} URLs".format(len(urls)))

    if MODE == "train":
        train_phishpedia(urls, labels)
        print("[Phishpedia] Training complete.")
        return

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = os.path.join(tmpdir, "test_sites")
        os.makedirs(test_dir, exist_ok=True)

        prepare_test_folder(urls, test_dir)

        output_file = os.path.join(tmpdir, "phishpedia_output.txt")
        ret = run_phishpedia(test_dir, output_file)

        if os.path.isfile(output_file):
            print("[Phishpedia] Output file: {} ({} bytes)".format(
                output_file, os.path.getsize(output_file)))
        else:
            print("[Phishpedia] WARNING: Output file not found: {}".format(output_file))
        predictions = parse_phishpedia_output(output_file, urls)
        n_unk = sum(1 for _, p, _ in predictions if p == "unknown")
        print("[Phishpedia] Parsed: {} total, {} unknown".format(len(predictions), n_unk))

    write_output(predictions, OUTPUT_DIR)
    print("[Phishpedia] Done.")


if __name__ == "__main__":
    main()
