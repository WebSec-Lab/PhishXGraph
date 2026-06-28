#!/usr/bin/env python3
"""
Curate a small, safe-to-share sample bundle from a full dataset.

Author-side utility. Given a pre-existing feature cache and the
status DB that indexes it (both produced by a full crawl), this
script:

  - picks 30 URLs with cached instrumentation (15 phish, 15 benign)
  - copies a redacted instrumentation.json for each (stack traces
    trimmed, value previews dropped)
  - writes the URL list to data/samples/input/sample_urls.csv

It does NOT copy raw page.html (live credential payloads). Point
``--features-dir`` and ``--status-db`` at wherever the full dataset
lives on your machine.

Run:
    python scripts/stage_samples.py \\
        --features-dir /path/to/data/features \\
        --status-db    /path/to/data/status.db \\
        --out-dir      data/samples
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sqlite3
import sys
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


# Common secret shapes we never want to ship in sample instrumentation.
# - Google API keys: AIzaSy... (39 chars)
# - OAuth access tokens: ya29....
# - OpenAI-style: sk-...
# - Generic query params that usually carry keys/tokens
_SECRET_TOKEN_RE = re.compile(
    r"(?:AIza[0-9A-Za-z_-]{30,}|ya29\.[0-9A-Za-z_-]{20,}|sk-[0-9A-Za-z_-]{20,})"
)
_SENSITIVE_QUERY_KEYS = {
    "key", "apikey", "api_key", "access_token", "accesstoken",
    "token", "auth", "authorization", "client_secret", "secret",
    "password", "pass", "pw", "signature", "sig",
}


def _scrub_url(url: str) -> str:
    """Return the URL with obvious secrets redacted from query + fragment."""
    if not url or not isinstance(url, str):
        return url
    # Substring replacement first (catches tokens in paths or fragments).
    url = _SECRET_TOKEN_RE.sub("<redacted>", url)
    try:
        parts = urlparse(url)
    except Exception:
        return url
    if not parts.query:
        return url
    scrubbed = []
    for k, v in parse_qsl(parts.query, keep_blank_values=True):
        if k.lower() in _SENSITIVE_QUERY_KEYS or _SECRET_TOKEN_RE.search(v):
            scrubbed.append((k, "<redacted>"))
        else:
            scrubbed.append((k, v))
    return urlunparse(parts._replace(query=urlencode(scrubbed)))


def pick_urls(db_path: str, n_phish: int, n_benign: int):
    conn = sqlite3.connect(db_path)
    q = """
    SELECT u.url, u.label
    FROM urls u
    JOIN feature_status fs1 ON u.url=fs1.url
        AND fs1.feature_type='html' AND fs1.status='ok'
    JOIN feature_status fs2 ON u.url=fs2.url
        AND fs2.feature_type='instrumentation' AND fs2.status='ok'
    WHERE u.label = ?
    LIMIT ?
    """
    phish = conn.execute(q, ("phish", n_phish)).fetchall()
    benign = conn.execute(q, ("benign", n_benign)).fetchall()
    conn.close()
    return phish, benign


def url_hash(url: str) -> str:
    import hashlib
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def redact_instrumentation(src: str, dst: str) -> None:
    """Write a redacted copy of instrumentation.json.

    Removes:
      - ``stack`` traces from every storage op (may contain local paths)
      - cookie ``value`` previews
      - API-key-shaped tokens and sensitive query parameters in every
        URL field of ``requests`` / ``redirects`` (a crawled phishing
        page often loads Google Maps / Firebase / etc. which appends
        third-party API keys as ``?key=AIza...``)
    """
    with open(src) as f:
        data = json.load(f)

    for op in data.get("storage_ops", []):
        op.pop("stack", None)
        if "value" in op:
            op["value"] = "<redacted>"
        # Firebase embeds API keys in storage keys, e.g.
        # ``firebase:authUser:AIzaSy…:[DEFAULT]`` — scrub here too.
        if "key" in op and isinstance(op["key"], str):
            op["key"] = _SECRET_TOKEN_RE.sub("<redacted>", op["key"])

    for req in data.get("requests", []):
        if "url" in req:
            req["url"] = _scrub_url(req["url"])

    for redir in data.get("redirects", []):
        for key in ("from_url", "to_url"):
            if key in redir:
                redir[key] = _scrub_url(redir[key])

    with open(dst, "w") as f:
        json.dump(data, f)


def main(argv=None):
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--features-dir", required=True)
    p.add_argument("--status-db", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-phish", type=int, default=15)
    p.add_argument("--n-benign", type=int, default=15)
    args = p.parse_args(argv)

    out_input = os.path.join(args.out_dir, "input")
    out_features = os.path.join(args.out_dir, "features")
    os.makedirs(out_input, exist_ok=True)
    os.makedirs(out_features, exist_ok=True)

    phish, benign = pick_urls(args.status_db, args.n_phish, args.n_benign)
    if len(phish) < args.n_phish or len(benign) < args.n_benign:
        print(f"WARN: only {len(phish)} phish, {len(benign)} benign available",
              file=sys.stderr)

    rows = phish + benign
    csv_path = os.path.join(out_input, "sample_urls.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["url", "label"])
        for u, l in rows:
            w.writerow([u, l])
    print(f"wrote {csv_path} with {len(rows)} URLs")

    # Redact and copy instrumentation. Skip page.html entirely.
    copied = 0
    for url, _ in rows:
        h = url_hash(url)
        src_dir = os.path.join(args.features_dir, h)
        dst_dir = os.path.join(out_features, h)
        os.makedirs(dst_dir, exist_ok=True)
        src_instr = os.path.join(src_dir, "instrumentation.json")
        if not os.path.isfile(src_instr):
            continue
        redact_instrumentation(src_instr, os.path.join(dst_dir, "instrumentation.json"))
        copied += 1
    print(f"staged {copied} redacted instrumentation payloads → {out_features}")

    readme = os.path.join(args.out_dir, "README.md")
    with open(readme, "w") as f:
        f.write(
            "# Sample bundle\n\n"
            f"- `input/sample_urls.csv` — {len(rows)} URL/label pairs\n"
            f"- `features/<hash>/instrumentation.json` — {copied} redacted\n"
            "  instrumentation payloads (stack traces and cookie value\n"
            "  previews removed)\n\n"
            "To score with the reference pipeline:\n\n"
            "```\n"
            "python -m phishxgraph.cli test \\\n"
            "    --input    data/samples/input/sample_urls.csv \\\n"
            "    --features-dir data/samples/features \\\n"
            "    --model-dir <path-to-your-trained-model> \\\n"
            "    --output   predictions.csv\n"
            "```\n\n"
            "Note: `page.html` is intentionally excluded. Structural and\n"
            "behavioral features that depend on the DOM cannot be computed\n"
            "without it; for a full re-run, regenerate the cache with\n"
            "`phishxgraph.cli collect`.\n"
        )
    print(f"wrote {readme}")


if __name__ == "__main__":
    main()
