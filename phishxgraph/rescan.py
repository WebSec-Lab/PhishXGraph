"""
Periodic blacklist re-scan (§5.6 deferred confirmation).

The paper's §5.6 study works like this:

  t=0      PhishXGraph detects URL from CertStream
  t=12min  URL appears on OpenPhish/PhishTank/GSB
  t=36min  URL appears on the slowest blacklist
  → PhishXGraph lead = 36 − 0 = 36 min (vs. earliest 12 min)

A single lookup at detection time is therefore *not enough*: most
positives are not yet listed when PhishXGraph flags them. To capture
later listings, re-scan the results CSV periodically.

Usage:

  # One-shot: refresh every row in a realtime_hits.csv whose
  # earliest_blacklist column is empty.
  python -m phishxgraph.rescan \\
      --input  data/results/realtime_hits.csv \\
      --output data/results/realtime_hits.csv   # in-place update

  # Loop mode: rescan every N seconds.
  python -m phishxgraph.rescan --input ... --output ... --interval 900
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from datetime import datetime, timezone

from .blacklists import (
    BlacklistPanel,
    GoogleSafeBrowsingAPI,
    OpenPhishFeed,
    PhishTankFeed,
)

log = logging.getLogger("phishxgraph.rescan")


FIELDNAMES = [
    "ts_detected", "url", "domain", "score", "prediction",
    "openphish_first_seen", "phishtank_first_seen", "gsb_first_seen",
    "earliest_blacklist", "lead_minutes",
]


def _parse_ts(s: str):
    """Parse an ISO-8601 timestamp and force it to be UTC-aware.

    Mixing aware and naive datetimes raises ``TypeError`` in ``min()``;
    both OpenPhish (aware) and historical CSV rows (naive) flow through
    this rescan, so we normalize here.
    """
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def rescan_rows(rows, panel: BlacklistPanel):
    """Update `earliest_blacklist` and per-source columns for unresolved rows.

    A row is considered "resolved" when all three blacklist columns are
    already populated — at that point no further rescanning is useful.
    Resolved rows are skipped without a network call.

    For unresolved rows, we only *fill in* missing values; we never
    overwrite an existing timestamp. The earliest_blacklist column is
    recomputed from whatever timestamps are present.
    """
    panel.refresh_feeds()
    updated = 0
    now_iso_ts = datetime.now(timezone.utc).isoformat()

    for row in rows:
        if row.get("prediction") != "phish":
            continue

        url = row["url"]
        current = {
            "openphish": row.get("openphish_first_seen") or "",
            "phishtank": row.get("phishtank_first_seen") or "",
            "gsb": row.get("gsb_first_seen") or "",
        }
        # All three already set? nothing to do.
        if all(current.values()):
            continue

        hits = panel.lookup(url)
        row_touched = False
        for src, hit in hits.items():
            col = f"{src}_first_seen"
            if current[src]:
                continue  # keep the original (earlier) timestamp
            if hit is None or hit.first_seen is None:
                continue
            # OpenPhish / GSB have no real first_seen → the timestamp we
            # record is the time *we* first saw them listed. Use the
            # rescan time, not feed fetch time (which is often seconds
            # fresher and misleading).
            if src in ("openphish", "gsb"):
                row[col] = now_iso_ts
            else:
                row[col] = hit.first_seen.isoformat()
            current[src] = row[col]
            row_touched = True

        # Recompute earliest_blacklist + lead_minutes from whatever we have.
        timestamps = [_parse_ts(t) for t in current.values() if t]
        timestamps = [t for t in timestamps if t is not None]
        if timestamps:
            earliest = min(timestamps)
            row["earliest_blacklist"] = earliest.isoformat()
            det = _parse_ts(row.get("ts_detected", ""))
            if det is not None:
                # Both earliest and det are UTC-aware thanks to _parse_ts.
                delta = earliest - det
                row["lead_minutes"] = f"{delta.total_seconds() / 60:.2f}"
            row_touched = True
        if row_touched:
            updated += 1
    return updated


def read_rows(path: str):
    with open(path) as f:
        return list(csv.DictReader(f))


def write_rows(rows, path: str):
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for row in rows:
            # Ensure we only write known columns.
            w.writerow({k: row.get(k, "") for k in FIELDNAMES})
    os.replace(tmp, path)


def make_panel_from_env(enable_gsb: bool = True,
                        phishtank_key: str = "") -> BlacklistPanel:
    return BlacklistPanel(
        openphish=OpenPhishFeed(),
        phishtank=PhishTankFeed(app_key=phishtank_key or None),
        gsb=(GoogleSafeBrowsingAPI() if enable_gsb else None),
    )


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [rescan] %(levelname)s: %(message)s",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True,
                   help="Can be the same path as --input for in-place update")
    p.add_argument("--interval", type=int, default=0,
                   help="Seconds between successive rescans (0 = run once)")
    p.add_argument("--disable-gsb", action="store_true",
                   help="Skip GSB lookups (useful when GOOGLE_API_KEY unset)")
    p.add_argument("--phishtank-key",
                   default=os.environ.get("PHISHTANK_KEY", ""))
    args = p.parse_args(argv)

    panel = make_panel_from_env(
        enable_gsb=not args.disable_gsb,
        phishtank_key=args.phishtank_key,
    )

    def _one_cycle():
        rows = read_rows(args.input)
        n = rescan_rows(rows, panel)
        write_rows(rows, args.output)
        log.info("rescan: %d/%d rows updated", n, len(rows))

    if args.interval <= 0:
        _one_cycle()
        return
    log.info("rescan loop: interval=%ds", args.interval)
    while True:
        try:
            _one_cycle()
        except Exception as exc:
            log.warning("rescan cycle failed: %s", exc)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
