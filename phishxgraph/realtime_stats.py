"""
Aggregate realtime detection logs into the §5.6 summary table.

Reads the CSV produced by :mod:`phishxgraph.realtime` (columns:
ts_detected, url, domain, score, prediction, openphish_first_seen,
phishtank_first_seen, gsb_first_seen, earliest_blacklist, lead_minutes)
and emits:

- total phishing detections
- how many were absent from every blacklist at detection time
- how many were later listed on at least one blacklist (confirmed)
- mean/median PhishXGraph lead over earliest blacklist (minutes)

This reproduces the §5.6 headline numbers ("2,058 detections, 1,933
absent at detection time, average 12 min vs blacklists' 36 min").
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import statistics
from datetime import datetime

log = logging.getLogger("phishxgraph.realtime_stats")


def _parse_ts(s: str):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def compute(rows):
    total = 0
    absent_at_detection = 0
    confirmed = 0
    lead_minutes = []
    by_src = {"openphish": 0, "phishtank": 0, "gsb": 0}

    for r in rows:
        if r.get("prediction") != "phish":
            continue
        total += 1
        det = _parse_ts(r.get("ts_detected", ""))
        earliest = _parse_ts(r.get("earliest_blacklist", ""))
        if earliest is None:
            absent_at_detection += 1
        else:
            confirmed += 1
            if det is not None:
                try:
                    lead_minutes.append(
                        (earliest.replace(tzinfo=None)
                         - det.replace(tzinfo=None)).total_seconds() / 60
                    )
                except Exception:
                    pass
        for src in by_src:
            if r.get(f"{src}_first_seen"):
                by_src[src] += 1

    mean_lead = statistics.mean(lead_minutes) if lead_minutes else float("nan")
    median_lead = statistics.median(lead_minutes) if lead_minutes else float("nan")
    pct_ahead = (
        sum(1 for x in lead_minutes if x > 0) / len(lead_minutes)
        if lead_minutes else float("nan")
    )

    return {
        "total_phish_detections": total,
        "absent_from_all_blacklists_at_detection": absent_at_detection,
        "confirmed_on_any_blacklist_later": confirmed,
        "listed_openphish": by_src["openphish"],
        "listed_phishtank": by_src["phishtank"],
        "listed_gsb": by_src["gsb"],
        "mean_lead_minutes": mean_lead,
        "median_lead_minutes": median_lead,
        "fraction_earlier_than_blacklist": pct_ahead,
    }


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [realtime_stats] %(levelname)s: %(message)s",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True,
                   help="CSV from phishxgraph.realtime")
    p.add_argument("--output", default="",
                   help="Optional CSV output; otherwise prints to stdout")
    args = p.parse_args(argv)

    with open(args.input) as f:
        rows = list(csv.DictReader(f))
    stats = compute(rows)

    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:40s}: {v:.3f}")
        else:
            print(f"  {k:40s}: {v}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for k, v in stats.items():
                w.writerow([k, v])
        log.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
