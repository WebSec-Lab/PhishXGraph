"""
Real-time emerging phishing detector (§5.6).

Subscribes to CertStream (Certificate Transparency feed) — the primary
source used in the paper — classifies newly observed hosts with a trained
PhishXGraph model, and writes positives to a CSV. Certificate Transparency
polling is provided as an optional fallback via ``--fallback-ct-logs``.

CertStream endpoint: wss://certstream.calidog.io/ (Calidog Labs, community
mirror of Google's CT logs).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import queue
import signal
import threading
import time
from urllib.parse import urlparse

from .blacklists import (
    BlacklistPanel,
    GoogleSafeBrowsingAPI,
    OpenPhishFeed,
    PhishTankFeed,
)
from .collect import DEFAULT_TIMEOUT_SEC, instrument_page, page_cache_dir
from .features import extract_all_features
from .graph import build_graph, extract_domain
from .model import PhishXGraphClassifier

try:
    import websocket  # websocket-client
    _HAS_WEBSOCKET = True
except ImportError:  # pragma: no cover
    _HAS_WEBSOCKET = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [realtime] %(levelname)s: %(message)s",
)
log = logging.getLogger("phishxgraph.realtime")

DEFAULT_CERTSTREAM_URL = "wss://certstream.calidog.io/"


class CertStreamSource:
    """Stream newly issued TLS certificates from CertStream.

    Each message yields one candidate URL (https://<domain>/) per domain in
    the certificate's CN + SAN list.
    """

    def __init__(self, url: str = DEFAULT_CERTSTREAM_URL,
                 queue_maxsize: int = 10000):
        if not _HAS_WEBSOCKET:
            raise RuntimeError(
                "websocket-client is required for CertStream. "
                "Install via `pip install websocket-client`."
            )
        self.url = url
        self.q: queue.Queue[str] = queue.Queue(maxsize=queue_maxsize)
        self._stop = threading.Event()

    def _on_message(self, _ws, message):
        try:
            data = json.loads(message)
        except Exception:
            return
        if data.get("message_type") != "certificate_update":
            return
        domains = (data.get("data", {}).get("leaf_cert", {}) or {}
                   ).get("all_domains", [])
        for d in domains:
            if not d or "*" in d:
                continue
            try:
                self.q.put_nowait(f"https://{d}/")
            except queue.Full:
                pass

    def _on_error(self, _ws, err):
        log.warning("certstream error: %s", err)

    def _on_close(self, _ws, *_args):
        log.info("certstream closed")

    def _on_open(self, _ws):
        log.info("certstream connected: %s", self.url)

    def run(self):
        while not self._stop.is_set():
            ws = websocket.WebSocketApp(
                self.url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open,
            )
            try:
                ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as exc:
                log.warning("certstream run_forever failed: %s", exc)
            if not self._stop.is_set():
                log.info("reconnecting in 5s…")
                time.sleep(5)

    def stop(self):
        self._stop.set()


class CTLogsFallback:
    """Polling fallback for environments that cannot reach CertStream."""

    def __init__(self, poll_interval: int = 60, queue_maxsize: int = 10000):
        self.poll_interval = poll_interval
        self.q: queue.Queue[str] = queue.Queue(maxsize=queue_maxsize)
        self._stop = threading.Event()

    def run(self):  # pragma: no cover — requires network
        import urllib.request
        from urllib.error import URLError
        endpoint = (
            "https://crt.sh/?q=%25&output=json&exclude=expired"
        )
        seen: set[str] = set()
        while not self._stop.is_set():
            try:
                req = urllib.request.Request(
                    endpoint, headers={"User-Agent": "phishxgraph-realtime"}
                )
                with urllib.request.urlopen(req, timeout=30) as r:
                    records = json.loads(r.read().decode("utf-8"))
            except (URLError, json.JSONDecodeError) as exc:
                log.warning("ct-logs fetch failed: %s", exc)
                time.sleep(self.poll_interval)
                continue
            for rec in records:
                name = rec.get("name_value", "")
                for domain in name.split("\n"):
                    domain = domain.strip()
                    if not domain or "*" in domain or domain in seen:
                        continue
                    seen.add(domain)
                    try:
                        self.q.put_nowait(f"https://{domain}/")
                    except queue.Full:
                        pass
            time.sleep(self.poll_interval)

    def stop(self):
        self._stop.set()


def worker(source, model: PhishXGraphClassifier,
           features_dir: str, output_csv: str,
           threshold: float, timeout_sec: int, duration_sec: int | None,
           blacklist_panel: BlacklistPanel | None = None,
           blacklist_refresh_sec: int = 900):
    """Consume URLs from `source`, score with PhishXGraph, and compare
    PhishXGraph's detection time against blacklist listing times (§5.6).

    Each phishing hit row records::

        ts_detected        — when PhishXGraph classified the URL
        openphish_first_seen, phishtank_first_seen, gsb_first_seen
                           — blacklist first-seen (ISO-8601) if present
        earliest_blacklist — earliest of the above, else empty
        lead_minutes       — earliest_blacklist - ts_detected, in minutes
                             (positive = PhishXGraph detected first)
    """
    from playwright.sync_api import sync_playwright

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    write_header = not os.path.isfile(output_csv)
    deadline = time.time() + duration_sec if duration_sec else None
    processed = 0
    hits = 0

    last_refresh = 0.0
    if blacklist_panel is not None:
        blacklist_panel.refresh_feeds()
        last_refresh = time.time()

    with sync_playwright() as pw, open(output_csv, "a", newline="") as fp:
        browser = pw.chromium.launch(headless=True)
        writer = csv.writer(fp)
        if write_header:
            writer.writerow([
                "ts_detected", "url", "domain", "score", "prediction",
                "openphish_first_seen", "phishtank_first_seen",
                "gsb_first_seen", "earliest_blacklist", "lead_minutes",
            ])
            fp.flush()

        while True:
            if deadline and time.time() >= deadline:
                log.info("duration reached, stopping")
                break
            try:
                url = source.q.get(timeout=10)
            except queue.Empty:
                continue

            payload = instrument_page(browser, url, timeout_sec=timeout_sec)
            if not payload or not payload.get("html"):
                continue
            out_dir = page_cache_dir(features_dir, url)
            from .collect import persist_instrumentation
            persist_instrumentation(out_dir, payload)

            domain = extract_domain(url)
            G, soup = build_graph(url, payload, domain)
            feats = extract_all_features(G, url, domain, soup)
            _, proba = model.predict([feats])
            score = float(proba[0])
            prediction = "phish" if score >= threshold else "benign"
            detected_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            processed += 1
            if prediction == "phish":
                hits += 1

                # Blacklist comparison (§5.6)
                bl_data = {"openphish": None, "phishtank": None, "gsb": None}
                earliest = ""
                lead_min = ""
                if blacklist_panel is not None:
                    # Refresh feeds periodically
                    if (time.time() - last_refresh) >= blacklist_refresh_sec:
                        blacklist_panel.refresh_feeds()
                        last_refresh = time.time()
                    hits_by_src = blacklist_panel.lookup(url)
                    for src_name, hit in hits_by_src.items():
                        if hit is not None and hit.first_seen is not None:
                            bl_data[src_name] = hit.first_seen.isoformat()
                    earliest_hit = blacklist_panel.earliest_hit(url)
                    if earliest_hit is not None and earliest_hit.first_seen:
                        earliest = earliest_hit.first_seen.isoformat()
                        # Positive = PhishXGraph earlier than blacklist
                        from datetime import datetime
                        try:
                            detected_dt = datetime.fromisoformat(
                                detected_at).replace(
                                tzinfo=earliest_hit.first_seen.tzinfo)
                            delta = earliest_hit.first_seen - detected_dt
                            lead_min = f"{delta.total_seconds() / 60:.2f}"
                        except Exception:
                            pass

                writer.writerow([
                    detected_at, url, domain, f"{score:.4f}", prediction,
                    bl_data["openphish"] or "",
                    bl_data["phishtank"] or "",
                    bl_data["gsb"] or "",
                    earliest, lead_min,
                ])
                fp.flush()
                log.info("PHISH score=%.3f lead_min=%s %s",
                         score, lead_min or "-", url)
            if processed % 50 == 0:
                log.info("processed=%d hits=%d", processed, hits)
        browser.close()
    log.info("done processed=%d hits=%d", processed, hits)


def main(argv=None):
    p = argparse.ArgumentParser(description="PhishXGraph realtime worker")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--features-dir", required=True)
    p.add_argument("--output", required=True,
                   help="CSV path; phishing hits are appended here")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SEC)
    p.add_argument("--duration", type=int, default=0,
                   help="Stop after N seconds (0 = run forever)")
    p.add_argument("--certstream-url", default=DEFAULT_CERTSTREAM_URL)
    p.add_argument("--fallback-ct-logs", action="store_true",
                   help="Use crt.sh polling instead of CertStream")
    p.add_argument("--disable-blacklists", action="store_true",
                   help="Skip OpenPhish/PhishTank/GSB comparison (§5.6)")
    p.add_argument("--phishtank-key", default=os.environ.get("PHISHTANK_KEY", ""),
                   help="Optional PhishTank application key")
    p.add_argument("--gsb-key", default=os.environ.get("GOOGLE_API_KEY", ""),
                   help="Google Safe Browsing v4 API key")
    p.add_argument("--blacklist-refresh-sec", type=int, default=900,
                   help="Seconds between feed refreshes (OpenPhish/PhishTank)")
    args = p.parse_args(argv)

    model = PhishXGraphClassifier.load(args.model_dir)

    if args.fallback_ct_logs:
        source = CTLogsFallback()
    else:
        source = CertStreamSource(url=args.certstream_url)

    panel: BlacklistPanel | None = None
    if not args.disable_blacklists:
        panel = BlacklistPanel(
            openphish=OpenPhishFeed(),
            phishtank=PhishTankFeed(app_key=args.phishtank_key or None),
            gsb=GoogleSafeBrowsingAPI(api_key=args.gsb_key or None),
        )

    stop_evt = threading.Event()

    def _sigint(*_):
        log.info("shutdown requested")
        stop_evt.set()
        source.stop()

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    t = threading.Thread(target=source.run, daemon=True)
    t.start()

    worker(
        source=source,
        model=model,
        features_dir=args.features_dir,
        output_csv=args.output,
        threshold=args.threshold,
        timeout_sec=args.timeout,
        duration_sec=args.duration or None,
        blacklist_panel=panel,
        blacklist_refresh_sec=args.blacklist_refresh_sec,
    )


if __name__ == "__main__":
    main()
