"""
Blacklist lookup helpers for the §5.6 real-time comparison.

Each lookup returns, for a given URL/hostname, the earliest time the URL
was listed on a particular blacklist — or ``None`` if it is not listed.
PhishXGraph's §5.6 experiment subtracts this from the time we first
predicted phish to compute lead time.

Supported sources (paper §5.6):
- OpenPhish  (feed.txt — free, refreshed ~12 min)
- PhishTank  (online-valid.csv — free but rate-limited; add API key for
             higher quota)
- Google Safe Browsing v4  (lookup API, requires GOOGLE_API_KEY)

For off-line reproduction we only retain *whether* a URL was ever on the
blacklist during the observation window; first-seen timestamps come from
each source's own metadata when available.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse

import urllib.request

log = logging.getLogger("phishxgraph.blacklists")

OPENPHISH_URL = "https://openphish.com/feed.txt"
PHISHTANK_URL = "https://data.phishtank.com/data/online-valid.csv"
GSB_API_URL = "https://safebrowsing.googleapis.com/v4/threatMatches:find"


@dataclass
class BlacklistHit:
    source: str
    first_seen: datetime | None
    listed: bool = True


def _http_get(url: str, timeout: int = 45,
              headers: dict | None = None) -> bytes:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _extract_host(url: str) -> str | None:
    try:
        p = urlparse(url if "://" in url else f"http://{url}")
        return (p.hostname or "").lower() or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# OpenPhish — free feed, URLs only (no timestamps). We approximate
# first_seen with the fetch time of the feed response.
# ---------------------------------------------------------------------------
class OpenPhishFeed:
    name = "openphish"

    def __init__(self):
        self._urls: set[str] = set()
        self._hosts: set[str] = set()
        self._fetched_at: datetime | None = None

    def refresh(self) -> None:
        body = _http_get(OPENPHISH_URL).decode("utf-8", errors="replace")
        urls = {
            l.strip() for l in body.splitlines()
            if l.strip() and not l.strip().startswith("#")
        }
        self._urls = urls
        self._hosts = {_extract_host(u) for u in urls}
        self._hosts.discard(None)
        self._fetched_at = datetime.now(timezone.utc)
        log.info("OpenPhish: %d urls, %d hosts @ %s",
                 len(self._urls), len(self._hosts), self._fetched_at)

    def lookup(self, url: str) -> BlacklistHit | None:
        if url in self._urls:
            return BlacklistHit(self.name, self._fetched_at)
        host = _extract_host(url)
        if host and host in self._hosts:
            return BlacklistHit(self.name, self._fetched_at)
        return None


# ---------------------------------------------------------------------------
# PhishTank — CSV with verification_time (per entry).
# ---------------------------------------------------------------------------
class PhishTankFeed:
    """PhishTank CSV feed with verification timestamps.

    Host-level matching on the raw CSV causes false positives for
    historically-abused legitimate domains (e.g. ``www.google.com``
    appearing in ``https://www.google.com/url?q=<phishing>`` entries from
    2022). To contain this, host-level lookups only succeed when the
    verification time is within ``host_match_max_age_days`` of ``now``.
    URL-exact matches are always returned regardless of age.
    """

    name = "phishtank"

    def __init__(self, app_key: str | None = None,
                 host_match_max_age_days: int = 60):
        self.app_key = app_key
        self.host_match_max_age_days = host_match_max_age_days
        self._by_url: dict[str, datetime] = {}
        self._by_host: dict[str, datetime] = {}

    def refresh(self) -> None:
        headers = {
            "User-Agent": "phishxgraph-realtime/1.0 "
                          "(research open-science release)"
        }
        if self.app_key:
            headers["Authorization"] = f"PhishTank {self.app_key}"
        body = _http_get(PHISHTANK_URL, headers=headers).decode(
            "utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(body))
        by_url, by_host = {}, {}
        for row in reader:
            url = row.get("url", "").strip()
            t = row.get("verification_time", "").strip()
            if not url:
                continue
            ts = None
            try:
                ts = datetime.fromisoformat(t.replace("Z", "+00:00")) if t else None
            except Exception:
                ts = None
            by_url[url] = ts or datetime.now(timezone.utc)
            h = _extract_host(url)
            if h:
                prev = by_host.get(h)
                by_host[h] = min(filter(None, [prev, by_url[url]])) \
                    if prev else by_url[url]
        self._by_url = by_url
        self._by_host = by_host
        log.info("PhishTank: %d urls, %d hosts", len(by_url), len(by_host))

    def lookup(self, url: str) -> BlacklistHit | None:
        if url in self._by_url:
            return BlacklistHit(self.name, self._by_url[url])
        host = _extract_host(url)
        if host and host in self._by_host:
            host_ts = self._by_host[host]
            # Guard against stale host matches from historically-abused
            # legitimate domains (e.g. google.com in old ?q= entries).
            age_days = (datetime.now(timezone.utc) - host_ts).days
            if age_days <= self.host_match_max_age_days:
                return BlacklistHit(self.name, host_ts)
        return None


# ---------------------------------------------------------------------------
# Google Safe Browsing v4 (Lookup API)
# ---------------------------------------------------------------------------
class GoogleSafeBrowsingAPI:
    """Query GSB v4 threatMatches:find API.

    Requires ``GOOGLE_API_KEY``. See
    https://developers.google.com/safe-browsing/v4 .
    """

    name = "gsb"

    def __init__(self, api_key: str | None = None,
                 client_id: str = "phishxgraph",
                 client_version: str = "1.0"):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self.client_id = client_id
        self.client_version = client_version

    def lookup(self, url: str) -> BlacklistHit | None:
        if not self.api_key:
            return None
        body = {
            "client": {
                "clientId": self.client_id,
                "clientVersion": self.client_version,
            },
            "threatInfo": {
                "threatTypes": [
                    "MALWARE", "SOCIAL_ENGINEERING",
                    "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION",
                ],
                "platformTypes": ["ANY_PLATFORM"],
                "threatEntryTypes": ["URL"],
                "threatEntries": [{"url": url}],
            },
        }
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{GSB_API_URL}?key={self.api_key}",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                resp = json.loads(r.read().decode("utf-8"))
        except Exception as exc:
            log.debug("gsb lookup failed for %s: %s", url, exc)
            return None
        if resp.get("matches"):
            # GSB API does not return a first-seen timestamp; record the
            # observation time instead, matching OpenPhish's semantics.
            return BlacklistHit(self.name, datetime.now(timezone.utc))
        return None


class BlacklistPanel:
    """Aggregate of OpenPhish, PhishTank, and GSB."""

    def __init__(self,
                 openphish: OpenPhishFeed | None = None,
                 phishtank: PhishTankFeed | None = None,
                 gsb: GoogleSafeBrowsingAPI | None = None):
        self.openphish = openphish
        self.phishtank = phishtank
        self.gsb = gsb

    def refresh_feeds(self) -> None:
        if self.openphish is not None:
            try:
                self.openphish.refresh()
            except Exception as exc:
                log.warning("openphish refresh failed: %s", exc)
        if self.phishtank is not None:
            try:
                self.phishtank.refresh()
            except Exception as exc:
                log.warning("phishtank refresh failed: %s", exc)

    def lookup(self, url: str) -> dict[str, BlacklistHit | None]:
        out: dict[str, BlacklistHit | None] = {}
        for src in (self.openphish, self.phishtank, self.gsb):
            if src is None:
                continue
            try:
                out[src.name] = src.lookup(url)
            except Exception as exc:
                log.debug("%s lookup failed: %s", src.name, exc)
                out[src.name] = None
        return out

    def earliest_hit(self, url: str) -> BlacklistHit | None:
        hits = [h for h in self.lookup(url).values()
                if h is not None and h.first_seen is not None]
        if not hits:
            return None
        return min(hits, key=lambda h: h.first_seen)
