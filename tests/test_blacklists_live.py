"""
Live network tests for blacklist integrations.

Skipped by default. Enable with::

    PHISHXGRAPH_LIVE_TESTS=1 pytest tests/test_blacklists_live.py -v

These tests make real HTTP requests to OpenPhish and PhishTank. GSB is
only exercised when ``GOOGLE_API_KEY`` is set.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from phishxgraph.blacklists import (  # noqa: E402
    BlacklistPanel,
    GoogleSafeBrowsingAPI,
    OpenPhishFeed,
    PhishTankFeed,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("PHISHXGRAPH_LIVE_TESTS", "") != "1",
    reason="live network tests; set PHISHXGRAPH_LIVE_TESTS=1 to run",
)


def test_openphish_roundtrip():
    feed = OpenPhishFeed()
    feed.refresh()
    assert len(feed._urls) > 0, "OpenPhish returned no URLs"
    sample = next(iter(feed._urls))
    hit = feed.lookup(sample)
    assert hit is not None
    assert hit.source == "openphish"
    # Known-benign must not match.
    assert feed.lookup("https://www.python.org/") is None


def test_phishtank_roundtrip_and_age_guard():
    feed = PhishTankFeed(host_match_max_age_days=60)
    feed.refresh()
    assert len(feed._by_url) > 0
    # URL-exact match preserved regardless of age.
    sample_url = next(iter(feed._by_url))
    assert feed.lookup(sample_url) is not None
    # Legit domains must NOT match via stale host fallback.
    assert feed.lookup("https://www.google.com/") is None
    assert feed.lookup("https://www.python.org/") is None


def test_gsb_key_missing_returns_none():
    """Without GOOGLE_API_KEY, GSB silently returns None — no crash."""
    gsb = GoogleSafeBrowsingAPI(api_key="")
    assert gsb.lookup("https://example.com/") is None


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set",
)
def test_gsb_live():
    gsb = GoogleSafeBrowsingAPI()
    # Google's documented GSB test URL — always classified as threat
    hit = gsb.lookup("http://malware.testing.google.test/testing/malware/")
    assert hit is not None and hit.source == "gsb"
    # Known benign
    assert gsb.lookup("https://www.google.com/") is None


def test_panel_earliest_hit_live():
    panel = BlacklistPanel(
        openphish=OpenPhishFeed(),
        phishtank=PhishTankFeed(),
        gsb=GoogleSafeBrowsingAPI(),
    )
    panel.refresh_feeds()
    # Pull a fresh OpenPhish entry; panel should return at least that hit.
    if not panel.openphish._urls:
        pytest.skip("OpenPhish returned no URLs")
    sample = next(iter(panel.openphish._urls))
    hits = panel.lookup(sample)
    assert hits["openphish"] is not None
    earliest = panel.earliest_hit(sample)
    assert earliest is not None
