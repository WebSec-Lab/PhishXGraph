"""
Unit tests for the additional paper-claim modules:
  - phishxgraph.ablation     (Table 7: category partitioning)
  - evasion.aggregate        (Table 6: ASR roll-up)
  - phishxgraph.blacklists   (§5.6: BlacklistPanel dispatch)
  - phishxgraph.realtime_stats (§5.6 summary stats)
"""
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from phishxgraph.ablation import (  # noqa: E402
    CATEGORY_MAP,
    _features_for_categories,
)
from phishxgraph.blacklists import (  # noqa: E402
    BlacklistHit,
    BlacklistPanel,
)
from phishxgraph.features import FEATURE_NAMES  # noqa: E402
from phishxgraph.realtime_stats import compute  # noqa: E402
from evasion.aggregate import build_table, SURFACE_OF  # noqa: E402


# ------------------- ablation -------------------

def test_category_map_covers_all_101():
    all_cats = sum(
        (_features_for_categories([c]) for c in ("C", "S", "B")), []
    )
    assert len(all_cats) == 101, len(all_cats)
    assert set(all_cats) == set(FEATURE_NAMES)


def test_individual_counts():
    assert len(_features_for_categories(["C"])) == 42
    assert len(_features_for_categories(["S"])) == 43
    assert len(_features_for_categories(["B"])) == 16


# ------------------- evasion aggregate -------------------

def _mkrow(attack, variant, url, label="phish", prediction="benign",
           status="ok", score="0.1"):
    return {
        "attack": attack, "variant": variant, "url": url, "label": label,
        "prediction": prediction, "score": score, "status": status,
    }


def test_surface_asr_union_semantics():
    # One URL evaded on URL surface (via A1 v1), another on Logo (A8 v2).
    # A different URL evaded nowhere.
    rows = [
        _mkrow("A1_url_shortener", "v1", "http://p1.test", prediction="benign"),
        _mkrow("A2_url_combosquat", "v1", "http://p1.test", prediction="phish"),
        _mkrow("A8_logo_masking", "v2", "http://p2.test", prediction="benign"),
        _mkrow("A1_url_shortener", "v1", "http://p2.test", prediction="phish"),
        _mkrow("A1_url_shortener", "v1", "http://p3.test", prediction="phish"),
        _mkrow("A8_logo_masking", "v2", "http://p3.test", prediction="phish"),
    ]
    table = build_table(rows)
    by_name = {r["scenario"]: r for r in table}
    # p1 evaded on URL only → URL ASR = 1/3
    assert abs(by_name["URL"]["asr"] - 1 / 3) < 1e-6
    # p2 evaded on Logo only → Logo ASR = 1/3
    assert abs(by_name["Logo"]["asr"] - 1 / 3) < 1e-6
    # HTML untouched
    assert by_name["HTML"]["asr"] == 0.0
    # URL+Logo combo: union(p1, p2) → 2/3
    assert abs(by_name["URL+Logo"]["asr"] - 2 / 3) < 1e-6
    # URL+HTML+Logo combo: still union of p1 and p2 → 2/3
    assert abs(by_name["URL+HTML+Logo"]["asr"] - 2 / 3) < 1e-6


def test_surface_of_covers_10_attacks():
    assert len(SURFACE_OF) == 10
    assert set(SURFACE_OF.values()) == {"URL", "HTML", "Logo"}


# ------------------- blacklist panel -------------------

class _FakeFeed:
    def __init__(self, name, hits):
        self.name = name
        self._hits = hits  # url -> datetime|None

    def refresh(self):
        pass

    def lookup(self, url):
        if url in self._hits:
            ts = self._hits[url]
            return BlacklistHit(self.name, ts) if ts else None
        return None


def test_panel_dispatch_and_earliest():
    t1 = datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 4, 21, 10, 5, tzinfo=timezone.utc)
    t3 = datetime(2026, 4, 21, 10, 2, tzinfo=timezone.utc)
    panel = BlacklistPanel(
        openphish=_FakeFeed("openphish", {"http://p.test": t1}),
        phishtank=_FakeFeed("phishtank", {"http://p.test": t2}),
        gsb=_FakeFeed("gsb", {"http://p.test": t3}),
    )
    hits = panel.lookup("http://p.test")
    assert hits["openphish"].first_seen == t1
    assert hits["phishtank"].first_seen == t2
    assert hits["gsb"].first_seen == t3
    assert panel.earliest_hit("http://p.test").first_seen == t1
    # unknown URL returns all None
    assert all(v is None for v in panel.lookup("http://x.test").values())
    assert panel.earliest_hit("http://x.test") is None


# ------------------- realtime_stats -------------------

def test_compute_summary():
    rows = [
        # Detection 1: absent from blacklists at detection time
        {
            "ts_detected": "2026-04-21T10:00:00",
            "url": "http://a.test",
            "prediction": "phish",
            "openphish_first_seen": "",
            "phishtank_first_seen": "",
            "gsb_first_seen": "",
            "earliest_blacklist": "",
            "lead_minutes": "",
        },
        # Detection 2: listed 12 min after detection
        {
            "ts_detected": "2026-04-21T10:00:00",
            "url": "http://b.test",
            "prediction": "phish",
            "openphish_first_seen": "2026-04-21T10:12:00",
            "phishtank_first_seen": "",
            "gsb_first_seen": "",
            "earliest_blacklist": "2026-04-21T10:12:00",
            "lead_minutes": "12.00",
        },
        # Detection 3: benign — should be ignored
        {
            "ts_detected": "2026-04-21T10:00:00",
            "url": "http://c.test",
            "prediction": "benign",
            "openphish_first_seen": "",
            "phishtank_first_seen": "",
            "gsb_first_seen": "",
            "earliest_blacklist": "",
            "lead_minutes": "",
        },
    ]
    stats = compute(rows)
    assert stats["total_phish_detections"] == 2
    assert stats["absent_from_all_blacklists_at_detection"] == 1
    assert stats["confirmed_on_any_blacklist_later"] == 1
    assert stats["listed_openphish"] == 1
    assert abs(stats["mean_lead_minutes"] - 12.0) < 1e-6
    assert stats["fraction_earlier_than_blacklist"] == 1.0
