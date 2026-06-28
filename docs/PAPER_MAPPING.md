# Paper → Code Mapping

This document traces every numbered claim in the PhishXGraph paper to the
file and function that implements it in `phishxgraph/`.

## §4.1 Graph Representation (Table 2)

| Paper concept | Code |
|---|---|
| HTML nodes (DOM elements + attributes) | [`graph.py`](../phishxgraph/graph.py) `add_dom()` |
| Script nodes (`<script>` + dynamically loaded) | [`graph.py`](../phishxgraph/graph.py) script-layer block |
| Network nodes (one per unique requested URL) | [`graph.py`](../phishxgraph/graph.py) request-layer block |
| Storage nodes (one per `(type, key)` item) | [`graph.py`](../phishxgraph/graph.py) storage-layer block |
| Structure edges | `EDGE_TYPE_STRUCTURE` in [`constants.py`](../phishxgraph/constants.py) |
| Execute edges | `EDGE_TYPE_EXECUTE` (renamed from legacy `"script"`) |
| Request edges (incl. 3xx redirect chains) | `EDGE_TYPE_REQUEST`, `is_redirect=True` |
| Access edges (storage read/write) | `EDGE_TYPE_ACCESS` (renamed from `"storage_access"`) |

## §4.2 Graph Construction

| Paper element | Code |
|---|---|
| Playwright + Chrome DevTools Protocol | [`collect.py`](../phishxgraph/collect.py) `instrument_page()` |
| 30-second per-page timeout | `DEFAULT_TIMEOUT_SEC = 30` |
| Rendered DOM parsed with BeautifulSoup | [`graph.py`](../phishxgraph/graph.py) `BeautifulSoup(html, "lxml")` |
| Storage API hook injected before page scripts | `_STORAGE_HOOK_JS` in [`collect.py`](../phishxgraph/collect.py) |
| Key-level granularity for cookie/localStorage/sessionStorage | `_STORAGE_HOOK_JS` emits `{type, storage, key, stack}` |
| 3xx redirect chains as Network→Network edges | [`graph.py`](../phishxgraph/graph.py) redirect loop |

## §4.3 Feature Extraction (Tables 3, 9–11)

All 101 features, grouped exactly as in the paper:

| Category | Count | Code |
|---|---|---|
| Content / URL lexical | 19 | `_content_features()` in [`features.py`](../phishxgraph/features.py) |
| Content / URL entropy & token | 5 | same |
| Content / HTML phishing cue | 8 | same |
| Content / HTML content | 10 | same |
| Structural / Graph topology | 9 | `_topology_features()` |
| Structural / DOM shape | 4 | same |
| Structural / Centrality & hierarchy | 12 | same |
| Structural / Node type distribution | 11 | `_distribution_features()` |
| Structural / Edge type distribution | 7 | `_edge_features()` |
| Behavioral / Subgraph & path | 8 | `_subgraph_features()` |
| Behavioral / Edge interaction | 2 | `_edge_features()` (`form_to_external_edge`, `script_to_form_path_exists`) |
| Behavioral / Data flow | 6 | `_flow_features()` |
| *Total* | *101* | — |

Feature names match the paper's tables verbatim (see `FEATURE_GROUPS` in
[`features.py`](../phishxgraph/features.py)). Runtime assertion
`len(FEATURE_NAMES) == 101` guards against drift. See
[`FEATURES.md`](FEATURES.md) for the full list with definitions.

### Feature renames applied during open-science refactor

The implementation previously used internal names that differed from the
paper. The release code uses the paper's names; aliases are not kept.

| Paper (Tables 9–11) | Legacy code name | Reason |
|---|---|---|
| `execute_edge_ratio` | `script_edge_ratio` | matches "Execute" edge type |
| `access_edge_ratio` | `storage_access_edge_ratio` | matches "Access" edge type |
| `exfil_path_length` | `critical_path_length` | describes §4.3 behavioral intent |
| `cookie_set_with_form` | `cookie_set_before_form` | exact paper text |

## §4.4 Classification Model

Paper default is *XGBoost* (gradient-boosted decision trees, [@Chen2016]).
The legacy code used `sklearn.ensemble.RandomForestClassifier`; the release
defaults to XGBoost and keeps Random Forest / SVM / MLP as explicit
ablation options (§5.5, Table 7) via `--classifier` on
[`cli.py`](../phishxgraph/cli.py).

## §5.4 Robustness (Table 6)

The ten adversarial attacks (three URL, four HTML, three logo — 60
variants) are bundled here in [`evasion/attacks/`](../evasion/attacks/),
fully self-contained:

| Attack | File | Surface |
|---|---|---|
| A1 URL shortener | `evasion/attacks/a1_url_shortener.py` | URL |
| A2 combosquatting | `evasion/attacks/a2_url_combosquat.py` | URL |
| A3 TLD substitution | `evasion/attacks/a3_url_tld_sub.py` | URL |
| A4 invisible link | `evasion/attacks/a4_html_invisible_link.py` | HTML |
| A5 object ratio | `evasion/attacks/a5_html_object_ratio.py` | HTML |
| A6 DOM expansion | `evasion/attacks/a6_html_dom_expand.py` | HTML |
| A7 input obfuscation | `evasion/attacks/a7_html_input_obfuscate.py` | HTML |
| A8 logo masking | `evasion/attacks/a8_logo_masking.py` | Logo |
| A9 logo manipulation | `evasion/attacks/a9_logo_manipulation.py` | Logo |
| A10 font substitution | `evasion/attacks/a10_logo_font_subst.py` | Logo |

Pipeline:

| Step | Code |
|---|---|
| Generate single-surface caches | [`evasion/generate.py`](../evasion/generate.py) |
| Generate Table 6 combos | [`evasion/generate_combos.py`](../evasion/generate_combos.py) |
| Score per (attack, variant, url) | [`evasion/score.py`](../evasion/score.py) |
| Roll up into surface / combo ASR | [`evasion/aggregate.py`](../evasion/aggregate.py) |

The aggregator implements the paper's union semantics for combos: a
phishing URL is counted as evaded on surface *S* if any variant of
any attack on *S* flips it to benign.

## §5.5 Ablation (Table 7)

Two axes reproduced by [`phishxgraph/ablation.py`](../phishxgraph/ablation.py):

- *Feature category*: C (42) / S (43) / B (16) and all pairwise plus
  full combination.
- *Classifier*: XGBoost (default, §4.4) / Random Forest / SVM (RBF) /
  MLP — each trained on the full 101-feature vector.

The category grouping is driven by `CATEGORY_MAP` which derives subsets
from the canonical `FEATURE_GROUPS` so there is exactly one source of
truth for "which features belong to C/S/B".

## §5.6 Real-World pipeline

[`realtime.py`](../phishxgraph/realtime.py) subscribes to CertStream
(`wss://certstream.calidog.io/`) by default, feeds candidate hostnames
through the same `instrument_page → build_graph → extract_all_features →
predict` pipeline, and appends phishing hits to CSV. `--fallback-ct-logs`
switches to crt.sh polling for environments that cannot reach CertStream.

For every phishing hit the worker also queries the three blacklists the
paper compares against, implemented in
[`phishxgraph/blacklists.py`](../phishxgraph/blacklists.py):

| Source | Implementation | Auth |
|---|---|---|
| OpenPhish | `feed.txt` polled every 15 min | none |
| PhishTank | `online-valid.csv` (per-entry `verification_time`) | optional app key |
| Google Safe Browsing | v4 `threatMatches:find` API | `GOOGLE_API_KEY` env var |

The output CSV has the per-source first-seen timestamp and the
`lead_minutes` column (positive = PhishXGraph detected before the
earliest blacklist listed the URL). Run
[`phishxgraph/realtime_stats.py`](../phishxgraph/realtime_stats.py) on
that CSV to get the §5.6 headline numbers:

- total phishing detections
- how many were absent from every blacklist at detection time
- how many were later listed (confirmed)
- mean / median lead over earliest blacklist

## Known Constraints preserved from the paper

- Pages that fail to render within 30 s are excluded, matching §5.1.
- Logo content itself is not analysed; only the request for logo
  resources is (noted in Table 1 footnote).
