# Full feature list (101)

Canonical list of the features extracted by
[`phishxgraph/features.py`](../phishxgraph/features.py) — kept in sync
with the paper's Tables 9–11 (§4.3).

`Type` column: **C** = continuous, **I** = integer, **B** = binary (0/1).

The module-load assertion `len(FEATURE_NAMES) == 101` guarantees this
list and the code stay locked together.

---

## Content features (42) — paper Table 9

### URL lexical (19)

| # | Name | Type | Definition |
|---|---|---|---|
| 1 | `url_length` | I | Number of characters in the URL |
| 2 | `domain_length` | I | Number of characters in the domain |
| 3 | `path_length` | I | Number of characters in the path component |
| 4 | `query_length` | I | Number of characters in the query string |
| 5 | `tld_length` | I | Number of characters in the top-level domain |
| 6 | `num_dots_in_url` | I | Count of `.` characters in the URL |
| 7 | `num_hyphens_in_domain` | I | Count of `-` characters in the domain |
| 8 | `num_digits_in_domain` | I | Count of digit characters in the domain |
| 9 | `num_digits_in_url` | I | Count of digit characters in the URL |
| 10 | `num_slashes_in_url` | I | Count of `/` characters in the URL |
| 11 | `num_slashes_in_domain` | I | Count of `/` characters in the domain |
| 12 | `num_subdomains` | I | Domain dot count minus one |
| 13 | `path_depth` | I | Number of non-empty segments in the path |
| 14 | `num_special_chars` | I | Count of `@_~!$&*` characters in the URL |
| 15 | `has_at_symbol` | B | URL contains `@` |
| 16 | `has_double_slash_redirect` | B | Path contains `//` |
| 17 | `has_ip_address` | B | Domain matches IPv4 pattern |
| 18 | `is_https` | B | Scheme is `https` |
| 19 | `has_suspicious_tld` | B | TLD ∈ `{.tk, .ml, .ga, .cf, .gq, .xyz, .top, .buzz, .club, .work, .info}` |

### URL entropy & token (5)

| # | Name | Type | Definition |
|---|---|---|---|
| 20 | `url_entropy` | C | Shannon entropy of the URL's character distribution |
| 21 | `domain_entropy` | C | Shannon entropy of the domain's character distribution |
| 22 | `num_url_tokens` | I | Token count after splitting on delimiters and filtering stop words (`http`, `www`, `com`, …) |
| 23 | `num_suspicious_tokens` | I | Tokens matching the suspicious word list (`login`, `verify`, `bank`, …) |
| 24 | `suspicious_token_ratio` | C | Fraction of URL tokens matching the suspicious word list |

### HTML phishing cue (8)

| # | Name | Type | Definition |
|---|---|---|---|
| 25 | `phishing_keyword_in_nodes` | C | Ratio of DOM nodes whose attributes contain phishing keywords (credential ∪ urgency ∪ brand) |
| 26 | `credential_keyword_depth` | C | Average DOM depth of elements containing credential keywords |
| 27 | `brand_keyword_in_nodes` | C | Ratio of DOM tags mentioning brand names (paypal, microsoft, …) |
| 28 | `brand_in_text_not_domain` | B | Brand name appears in page text but not in domain |
| 29 | `urgency_keyword_ratio` | C | Ratio of DOM tags containing urgency language (verify, suspend, …) |
| 30 | `external_brand_resource` | B | Logo/icon resource references a brand and loads from an external domain |
| 31 | `action_keyword_in_form` | C | Ratio of forms containing action keywords (submit, continue, click, …) |
| 32 | `obfuscation_indicator` | C | Ratio of inline scripts using obfuscation patterns (`eval(`, `atob(`, `\xNN`) |

### HTML content (10)

| # | Name | Type | Definition |
|---|---|---|---|
| 33 | `data_uri_node_ratio` | C | Fraction of nodes using `data:` URIs |
| 34 | `empty_link_node_ratio` | C | Fraction of `<a>` tags with empty / `javascript:` href |
| 35 | `html_length` | I | Total length of the rendered HTML source |
| 36 | `num_password_inputs` | I | Count of `<input type="password">` |
| 37 | `num_hidden_inputs` | I | Count of `<input type="hidden">` |
| 38 | `num_iframes` | I | Count of `<iframe>` elements |
| 39 | `has_title` | B | `<title>` element exists |
| 40 | `title_length` | I | Character length of `<title>` text |
| 41 | `has_favicon` | B | `<link rel="icon">` element exists |
| 42 | `favicon_external` | B | Favicon URL is on a different domain |

---

## Structural features (43) — paper Table 10

### Graph topology (9)

| # | Name | Type | Definition |
|---|---|---|---|
| 43 | `graph_density` | C | Ratio of edges to all possible directed edges |
| 44 | `graph_diameter` | I | Diameter of the largest weakly connected component (undirected) |
| 45 | `graph_avg_shortest_path` | C | Avg. shortest path length in the largest weakly connected component |
| 46 | `graph_clustering_coefficient` | C | Avg. clustering coefficient (undirected) |
| 47 | `graph_num_components` | I | Number of weakly connected components |
| 48 | `graph_largest_component_ratio` | C | Fraction of nodes in the largest weakly connected component |
| 49 | `graph_assortativity` | C | Degree assortativity coefficient |
| 50 | `graph_transitivity` | C | Transitivity (fraction of possible triangles) |
| 51 | `average_degree_connectivity` | C | Mean of the average neighbor degree per degree class |

### DOM shape (4)

| # | Name | Type | Definition |
|---|---|---|---|
| 52 | `dom_depth_max` | I | Maximum depth of the DOM tree via structure edges |
| 53 | `dom_depth_avg` | C | Average depth across all HTML nodes |
| 54 | `dom_breadth_max` | I | Maximum number of children of any HTML node |
| 55 | `dom_leaf_ratio` | C | Fraction of nodes with out-degree zero |

### Centrality & hierarchy (12)

| # | Name | Type | Definition |
|---|---|---|---|
| 56 | `closeness_centrality_mean` | C | Mean closeness centrality over all nodes |
| 57 | `closeness_centrality_std` | C | Std. dev. of closeness centrality |
| 58 | `closeness_centrality_max` | C | Maximum closeness centrality |
| 59 | `closeness_centrality_min` | C | Minimum closeness centrality |
| 60 | `ancestors_mean` | C | Mean ancestor count per node (directed) |
| 61 | `ancestors_std` | C | Std. dev. of ancestor counts |
| 62 | `ancestors_max` | I | Maximum ancestor count |
| 63 | `ancestors_min` | I | Minimum ancestor count |
| 64 | `descendants_mean` | C | Mean descendant count per node (directed) |
| 65 | `descendants_std` | C | Std. dev. of descendant counts |
| 66 | `descendants_max` | I | Maximum descendant count |
| 67 | `descendants_min` | I | Minimum descendant count |

### Node type distribution (11)

| # | Name | Type | Definition |
|---|---|---|---|
| 68 | `html_node_ratio` | C | Fraction of nodes of type `html` |
| 69 | `script_node_ratio` | C | Fraction of nodes of type `script` |
| 70 | `request_node_ratio` | C | Fraction of nodes of type `network` |
| 71 | `storage_node_ratio` | C | Fraction of nodes of type `storage` |
| 72 | `form_node_ratio` | C | Fraction of HTML nodes with tag ∈ `{form, fieldset, legend}` |
| 73 | `input_node_ratio` | C | Fraction of HTML nodes with tag ∈ `{input, textarea, select, option}` |
| 74 | `interactive_node_ratio` | C | Fraction of HTML nodes with tag ∈ `{button, a, select, input, label}` |
| 75 | `semantic_tag_ratio` | C | Fraction of HTML nodes with tag ∈ `{header, nav, main, article, section, aside, footer}` |
| 76 | `div_span_ratio` | C | Fraction of HTML nodes with tag ∈ `{div, span}` |
| 77 | `media_node_ratio` | C | Fraction of HTML nodes with tag ∈ `{img, video, audio, svg, canvas, picture}` |
| 78 | `table_node_ratio` | C | Fraction of HTML nodes with tag ∈ `{table, tr, td, th, thead, tbody}` |

### Edge type distribution (7)

| # | Name | Type | Definition |
|---|---|---|---|
| 79 | `structure_edge_ratio` | C | Fraction of edges of type `structure` |
| 80 | `request_edge_ratio` | C | Fraction of edges of type `request` |
| 81 | `execute_edge_ratio` | C | Fraction of edges of type `execute` |
| 82 | `access_edge_ratio` | C | Fraction of edges of type `access` (paper Table 10 prints this as "acess_edge_ratio" — typo) |
| 83 | `redirect_edge_ratio` | C | Fraction of all edges that are redirect requests |
| 84 | `cross_domain_edge_ratio` | C | Fraction of request edges targeting third-party domains |
| 85 | `avg_edge_type_entropy` | C | Shannon entropy of the edge type distribution |

---

## Behavioral features (16) — paper Table 11

### Subgraph & path analysis (8)

| # | Name | Type | Definition |
|---|---|---|---|
| 86 | `form_subgraph_size` | I | Total nodes reachable from any form node via directed edges |
| 87 | `form_subgraph_ratio` | C | Fraction of graph nodes reachable from any form node |
| 88 | `form_subgraph_depth` | I | Max depth of form subtrees along structure edges |
| 89 | `input_to_root_avg_distance` | C | Avg. shortest path from input nodes to root (reversed graph) |
| 90 | `script_to_input_path_length` | C | Avg. shortest path from JavaScript nodes to input nodes; `-1` when no path exists |
| 91 | `resource_dependency_depth` | I | Max depth of chained `request → request` edges (including redirects) |
| 92 | `exfil_path_length` | I | Shortest `root → form → input → request` path |
| 93 | `non_form_content_ratio` | C | Fraction of text-content tags (`p`, `h1–h6`, `li`, …) outside the form subgraph |

### Edge interaction patterns (2)

| # | Name | Type | Definition |
|---|---|---|---|
| 94 | `form_to_external_edge` | B | Any form node has a direct third-party request successor |
| 95 | `script_to_form_path_exists` | B | Any JavaScript node can reach a form node via a directed path |

### Data flow (6)

| # | Name | Type | Definition |
|---|---|---|---|
| 96 | `form_data_exfil_path` | B | Directed path from form → third-party request |
| 97 | `input_to_request_flow` | B | Directed path from input → any request |
| 98 | `cookie_set_with_form` | B | Page sets cookies and contains ≥ 1 form |
| 99 | `storage_to_request_flow` | B | Directed path from storage → third-party request |
| 100 | `redirect_to_form_pattern` | B | Directed path from redirect request node → form |
| 101 | `form_to_redirect_pattern` | B | Directed path from form → redirect request node |

---

## Programmatic access

```python
from phishxgraph.features import FEATURE_NAMES, FEATURE_GROUPS

assert len(FEATURE_NAMES) == 101

# All feature names in the canonical order used by the classifier:
for name in FEATURE_NAMES:
    print(name)

# Grouped by paper category:
for group, names in FEATURE_GROUPS.items():
    print(f"{group} ({len(names)})")
    for n in names:
        print(f"  {n}")
```

Dump the table to CSV:

```bash
python -c "
from phishxgraph.features import FEATURE_NAMES, FEATURE_GROUPS
print('index,group,name')
i = 0
for group, names in FEATURE_GROUPS.items():
    for n in names:
        i += 1
        print(f'{i},{group},{n}')
"
```

## Paper numbering note

Paper Tables 10–11 contain two minor typos in the numbering column (the
numbers `70` and `80` each appear twice). This document re-numbers 1–101
to remove the ambiguity while preserving the intended category ordering.
The feature *names* are the exact ones used in the code and in the paper
text.
