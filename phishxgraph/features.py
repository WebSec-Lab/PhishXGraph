"""
Feature extraction (§4.3, Tables 9-11).

101 features total, split into three categories:
- Content (42): URL lexical/entropy/token + HTML phishing cue/content
- Structural (43): Graph topology, DOM shape, centrality/hierarchy,
  node type distribution, edge type distribution
- Behavioral (16): Subgraph/path, edge interaction, data flow

Feature names match the paper exactly. The constants
:data:`FEATURE_NAMES` (ordered) and :data:`FEATURE_GROUPS` (category map)
expose the canonical ordering so train/test matrices stay consistent.
"""
import math
import re
from collections import Counter
from urllib.parse import urlparse

import networkx as nx
import numpy as np

from .constants import (
    ACTION_KEYWORDS,
    ALL_PHISHING_KEYWORDS,
    BPE_SUSPICIOUS_WORDS,
    BRAND_KEYWORDS,
    CONTENT_TEXT_TAGS,
    CREDENTIAL_KEYWORDS,
    DIV_SPAN_TAGS,
    EDGE_TYPE_ACCESS,
    EDGE_TYPE_EXECUTE,
    EDGE_TYPE_REQUEST,
    EDGE_TYPE_STRUCTURE,
    FORM_TAGS,
    INPUT_TAGS,
    INTERACTIVE_TAGS,
    MEDIA_TAGS,
    NODE_TYPE_HTML,
    NODE_TYPE_NETWORK,
    NODE_TYPE_SCRIPT,
    NODE_TYPE_STORAGE,
    SEMANTIC_TAGS,
    SUSPICIOUS_TLDS,
    TABLE_TAGS,
    URGENCY_KEYWORDS,
    URL_TOKEN_STOP_WORDS,
)


def _safe_div(a, b, default=0.0):
    return a / b if b > 0 else default


def _shannon_entropy(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return entropy


# ============================================================
# Content features (42)  — §4.3, Table 9
# ============================================================

def _content_features(G, url, page_domain, soup):
    parsed = urlparse(url)
    domain = parsed.netloc.split(":")[0]

    feats = {
        # HTML phishing cue (8)
        "phishing_keyword_in_nodes": 0,
        "credential_keyword_depth": 0,
        "brand_keyword_in_nodes": 0,
        "brand_in_text_not_domain": 0,
        "urgency_keyword_ratio": 0,
        "external_brand_resource": 0,
        "action_keyword_in_form": 0,
        "obfuscation_indicator": 0,
        # HTML content (10) — partial (rest assigned below)
        "data_uri_node_ratio": 0,
        "empty_link_node_ratio": 0,
        # URL lexical (19)
        "url_length": len(url),
        "domain_length": len(domain),
        "path_length": len(parsed.path),
        "num_dots_in_url": url.count("."),
        "num_hyphens_in_domain": domain.count("-"),
        "num_digits_in_domain": sum(c.isdigit() for c in domain),
        "num_digits_in_url": sum(c.isdigit() for c in url),
        "num_slashes_in_url": url.count("/"),
        "num_slashes_in_domain": domain.count("/"),
        "has_at_symbol": int("@" in url),
        "has_double_slash_redirect": int("//" in parsed.path),
        "has_ip_address": int(bool(re.match(
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", domain))),
        "is_https": int(parsed.scheme == "https"),
        "has_suspicious_tld": int(any(
            domain.endswith(t) for t in SUSPICIOUS_TLDS)),
        "num_subdomains": max(0, domain.count(".") - 1),
        "path_depth": len([t for t in parsed.path.split("/") if t]),
        "query_length": len(parsed.query),
        "tld_length": len(domain.split(".")[-1]) if domain else 0,
        "num_special_chars": sum(1 for c in url if c in "@_~!$&*"),
        # URL entropy / token (5)
        "url_entropy": 0.0,
        "domain_entropy": 0.0,
        "num_url_tokens": 0,
        "num_suspicious_tokens": 0,
        "suspicious_token_ratio": 0.0,
        # HTML content (10) remainder
        "html_length": 0,
        "num_password_inputs": 0,
        "num_hidden_inputs": 0,
        "num_iframes": 0,
        "has_title": 0,
        "title_length": 0,
        "has_favicon": 0,
        "favicon_external": 0,
    }

    if url:
        freq = Counter(url)
        length = len(url)
        feats["url_entropy"] = round(-sum(
            (c / length) * math.log2(c / length)
            for c in freq.values() if c > 0
        ), 4)

    if domain:
        freq = Counter(domain.lower())
        length = len(domain)
        feats["domain_entropy"] = round(-sum(
            (c / length) * math.log2(c / length)
            for c in freq.values() if c > 0
        ), 4)

    tokens = re.split(r"[./\-_?=&%#@:]+", url)
    substrings = [t.lower() for t in tokens
                  if len(t) > 2 and t.lower() not in URL_TOKEN_STOP_WORDS]
    matching = set(substrings) & BPE_SUSPICIOUS_WORDS
    feats["num_url_tokens"] = len(substrings)
    feats["num_suspicious_tokens"] = len(matching)
    feats["suspicious_token_ratio"] = round(
        _safe_div(len(matching), max(len(substrings), 1)), 4)

    if soup is None:
        return feats

    n_nodes = max(G.number_of_nodes(), 1)
    page_text = soup.get_text(separator=" ", strip=True).lower()

    keyword_node_count = 0
    for _, data in G.nodes(data=True):
        if data.get("type") != NODE_TYPE_HTML:
            continue
        node_text = " ".join(
            str(data.get(a, "")).lower()
            for a in ("id", "class", "src", "href", "action")
        )
        if any(kw in node_text for kw in ALL_PHISHING_KEYWORDS):
            keyword_node_count += 1
    feats["phishing_keyword_in_nodes"] = round(
        _safe_div(keyword_node_count, n_nodes), 4)

    cred_depths = []
    for tag in soup.find_all(True):
        tag_text = tag.get_text(strip=True).lower()
        if any(kw in tag_text for kw in CREDENTIAL_KEYWORDS):
            depth = 0
            parent = tag.parent
            while parent and parent.name and parent.name != "[document]":
                depth += 1
                parent = parent.parent
            cred_depths.append(depth)
    feats["credential_keyword_depth"] = round(
        _safe_div(sum(cred_depths), len(cred_depths)), 2
    ) if cred_depths else 0

    brand_count = 0
    all_tags = soup.find_all(True)
    total_tags = max(len(all_tags), 1)
    for tag in all_tags:
        tag_text = (tag.get_text(strip=True) + " " +
                    str(tag.get("alt", "")) + " " +
                    str(tag.get("title", ""))).lower()
        if any(kw in tag_text for kw in BRAND_KEYWORDS):
            brand_count += 1
    feats["brand_keyword_in_nodes"] = round(
        _safe_div(brand_count, total_tags), 4)

    domain_lower = page_domain.lower()
    feats["brand_in_text_not_domain"] = int(any(
        kw in page_text and kw not in domain_lower
        for kw in BRAND_KEYWORDS
    ))

    urgency_count = 0
    for tag in all_tags:
        if any(kw in tag.get_text(strip=True).lower() for kw in URGENCY_KEYWORDS):
            urgency_count += 1
    feats["urgency_keyword_ratio"] = round(
        _safe_div(urgency_count, total_tags), 4)

    logo_keywords = ["logo", "brand", "icon"]
    for tag in soup.find_all(["img", "link", "svg"]):
        tag_attrs = " ".join(
            str(tag.get(a, "")).lower()
            for a in ("class", "alt", "id", "rel", "aria-label")
        )
        if any(kw in tag_attrs for kw in logo_keywords):
            src = tag.get("src", "") or tag.get("href", "")
            if src and src.startswith("http") and page_domain not in src:
                feats["external_brand_resource"] = 1
                break

    forms = soup.find_all("form")
    form_with_action_kw = 0
    for form in forms:
        form_text = form.get_text(strip=True).lower()
        if any(kw in form_text for kw in ACTION_KEYWORDS):
            form_with_action_kw += 1
    feats["action_keyword_in_form"] = round(
        _safe_div(form_with_action_kw, max(len(forms), 1)), 4)

    obfuscation_patterns = re.compile(
        r"(eval\s*\(|atob\s*\(|fromCharCode|unescape\s*\(|\\x[0-9a-f]{2})", re.I)
    inline_scripts = soup.find_all("script")
    obfus_count = 0
    total_scripts = 0
    for s in inline_scripts:
        code = s.string or ""
        if code.strip():
            total_scripts += 1
            if obfuscation_patterns.search(code):
                obfus_count += 1
    feats["obfuscation_indicator"] = round(
        _safe_div(obfus_count, max(total_scripts, 1)), 4)

    data_uri_count = 0
    for _, data in G.nodes(data=True):
        for attr in ("src", "href"):
            val = str(data.get(attr, ""))
            if val.startswith("data:"):
                data_uri_count += 1
                break
    feats["data_uri_node_ratio"] = round(
        _safe_div(data_uri_count, n_nodes), 4)

    link_nodes = [
        (n, d) for n, d in G.nodes(data=True) if d.get("tag") == "a"
    ]
    empty_link_count = 0
    for _, d in link_nodes:
        href = str(d.get("href", "")).strip()
        if href in ("#", "") or href.startswith("javascript:"):
            empty_link_count += 1
    feats["empty_link_node_ratio"] = round(
        _safe_div(empty_link_count, max(len(link_nodes), 1)), 4)

    feats["html_length"] = len(str(soup))
    feats["num_password_inputs"] = len(
        soup.find_all("input", {"type": "password"}))
    feats["num_hidden_inputs"] = len(
        soup.find_all("input", {"type": "hidden"}))
    feats["num_iframes"] = len(soup.find_all("iframe"))

    title = soup.find("title")
    feats["has_title"] = int(title is not None)
    feats["title_length"] = (
        len(title.string) if title and title.string else 0)

    favicon = soup.find("link", rel=re.compile(r"icon", re.I))
    feats["has_favicon"] = int(favicon is not None)
    if favicon:
        href = favicon.get("href", "")
        feats["favicon_external"] = int(
            href.startswith("http") and page_domain not in href)

    return feats


# ============================================================
# Structural features (43) — §4.3, Table 10
# ============================================================

def _find_root(G):
    for n, d in G.nodes(data=True):
        if d.get("type") == NODE_TYPE_HTML and G.in_degree(n) == 0:
            return n
    for n, d in G.nodes(data=True):
        if d.get("type") == NODE_TYPE_HTML:
            return n
    return None


def _bfs_depths(G, source, edge_type=None):
    depths = {source: 0}
    queue = [source]
    while queue:
        current = queue.pop(0)
        for succ in G.successors(current):
            if succ in depths:
                continue
            if edge_type is not None:
                edge_data = G.edges[current, succ]
                if edge_data.get("type") != edge_type:
                    continue
            depths[succ] = depths[current] + 1
            queue.append(succ)
    return depths


def _topology_features(G):
    """Graph topology (9) + DOM shape (4) + Centrality/hierarchy (12) = 25."""
    feats = {
        # Graph topology (9)
        "graph_density": 0,
        "graph_diameter": 0,
        "graph_avg_shortest_path": 0,
        "graph_clustering_coefficient": 0,
        "graph_num_components": 0,
        "graph_largest_component_ratio": 0,
        "graph_assortativity": 0,
        "graph_transitivity": 0,
        "average_degree_connectivity": 0,
        # DOM shape (4)
        "dom_depth_max": 0,
        "dom_depth_avg": 0,
        "dom_breadth_max": 0,
        "dom_leaf_ratio": 0,
        # Centrality / hierarchy (12)
        "closeness_centrality_mean": 0,
        "closeness_centrality_std": 0,
        "closeness_centrality_max": 0,
        "closeness_centrality_min": 0,
        "ancestors_mean": 0,
        "ancestors_std": 0,
        "ancestors_max": 0,
        "ancestors_min": 0,
        "descendants_mean": 0,
        "descendants_std": 0,
        "descendants_max": 0,
        "descendants_min": 0,
    }
    n_nodes = G.number_of_nodes()
    if n_nodes == 0:
        return feats

    feats["graph_density"] = round(nx.density(G), 6)

    wcc = list(nx.weakly_connected_components(G))
    feats["graph_num_components"] = len(wcc)
    largest_cc_nodes = max(wcc, key=len) if wcc else set()
    feats["graph_largest_component_ratio"] = round(
        _safe_div(len(largest_cc_nodes), n_nodes), 4)

    if len(largest_cc_nodes) > 1:
        sub = G.subgraph(largest_cc_nodes).to_undirected()
        try:
            feats["graph_diameter"] = nx.diameter(sub)
            feats["graph_avg_shortest_path"] = round(
                nx.average_shortest_path_length(sub), 4)
        except Exception:
            pass

    try:
        feats["graph_clustering_coefficient"] = round(
            nx.average_clustering(G.to_undirected()), 4)
    except Exception:
        pass

    root_nid = _find_root(G)
    html_nodes = [n for n, d in G.nodes(data=True)
                  if d.get("type") == NODE_TYPE_HTML]
    if root_nid and html_nodes:
        depths = _bfs_depths(G, root_nid, edge_type=EDGE_TYPE_STRUCTURE)
        html_depths = [depths[n] for n in html_nodes if n in depths]
        if html_depths:
            feats["dom_depth_max"] = max(html_depths)
            feats["dom_depth_avg"] = round(
                sum(html_depths) / len(html_depths), 2)

    max_children = 0
    for n in html_nodes:
        children = [
            s for s in G.successors(n)
            if G.nodes[s].get("type") == NODE_TYPE_HTML
            and G.edges[n, s].get("type") == EDGE_TYPE_STRUCTURE
        ]
        max_children = max(max_children, len(children))
    feats["dom_breadth_max"] = max_children

    leaf_count = sum(1 for n in G.nodes() if G.out_degree(n) == 0)
    feats["dom_leaf_ratio"] = round(_safe_div(leaf_count, n_nodes), 4)

    try:
        feats["graph_assortativity"] = round(
            nx.degree_assortativity_coefficient(G), 4)
    except Exception:
        pass

    try:
        feats["graph_transitivity"] = round(
            nx.transitivity(G.to_undirected()), 4)
    except Exception:
        pass

    try:
        adc = nx.average_degree_connectivity(G)
        feats["average_degree_connectivity"] = (
            round(sum(adc.values()) / len(adc), 4) if adc else 0)
    except Exception:
        pass

    try:
        cc = nx.closeness_centrality(G)
        if cc:
            vals = np.array(list(cc.values()))
            feats["closeness_centrality_mean"] = round(float(vals.mean()), 4)
            feats["closeness_centrality_std"] = round(float(vals.std()), 4)
            feats["closeness_centrality_max"] = round(float(vals.max()), 4)
            feats["closeness_centrality_min"] = round(float(vals.min()), 4)
    except Exception:
        pass

    try:
        anc_counts = np.array([len(nx.ancestors(G, n)) for n in G.nodes()])
        if len(anc_counts) > 0:
            feats["ancestors_mean"] = round(float(anc_counts.mean()), 4)
            feats["ancestors_std"] = round(float(anc_counts.std()), 4)
            feats["ancestors_max"] = int(anc_counts.max())
            feats["ancestors_min"] = int(anc_counts.min())
    except Exception:
        pass

    try:
        desc_counts = np.array([len(nx.descendants(G, n)) for n in G.nodes()])
        if len(desc_counts) > 0:
            feats["descendants_mean"] = round(float(desc_counts.mean()), 4)
            feats["descendants_std"] = round(float(desc_counts.std()), 4)
            feats["descendants_max"] = int(desc_counts.max())
            feats["descendants_min"] = int(desc_counts.min())
    except Exception:
        pass

    return feats


def _distribution_features(G):
    """Node type distribution (11)."""
    feats = {
        "html_node_ratio": 0.0,
        "script_node_ratio": 0.0,
        "request_node_ratio": 0.0,
        "storage_node_ratio": 0.0,
        "form_node_ratio": 0.0,
        "input_node_ratio": 0.0,
        "interactive_node_ratio": 0.0,
        "semantic_tag_ratio": 0.0,
        "div_span_ratio": 0.0,
        "media_node_ratio": 0.0,
        "table_node_ratio": 0.0,
    }
    n_nodes = max(G.number_of_nodes(), 1)

    type_counts = Counter()
    tag_counts = Counter()
    html_count = 0

    for _, data in G.nodes(data=True):
        ntype = data.get("type", NODE_TYPE_HTML)
        type_counts[ntype] += 1
        if ntype == NODE_TYPE_HTML:
            html_count += 1
            tag = data.get("tag", "").lower()
            if tag:
                tag_counts[tag] += 1

    html_count = max(html_count, 1)

    feats["html_node_ratio"] = round(
        _safe_div(type_counts[NODE_TYPE_HTML], n_nodes), 4)
    feats["script_node_ratio"] = round(
        _safe_div(type_counts[NODE_TYPE_SCRIPT], n_nodes), 4)
    feats["request_node_ratio"] = round(
        _safe_div(type_counts[NODE_TYPE_NETWORK], n_nodes), 4)
    feats["storage_node_ratio"] = round(
        _safe_div(type_counts[NODE_TYPE_STORAGE], n_nodes), 4)

    def _tag_group_ratio(tag_set):
        return round(_safe_div(
            sum(tag_counts.get(t, 0) for t in tag_set), html_count), 4)

    feats["form_node_ratio"] = _tag_group_ratio(FORM_TAGS)
    feats["input_node_ratio"] = _tag_group_ratio(INPUT_TAGS)
    feats["interactive_node_ratio"] = _tag_group_ratio(INTERACTIVE_TAGS)
    feats["semantic_tag_ratio"] = _tag_group_ratio(SEMANTIC_TAGS)
    feats["div_span_ratio"] = _tag_group_ratio(DIV_SPAN_TAGS)
    feats["media_node_ratio"] = _tag_group_ratio(MEDIA_TAGS)
    feats["table_node_ratio"] = _tag_group_ratio(TABLE_TAGS)

    return feats


def _edge_features(G):
    """Edge type distribution (7) + edge interaction (2) = 9.

    Paper names used:
      - execute_edge_ratio  (paper §4.3 Table 10)
      - access_edge_ratio   (paper §4.3 Table 10; printed "acess_edge_ratio"
                             in the PDF due to a typo)
    """
    feats = {
        # Edge type distribution (7)
        "structure_edge_ratio": 0,
        "request_edge_ratio": 0,
        "execute_edge_ratio": 0,
        "access_edge_ratio": 0,
        "redirect_edge_ratio": 0,
        "cross_domain_edge_ratio": 0,
        "avg_edge_type_entropy": 0,
        # Edge interaction (2) — paper Table 11
        "form_to_external_edge": 0,
        "script_to_form_path_exists": 0,
    }
    n_edges = max(G.number_of_edges(), 1)

    edge_type_counts = Counter()
    for _, _, data in G.edges(data=True):
        edge_type_counts[data.get("type", "unknown")] += 1

    feats["structure_edge_ratio"] = round(
        _safe_div(edge_type_counts.get(EDGE_TYPE_STRUCTURE, 0), n_edges), 4)
    feats["request_edge_ratio"] = round(
        _safe_div(edge_type_counts.get(EDGE_TYPE_REQUEST, 0), n_edges), 4)
    feats["execute_edge_ratio"] = round(
        _safe_div(edge_type_counts.get(EDGE_TYPE_EXECUTE, 0), n_edges), 4)
    feats["access_edge_ratio"] = round(
        _safe_div(edge_type_counts.get(EDGE_TYPE_ACCESS, 0), n_edges), 4)

    redirect_edge_count = sum(
        1 for _, _, d in G.edges(data=True)
        if d.get("type") == EDGE_TYPE_REQUEST and d.get("is_redirect", False)
    )
    feats["redirect_edge_ratio"] = round(
        _safe_div(redirect_edge_count, n_edges), 4)

    request_edges = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get("type") == EDGE_TYPE_REQUEST
    ]
    tp_count = sum(
        1 for _, v in request_edges
        if G.nodes[v].get("is_third_party", False)
    )
    feats["cross_domain_edge_ratio"] = round(
        _safe_div(tp_count, max(len(request_edges), 1)), 4)

    form_nodes = [n for n, d in G.nodes(data=True) if d.get("tag") == "form"]
    for fn in form_nodes:
        for succ in G.successors(fn):
            if (G.nodes[succ].get("type") == NODE_TYPE_NETWORK
                    and G.nodes[succ].get("is_third_party", False)):
                feats["form_to_external_edge"] = 1
                break
        if feats["form_to_external_edge"]:
            break

    js_nodes = [n for n, d in G.nodes(data=True)
                if d.get("type") == NODE_TYPE_SCRIPT]
    for js in js_nodes:
        for fn in form_nodes:
            try:
                if nx.has_path(G, js, fn):
                    feats["script_to_form_path_exists"] = 1
                    break
            except Exception:
                pass
        if feats["script_to_form_path_exists"]:
            break

    feats["avg_edge_type_entropy"] = round(_shannon_entropy(edge_type_counts), 4)
    return feats


# ============================================================
# Behavioral features (16) — §4.3, Table 11
# ============================================================

def _subgraph_features(G):
    """Subgraph/path (8)."""
    feats = {
        "form_subgraph_size": 0,
        "form_subgraph_ratio": 0,
        "form_subgraph_depth": 0,
        "input_to_root_avg_distance": 0,
        "script_to_input_path_length": -1,
        "resource_dependency_depth": 0,
        "exfil_path_length": 0,
        "non_form_content_ratio": 0,
    }
    n_nodes = max(G.number_of_nodes(), 1)

    form_nodes = [n for n, d in G.nodes(data=True) if d.get("tag") == "form"]
    input_nodes = [n for n, d in G.nodes(data=True)
                   if d.get("tag") in INPUT_TAGS]
    request_nodes = [n for n, d in G.nodes(data=True)
                     if d.get("type") == NODE_TYPE_NETWORK]
    root_nid = _find_root(G)

    all_form_descendants = set()
    max_form_depth = 0
    for fn in form_nodes:
        descendants = set(nx.descendants(G, fn))
        all_form_descendants.update(descendants)
        all_form_descendants.add(fn)
        depths = _bfs_depths(G, fn, edge_type=EDGE_TYPE_STRUCTURE)
        if depths:
            max_form_depth = max(max_form_depth, max(depths.values()))

    feats["form_subgraph_size"] = len(all_form_descendants)
    feats["form_subgraph_ratio"] = round(
        _safe_div(len(all_form_descendants), n_nodes), 4)
    feats["form_subgraph_depth"] = max_form_depth

    if root_nid and input_nodes:
        distances = []
        G_rev = G.reverse()
        for inp in input_nodes:
            try:
                d = nx.shortest_path_length(G_rev, inp, root_nid)
                distances.append(d)
            except nx.NetworkXNoPath:
                pass
        if distances:
            feats["input_to_root_avg_distance"] = round(
                _safe_div(sum(distances), len(distances)), 2)

    js_nodes = [n for n, d in G.nodes(data=True)
                if d.get("type") == NODE_TYPE_SCRIPT]
    if js_nodes and input_nodes:
        path_lengths = []
        for js in js_nodes:
            for inp in input_nodes:
                try:
                    pl = nx.shortest_path_length(G, js, inp)
                    path_lengths.append(pl)
                except nx.NetworkXNoPath:
                    pass
        if path_lengths:
            feats["script_to_input_path_length"] = round(
                _safe_div(sum(path_lengths), len(path_lengths)), 2)

    max_req_depth = 0
    for rn in request_nodes:
        depth = 0
        current = rn
        visited = set()
        while current not in visited:
            visited.add(current)
            req_succs = [
                s for s in G.successors(current)
                if G.nodes[s].get("type") == NODE_TYPE_NETWORK
            ]
            if req_succs:
                current = req_succs[0]
                depth += 1
            else:
                break
        max_req_depth = max(max_req_depth, depth)
    feats["resource_dependency_depth"] = max_req_depth

    if root_nid and form_nodes and input_nodes:
        for fn in form_nodes:
            try:
                r2f = nx.shortest_path_length(G, root_nid, fn)
            except nx.NetworkXNoPath:
                continue
            form_inputs = [inp for inp in input_nodes
                           if inp in all_form_descendants]
            for inp in form_inputs:
                try:
                    f2i = nx.shortest_path_length(G, fn, inp)
                except nx.NetworkXNoPath:
                    continue
                for rn in request_nodes:
                    try:
                        i2r = nx.shortest_path_length(G, inp, rn)
                        total = r2f + f2i + i2r
                        if feats["exfil_path_length"] == 0 or total < feats["exfil_path_length"]:
                            feats["exfil_path_length"] = total
                    except nx.NetworkXNoPath:
                        pass
                for rn in request_nodes:
                    try:
                        f2r = nx.shortest_path_length(G, fn, rn)
                        total = r2f + f2r
                        if feats["exfil_path_length"] == 0 or total < feats["exfil_path_length"]:
                            feats["exfil_path_length"] = total
                    except nx.NetworkXNoPath:
                        pass

    content_outside = 0
    total_content = 0
    for n, d in G.nodes(data=True):
        if d.get("tag") in CONTENT_TEXT_TAGS:
            total_content += 1
            if n not in all_form_descendants:
                content_outside += 1
    feats["non_form_content_ratio"] = round(
        _safe_div(content_outside, max(total_content, 1)), 4)

    return feats


def _flow_features(G):
    """Data flow (6)."""
    feats = {
        "form_data_exfil_path": 0,
        "input_to_request_flow": 0,
        "cookie_set_with_form": 0,
        "storage_to_request_flow": 0,
        "redirect_to_form_pattern": 0,
        "form_to_redirect_pattern": 0,
    }

    form_nodes = [n for n, d in G.nodes(data=True) if d.get("tag") == "form"]
    input_nodes = [n for n, d in G.nodes(data=True)
                   if d.get("tag") in INPUT_TAGS]
    request_nodes = [n for n, d in G.nodes(data=True)
                     if d.get("type") == NODE_TYPE_NETWORK]
    storage_nodes = [n for n, d in G.nodes(data=True)
                     if d.get("type") == NODE_TYPE_STORAGE]
    redirect_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("type") == NODE_TYPE_NETWORK and d.get("is_redirect", False)
    ]
    tp_requests = [
        n for n in request_nodes if G.nodes[n].get("is_third_party", False)
    ]

    for fn in form_nodes:
        for rn in tp_requests:
            try:
                if nx.has_path(G, fn, rn):
                    feats["form_data_exfil_path"] = 1
                    break
            except Exception:
                pass
        if feats["form_data_exfil_path"]:
            break

    for inp in input_nodes:
        for rn in request_nodes:
            try:
                if nx.has_path(G, inp, rn):
                    feats["input_to_request_flow"] = 1
                    break
            except Exception:
                pass
        if feats["input_to_request_flow"]:
            break

    has_cookie_set = any(
        G.nodes[n].get("set_cookie", 0) > 0 for n in storage_nodes
    )
    feats["cookie_set_with_form"] = int(has_cookie_set and len(form_nodes) > 0)

    for sn in storage_nodes:
        for rn in tp_requests:
            try:
                if nx.has_path(G, sn, rn):
                    feats["storage_to_request_flow"] = 1
                    break
            except Exception:
                pass
        if feats["storage_to_request_flow"]:
            break

    for rn in redirect_nodes:
        for fn in form_nodes:
            try:
                if nx.has_path(G, rn, fn):
                    feats["redirect_to_form_pattern"] = 1
                    break
            except Exception:
                pass
        if feats["redirect_to_form_pattern"]:
            break

    for fn in form_nodes:
        for rn in redirect_nodes:
            try:
                if nx.has_path(G, fn, rn):
                    feats["form_to_redirect_pattern"] = 1
                    break
            except Exception:
                pass
        if feats["form_to_redirect_pattern"]:
            break

    return feats


# ============================================================
# Top-level API
# ============================================================

def extract_all_features(G, url, page_domain, soup):
    """Return a dict with all 101 features (paper §4.3)."""
    feats = {}
    feats.update(_content_features(G, url, page_domain, soup))
    feats.update(_topology_features(G))
    feats.update(_distribution_features(G))
    feats.update(_edge_features(G))
    feats.update(_subgraph_features(G))
    feats.update(_flow_features(G))
    return feats


# Canonical feature ordering (used so train/test matrices align).
# Derived by running extract_all_features on an empty graph once.
FEATURE_GROUPS = {
    "content_url_lexical": [
        "url_length", "domain_length", "path_length", "query_length",
        "tld_length", "num_dots_in_url", "num_hyphens_in_domain",
        "num_digits_in_domain", "num_digits_in_url", "num_slashes_in_url",
        "num_slashes_in_domain", "num_subdomains", "path_depth",
        "num_special_chars", "has_at_symbol", "has_double_slash_redirect",
        "has_ip_address", "is_https", "has_suspicious_tld",
    ],
    "content_url_entropy_token": [
        "url_entropy", "domain_entropy", "num_url_tokens",
        "num_suspicious_tokens", "suspicious_token_ratio",
    ],
    "content_html_phishing_cue": [
        "phishing_keyword_in_nodes", "credential_keyword_depth",
        "brand_keyword_in_nodes", "brand_in_text_not_domain",
        "urgency_keyword_ratio", "external_brand_resource",
        "action_keyword_in_form", "obfuscation_indicator",
    ],
    "content_html_content": [
        "data_uri_node_ratio", "empty_link_node_ratio", "html_length",
        "num_password_inputs", "num_hidden_inputs", "num_iframes",
        "has_title", "title_length", "has_favicon", "favicon_external",
    ],
    "structural_graph_topology": [
        "graph_density", "graph_diameter", "graph_avg_shortest_path",
        "graph_clustering_coefficient", "graph_num_components",
        "graph_largest_component_ratio", "graph_assortativity",
        "graph_transitivity", "average_degree_connectivity",
    ],
    "structural_dom_shape": [
        "dom_depth_max", "dom_depth_avg", "dom_breadth_max", "dom_leaf_ratio",
    ],
    "structural_centrality_hierarchy": [
        "closeness_centrality_mean", "closeness_centrality_std",
        "closeness_centrality_max", "closeness_centrality_min",
        "ancestors_mean", "ancestors_std", "ancestors_max", "ancestors_min",
        "descendants_mean", "descendants_std", "descendants_max",
        "descendants_min",
    ],
    "structural_node_type_distribution": [
        "html_node_ratio", "script_node_ratio", "request_node_ratio",
        "storage_node_ratio", "form_node_ratio", "input_node_ratio",
        "interactive_node_ratio", "semantic_tag_ratio", "div_span_ratio",
        "media_node_ratio", "table_node_ratio",
    ],
    "structural_edge_type_distribution": [
        "structure_edge_ratio", "request_edge_ratio", "execute_edge_ratio",
        "access_edge_ratio", "redirect_edge_ratio",
        "cross_domain_edge_ratio", "avg_edge_type_entropy",
    ],
    "behavioral_subgraph_path": [
        "form_subgraph_size", "form_subgraph_ratio", "form_subgraph_depth",
        "input_to_root_avg_distance", "script_to_input_path_length",
        "resource_dependency_depth", "exfil_path_length",
        "non_form_content_ratio",
    ],
    "behavioral_edge_interaction": [
        "form_to_external_edge", "script_to_form_path_exists",
    ],
    "behavioral_data_flow": [
        "form_data_exfil_path", "input_to_request_flow",
        "cookie_set_with_form", "storage_to_request_flow",
        "redirect_to_form_pattern", "form_to_redirect_pattern",
    ],
}

FEATURE_NAMES = [n for group in FEATURE_GROUPS.values() for n in group]
assert len(FEATURE_NAMES) == 101, (
    f"Expected 101 features per paper, got {len(FEATURE_NAMES)}"
)
