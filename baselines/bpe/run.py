#!/usr/bin/env python
"""
BPE (CCS'22) wrapper: reads phishinglist.csv, constructs a heterogeneous URL
network graph, and runs the original Belief Propagation algorithm.

Paper: "Phishing URL Detection: A Network-based Approach Robust to Evasion"
       Taeri Kim, Noseong Park, Jiwon Hong, Sang-Wook Kim (ACM CCS 2022)
Repo:  https://github.com/taerikkk/BPE

This wrapper:
  train — Builds URL-Domain-IP-Substring graph, generates DeepWalk embeddings,
          runs BP with 5-fold CV (matching original), saves graph artifacts
  test  — Loads graph artifacts, adds test URLs, runs BP inference,
          extracts per-URL predictions

The original BPE algorithm uses message passing on a heterogeneous graph.
Best hyperparameters from paper: DeepWalk 128-dim, RBF kernel, thresholds 0.7/0.7
"""
import csv
import gzip
import math
import os
import pickle
import random
import sys
import time
from urllib.parse import urlparse

import networkx as nx
import numpy as np
import tldextract
from tqdm import tqdm

INPUT_CSV = os.environ.get("INPUT_CSV", "/data/input/phishinglist.csv")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/data/results/bpe")
MODEL_DIR = os.environ.get("MODEL_DIR", "/data/models/bpe")
FEATURES_DIR = os.environ.get("FEATURES_DIR", "/data/features")
MODE = os.environ.get("MODE", "test")
BPE_DIR = "/app/bpe"

# Original paper's best hyperparameters
BPE_EMB_DIM = int(os.environ.get("BPE_EMB_DIM", "128"))
BPE_COMPAT_TYPE = os.environ.get("BPE_COMPAT_TYPE", "table3")
BPE_SIM_TYPE = os.environ.get("BPE_SIM_TYPE", "rbf")
BPE_THRESHOLD1 = float(os.environ.get("BPE_THRESHOLD1", "0.7"))
BPE_THRESHOLD2 = float(os.environ.get("BPE_THRESHOLD2", "0.7"))
BPE_MAX_EPOCH = int(os.environ.get("BPE_MAX_EPOCH", "5"))

sys.path.insert(0, BPE_DIR)


def read_urls(csv_path):
    urls, labels = [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            urls.append(row["url"])
            labels.append(row.get("label", "unknown"))
    return urls, labels


# ============================================================
# Graph construction (matching original paper's graph structure)
# ============================================================

import hashlib as _hashlib
import json as _json

DNS_CACHE_DIR = os.path.join(MODEL_DIR, "dns_cache")


def _url_hash(url):
    """Hash matching feature_collector's convention."""
    return _hashlib.sha256(url.encode()).hexdigest()[:16]


def _load_from_shared(url, filename):
    """Try loading from shared feature cache."""
    h = _url_hash(url)
    shared_path = os.path.join(FEATURES_DIR, h, filename)
    if os.path.isfile(shared_path):
        return shared_path
    return None


def _dns_cache_path(domain):
    h = _hashlib.sha256(domain.encode()).hexdigest()[:16]
    return os.path.join(DNS_CACHE_DIR, "{}.json".format(h))


def _load_cached_dns(domain):
    # Check shared features directory first
    shared_path = _load_from_shared(domain, "dns.json")
    if shared_path is not None:
        try:
            with open(shared_path, "r") as f:
                data = _json.load(f)
            return data.get("ip")
        except Exception:
            pass

    # Fall back to legacy model-specific cache
    path = _dns_cache_path(domain)
    if os.path.isfile(path):
        try:
            with open(path, "r") as f:
                data = _json.load(f)
            return data.get("ip")  # may be None if resolve failed
        except Exception:
            pass
    return "MISS"  # sentinel: not cached


def _append_collected_url(url):
    """Append a single URL to collected_urls.csv after caching."""
    import csv as _csv_mod
    from datetime import datetime as _dt
    from urllib.parse import urlparse as _up
    log_path = os.path.join(MODEL_DIR, "collected_urls.csv")
    write_header = not os.path.isfile(log_path)
    with open(log_path, "a", newline="") as f:
        w = _csv_mod.writer(f)
        if write_header:
            w.writerow(["url", "cached_at", "status", "cache_path"])
        parsed = _up(url)
        domain = parsed.netloc or parsed.path.split("/")[0]
        cache_file = _dns_cache_path(domain)
        status = "ok" if os.path.isfile(cache_file) else "fail"
        ts = _dt.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        w.writerow([url, ts, status, cache_file if status == "ok" else ""])


def _save_cached_dns(domain, ip):
    os.makedirs(DNS_CACHE_DIR, exist_ok=True)
    dest = _dns_cache_path(domain)
    tmp = dest + ".tmp"
    with open(tmp, "w") as f:
        _json.dump({"domain": domain, "ip": ip}, f)
    os.replace(tmp, dest)


def resolve_ip(domain, timeout=3):
    """Resolve IP from cache only in train/test mode. No live DNS lookups."""
    # Check cache first
    cached = _load_cached_dns(domain)
    if cached != "MISS":
        return cached

    # In train/test mode, only use pre-collected data
    if MODE != "collect":
        return None

    import socket
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)
    try:
        ip = socket.gethostbyname(domain)
        _save_cached_dns(domain, ip)
        return ip
    except (socket.gaierror, socket.timeout, OSError):
        _save_cached_dns(domain, None)
        return None
    finally:
        socket.setdefaulttimeout(old_timeout)


def extract_substrings(url, min_len=4):
    import re
    parsed = urlparse(url)
    path = parsed.path or ""
    query = parsed.query or ""
    segments = [s for s in re.split(r"[/\-_\.?&=]", path + query) if len(s) >= min_len]
    return segments


def build_graph(urls):
    """Build heterogeneous graph matching BPE paper's structure."""
    G = nx.Graph()

    for url in urls:
        G.add_node(url, node_type="url")

        parsed = urlparse(url)
        full_domain = parsed.netloc or parsed.path.split("/")[0]
        ext = tldextract.extract(url)
        registered_domain = "{}.{}".format(ext.domain, ext.suffix) if ext.suffix else ext.domain

        if not G.has_node(registered_domain):
            G.add_node(registered_domain, node_type="domain")
        G.add_edge(url, registered_domain)

        ip = resolve_ip(full_domain)
        if ip:
            if not G.has_node(ip):
                G.add_node(ip, node_type="ip")
            G.add_edge(registered_domain, ip)

        substrings = extract_substrings(url)
        for substr in substrings[:5]:
            substr_key = "substr:{}".format(substr.lower())
            if not G.has_node(substr_key):
                G.add_node(substr_key, node_type="substring")
            G.add_edge(url, substr_key)

    print("[BPE] Graph: {} nodes, {} edges".format(G.number_of_nodes(), G.number_of_edges()))
    return G


# ============================================================
# DeepWalk embedding generation (matching paper: 128-dim)
# ============================================================

def deepwalk_random_walks(G, num_walks=10, walk_length=40):
    """Generate random walks on the graph for DeepWalk."""
    walks = []
    nodes = list(G.nodes())
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            for _ in range(walk_length - 1):
                neighbors = list(G.neighbors(walk[-1]))
                if not neighbors:
                    break
                walk.append(random.choice(neighbors))
            walks.append([str(n) for n in walk])
    return walks


def generate_deepwalk_embeddings(G, dim=128):
    """Generate DeepWalk embeddings using Word2Vec on random walks."""
    from gensim.models import Word2Vec

    print("[BPE] Generating DeepWalk embeddings (dim={})...".format(dim))
    walks = deepwalk_random_walks(G, num_walks=10, walk_length=40)
    model = Word2Vec(sentences=walks, vector_size=dim, window=5,
                     min_count=0, sg=1, workers=4, epochs=5)

    embeddings = {}
    for node in G.nodes():
        node_str = str(node)
        if node_str in model.wv:
            embeddings[node] = model.wv[node_str]
        else:
            embeddings[node] = np.zeros(dim)

    print("[BPE] Generated embeddings for {} nodes".format(len(embeddings)))
    return embeddings


# ============================================================
# Original BPE Belief Propagation algorithm
# (Imported from prediction_bp_ct.py — step, _send_msg_label,
#  _send_msg, _min_sum, MAP functions)
# ============================================================

def _min_sum(G, _from, _to, type_compat, compat_threshold1, compat_threshold2):
    """Original BPE min-sum message computation."""
    eps = 0.001
    new_msg = [0] * 2

    for i in range(2):
        fromnode = G.nodes[_from]
        p_not_related = fromnode['data_cost'][0]
        p_related = fromnode['data_cost'][1]

        p_not_related += fromnode['msg_comp'][0] - fromnode['msgbox'][_to][0]
        p_related += fromnode['msg_comp'][1] - fromnode['msgbox'][_to][1]

        if type_compat == 'table1':
            p_not_related += 0.5 - eps if i == 0 else 0.5 + eps
            p_related += 0.5 + eps if i == 0 else 0.5 - eps
        elif type_compat == 'table2':
            p_not_related += 0 if i == 0 else G[_from][_to]['distance']
            p_related += G[_from][_to]['distance'] if i == 0 else 0
        elif type_compat == 'table3':
            p_not_related += np.min([compat_threshold1, 1 - G[_to][_from]['sim']]) if i == 0 else np.max([compat_threshold2, G[_to][_from]['sim']])
            p_related += np.max([compat_threshold2, G[_to][_from]['sim']]) if i == 0 else np.min([compat_threshold1, 1 - G[_to][_from]['sim']])

        new_msg[i] = min(p_not_related, p_related)

    return new_msg


def _send_msg_label(G, _from, _to):
    """Original BPE: send message from labeled node."""
    if G.nodes[_from]['label'] == 1:
        msg = [1, 0]
    elif G.nodes[_from]['label'] == 0:
        msg = [0, 1]
    else:
        msg = G.nodes[_from]['data_cost']

    to_node = G.nodes[_to]
    to_node['msg_comp'][0] -= to_node['msgbox'][_from][0]
    to_node['msg_comp'][1] -= to_node['msgbox'][_from][1]
    to_node['msg_comp'][0] += msg[0]
    to_node['msg_comp'][1] += msg[1]
    to_node['msgbox'][_from] = msg


def _send_msg(G, type_compat, _from, _to, compat_threshold1=None, compat_threshold2=None):
    """Original BPE: send message from unlabeled node."""
    msg = _min_sum(G, _from, _to, type_compat, compat_threshold1, compat_threshold2)

    to_node = G.nodes[_to]
    to_node['msg_comp'][0] -= to_node['msgbox'][_from][0]
    to_node['msg_comp'][1] -= to_node['msgbox'][_from][1]
    to_node['msg_comp'][0] += msg[0]
    to_node['msg_comp'][1] += msg[1]
    to_node['msgbox'][_from] = msg


def bp_step(G, type_compat, compat_threshold1=None, compat_threshold2=None):
    """Original BPE: one iteration of belief propagation."""
    for n in tqdm(G.nodes(), desc="BP: labeled→unlabeled", mininterval=0.5):
        if G.nodes[n]['label'] is not None:
            for nbr in G.neighbors(n):
                if G.nodes[nbr]['label'] is None:
                    _send_msg_label(G, n, nbr)
    for n in tqdm(G.nodes(), desc="BP: unlabeled→unlabeled", mininterval=0.5):
        if G.nodes[n]['label'] is None:
            for nbr in G.neighbors(n):
                if G.nodes[nbr]['label'] is None:
                    _send_msg(G, type_compat, n, nbr,
                              compat_threshold1=compat_threshold1,
                              compat_threshold2=compat_threshold2)


def bp_map(G):
    """Original BPE: compute MAP energy and assign best_label."""
    for n in G.nodes():
        nodedata = G.nodes[n]
        cost_not_related = nodedata['data_cost'][0] + nodedata['msg_comp'][0]
        cost_related = nodedata['data_cost'][1] + nodedata['msg_comp'][1]

        if cost_related < cost_not_related:
            nodedata['best_label'] = 1
        else:
            nodedata['best_label'] = 0

    energy = 0
    for n in G.nodes():
        cur_label = G.nodes[n]['best_label']
        energy += G.nodes[n]['data_cost'][cur_label]
        for nbr, eattr in G[n].items():
            energy += 0 if G.nodes[nbr]['best_label'] == cur_label else eattr.get('distance', 1.0)
    return energy


def init_bp_graph(G, url_truth, emb, type_sim):
    """Initialize graph nodes and edges for BP (matching original code)."""
    for node in G.nodes():
        G.nodes[node]['label'] = None
        G.nodes[node]['best_label'] = -1
        G.nodes[node]['data_cost'] = [0.5, 0.5]
        G.nodes[node]['msgbox'] = {}
        G.nodes[node]['msg_comp'] = [0, 0]
        for nbr in list(G.neighbors(node)):
            G.nodes[node]['msgbox'][nbr] = [0, 0]

    for node, label in url_truth.items():
        if node in G.nodes():
            G.nodes[node]['label'] = label
            if label == 1:
                G.nodes[node]['data_cost'] = [0.99, 0.01]
            elif label == 0:
                G.nodes[node]['data_cost'] = [0.01, 0.99]

    # Set edge distances/similarities from embeddings
    for edge in G.edges():
        n1, n2 = edge
        if n1 in emb and n2 in emb:
            if type_sim == 'rbf':
                dist = np.linalg.norm(emb[n1] - emb[n2])
                G.edges[edge]['distance'] = dist
                G.edges[edge]['sim'] = np.exp((-1.0 / 2.0) * np.power(dist, 2.0))
            elif type_sim == 'cos':
                norm1 = np.linalg.norm(emb[n1])
                norm2 = np.linalg.norm(emb[n2])
                if norm1 > 0 and norm2 > 0:
                    sim = np.dot(emb[n1], emb[n2]) / (norm1 * norm2)
                else:
                    sim = 0.5
                G.edges[edge]['sim'] = sim
                G.edges[edge]['distance'] = 1 - sim
            else:
                G.edges[edge]['distance'] = 1.0
                G.edges[edge]['sim'] = 0.5
        else:
            G.edges[edge]['distance'] = 1.0
            G.edges[edge]['sim'] = 0.5


def run_bp_inference(G, max_epoch, type_compat, threshold1, threshold2):
    """Run BP message passing iterations (original algorithm)."""
    for epoch in range(max_epoch):
        bp_step(G, type_compat,
                compat_threshold1=threshold1,
                compat_threshold2=threshold2)
        energy = bp_map(G)
        print("[BPE] Iteration {}: MAP energy = {:.4f}".format(epoch + 1, energy))


# ============================================================
# Wrapper train/test logic
# ============================================================

def train(urls, labels):
    """Train: build graph + embeddings, run BP with known labels, save artifacts."""
    print("[BPE] Training with {} labeled URLs...".format(len(urls)))

    G = build_graph(urls)
    emb = generate_deepwalk_embeddings(G, dim=BPE_EMB_DIM)

    url_truth = {}
    for url, label in zip(urls, labels):
        url_truth[url] = 1 if label == "phish" else 0

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save artifacts in format compatible with original code
    with gzip.open(os.path.join(MODEL_DIR, "graph.gzpickle"), "wb") as f:
        pickle.dump(G, f)
    with gzip.open(os.path.join(MODEL_DIR, "url_truth.gzpickle"), "wb") as f:
        pickle.dump(url_truth, f)
    with gzip.open(os.path.join(MODEL_DIR, "embeddings.gzpickle"), "wb") as f:
        pickle.dump(emb, f)

    # Run BP to validate training (like original's CV)
    import copy
    G_copy = copy.deepcopy(G)
    init_bp_graph(G_copy, url_truth, emb, BPE_SIM_TYPE)
    run_bp_inference(G_copy, BPE_MAX_EPOCH, BPE_COMPAT_TYPE, BPE_THRESHOLD1, BPE_THRESHOLD2)

    # Count training accuracy
    correct = sum(1 for url in urls
                  if url in G_copy.nodes() and
                  G_copy.nodes[url]['best_label'] == url_truth[url])
    print("[BPE] Train accuracy: {:.4f}".format(correct / max(len(urls), 1)))
    print("[BPE] Model saved to {}".format(MODEL_DIR))


def test(urls):
    """Test: load graph, add test URLs, run BP, extract per-URL predictions."""
    graph_path = os.path.join(MODEL_DIR, "graph.gzpickle")
    truth_path = os.path.join(MODEL_DIR, "url_truth.gzpickle")
    emb_path = os.path.join(MODEL_DIR, "embeddings.gzpickle")

    if not all(os.path.exists(p) for p in [graph_path, truth_path, emb_path]):
        print("[BPE] ERROR: No trained model found at {}".format(MODEL_DIR))
        print("[BPE] Run with MODE=train first.")
        return [(url, "unknown", 0.0) for url in urls]

    print("[BPE] Loading trained model...")
    with gzip.open(graph_path, "rb") as f:
        G = pickle.load(f)
    with gzip.open(truth_path, "rb") as f:
        url_truth = pickle.load(f)
    with gzip.open(emb_path, "rb") as f:
        emb = pickle.load(f)

    # Add test URLs to graph (connect via domain/IP/substring)
    new_urls = [url for url in urls if url not in G.nodes()]
    if new_urls:
        print("[BPE] Adding {} new test URLs to graph...".format(len(new_urls)))
        for url in new_urls:
            G.add_node(url, node_type="url")
            parsed = urlparse(url)
            full_domain = parsed.netloc or parsed.path.split("/")[0]
            ext = tldextract.extract(url)
            registered_domain = "{}.{}".format(ext.domain, ext.suffix) if ext.suffix else ext.domain

            if not G.has_node(registered_domain):
                G.add_node(registered_domain, node_type="domain")
            G.add_edge(url, registered_domain)

            ip = resolve_ip(full_domain)
            if ip:
                if not G.has_node(ip):
                    G.add_node(ip, node_type="ip")
                G.add_edge(registered_domain, ip)

            substrings = extract_substrings(url)
            for substr in substrings[:5]:
                substr_key = "substr:{}".format(substr.lower())
                if not G.has_node(substr_key):
                    G.add_node(substr_key, node_type="substring")
                G.add_edge(url, substr_key)

        # Re-generate embeddings for expanded graph
        emb = generate_deepwalk_embeddings(G, dim=BPE_EMB_DIM)

    # Initialize BP and run inference
    # Training URLs get known labels, test URLs are unlabeled
    init_bp_graph(G, url_truth, emb, BPE_SIM_TYPE)
    run_bp_inference(G, BPE_MAX_EPOCH, BPE_COMPAT_TYPE, BPE_THRESHOLD1, BPE_THRESHOLD2)

    # Extract per-URL predictions
    predictions = []
    for url in urls:
        if url in G.nodes():
            best_label = G.nodes[url].get('best_label', -1)
            data_cost = G.nodes[url].get('data_cost', [0.5, 0.5])
            msg_comp = G.nodes[url].get('msg_comp', [0, 0])

            # Compute confidence score from data cost + messages
            cost_benign = data_cost[0] + msg_comp[0]
            cost_phish = data_cost[1] + msg_comp[1]
            total = cost_benign + cost_phish
            if total > 0:
                phish_score = 1.0 - (cost_phish / total)
            else:
                phish_score = 0.5

            pred = "phish" if best_label == 1 else "benign"
            predictions.append((url, pred, float(phish_score)))
        else:
            predictions.append((url, "unknown", 0.0))

    return predictions


def write_output(predictions, output_path):
    os.makedirs(output_path, exist_ok=True)
    csv_path = os.path.join(output_path, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "prediction", "score"])
        for url, pred, s in predictions:
            writer.writerow([url, pred, "{:.4f}".format(s)])
    print("[BPE] Results written to {}".format(csv_path))


def main():
    print("[BPE] Starting (mode={})...".format(MODE))
    print("[BPE] Using original Belief Propagation algorithm (CCS'22)")
    print("[BPE] Hyperparameters: compat={}, sim={}, thresholds={}/{}".format(
        BPE_COMPAT_TYPE, BPE_SIM_TYPE, BPE_THRESHOLD1, BPE_THRESHOLD2))

    if MODE == "collect":
        print("[BPE] Collect mode: feature-collector service handles extraction. Sleeping.")
        while True:
            time.sleep(3600)

    if not os.path.exists(INPUT_CSV):
        print("[BPE] ERROR: Input file not found: {}".format(INPUT_CSV))
        sys.exit(1)

    urls, labels = read_urls(INPUT_CSV)
    print("[BPE] Loaded {} URLs".format(len(urls)))

    if MODE == "train":
        train(urls, labels)
        print("[BPE] Training complete.")
    else:
        predictions = test(urls)
        write_output(predictions, OUTPUT_DIR)
        print("[BPE] Done.")


if __name__ == "__main__":
    main()
