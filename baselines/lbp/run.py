#!/usr/bin/env python
"""
LBP (ICC'25) wrapper: Loopy Belief Propagation for phishing URL detection.

Paper: "Efficient Phishing URL Detection Using Graph-based ML and LBP" (ICC 2025)
Repo:  https://github.com/wenyeguo/LBP

This wrapper:
  train — Builds URL-Domain-IP-Substring graph, generates Word2Vec embeddings,
          computes similarity matrix, saves all artifacts for LBP inference
  test  — Loads graph artifacts, adds test URLs, runs LBP message passing
          using original module system, extracts per-URL predictions

The original LBP algorithm uses the module system from the LBP repo:
  - GraphNode: initializes node labels and prior probabilities
  - Message: min-sum message passing between nodes
  - Worker: orchestrates BP iterations with cycle deletion
  - Similarity: edge similarity computation from embeddings

Default hyperparameters from run.sh: word2vec, rbf, sim, threshold 0.5, t1=0.6, t2=1.0
"""
import copy
import csv
import gzip
import os
import pickle
import random
import sys
import time
from urllib.parse import urlparse

import networkx as nx
import numpy as np
import tldextract

INPUT_CSV = os.environ.get("INPUT_CSV", "/data/input/phishinglist.csv")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/data/results/lbp")
MODEL_DIR = os.environ.get("MODEL_DIR", "/data/models/lbp")
FEATURES_DIR = os.environ.get("FEATURES_DIR", "/data/features")
MODE = os.environ.get("MODE", "test")
LBP_DIR = "/app/lbp"

# Original paper's default hyperparameters (from run.sh)
LBP_EMB_DIM = int(os.environ.get("LBP_EMB_DIM", "128"))
LBP_SIM_TYPE = os.environ.get("LBP_SIM_TYPE", "rbf")
LBP_EDGE_POTENTIAL = os.environ.get("LBP_EDGE_POTENTIAL", "sim")
LBP_DELETE_CYCLE = os.environ.get("LBP_DELETE_CYCLE", "False").lower() == "true"
LBP_PRIOR_PROB = os.environ.get("LBP_PRIOR_PROB", "True").lower() == "true"
LBP_CLASSIFY_THRESHOLD = float(os.environ.get("LBP_CLASSIFY_THRESHOLD", "0.5"))
LBP_THRESHOLD1 = float(os.environ.get("LBP_THRESHOLD1", "0.6"))
LBP_THRESHOLD2 = float(os.environ.get("LBP_THRESHOLD2", "1.0"))

# Add LBP repo to path so we can import original modules
sys.path.insert(0, LBP_DIR)


def read_urls(csv_path):
    urls, labels = [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            urls.append(row["url"])
            labels.append(row.get("label", "unknown"))
    return urls, labels


# ============================================================
# Graph construction
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
            return data.get("ip")
        except Exception:
            pass
    return "MISS"


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
    """Build heterogeneous graph matching LBP paper's structure.
    Edge types: URL-Domain, Domain-IP, URL-Substring (matching create_graph.py)"""
    G = nx.Graph()

    for url in urls:
        G.add_node(url, node_type="url")

        parsed = urlparse(url)
        full_domain = parsed.netloc or parsed.path.split("/")[0]
        ext = tldextract.extract(url)
        registered_domain = "{}.{}".format(ext.domain, ext.suffix) if ext.suffix else ext.domain

        # URL <-> Domain edge (matching add_edge_between_url_domain)
        if not G.has_node(registered_domain):
            G.add_node(registered_domain, node_type="domain")
        if not G.has_edge(url, registered_domain):
            G.add_edge(url, registered_domain)

        # Domain <-> IP edge (matching add_edge_between_domain_ip)
        ip = resolve_ip(full_domain)
        if ip:
            if not G.has_node(ip):
                G.add_node(ip, node_type="ip")
            if not G.has_edge(registered_domain, ip):
                G.add_edge(registered_domain, ip)

        # URL <-> Substring edges (matching add_edge_between_url_substring)
        substrings = extract_substrings(url)
        for substr in substrings[:5]:
            substr_key = "substr:{}".format(substr.lower())
            if not G.has_node(substr_key):
                G.add_node(substr_key, node_type="substring")
            if not G.has_edge(url, substr_key):
                G.add_edge(url, substr_key)

    print("[LBP] Graph: {} nodes, {} edges".format(G.number_of_nodes(), G.number_of_edges()))
    return G


# ============================================================
# Word2Vec embedding (matching original embedding/wordbased/)
# ============================================================

def generate_word2vec_embeddings(G, urls, dim=128):
    """Generate Word2Vec embeddings matching original word_doc_emb.py approach.
    Uses DeepWalk-style random walks to generate training sentences."""
    from gensim.models import Word2Vec

    print("[LBP] Generating Word2Vec embeddings (dim={})...".format(dim))
    # Random walks on graph (DeepWalk approach used in original)
    walks = []
    nodes = list(G.nodes())
    for _ in range(10):
        random.shuffle(nodes)
        for node in nodes:
            walk = [str(node)]
            for _ in range(39):
                neighbors = list(G.neighbors(walk[-1]) if walk[-1] in G else [])
                if not neighbors:
                    break
                walk.append(str(random.choice(neighbors)))
            walks.append(walk)

    model = Word2Vec(sentences=walks, vector_size=dim, window=5,
                     min_count=0, sg=1, workers=4, epochs=10)

    embeddings = {}
    for node in G.nodes():
        node_str = str(node)
        if node_str in model.wv:
            embeddings[node] = model.wv[node_str]
        else:
            embeddings[node] = np.zeros(dim)

    print("[LBP] Generated embeddings for {} nodes".format(len(embeddings)))
    return embeddings


# ============================================================
# Similarity computation (matching original calculate_similarities.py)
# ============================================================

def compute_edge_similarity(G, embeddings, sim_type="rbf"):
    """Compute edge similarity matching original Similarity module."""
    similarity = {}
    for edge in G.edges():
        n1, n2 = edge
        edge_key = tuple(sorted(edge))
        v1 = embeddings.get(n1, np.zeros(128))
        v2 = embeddings.get(n2, np.zeros(128))

        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 > 0:
            v1 = v1 / norm1
        if norm2 > 0:
            v2 = v2 / norm2

        if sim_type == "rbf":
            distance = np.linalg.norm(v1 - v2)
            sim = np.exp((-1.0 / 2.0) * np.power(distance, 2.0))
        elif sim_type == "cos":
            if norm1 > 0 and norm2 > 0:
                sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            else:
                sim = 0.0
        else:
            sim = 0.5

        similarity[edge_key] = sim

    print("[LBP] Computed similarity for {} edges".format(len(similarity)))
    return similarity


# ============================================================
# LBP message passing (using original module system if available,
# fallback to reimplementation matching original algorithm)
# ============================================================

def try_import_original_modules():
    """Try to import original LBP modules from cloned repo."""
    try:
        from module.graphNodeModule import GraphNode
        from module.messageModule import Message
        from module.workerModule import Worker
        from module.metricModule import Metric
        from module.predictModule import assign_node_predict_label
        print("[LBP] Successfully imported original LBP modules")
        return True
    except ImportError as e:
        print("[LBP] Could not import original modules: {}".format(e))
        return False


def run_lbp_with_original_modules(G, url_labels, similarity, train_urls, test_urls):
    """Run LBP using original module system."""
    from module.graphNodeModule import GraphNode
    from module.messageModule import Message
    from module.predictModule import assign_node_predict_label

    data = {"train": set(train_urls), "test": set(test_urls)}
    data_suffix = "wrapper"

    # Initialize graph nodes (matching original GraphNode.init_nodes)
    graph_node = GraphNode(G, data_suffix)
    graph_node.set_classify_threshold(LBP_CLASSIFY_THRESHOLD)
    graph_node.init_nodes(data, url_labels)

    if LBP_DELETE_CYCLE:
        # Delete cycles and run message passing (matching worker_cycle)
        G = graph_node.remove_cycles()
        print("[LBP] Removed cycles from graph")

    # Message passing iterations
    unknown_nodes = [n for n in list(G.nodes()) if G.nodes[n]['label'] == 0.5]
    print("[LBP] Running message passing on {} unknown nodes...".format(len(unknown_nodes)))

    max_iterations = 6
    for iteration in range(max_iterations):
        message = Message(G, LBP_EDGE_POTENTIAL, similarity)
        message.set_thresholds(LBP_THRESHOLD1, LBP_THRESHOLD2)
        message.set_message_type('normal')
        outputs = [message.receive_message(node) for node in unknown_nodes]
        message.update_graph_nodes_message(outputs)
        converged = message.count_converged_nodes(unknown_nodes)
        print("[LBP] Iteration {}: converged {}/{}".format(
            iteration, converged, len(unknown_nodes)))

        if converged == len(unknown_nodes):
            message.assignNodeLabels(unknown_nodes, LBP_CLASSIFY_THRESHOLD)
            break
    else:
        # Assign labels after max iterations
        message.assignNodeLabels(unknown_nodes, LBP_CLASSIFY_THRESHOLD)

    if LBP_DELETE_CYCLE:
        # Add cycles back and continue (matching worker_cycle flow)
        graph_node.assign_predicted_label()

    return G


def run_lbp_fallback(G, url_labels, similarity, train_urls, test_urls):
    """Fallback: reimplementation matching original LBP algorithm
    when original modules can't be imported."""
    from module.predictModule import assign_node_predict_label

    # Initialize all nodes (matching GraphNode.init_all_graph_nodes)
    for node in G.nodes():
        G.nodes[node]["label"] = 0.5
        G.nodes[node]["predict_label"] = -1
        G.nodes[node]["prior_probability"] = [0.5, 0.5]
        G.nodes[node]["msg_sum"] = [0, 0]
        G.nodes[node]["msg_nbr"] = {}
        for nbr in list(G.neighbors(node)):
            G.nodes[node]["msg_nbr"][nbr] = [0, 0]

    # Set train node labels (matching GraphNode.init_train_node)
    for node in train_urls:
        if node in G.nodes() and node in url_labels:
            label = url_labels[node]
            G.nodes[node]['label'] = label
            G.nodes[node]['predict_label'] = label
            if label == 1:
                G.nodes[node]['prior_probability'] = [0, 1]
            elif label == 0:
                G.nodes[node]['prior_probability'] = [1, 0]

    # Message passing (simplified version matching Message.min_sum)
    unknown_nodes = [n for n in list(G.nodes()) if G.nodes[n]['label'] == 0.5]
    print("[LBP] Running fallback message passing on {} unknown nodes...".format(len(unknown_nodes)))

    for iteration in range(6):
        prev_graph = copy.deepcopy(G)

        for receiver in unknown_nodes:
            for sender in list(G.neighbors(receiver)):
                msg = _lbp_min_sum(G, sender, receiver, similarity)
                # Update receiver messages
                for i in range(2):
                    G.nodes[receiver]['msg_sum'][i] = round(
                        G.nodes[receiver]['msg_sum'][i]
                        + (-G.nodes[receiver]['msg_nbr'][sender][i])
                        + msg[i], 10)
                    G.nodes[receiver]['msg_nbr'][sender][i] = round(msg[i], 10)

        # Check convergence
        converged = 0
        for node in unknown_nodes:
            k = sum(1 for i in range(2) if abs(G.nodes[node]['msg_sum'][i] - prev_graph.nodes[node]['msg_sum'][i]) < 0.001)
            if k == 2:
                converged += 1
        print("[LBP] Iteration {}: converged {}/{}".format(iteration, converged, len(unknown_nodes)))
        if converged == len(unknown_nodes):
            break

    # Assign predicted labels (matching predictModule.assign_node_predict_label)
    for node in unknown_nodes:
        predict_label = assign_node_predict_label(G.nodes[node], LBP_CLASSIFY_THRESHOLD)
        G.nodes[node]['predict_label'] = predict_label

    return G


def _lbp_min_sum(G, sender, receiver, similarity):
    """Compute min-sum message (matching original Message.min_sum)."""
    send_prior_benign = G.nodes[sender]['prior_probability'][0]
    send_prior_malicious = G.nodes[sender]['prior_probability'][1]
    msg = [0] * 2

    # Sum of neighbor messages excluding receiver
    sum_of_send = [0, 0]
    if G.nodes[sender]['label'] == 0.5:
        sum_of_send[0] = G.nodes[sender]['msg_sum'][0] - G.nodes[sender]['msg_nbr'][receiver][0]
        sum_of_send[1] = G.nodes[sender]['msg_sum'][1] - G.nodes[sender]['msg_nbr'][receiver][1]

    for assume_label in range(2):
        # Edge potential
        edge_key = tuple(sorted([sender, receiver]))
        sim = similarity.get(edge_key, 0.5)

        if LBP_EDGE_POTENTIAL == 't1':
            e = 0.0001
            if assume_label == 0:
                edges = [0.5 - e, 0.5 + e]
            else:
                edges = [0.5 + e, 0.5 - e]
        elif LBP_EDGE_POTENTIAL == 'sim_only':
            if assume_label == 0:
                edges = [1 - sim, sim]
            else:
                edges = [sim, 1 - sim]
        elif LBP_EDGE_POTENTIAL == 'sim':
            if assume_label == 0:
                edges = [min(LBP_THRESHOLD1, 1 - sim), max(LBP_THRESHOLD2, sim)]
            else:
                edges = [max(LBP_THRESHOLD2, sim), min(LBP_THRESHOLD1, 1 - sim)]
        else:
            edges = [0.5, 0.5]

        msg_benign = round(1 - send_prior_benign + edges[0] + sum_of_send[0], 10)
        msg_phishing = round(1 - send_prior_malicious + edges[1] + sum_of_send[1], 10)
        msg[assume_label] = min(msg_benign, msg_phishing)

    return msg


# ============================================================
# Wrapper train/test logic
# ============================================================

def train(urls, labels):
    """Train: build graph + embeddings + similarity, save artifacts."""
    print("[LBP] Training with {} labeled URLs...".format(len(urls)))

    G = build_graph(urls)
    emb = generate_word2vec_embeddings(G, urls, dim=LBP_EMB_DIM)
    similarity = compute_edge_similarity(G, emb, sim_type=LBP_SIM_TYPE)

    url_labels = {}
    for url, label in zip(urls, labels):
        url_labels[url] = 1 if label == "phish" else 0

    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(os.path.join(MODEL_DIR, "graph.pickle"), "wb") as f:
        pickle.dump(G, f)
    with open(os.path.join(MODEL_DIR, "url_labels.pickle"), "wb") as f:
        pickle.dump(url_labels, f)
    with open(os.path.join(MODEL_DIR, "similarity.pickle"), "wb") as f:
        pickle.dump(similarity, f)
    with open(os.path.join(MODEL_DIR, "embeddings.pickle"), "wb") as f:
        pickle.dump(emb, f)

    print("[LBP] Model artifacts saved to {}".format(MODEL_DIR))


def test(urls):
    """Test: load graph, add test URLs, run LBP, extract predictions."""
    graph_path = os.path.join(MODEL_DIR, "graph.pickle")
    labels_path = os.path.join(MODEL_DIR, "url_labels.pickle")
    sim_path = os.path.join(MODEL_DIR, "similarity.pickle")
    emb_path = os.path.join(MODEL_DIR, "embeddings.pickle")

    if not all(os.path.exists(p) for p in [graph_path, labels_path, sim_path]):
        print("[LBP] ERROR: No trained model found at {}".format(MODEL_DIR))
        print("[LBP] Run with MODE=train first.")
        return [(url, "unknown", 0.0) for url in urls]

    print("[LBP] Loading trained model...")
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    with open(labels_path, "rb") as f:
        url_labels = pickle.load(f)
    with open(sim_path, "rb") as f:
        similarity = pickle.load(f)

    emb = {}
    if os.path.exists(emb_path):
        with open(emb_path, "rb") as f:
            emb = pickle.load(f)

    # Add test URLs to graph
    new_urls = [url for url in urls if url not in G.nodes()]
    if new_urls:
        print("[LBP] Adding {} new test URLs to graph...".format(len(new_urls)))
        for url in new_urls:
            G.add_node(url, node_type="url")
            parsed = urlparse(url)
            full_domain = parsed.netloc or parsed.path.split("/")[0]
            ext = tldextract.extract(url)
            registered_domain = "{}.{}".format(ext.domain, ext.suffix) if ext.suffix else ext.domain

            if not G.has_node(registered_domain):
                G.add_node(registered_domain, node_type="domain")
            if not G.has_edge(url, registered_domain):
                G.add_edge(url, registered_domain)

            ip = resolve_ip(full_domain)
            if ip:
                if not G.has_node(ip):
                    G.add_node(ip, node_type="ip")
                if not G.has_edge(registered_domain, ip):
                    G.add_edge(registered_domain, ip)

            substrings = extract_substrings(url)
            for substr in substrings[:5]:
                substr_key = "substr:{}".format(substr.lower())
                if not G.has_node(substr_key):
                    G.add_node(substr_key, node_type="substring")
                if not G.has_edge(url, substr_key):
                    G.add_edge(url, substr_key)

        # Re-generate embeddings and similarity for expanded graph
        emb = generate_word2vec_embeddings(G, urls, dim=LBP_EMB_DIM)
        similarity = compute_edge_similarity(G, emb, sim_type=LBP_SIM_TYPE)

    # Separate train URLs (labeled) and test URLs (unlabeled)
    train_urls = list(url_labels.keys())

    # Try original modules first, fallback to reimplementation
    use_original = try_import_original_modules()
    if use_original:
        G = run_lbp_with_original_modules(G, url_labels, similarity, train_urls, urls)
    else:
        G = run_lbp_fallback(G, url_labels, similarity, train_urls, urls)

    # Extract per-URL predictions
    predictions = []
    for url in urls:
        if url in G.nodes():
            predict_label = G.nodes[url].get('predict_label', -1)
            prior = G.nodes[url].get('prior_probability', [0.5, 0.5])
            msg_sum = G.nodes[url].get('msg_sum', [0, 0])

            # Compute confidence from prior probability + message sum
            cost_benign = (1 - prior[0]) + msg_sum[0]
            cost_phish = (1 - prior[1]) + msg_sum[1]
            total = cost_benign + cost_phish
            if total > 0:
                phish_score = 1.0 - (cost_benign / total)
            else:
                phish_score = 0.5

            if predict_label == -1:
                pred = "unknown"
            else:
                pred = "phish" if predict_label == 1 else "benign"
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
    print("[LBP] Results written to {}".format(csv_path))


def main():
    print("[LBP] Starting (mode={})...".format(MODE))
    print("[LBP] Using original Loopy Belief Propagation algorithm (ICC'25)")
    print("[LBP] Hyperparameters: sim={}, edge_potential={}, threshold={}, t1={}, t2={}".format(
        LBP_SIM_TYPE, LBP_EDGE_POTENTIAL, LBP_CLASSIFY_THRESHOLD,
        LBP_THRESHOLD1, LBP_THRESHOLD2))

    if MODE == "collect":
        print("[LBP] Collect mode: feature-collector service handles extraction. Sleeping.")
        while True:
            time.sleep(3600)

    if not os.path.exists(INPUT_CSV):
        print("[LBP] ERROR: Input file not found: {}".format(INPUT_CSV))
        sys.exit(1)

    urls, labels = read_urls(INPUT_CSV)
    print("[LBP] Loaded {} URLs".format(len(urls)))

    if MODE == "train":
        train(urls, labels)
        print("[LBP] Training complete.")
    else:
        predictions = test(urls)
        write_output(predictions, OUTPUT_DIR)
        print("[LBP] Done.")


if __name__ == "__main__":
    main()
