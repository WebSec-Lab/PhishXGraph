"""
Cross-layer interaction graph construction (§4.1-4.2).

Builds a heterogeneous directed graph with 4 node types (HTML, Script,
Network, Storage) and 4 edge types (Structure, Execute, Request, Access),
consistent with Table 2 of the paper.
"""
import re
from urllib.parse import urlparse

import networkx as nx
from bs4 import BeautifulSoup, Tag

from .constants import (
    EDGE_TYPE_ACCESS,
    EDGE_TYPE_EXECUTE,
    EDGE_TYPE_REQUEST,
    EDGE_TYPE_STRUCTURE,
    NODE_TYPE_HTML,
    NODE_TYPE_NETWORK,
    NODE_TYPE_SCRIPT,
    NODE_TYPE_STORAGE,
)

EMPTY_INSTRUMENTATION = {"html": "", "requests": [], "redirects": [], "storage_ops": []}


def extract_domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        return parsed.netloc or parsed.path.split("/")[0]
    except Exception:
        return url


def build_graph(url, instr_data, page_domain):
    """Build the cross-layer interaction graph for a page.

    Parameters
    ----------
    url : str
        Page URL.
    instr_data : dict
        Instrumentation payload with keys: html, requests, redirects, storage_ops.
    page_domain : str
        Domain of the page URL, used to flag third-party requests.

    Returns
    -------
    (networkx.DiGraph, bs4.BeautifulSoup|None)
        Graph and parsed soup (None if HTML parse failed).
    """
    G = nx.DiGraph()

    if isinstance(instr_data, str):
        instr_data = {"html": instr_data, "requests": [],
                      "redirects": [], "storage_ops": []}

    html = instr_data.get("html", "")
    captured_requests = instr_data.get("requests", [])
    redirects = instr_data.get("redirects", [])
    storage_ops = instr_data.get("storage_ops", [])

    if not html:
        G.add_node("page_root", type=NODE_TYPE_HTML, tag="document")
        return G, None

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        G.add_node("page_root", type=NODE_TYPE_HTML, tag="document")
        return G, None

    nid_counter = [0]

    # ---- HTML layer: DOM nodes + structure edges (§4.2) ----
    def add_dom(element, parent_nid=None):
        nid = f"html_{nid_counter[0]}"
        nid_counter[0] += 1
        attrs = {"type": NODE_TYPE_HTML}
        if isinstance(element, Tag):
            attrs["tag"] = element.name
            for a in ("id", "src", "href", "action"):
                v = element.get(a, "")
                if v:
                    attrs[a] = str(v) if not isinstance(v, str) else v
            cls = element.get("class", [])
            if cls:
                attrs["class"] = " ".join(cls) if isinstance(cls, list) else str(cls)
            for a in element.attrs:
                if str(a).startswith("on"):
                    attrs["has_event_handler"] = True
        else:
            attrs["tag"] = "#text"

        G.add_node(nid, **attrs)
        if parent_nid:
            G.add_edge(parent_nid, nid, type=EDGE_TYPE_STRUCTURE)

        if isinstance(element, Tag):
            for child in element.children:
                if isinstance(child, Tag):
                    add_dom(child, nid)
        return nid

    root_nid = add_dom(soup.html if soup.html else soup)

    # ---- JS layer: script nodes + execute edges (§4.2) ----
    scripts = soup.find_all("script")
    script_url_to_nid = {}
    js_counter = 0
    for script in scripts:
        js_nid = f"js_{js_counter}"
        js_counter += 1
        src = script.get("src", "")
        inline_code = script.string or ""
        G.add_node(js_nid, type=NODE_TYPE_SCRIPT, src=src,
                   is_inline=len(inline_code.strip()) > 0,
                   has_inline_code=len(inline_code.strip()) > 0)
        if src:
            script_url_to_nid[src] = js_nid
        parent = script.parent
        if parent and isinstance(parent, Tag):
            candidates = [
                n for n, d in G.nodes(data=True)
                if d.get("tag") == parent.name and d.get("type") == NODE_TYPE_HTML
            ]
            if candidates:
                G.add_edge(candidates[-1], js_nid, type=EDGE_TYPE_EXECUTE)

    # Dynamically loaded scripts: script-type requests not tied to <script>
    for req in captured_requests:
        if req.get("resource_type") == "script":
            req_url = req.get("url", "")
            if req_url and req_url not in script_url_to_nid:
                js_nid = f"js_{js_counter}"
                js_counter += 1
                G.add_node(js_nid, type=NODE_TYPE_SCRIPT, src=req_url,
                           is_inline=False, has_inline_code=False)
                script_url_to_nid[req_url] = js_nid
                G.add_edge(root_nid, js_nid, type=EDGE_TYPE_EXECUTE)

    # ---- Network layer: request nodes + request edges (§4.2) ----
    req_counter = [0]
    request_url_to_nid = {}

    if captured_requests:
        js_fallback = None
        for n, d in G.nodes(data=True):
            if d.get("type") == NODE_TYPE_SCRIPT and d.get("src"):
                js_fallback = n
                break
        if js_fallback is None:
            for n, d in G.nodes(data=True):
                if d.get("type") == NODE_TYPE_SCRIPT:
                    js_fallback = n
                    break

        for req in captured_requests:
            req_url = req.get("url", "")
            if not req_url or not req_url.startswith("http"):
                continue
            if req.get("resource_type") == "script":
                continue

            is_nav = bool(req.get("is_navigation"))
            req_nid = f"request_{req_counter[0]}"
            req_counter[0] += 1
            res_domain = extract_domain(req_url)
            G.add_node(
                req_nid, type=NODE_TYPE_NETWORK,
                url=req_url, domain=res_domain,
                resource_type=req.get("resource_type", "other"),
                status=req.get("status", 0),
                is_third_party=res_domain != page_domain,
                is_navigation=is_nav,
            )
            request_url_to_nid[req_url] = req_nid

            if is_nav:
                if req_url == url:
                    G.add_edge(req_nid, root_nid, type=EDGE_TYPE_REQUEST)
                continue

            rtype = req.get("resource_type", "")
            linked = False
            for node, data in G.nodes(data=True):
                if data.get("type") != NODE_TYPE_HTML:
                    continue
                for attr in ("src", "href", "action"):
                    if data.get(attr, "") == req_url:
                        G.add_edge(node, req_nid, type=EDGE_TYPE_REQUEST)
                        linked = True
                        break
                if linked:
                    break
            if not linked:
                if (rtype in ("xhr", "fetch", "ping", "beacon", "image")
                        and js_fallback is not None):
                    G.add_edge(js_fallback, req_nid, type=EDGE_TYPE_REQUEST)
                else:
                    G.add_edge(root_nid, req_nid, type=EDGE_TYPE_REQUEST)

        # 3xx redirect chains as network-to-network edges
        for redir in redirects:
            from_nid = request_url_to_nid.get(redir.get("from_url"))
            to_nid = request_url_to_nid.get(redir.get("to_url"))
            if from_nid and to_nid:
                G.add_edge(from_nid, to_nid, type=EDGE_TYPE_REQUEST,
                           is_redirect=True)
                G.nodes[to_nid]["is_redirect"] = True
    else:
        # Static fallback when no instrumentation captured
        for node, data in list(G.nodes(data=True)):
            for attr in ("src", "href", "action"):
                resource_url = data.get(attr, "")
                if resource_url and resource_url.startswith("http"):
                    req_nid = f"request_{req_counter[0]}"
                    req_counter[0] += 1
                    res_domain = extract_domain(resource_url)
                    G.add_node(
                        req_nid, type=NODE_TYPE_NETWORK,
                        url=resource_url, domain=res_domain,
                        is_third_party=res_domain != page_domain,
                    )
                    G.add_edge(node, req_nid, type=EDGE_TYPE_REQUEST)

    # ---- Storage layer: storage nodes + access edges (§4.2) ----
    if storage_ops:
        for i, op in enumerate(storage_ops):
            op_type = op.get("type", "")
            storage_nid = f"storage_{i}"
            G.add_node(storage_nid, type=NODE_TYPE_STORAGE, op_type=op_type,
                       key=op.get("key", ""),
                       storage_name=op.get("storage",
                                           "cookie" if "cookie" in op_type else ""))

            stack = op.get("stack", "")
            linked = False
            if stack:
                for s_url, js_nid in script_url_to_nid.items():
                    if s_url and s_url in stack:
                        G.add_edge(storage_nid, js_nid, type=EDGE_TYPE_ACCESS,
                                   action=op_type)
                        linked = True
                        break
            if not linked:
                for n, d in G.nodes(data=True):
                    if d.get("type") == NODE_TYPE_SCRIPT and d.get("has_inline_code"):
                        G.add_edge(storage_nid, n, type=EDGE_TYPE_ACCESS,
                                   action=op_type)
                        break
    else:
        html_str = str(soup)
        static_counts = {
            "get_cookie": len(re.findall(r"document\.cookie", html_str)),
            "set_cookie": len(re.findall(r"document\.cookie\s*=", html_str)),
            "get_storage": len(re.findall(
                r"(localStorage|sessionStorage)\.getItem", html_str)),
            "set_storage": len(re.findall(
                r"(localStorage|sessionStorage)\.setItem", html_str)),
        }
        if any(v > 0 for v in static_counts.values()):
            storage_nid = "storage_0"
            G.add_node(storage_nid, type=NODE_TYPE_STORAGE, **static_counts)
            for js_nid in [n for n, d in G.nodes(data=True)
                           if d.get("type") == NODE_TYPE_SCRIPT]:
                G.add_edge(storage_nid, js_nid, type=EDGE_TYPE_ACCESS)

    return G, soup
