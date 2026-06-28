"""
Playwright + CDP instrumentation (§4.2).

Visits a page with a JS storage-API hook installed before page scripts run,
captures every HTTP request/response/redirect and every
document.cookie/localStorage/sessionStorage access, and persists the
rendered DOM and instrumentation payload to disk.

Paper default timeout: 30 seconds per page (§5.1, "We set the graph
construction timeout to 30 seconds per phishing page and exclude pages
that fail to load within this time from the dataset.").
"""
from __future__ import annotations

import hashlib
import json
import logging
import os

_STORAGE_HOOK_JS = """(function() {
    window.__phgraph_storage_ops__ = [];
    try {
        var cd = Object.getOwnPropertyDescriptor(Document.prototype, 'cookie')
              || Object.getOwnPropertyDescriptor(HTMLDocument.prototype, 'cookie');
        if (cd) {
            Object.defineProperty(document, 'cookie', {
                get: function() {
                    try { window.__phgraph_storage_ops__.push(
                        {type:'get_cookie', stack: new Error().stack||'', ts:Date.now()}
                    ); } catch(e) {}
                    return cd.get.call(this);
                },
                set: function(v) {
                    try { window.__phgraph_storage_ops__.push(
                        {type:'set_cookie', value:String(v).substring(0,200),
                         stack: new Error().stack||'', ts:Date.now()}
                    ); } catch(e) {}
                    return cd.set.call(this, v);
                },
                configurable: true
            });
        }
    } catch(e) {}
    ['localStorage','sessionStorage'].forEach(function(name) {
        try {
            var s = window[name]; if (!s) return;
            var _set = s.__proto__.setItem, _get = s.__proto__.getItem;
            s.__proto__.setItem = function(k,v) {
                try { window.__phgraph_storage_ops__.push(
                    {type:'set_storage', storage:name, key:k,
                     stack: new Error().stack||'', ts:Date.now()}
                ); } catch(e) {}
                return _set.call(this,k,v);
            };
            s.__proto__.getItem = function(k) {
                try { window.__phgraph_storage_ops__.push(
                    {type:'get_storage', storage:name, key:k,
                     stack: new Error().stack||'', ts:Date.now()}
                ); } catch(e) {}
                return _get.call(this,k);
            };
        } catch(e) {}
    });
})();"""

# Paper default (§5.1)
DEFAULT_TIMEOUT_SEC = 30

log = logging.getLogger("phishxgraph.collect")


def url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def page_cache_dir(features_dir: str, url: str) -> str:
    d = os.path.join(features_dir, url_hash(url))
    os.makedirs(d, exist_ok=True)
    return d


def instrument_page(browser, url: str, timeout_sec: int = DEFAULT_TIMEOUT_SEC):
    """Visit `url` and return an instrumentation payload dict or None.

    The dict has keys: html, requests, redirects, storage_ops. Returning
    None means the page failed to load within the timeout.
    """
    timeout_ms = timeout_sec * 1000
    captured_requests = []
    captured_responses = {}
    context = None
    try:
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            ignore_https_errors=True,
        )
        context.add_init_script(_STORAGE_HOOK_JS)
        page = context.new_page()

        def _on_request(req):
            captured_requests.append({
                "url": req.url,
                "method": req.method,
                "resource_type": req.resource_type,
                "is_navigation": req.is_navigation_request(),
            })

        def _on_response(resp):
            redir = resp.request.redirected_from
            captured_responses[resp.url] = {
                "status": resp.status,
                "redirected_from": redir.url if redir else None,
            }

        page.on("request", _on_request)
        page.on("response", _on_response)
        page.goto(url, timeout=timeout_ms, wait_until="networkidle")
        html = page.content()
        try:
            storage_ops = page.evaluate("window.__phgraph_storage_ops__ || []")
        except Exception:
            storage_ops = []

        redirects = []
        for resp_url, rd in captured_responses.items():
            if rd["redirected_from"]:
                redirects.append({
                    "from_url": rd["redirected_from"],
                    "to_url": resp_url,
                    "status": rd["status"],
                })

        requests_data = []
        for req in captured_requests:
            resp = captured_responses.get(req["url"], {})
            requests_data.append({
                "url": req["url"],
                "method": req["method"],
                "resource_type": req["resource_type"],
                "is_navigation": req["is_navigation"],
                "status": resp.get("status", 0),
            })

        context.close()
        return {
            "html": html,
            "requests": requests_data,
            "redirects": redirects,
            "storage_ops": storage_ops,
        }
    except Exception as exc:
        log.warning("instrument_page failed for %s: %s", url, exc)
        if context:
            try:
                context.close()
            except Exception:
                pass
        return None


def persist_instrumentation(out_dir: str, payload: dict) -> None:
    """Write page.html + instrumentation.json to `out_dir`."""
    html_path = os.path.join(out_dir, "page.html")
    with open(html_path + ".tmp", "w", errors="replace") as f:
        f.write(payload.get("html", ""))
    os.replace(html_path + ".tmp", html_path)

    meta = {
        "requests": payload.get("requests", []),
        "redirects": payload.get("redirects", []),
        "storage_ops": payload.get("storage_ops", []),
    }
    meta_path = os.path.join(out_dir, "instrumentation.json")
    with open(meta_path + ".tmp", "w") as f:
        json.dump(meta, f)
    os.replace(meta_path + ".tmp", meta_path)


def load_cached_instrumentation(features_dir: str, url: str):
    """Return dict with html/requests/redirects/storage_ops, or None."""
    d = page_cache_dir(features_dir, url)
    html_path = os.path.join(d, "page.html")
    meta_path = os.path.join(d, "instrumentation.json")
    if not os.path.isfile(html_path):
        return None
    with open(html_path, errors="replace") as f:
        html = f.read()
    if not html:
        return None
    meta = {"requests": [], "redirects": [], "storage_ops": []}
    if os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            pass
    return {
        "html": html,
        "requests": meta.get("requests", []),
        "redirects": meta.get("redirects", []),
        "storage_ops": meta.get("storage_ops", []),
    }


def collect_urls(urls, features_dir: str,
                 timeout_sec: int = DEFAULT_TIMEOUT_SEC,
                 headless: bool = True) -> dict:
    """Instrument a list of URLs and write results under `features_dir`.

    Returns a dict of stats: {"ok": n, "failed": n, "total": n}.
    """
    from playwright.sync_api import sync_playwright

    ok, failed = 0, 0
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        for i, url in enumerate(urls):
            out_dir = page_cache_dir(features_dir, url)
            if os.path.isfile(os.path.join(out_dir, "page.html")):
                log.info("[%d/%d] cached: %s", i + 1, len(urls), url)
                ok += 1
                continue
            log.info("[%d/%d] %s", i + 1, len(urls), url)
            payload = instrument_page(browser, url, timeout_sec=timeout_sec)
            if payload and payload.get("html"):
                persist_instrumentation(out_dir, payload)
                ok += 1
            else:
                failed += 1
        browser.close()
    return {"ok": ok, "failed": failed, "total": len(urls)}
