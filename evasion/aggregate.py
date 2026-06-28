"""
Collapse per-variant evasion scoring into Table 6 of the paper.

Table 6 reports Attack Success Rate (ASR) per attack **surface** and per
combination (URL, HTML, Logo, URL+HTML, URL+Logo, HTML+Logo,
URL+HTML+Logo). ASR is defined (§5.4) as the fraction of true-positive
phishing samples that flip to "benign" under the attack.

Input: long-format CSV produced by :mod:`phishxgraph.evasion.score`.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
from collections import defaultdict
from itertools import chain, combinations

log = logging.getLogger("phishxgraph.evasion.aggregate")

SURFACE_OF = {
    "A1_url_shortener": "URL",
    "A2_url_combosquat": "URL",
    "A3_url_tld_sub": "URL",
    "A4_html_invisible_link": "HTML",
    "A5_html_object_ratio": "HTML",
    "A6_html_dom_expand": "HTML",
    "A7_html_input_obfuscate": "HTML",
    "A8_logo_masking": "Logo",
    "A9_logo_manipulation": "Logo",
    "A10_logo_font_subst": "Logo",
}

SURFACES = ["URL", "HTML", "Logo"]


def _read(path: str):
    with open(path) as f:
        return list(csv.DictReader(f))


def _is_phish_url(row):
    return str(row.get("label", "")).lower() in ("phish", "1", "phishing")


def compute_asr_per_attack(rows):
    """ASR per (attack, variant) and then the max across variants.

    Paper §5.4: "we measure aggregated attack success rate for each
    perturbation surface ... by jointly considering all attacks generated
    on that surface." We follow the same logic: a phishing URL is counted
    as "evaded" on surface S iff *any* variant of *any* attack on S
    misclassifies it as benign.
    """
    by_attack_url = defaultdict(dict)  # attack -> url -> best_score_phish_flip
    true_phish_urls = set()

    for r in rows:
        if not _is_phish_url(r):
            continue
        if r.get("status") != "ok":
            continue
        true_phish_urls.add(r["url"])
        evaded = r.get("prediction") == "benign"
        prev = by_attack_url[r["attack"]].get(r["url"], False)
        by_attack_url[r["attack"]][r["url"]] = prev or evaded
    return by_attack_url, true_phish_urls


def compute_surface_asr(by_attack_url, true_phish_urls):
    """Per-surface ASR using union semantics over attacks on that surface."""
    by_surface_url = {s: defaultdict(bool) for s in SURFACES}
    for attack, url_to_evaded in by_attack_url.items():
        surface = SURFACE_OF.get(attack)
        if surface is None:
            continue
        for url, evaded in url_to_evaded.items():
            by_surface_url[surface][url] = by_surface_url[surface][url] or evaded
    results = {}
    n_phish = max(len(true_phish_urls), 1)
    for s in SURFACES:
        evaded_urls = {u for u, v in by_surface_url[s].items() if v}
        results[s] = {
            "asr": len(evaded_urls) / n_phish,
            "n_evaded": len(evaded_urls),
            "n_phish": len(true_phish_urls),
        }
    return results, by_surface_url


def compute_combo_asr(by_surface_url, true_phish_urls, surfaces):
    """Combo ASR = union of evaded URL sets over chosen surfaces."""
    n_phish = max(len(true_phish_urls), 1)
    evaded_urls = set()
    for s in surfaces:
        evaded_urls |= {u for u, v in by_surface_url.get(s, {}).items() if v}
    return {
        "asr": len(evaded_urls) / n_phish,
        "n_evaded": len(evaded_urls),
        "n_phish": len(true_phish_urls),
    }


def build_table(rows):
    by_attack_url, true_phish = compute_asr_per_attack(rows)
    surface_asr, by_surface_url = compute_surface_asr(by_attack_url, true_phish)

    combos = []
    # Paper Table 6 columns: URL, HTML, Logo, URL+HTML, URL+Logo, HTML+Logo, URL+HTML+Logo
    for k in (1, 2, 3):
        for combo in combinations(SURFACES, k):
            if k == 1:
                name = combo[0]
                result = surface_asr[name]
            else:
                name = "+".join(combo)
                result = compute_combo_asr(by_surface_url, true_phish, combo)
            combos.append({"scenario": name, **result})
    return combos


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [evasion.aggregate] %(levelname)s: %(message)s",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True,
                   help="Per-variant CSV from phishxgraph.evasion.score")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    rows = _read(args.input)
    table = build_table(rows)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["scenario", "asr", "n_evaded", "n_phish"]
        )
        w.writeheader()
        for row in table:
            w.writerow({
                "scenario": row["scenario"],
                "asr": f"{row['asr']:.4f}",
                "n_evaded": row["n_evaded"],
                "n_phish": row["n_phish"],
            })
    log.info("wrote %s", args.output)
    for row in table:
        log.info("  %-15s ASR=%.4f (%d / %d)",
                 row["scenario"], row["asr"], row["n_evaded"], row["n_phish"])


if __name__ == "__main__":
    main()
