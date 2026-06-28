"""
Phase 2 — generate combined-surface adversarial caches (paper §5.4).

Paper Table 6 combos: URL+HTML, URL+Logo, HTML+Logo, URL+HTML+Logo.
Each uses the strongest single-surface winner on that surface:

    URL  = A1_url_shortener     v1  (bit.ly-style path compression)
    HTML = A7_html_input_obfuscate v1 (type=password → type=text)
    Logo = A10_logo_font_subst   v1 (DejaVu Sans re-render)

Output layout (same as Phase 1 so :mod:`evasion.score` consumes both):

    <output-dir>/{scenario}/<sha256(url)[:16]>/{page.html, screenshot.png,
                                                 instrumentation.json,
                                                 adversarial_url.txt?}

For scenarios with a URL attack, we also emit
``<output-dir>/{scenario}/adversarial_input.csv`` and symlink
``hash(adv_url)`` → ``hash(orig_url)`` so scorers can look up either form.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import shutil
from pathlib import Path

from evasion.attacks import ATTACKS

log = logging.getLogger("evasion.generate_combos")

WINNERS = {
    "url": ("A1_url_shortener", "v1"),
    "html": ("A7_html_input_obfuscate", "v1"),
    "logo": ("A10_logo_font_subst", "v1"),
}

SCENARIOS = [
    ("URL_HTML", ["url", "html"]),
    ("URL_Logo", ["url", "logo"]),
    ("HTML_Logo", ["html", "logo"]),
    ("URL_HTML_Logo", ["url", "html", "logo"]),
]


def url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def load_rows(csv_path: str):
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def apply_combo(surfaces, feature_dir: Path, out_dir: Path, url: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Seed with a copy of every original feature so unmodified files exist
    # even if one surface's attack fails.
    if feature_dir.is_dir():
        for f in feature_dir.iterdir():
            if f.is_file():
                dst = out_dir / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)

    adv_url = None
    metas = []
    for surf in surfaces:
        attack_name, variant = WINNERS[surf]
        attack = ATTACKS[attack_name]()
        result = attack.apply(feature_dir, out_dir, url, variant)
        metas.append({
            "surface": surf, "attack": attack_name, "variant": variant,
            "success": result.success, "metadata": result.metadata,
        })
        if surf == "url" and result.success:
            adv_url = result.metadata.get("adv_url")
    return adv_url, metas


def run(args):
    rows = load_rows(args.input)
    features_dir = Path(args.features_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for scenario_name, surfaces in SCENARIOS:
        log.info("=== %s (%s) ===", scenario_name, ",".join(surfaces))
        scenario_dir = out_root / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        has_url = "url" in surfaces
        adv_rows, success_count, symlink_count = [], 0, 0

        for row in rows:
            url = row["url"]
            label = row.get("label", "phish")
            h = url_hash(url)
            feat_dir = features_dir / h
            out_dir = scenario_dir / h
            try:
                adv_url, metas = apply_combo(surfaces, feat_dir, out_dir, url)
            except Exception as exc:
                log.warning("%s: %s", url[:60], exc)
                metas, adv_url = [], None

            if has_url:
                effective_url = adv_url or url
                adv_rows.append({"url": effective_url, "label": label})
                if adv_url and adv_url != url:
                    h_adv = url_hash(adv_url)
                    if h_adv != h:
                        link = scenario_dir / h_adv
                        if not link.exists() and not link.is_symlink():
                            try:
                                link.symlink_to(h)
                                symlink_count += 1
                            except (FileExistsError, OSError):
                                pass
            if metas and all(m["success"] for m in metas):
                success_count += 1

        log.info("%s: %d/%d fully successful, +%d symlinks",
                 scenario_name, success_count, len(rows), symlink_count)

        if has_url:
            csv_out = scenario_dir / "adversarial_input.csv"
            with open(csv_out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["url", "label"])
                w.writeheader()
                w.writerows(adv_rows)
            log.info("  wrote %s", csv_out)


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [evasion.combos] %(levelname)s: %(message)s",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--features-dir", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
