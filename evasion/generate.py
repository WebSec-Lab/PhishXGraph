"""
Generate adversarial feature caches for Table 6 (§5.4).

For each attack in :data:`evasion.attacks.ATTACKS` and each of its six
variants, materialise an attacked version of the page's feature cache
under::

    <output-dir>/{attack}/{variant}/<sha256(url)[:16]>/

URLs for which the attack fails (e.g. no logo detected) are symlinked
from the clean feature cache so scoring sees the full test set.


CLI:

    python -m evasion.generate \\
        --input        data/input/test_evasion.csv \\
        --features-dir data/features \\
        --output-dir   data/evasion_features/single \\
        --attacks all \\
        --workers      4

Then score & aggregate:

    python -m evasion.score     --test ... --attacks-dir ... --model-dir ...
    python -m evasion.aggregate --input ...  --output ...
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from evasion.attacks import ATTACKS

log = logging.getLogger("evasion.generate")


def url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def load_rows(csv_path: str) -> list[tuple[str, str]]:
    with open(csv_path) as f:
        return [(r["url"], r.get("label", "phish"))
                for r in csv.DictReader(f)]


def _apply_one(attack_cls, features_dir, output_root, url, variant):
    h = url_hash(url)
    feat_dir = Path(features_dir) / h
    out_dir = Path(output_root) / attack_cls.name / variant / h
    # Skip if we already produced something for this (attack, variant, url).
    existing = any((out_dir / f).exists()
                   for f in ("page.html", "screenshot.png",
                             "adversarial_url.txt"))
    if existing:
        return {"url": url, "attack": attack_cls.name, "variant": variant,
                "success": True, "skipped": True}
    try:
        r = attack_cls().apply(feat_dir, out_dir, url, variant)
        return {
            "url": url, "attack": r.attack_name, "variant": r.variant,
            "success": r.success,
            "adv_path": r.adversarial_path,
            "metadata": json.dumps(r.metadata, default=str),
        }
    except Exception as exc:
        return {"url": url, "attack": attack_cls.name, "variant": variant,
                "success": False, "error": f"{type(exc).__name__}: {exc}"}


def _symlink_non_attacked(features_dir, output_dir,
                          attacked_hashes_by_av, all_rows):
    features_dir = Path(features_dir)
    output_dir = Path(output_dir)
    for (attack_name, variant), attacked in attacked_hashes_by_av.items():
        var_dir = output_dir / attack_name / variant
        if not var_dir.exists():
            continue
        for url, _ in all_rows:
            h = url_hash(url)
            if h in attacked:
                continue
            link = var_dir / h
            src = features_dir / h
            if not src.is_dir() or link.exists():
                continue
            try:
                link.symlink_to(os.path.relpath(src, link.parent))
            except (FileExistsError, OSError):
                pass


def run(args):
    if args.attacks == "all":
        names = list(ATTACKS.keys())
    else:
        names = [a.strip() for a in args.attacks.split(",") if a.strip()]
        for n in names:
            if n not in ATTACKS:
                print(f"[ERR] unknown attack: {n}", file=sys.stderr)
                sys.exit(2)

    rows = load_rows(args.input)
    if args.max_samples:
        rows = rows[: args.max_samples]

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = Path(args.output_dir) / "_generation_manifest.csv"

    work = []
    for name in names:
        cls = ATTACKS[name]
        for v in cls().get_variants():
            for url, _ in rows:
                work.append((cls, url, v))

    log.info("generating %d (attack, url, variant) jobs with %d workers",
             len(work), args.workers)
    t0 = time.time()
    attacked_map: dict[tuple[str, str], set[str]] = {}

    with open(manifest_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["url", "attack", "variant", "success", "error"])
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [
                ex.submit(_apply_one, cls, args.features_dir,
                          args.output_dir, url, v)
                for cls, url, v in work
            ]
            for fut in as_completed(futs):
                r = fut.result()
                done += 1
                if done % 500 == 0 or done == len(work):
                    dt = time.time() - t0
                    rate = done / dt if dt > 0 else 0
                    eta_min = (len(work) - done) / rate / 60 if rate > 0 else 0
                    log.info("progress %d/%d  %.1f/s  ETA %.1f min",
                             done, len(work), rate, eta_min)
                w.writerow([
                    r.get("url", ""), r.get("attack", ""),
                    r.get("variant", ""), int(bool(r.get("success"))),
                    r.get("error", ""),
                ])
                if r.get("success"):
                    attacked_map.setdefault(
                        (r["attack"], r["variant"]), set()
                    ).add(url_hash(r["url"]))

    log.info("symlinking non-attacked URLs …")
    _symlink_non_attacked(args.features_dir, args.output_dir,
                          attacked_map, rows)
    log.info("done in %.1f min; manifest → %s",
             (time.time() - t0) / 60, manifest_path)


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [evasion.generate] %(levelname)s: %(message)s",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="URL list CSV")
    p.add_argument("--features-dir", required=True,
                   help="Clean feature cache (from phishxgraph.cli collect)")
    p.add_argument("--output-dir", required=True,
                   help="Root for {attack}/{variant}/{hash}/ outputs")
    p.add_argument("--attacks", default="all",
                   help="Comma-separated attack names or 'all'")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
