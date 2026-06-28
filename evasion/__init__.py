"""
Adversarial robustness evaluation (paper §5.4, Table 6).

Self-contained: the 10 attacks × 6 variants live in
:mod:`evasion.attacks`, the cache generators in :mod:`evasion.generate`
and :mod:`evasion.generate_combos`, and the scoring + aggregation in
:mod:`evasion.score` and :mod:`evasion.aggregate`.

Full pipeline (from the ``PhishXGraph/`` root):

    # 1) build clean feature cache for the test set
    python -m phishxgraph.cli collect \\
        --input data/input/test.csv --features-dir data/features

    # 2) generate single-surface adversarial caches (10 attacks × 6 variants)
    python -m evasion.generate \\
        --input        data/input/test.csv \\
        --features-dir data/features \\
        --output-dir   data/evasion_features/single

    # 3) (optional) generate paper Table 6 combos
    python -m evasion.generate_combos \\
        --input        data/input/test.csv \\
        --features-dir data/features \\
        --output-dir   data/evasion_features/combo

    # 4) score PhishXGraph on the adversarial caches
    python -m evasion.score \\
        --test         data/input/test.csv \\
        --attacks-dir  data/evasion_features/single \\
        --model-dir    data/models/phishxgraph \\
        --output       data/results/evasion_phishxgraph.csv

    # 5) collapse into Table 6 (URL, HTML, Logo, combos)
    python -m evasion.aggregate \\
        --input  data/results/evasion_phishxgraph.csv \\
        --output data/results/table6.csv
"""
from evasion.attacks import ATTACKS, ATTACKS_BY_SURFACE, SURFACE_OF

__all__ = ["ATTACKS", "ATTACKS_BY_SURFACE", "SURFACE_OF"]
