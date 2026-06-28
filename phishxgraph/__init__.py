"""
PhishXGraph — cross-layer interaction graphs for phishing detection.

Reference implementation for the paper:
  Bae, Kim, Kim, Wi. "PhishXGraph: Robust Phishing Website Detection
  using Cross-Layer Interaction Graphs."

Modules
-------
- constants: keyword/tag sets shared across feature extraction
- graph: cross-layer interaction graph construction (§4.1, §4.2)
- features: 101 feature extraction (§4.3, Tables 9-11)
- model: XGBoost classifier wrapper (§4.4)
- cli: train/test entry points
"""

__version__ = "1.0.0"

from .graph import build_graph, extract_domain
from .features import extract_all_features, FEATURE_NAMES, FEATURE_GROUPS
from .model import PhishXGraphClassifier
