"""
Classification model wrapper (§4.4).

Paper default: XGBoost (gradient-boosted decision trees).
Ablation options: RandomForest, SVM, MLP (§5.5, Table 7).
"""
import os
import pickle

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .features import FEATURE_NAMES

try:  # Paper default
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    _HAS_XGBOOST = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def _make_classifier(name: str):
    name = (name or "xgboost").lower()
    if name == "xgboost":
        if not _HAS_XGBOOST:
            raise RuntimeError(
                "xgboost is not installed. `pip install xgboost` or pass "
                "--classifier random_forest."
            )
        return XGBClassifier(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="logloss",
        )
    if name in {"random_forest", "randomforest", "rf"}:
        return RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
        )
    if name == "svm":
        return SVC(kernel="rbf", probability=True, random_state=42)
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=200, random_state=42
        )
    raise ValueError(f"Unknown classifier: {name}")


class PhishXGraphClassifier:
    """Paper-aligned classifier wrapper with fit/predict/save/load."""

    def __init__(self, classifier: str = "xgboost"):
        self.classifier_name = classifier
        self.clf = _make_classifier(classifier)
        self.feature_names = list(FEATURE_NAMES)

    def _matrix(self, feature_dicts):
        X = np.array(
            [[fd.get(k, 0) for k in self.feature_names] for fd in feature_dicts],
            dtype=float,
        )
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, feature_dicts, labels):
        X = self._matrix(feature_dicts)
        y = np.array(
            [1 if str(l).lower() in ("phish", "1", "true", "phishing") else 0
             for l in labels]
        )
        self.clf.fit(X, y)
        y_pred = self.clf.predict(X)
        return {
            "train_accuracy": float(accuracy_score(y, y_pred)),
            "train_f1": float(f1_score(y, y_pred)) if y.sum() > 0 else 0.0,
            "n_samples": int(len(y)),
            "n_phishing": int(y.sum()),
        }

    def predict(self, feature_dicts):
        X = self._matrix(feature_dicts)
        y_pred = self.clf.predict(X)
        try:
            proba = self.clf.predict_proba(X)[:, 1]
        except Exception:
            proba = y_pred.astype(float)
        return y_pred, proba

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
            pickle.dump(self.clf, f)
        with open(os.path.join(model_dir, "feature_names.pkl"), "wb") as f:
            pickle.dump(self.feature_names, f)
        with open(os.path.join(model_dir, "classifier.txt"), "w") as f:
            f.write(self.classifier_name + "\n")

    @classmethod
    def load(cls, model_dir: str):
        with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
            clf = pickle.load(f)
        with open(os.path.join(model_dir, "feature_names.pkl"), "rb") as f:
            feature_names = pickle.load(f)
        obj = cls.__new__(cls)
        obj.clf = clf
        obj.feature_names = feature_names
        classifier_path = os.path.join(model_dir, "classifier.txt")
        obj.classifier_name = (
            open(classifier_path).read().strip()
            if os.path.exists(classifier_path) else "unknown"
        )
        return obj
