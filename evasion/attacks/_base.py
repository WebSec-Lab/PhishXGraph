"""Base classes and helpers for v2 attacks."""

import hashlib
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AttackResult:
    url: str
    attack_name: str
    variant: str
    original_path: str
    adversarial_path: str
    success: bool
    metadata: dict = field(default_factory=dict)


class BaseAttack(ABC):
    name: str = "base"
    surface: str = ""  # url / html / logo
    target_file: str = ""  # e.g. page.html / screenshot.png / (empty for url)

    @abstractmethod
    def get_variants(self) -> list[str]:
        """Return exactly 6 variant names: v1..v6."""
        ...

    @abstractmethod
    def apply(self, feature_dir: Path, output_dir: Path, url: str, variant: str) -> AttackResult:
        ...


def copy_features_except(feature_dir: Path, output_dir: Path, skip_names: set[str]):
    """Copy every file from feature_dir to output_dir except the ones we modified."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for f in feature_dir.iterdir():
        if f.name in skip_names or not f.is_file():
            continue
        dst = output_dir / f.name
        if dst.exists():
            continue
        shutil.copy2(f, dst)


def deterministic_hash(s: str, n: int) -> str:
    """Short deterministic hash for synthetic shortener-style paths."""
    return hashlib.md5(s.encode()).hexdigest()[:n]
