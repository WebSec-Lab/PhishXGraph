"""A1: URL Shortening. Synthetic shortener substitution (no network).

v1 bit.ly/{h8}   v2 tinyurl.com/{h7}   v3 goo.gl/{h5}
v4 t.co/{h10}    v5 ow.ly/{h6}         v6 is.gd/{h4}
"""

from pathlib import Path
from urllib.parse import urlparse

from ._base import AttackResult, BaseAttack, copy_features_except, deterministic_hash

_VARIANTS = {
    "v1": ("bit.ly", 8),
    "v2": ("tinyurl.com", 7),
    "v3": ("goo.gl", 5),
    "v4": ("t.co", 10),
    "v5": ("ow.ly", 6),
    "v6": ("is.gd", 4),
}


class A1UrlShortener(BaseAttack):
    name = "A1_url_shortener"
    surface = "url"
    target_file = ""

    def get_variants(self) -> list[str]:
        return list(_VARIANTS.keys())

    def apply(self, feature_dir: Path, output_dir: Path, url: str, variant: str) -> AttackResult:
        host, n = _VARIANTS[variant]
        scheme = urlparse(url).scheme or "http"
        adv = f"{scheme}://{host}/{deterministic_hash(url, n)}"

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "adversarial_url.txt").write_text(adv, encoding="utf-8")
        if feature_dir.is_dir():
            copy_features_except(feature_dir, output_dir, skip_names=set())

        return AttackResult(
            url=url, attack_name=self.name, variant=variant,
            original_path=url, adversarial_path=adv,
            success=True,
            metadata={"shortener": host, "orig_url": url, "adv_url": adv},
        )
