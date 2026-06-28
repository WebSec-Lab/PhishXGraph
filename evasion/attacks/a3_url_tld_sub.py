"""A3: TLD Substitution.

Suspicious TLDs: v1 .xyz  v2 .top  v3 .tk
Legit-looking:   v4 .co   v5 .io   v6 .net
"""

from pathlib import Path
from urllib.parse import urlparse, urlunparse

from ._base import AttackResult, BaseAttack, copy_features_except

_VARIANTS = {
    "v1": "xyz", "v2": "top", "v3": "tk",
    "v4": "co",  "v5": "io",  "v6": "net",
}


class A3UrlTldSub(BaseAttack):
    name = "A3_url_tld_sub"
    surface = "url"
    target_file = ""

    def get_variants(self) -> list[str]:
        return list(_VARIANTS.keys())

    def apply(self, feature_dir: Path, output_dir: Path, url: str, variant: str) -> AttackResult:
        new_tld = _VARIANTS[variant]
        p = urlparse(url if "://" in url else f"http://{url}")
        netloc = p.netloc or p.path.split("/")[0]

        # Split into host:port then swap only the last label of host
        host_port = netloc.split(":", 1)
        host = host_port[0]
        port = host_port[1] if len(host_port) > 1 else None
        parts = host.split(".")
        if len(parts) < 2:
            new_host = f"{host}.{new_tld}"
        else:
            parts[-1] = new_tld
            new_host = ".".join(parts)
        new_netloc = f"{new_host}:{port}" if port else new_host

        adv = urlunparse((p.scheme or "http", new_netloc, p.path, p.params, p.query, p.fragment))

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "adversarial_url.txt").write_text(adv, encoding="utf-8")
        if feature_dir.is_dir():
            copy_features_except(feature_dir, output_dir, skip_names=set())

        return AttackResult(
            url=url, attack_name=self.name, variant=variant,
            original_path=url, adversarial_path=adv, success=True,
            metadata={"new_tld": new_tld, "orig_url": url, "adv_url": adv},
        )
