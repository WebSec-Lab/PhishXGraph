"""A2: URL Combosquatting. Inject trust keywords via path or subdomain.

v1 path /secure-login/
v2 path /account-verify/
v3 path /auth/signin/
v4 subdomain secure.
v5 subdomain account.
v6 subdomain login-verify.
"""

from pathlib import Path
from urllib.parse import urlparse, urlunparse

from ._base import AttackResult, BaseAttack, copy_features_except

_VARIANTS = {
    "v1": ("path", "/secure-login/"),
    "v2": ("path", "/account-verify/"),
    "v3": ("path", "/auth/signin/"),
    "v4": ("sub", "secure"),
    "v5": ("sub", "account"),
    "v6": ("sub", "login-verify"),
}


class A2UrlCombosquat(BaseAttack):
    name = "A2_url_combosquat"
    surface = "url"
    target_file = ""

    def get_variants(self) -> list[str]:
        return list(_VARIANTS.keys())

    def apply(self, feature_dir: Path, output_dir: Path, url: str, variant: str) -> AttackResult:
        kind, val = _VARIANTS[variant]
        p = urlparse(url if "://" in url else f"http://{url}")
        netloc = p.netloc or p.path.split("/")[0]

        if kind == "path":
            new_path = val.rstrip("/") + (p.path or "/")
            adv = urlunparse((p.scheme or "http", netloc, new_path, p.params, p.query, p.fragment))
        else:
            new_netloc = f"{val}.{netloc}"
            adv = urlunparse((p.scheme or "http", new_netloc, p.path, p.params, p.query, p.fragment))

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "adversarial_url.txt").write_text(adv, encoding="utf-8")
        if feature_dir.is_dir():
            copy_features_except(feature_dir, output_dir, skip_names=set())

        return AttackResult(
            url=url, attack_name=self.name, variant=variant,
            original_path=url, adversarial_path=adv, success=True,
            metadata={"kind": kind, "value": val, "orig_url": url, "adv_url": adv},
        )
