"""A7: Input Obfuscation. Hide credential-requesting inputs from CRP detectors.

v1 password -> text
v2 password -> email
v3 password -> text + data-role=password kept
v4 remove <form> wrappers (inputs become orphan)
v5 rename name=password -> name=pw, rewrite form action to relative
v6 hide real inputs (display:none) + add fake visual text inputs
"""

import re
from pathlib import Path

from bs4 import BeautifulSoup

from ._base import AttackResult, BaseAttack, copy_features_except


def _v1(soup):
    n = 0
    for inp in soup.find_all("input", {"type": "password"}):
        inp["type"] = "text"; n += 1
    return {"pwd_to_text": n}


def _v2(soup):
    n = 0
    for inp in soup.find_all("input", {"type": "password"}):
        inp["type"] = "email"; n += 1
    return {"pwd_to_email": n}


def _v3(soup):
    n = 0
    for inp in soup.find_all("input", {"type": "password"}):
        inp["type"] = "text"; inp["data-role"] = "password"; n += 1
    return {"pwd_to_text_plus_dataattr": n}


def _v4(soup):
    n = 0
    for form in soup.find_all("form"):
        form.unwrap(); n += 1
    return {"forms_unwrapped": n}


def _v5(soup):
    n_names = 0
    for inp in soup.find_all("input"):
        name = inp.get("name", "")
        if re.search(r"password|passwd|pwd", name, re.IGNORECASE):
            inp["name"] = "pw"; n_names += 1
    n_actions = 0
    for form in soup.find_all("form"):
        if form.get("action", "").startswith(("http://", "https://")):
            form["action"] = "/submit"; n_actions += 1
    return {"names_renamed": n_names, "actions_relativised": n_actions}


def _v6(soup):
    n_hidden = 0
    for inp in soup.find_all("input", {"type": "password"}):
        inp["style"] = (inp.get("style", "") + ";display:none").lstrip(";")
        n_hidden += 1
        # Add fake visual input after each hidden password
        fake = soup.new_tag("input", type="text", placeholder="Enter password")
        fake["aria-hidden"] = "true"
        inp.insert_after(fake)
    return {"pwds_hidden_with_fake": n_hidden}


_VARIANTS = {"v1": _v1, "v2": _v2, "v3": _v3, "v4": _v4, "v5": _v5, "v6": _v6}


class A7HtmlInputObfuscate(BaseAttack):
    name = "A7_html_input_obfuscate"
    surface = "html"
    target_file = "page.html"

    def get_variants(self) -> list[str]:
        return list(_VARIANTS.keys())

    def apply(self, feature_dir: Path, output_dir: Path, url: str, variant: str) -> AttackResult:
        src = feature_dir / "page.html"
        if not src.is_file():
            return AttackResult(
                url=url, attack_name=self.name, variant=variant,
                original_path=str(src), adversarial_path="", success=False,
                metadata={"error": "page.html missing"},
            )

        html = src.read_text(errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        info = _VARIANTS[variant](soup)

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "page.html").write_text(str(soup), encoding="utf-8")
        copy_features_except(feature_dir, output_dir, skip_names={"page.html"})

        return AttackResult(
            url=url, attack_name=self.name, variant=variant,
            original_path=str(src), adversarial_path=str(output_dir / "page.html"),
            success=True, metadata=info,
        )
