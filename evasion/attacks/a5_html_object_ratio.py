"""A5: Object Ratio Manipulation. Add dummy structural tags.

v1 empty <div> x 10
v2 empty <div> x 50
v3 HTML5 semantic tags x 10 (header/nav/main/footer/article/section/aside cycle)
v4 <span> x 25 with short text
v5 nested <div> depth 10 (single deep tree)
v6 mixed: 5 each of img/script/style/meta/link
"""

from pathlib import Path

from bs4 import BeautifulSoup

from ._base import AttackResult, BaseAttack, copy_features_except

_SEMANTIC = ["header", "nav", "main", "footer", "article", "section", "aside"]


def _v1(soup: BeautifulSoup, body):
    for _ in range(10):
        body.append(soup.new_tag("div"))


def _v2(soup: BeautifulSoup, body):
    for _ in range(50):
        body.append(soup.new_tag("div"))


def _v3(soup: BeautifulSoup, body):
    for i in range(10):
        body.append(soup.new_tag(_SEMANTIC[i % len(_SEMANTIC)]))


def _v4(soup: BeautifulSoup, body):
    for i in range(25):
        s = soup.new_tag("span")
        s.string = f"t{i}"
        body.append(s)


def _v5(soup: BeautifulSoup, body):
    cur = body
    for _ in range(10):
        child = soup.new_tag("div")
        cur.append(child)
        cur = child


def _v6(soup: BeautifulSoup, body):
    for _ in range(5):
        body.append(soup.new_tag("img", src="data:image/gif;base64,R0lGODlhAQABAAAAACw="))
    for _ in range(5):
        body.append(soup.new_tag("script"))
    for _ in range(5):
        body.append(soup.new_tag("style"))
    head = soup.find("head") or body
    for _ in range(5):
        head.append(soup.new_tag("meta", attrs={"name": "filler", "content": "x"}))
    for _ in range(5):
        head.append(soup.new_tag("link", rel="preconnect", href="https://example.com"))


_VARIANTS = {"v1": _v1, "v2": _v2, "v3": _v3, "v4": _v4, "v5": _v5, "v6": _v6}


class A5HtmlObjectRatio(BaseAttack):
    name = "A5_html_object_ratio"
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
        body = soup.find("body") or soup
        _VARIANTS[variant](soup, body)

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "page.html").write_text(str(soup), encoding="utf-8")
        copy_features_except(feature_dir, output_dir, skip_names={"page.html"})

        return AttackResult(
            url=url, attack_name=self.name, variant=variant,
            original_path=str(src), adversarial_path=str(output_dir / "page.html"),
            success=True, metadata={"variant": variant},
        )
