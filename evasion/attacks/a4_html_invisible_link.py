"""A4: Invisible Link Injection. Add hidden anchors with varying counts/hide methods.

v1 10 anchors, display:none
v2 25 anchors, display:none
v3 50 anchors, display:none
v4 25 anchors, visibility:hidden
v5 25 anchors, position:absolute;left:-9999px
v6 25 anchors, font-size:1px;color:transparent
"""

from pathlib import Path

from bs4 import BeautifulSoup

from ._base import AttackResult, BaseAttack, copy_features_except

_VARIANTS = {
    "v1": (10, "display:none"),
    "v2": (25, "display:none"),
    "v3": (50, "display:none"),
    "v4": (25, "visibility:hidden"),
    "v5": (25, "position:absolute;left:-9999px;top:-9999px"),
    "v6": (25, "font-size:1px;color:transparent"),
}

_INTERNAL_PATHS = [
    "/home", "/about", "/products", "/services", "/contact", "/blog",
    "/news", "/help", "/support", "/faq", "/login", "/signup", "/terms",
    "/privacy", "/careers", "/team", "/press", "/investor", "/sitemap",
    "/category/electronics", "/category/books", "/category/clothing",
    "/archive/2024", "/archive/2023", "/archive/2022", "/tag/featured",
    "/tag/popular", "/tag/new", "/tag/sale", "/page/1", "/page/2",
    "/page/3", "/page/4", "/page/5", "/section/weekly", "/section/daily",
    "/author/admin", "/author/editor", "/feed/rss", "/feed/atom",
    "/subscribe", "/newsletter", "/rss", "/partners", "/affiliates",
    "/developers", "/api", "/docs", "/status", "/security", "/compliance",
    "/accessibility",
]


class A4HtmlInvisibleLink(BaseAttack):
    name = "A4_html_invisible_link"
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

        count, style = _VARIANTS[variant]
        html = src.read_text(errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        body = soup.find("body")
        if body is None:
            body = soup

        for i in range(count):
            a = soup.new_tag("a", href=_INTERNAL_PATHS[i % len(_INTERNAL_PATHS)], style=style)
            a.string = f"nav{i}"
            body.append(a)

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "page.html").write_text(str(soup), encoding="utf-8")
        copy_features_except(feature_dir, output_dir, skip_names={"page.html"})

        return AttackResult(
            url=url, attack_name=self.name, variant=variant,
            original_path=str(src), adversarial_path=str(output_dir / "page.html"),
            success=True,
            metadata={"count": count, "style": style},
        )
