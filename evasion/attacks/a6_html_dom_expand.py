"""A6: DOM Structure Manipulation. Inject benign content to dilute phishing signals.

v1 5 benign paragraphs
v2 15 benign paragraphs
v3 5 HTML comments
v4 20 HTML comments
v5 hidden nav with 10 benign anchors
v6 combined: 5 paragraphs + 5 comments + 1 nav
"""

from pathlib import Path

from bs4 import BeautifulSoup, Comment

from ._base import AttackResult, BaseAttack, copy_features_except

_BENIGN_PARAGRAPHS = [
    "Welcome to our community forum. Explore trending topics and share ideas.",
    "Discover new recipes, travel guides, and lifestyle tips on our blog.",
    "Learn about the latest technology innovations and product reviews.",
    "Stay updated with sports news, game schedules, and player stats.",
    "Find helpful tutorials on photography, gardening, and home improvement.",
    "Read inspiring stories from around the world every day.",
    "Browse our library of articles on science, health, and culture.",
    "Join thousands of readers enjoying thoughtful journalism each week.",
    "Explore local events, festivals, and community gatherings near you.",
    "Subscribe to our newsletter for weekly curated content.",
    "Meet the authors behind your favourite columns and series.",
    "Visit our archive to read pieces from previous years and issues.",
    "Our editorial team values accuracy, fairness, and transparency.",
    "Check the site map for a full overview of our coverage areas.",
    "Contact us with story tips, feedback, or partnership opportunities.",
]

_BENIGN_ANCHORS = [
    ("Home", "/"), ("About", "/about"), ("Products", "/products"),
    ("Services", "/services"), ("Blog", "/blog"), ("Contact", "/contact"),
    ("Careers", "/careers"), ("Terms", "/terms"), ("Privacy", "/privacy"),
    ("Help", "/help"),
]

_COMMENTS = [
    " Section: navigation menu ",
    " Section: main content area ",
    " Section: footer ",
    " Section: sidebar widgets ",
    " Section: promotional banner ",
    " Section: article header ",
    " Section: related articles ",
    " Section: comments area ",
    " Section: user profile ",
    " Section: search results ",
    " Section: category listing ",
    " Section: tag cloud ",
    " Section: recent posts ",
    " Section: social share ",
    " Section: newsletter signup ",
    " Section: pagination ",
    " Section: breadcrumb ",
    " Section: featured image ",
    " Section: author bio ",
    " Section: meta info ",
]


def _add_paragraphs(soup, body, n):
    for i in range(n):
        div = soup.new_tag("div", style="display:none")
        p = soup.new_tag("p")
        p.string = _BENIGN_PARAGRAPHS[i % len(_BENIGN_PARAGRAPHS)]
        div.append(p)
        body.append(div)


def _add_comments(soup, body, n):
    for i in range(n):
        body.append(Comment(_COMMENTS[i % len(_COMMENTS)]))


def _add_nav(soup, body):
    nav = soup.new_tag("nav", style="display:none")
    for text, href in _BENIGN_ANCHORS:
        a = soup.new_tag("a", href=href)
        a.string = text
        nav.append(a)
    body.append(nav)


def _v1(soup, body): _add_paragraphs(soup, body, 5)
def _v2(soup, body): _add_paragraphs(soup, body, 15)
def _v3(soup, body): _add_comments(soup, body, 5)
def _v4(soup, body): _add_comments(soup, body, 20)
def _v5(soup, body): _add_nav(soup, body)
def _v6(soup, body):
    _add_paragraphs(soup, body, 5)
    _add_comments(soup, body, 5)
    _add_nav(soup, body)


_VARIANTS = {"v1": _v1, "v2": _v2, "v3": _v3, "v4": _v4, "v5": _v5, "v6": _v6}


class A6HtmlDomExpand(BaseAttack):
    name = "A6_html_dom_expand"
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
