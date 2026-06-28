"""Keyword and tag sets used across feature extraction (§4.3)."""

CREDENTIAL_KEYWORDS = [
    "login", "signin", "sign-in", "log-in", "password", "passwd",
    "credential", "authenticate", "auth", "sso", "oauth",
]
URGENCY_KEYWORDS = [
    "verify", "confirm", "suspend", "expire", "urgent", "immediate",
    "alert", "warning", "locked", "unlock", "restrict", "restore",
    "update", "secure", "unusual", "unauthorized", "limited",
]
TARGET_KEYWORDS = [
    "bank", "banking", "paypal", "wallet", "payment", "billing",
    "invoice", "transaction", "transfer", "refund",
]
ACTION_KEYWORDS = [
    "submit", "continue", "proceed", "click", "download", "install",
    "enable", "activate", "accept",
]
BRAND_KEYWORDS = [
    "microsoft", "apple", "google", "amazon", "facebook", "netflix",
    "linkedin", "dropbox", "adobe", "paypal", "chase", "wellsfargo",
    "dhl", "usps", "fedex", "instagram", "whatsapp", "telegram",
]
ALL_PHISHING_KEYWORDS = CREDENTIAL_KEYWORDS + URGENCY_KEYWORDS + TARGET_KEYWORDS

FORM_TAGS = {"form", "fieldset", "legend"}
INPUT_TAGS = {"input", "textarea", "select", "option"}
INTERACTIVE_TAGS = {"button", "a", "select", "input", "label"}
SEMANTIC_TAGS = {"header", "nav", "main", "article", "section", "aside", "footer"}
DIV_SPAN_TAGS = {"div", "span"}
MEDIA_TAGS = {"img", "video", "audio", "svg", "canvas", "picture"}
TABLE_TAGS = {"table", "tr", "td", "th", "thead", "tbody"}
CONTENT_TEXT_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "dt", "dd"}

SUSPICIOUS_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top",
    ".buzz", ".club", ".work", ".info",
}

URL_TOKEN_STOP_WORDS = {
    "http", "https", "www", "com", "org", "net", "html", "php", "asp",
}
BPE_SUSPICIOUS_WORDS = {
    "login", "verify", "secure", "update", "account", "signin",
    "bank", "paypal", "confirm", "password", "credential", "alert",
    "wallet", "billing", "suspend", "restore",
}

# Node types (§4.1, Table 2)
NODE_TYPE_HTML = "html"
NODE_TYPE_SCRIPT = "script"
NODE_TYPE_NETWORK = "network"
NODE_TYPE_STORAGE = "storage"

# Edge types (§4.1, Table 2) — paper uses "Execute" and "Access"
EDGE_TYPE_STRUCTURE = "structure"
EDGE_TYPE_EXECUTE = "execute"
EDGE_TYPE_REQUEST = "request"
EDGE_TYPE_ACCESS = "access"
