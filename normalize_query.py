"""
normalize_query.py — Text normalization for consistent query generation and matching.

Normalization ensures that "Vidio Premium", "VIDIO.COM PREMIUM", and "vidio prem1um"
all resolve to the same canonical form for dedup and fuzzy matching.
"""

import re
import unicodedata
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

try:
    from rapidfuzz.fuzz import token_set_ratio as _token_set_ratio
    _RAPIDFUZZ = True
except ImportError:
    from difflib import SequenceMatcher as _SequenceMatcher
    _RAPIDFUZZ = False


# Leet-speak substitution map — extend as new evasion patterns are observed
# All replacements are single-char → single-char so str.translate() is used (faster than replace loop)
LEET_MAP: dict[str, str] = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "@": "a",
    "$": "s",
    "!": "i",
    "|": "l",
}

# Pre-built translation table for O(n) leet decoding
_LEET_TABLE = str.maketrans(LEET_MAP)

# Noise words to strip after normalization.
# Quality markers are stored in their post-leet form because leet is applied before
# stopword removal (e.g. "1080p" → "io8op", "720p" → "t2op", "480p" → "a8op").
STOPWORDS: set[str] = {
    "streaming", "live", "gratis", "free", "nonton", "bisa", "link",
    "tonton", "full", "hd", "fhd", "ntn",
    # Quality markers in post-leet form
    "io8op",   # 1080p
    "t2op",    # 720p
    "a8op",    # 480p
}

# Default brand terms appended when generating crawler queries
_DEFAULT_BRAND_TERMS: list[str] = ["vidio", "vidio.com"]

# Known pirate evasion spellings per brand term.
# Extend this list as new patterns are spotted in the wild.
BRAND_EVASION_VARIANTS: dict[str, list[str]] = {
    "vidio": ["v1dio", "vid1o", "v1d1o", "v1d10", "vidie", "vidi0"],
}

# URL query params that are tracking noise and should be stripped before hashing
_URL_NOISE_PARAMS: frozenset[str] = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "_rdc", "_rdr", "igshid", "yclid", "gclid", "mcid", "ncid",
    "ref", "refid", "fref", "hc_ref", "__tn__", "__cft__",
})

# Facebook domain variants → canonical www.facebook.com
_FB_DOMAIN_CANON: dict[str, str] = {
    "m.facebook.com": "www.facebook.com",
    "fb.com": "www.facebook.com",
    "web.facebook.com": "www.facebook.com",
    "l.facebook.com": "www.facebook.com",   # link redirect wrapper
    "mbasic.facebook.com": "www.facebook.com",
}


def normalize(text: str) -> str:
    """
    Full normalization pipeline for a query or title string.

    Steps applied in order:
      1. Unicode normalization (NFKC) — collapse ligatures, fullwidth chars, etc.
      2. Lowercase
      3. Leet-speak substitution — @→a, $→s, 0→o, 1→i, etc.
      4. Strip non-alphanumeric except spaces — removes punctuation, emojis, etc.
      5. Tokenize and remove stopwords
      6. Collapse whitespace

    Args:
        text: raw title or query from crawler output

    Returns:
        Normalized string suitable for exact or fuzzy comparison.
        Returns empty string for empty or whitespace-only input.
    """
    if not text or not text.strip():
        return ""

    # Steps 1–2: unicode + lowercase
    text = unicodedata.normalize("NFKC", text).lower()

    # Step 3: leet — must come before stripping so @, $, | etc. → letters
    text = _apply_leet(text)

    # Step 4: keep only a-z, 0-9, space (post-leet residual digits are real, e.g. "8" in "io8op")
    text = re.sub(r"[^a-z0-9 ]", " ", text)

    # Steps 5–6: stopwords + collapse
    tokens = _strip_stopwords(text.split())
    return " ".join(tokens)


def normalize_url(url: str) -> str:
    """
    Normalize a URL for deduplication purposes.

    - Strips UTM params, FB click IDs, and other tracking noise
    - Remaps Facebook mobile/alternate domains to www.facebook.com
    - Lowercases scheme and host
    - Strips trailing slashes from path

    Args:
        url: raw URL from crawler output

    Returns:
        Normalized URL string. Returns lowercased input on parse failure.
    """
    try:
        parsed = urlparse(url.strip())
    except Exception:
        return url.strip().lower()

    scheme = (parsed.scheme or "https").lower()
    host = parsed.netloc.lower()
    host = _FB_DOMAIN_CANON.get(host, host)

    # Strip noise params; preserve meaningful path params (e.g. FB video ID)
    clean_params = {
        k: v
        for k, v in parse_qs(parsed.query, keep_blank_values=False).items()
        if k not in _URL_NOISE_PARAMS
    }

    path = parsed.path.rstrip("/") or "/"
    return urlunparse((scheme, host, path, "", urlencode(clean_params, doseq=True), ""))


def expand_to_queries(
    content_titles: list[str],
    brand_terms: list[str] | None = None,
    max_queries: int = 50,
) -> list[str]:
    """
    Expand a list of Vidio content titles into search queries for crawlers.

    For each title, generates (in priority order):
      1. title + brand_term  (e.g. "sinetron x vidio")
      2. title alone         (broader, catches unlabeled re-uploads)
      3. title + evasion variant (e.g. "sinetron x v1d10")

    Priority matters: if max_queries is hit, the highest-signal queries survive.

    Args:
        content_titles: list of canonical Vidio content titles (raw text OK)
        brand_terms: brand strings to append; defaults to ["vidio", "vidio.com"]
        max_queries: hard cap on output count to avoid overwhelming crawlers

    Returns:
        Deduplicated, priority-ordered list of query strings ready for crawler input.
    """
    if brand_terms is None:
        brand_terms = _DEFAULT_BRAND_TERMS

    seen: set[str] = set()
    brand_queries: list[str] = []
    plain_queries: list[str] = []
    evasion_queries: list[str] = []

    for title in content_titles:
        norm = normalize(title)
        if not norm:
            continue

        # Priority 1: brand queries
        for term in brand_terms:
            q = f"{norm} {term}"
            if q not in seen:
                seen.add(q)
                brand_queries.append(q)

        # Priority 2: plain title
        if norm not in seen:
            seen.add(norm)
            plain_queries.append(norm)

        # Priority 3: evasion variants
        for term in brand_terms:
            for variant in BRAND_EVASION_VARIANTS.get(term, []):
                q = f"{norm} {variant}"
                if q not in seen:
                    seen.add(q)
                    evasion_queries.append(q)

    ordered = brand_queries + plain_queries + evasion_queries
    return ordered[:max_queries]


def similarity(a: str, b: str) -> float:
    """
    Compute normalized similarity score between two strings (0.0–1.0).

    Both strings are normalized via normalize() before comparison.
    Uses rapidfuzz token_set_ratio (order-insensitive) when available,
    falls back to difflib SequenceMatcher if rapidfuzz is not installed.

    Token-set ratio treats {"foo", "bar"} == {"bar", "foo"}, which handles
    titles where word order varies across platforms.

    Args:
        a, b: strings to compare (raw or pre-normalized)

    Returns:
        1.0 if identical, 0.0 if no token overlap.
        Both-empty returns 1.0 (they are equal — the empty set).
    """
    a_norm = normalize(a)
    b_norm = normalize(b)

    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0

    if _RAPIDFUZZ:
        return _token_set_ratio(a_norm, b_norm) / 100.0
    return _SequenceMatcher(None, a_norm, b_norm).ratio()


def _apply_leet(text: str) -> str:
    """Replace leet characters with their alphabetic equivalents using str.translate."""
    return text.translate(_LEET_TABLE)


def _strip_stopwords(tokens: list[str]) -> list[str]:
    """Remove tokens that appear in STOPWORDS (case-sensitive on already-lowercased input)."""
    return [t for t in tokens if t not in STOPWORDS]
