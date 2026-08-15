"""
identity.py — Resolve offender identity to a permanent, platform-stable ID.

Display names change. Permanent IDs (numeric entity IDs, channel handles, etc.)
do not. This module is the canonical source of identity resolution for the pipeline.

Architecture: two-phase resolution per platform.
  Phase 1 — URL parsing (pure, no I/O, confidence 1.0):
    Extracts the permanent ID directly from the URL structure.
    Covers the majority of real-world cases.

  Phase 2 — HTML scraping (requires PageFetcher, confidence 0.9):
    Fetches the profile page and extracts the ID from embedded metadata.
    Only runs when Phase 1 cannot extract the ID.

Platform permanent-ID notes:
  Facebook : numeric entity ID (page, profile, group). Never changes.
  YouTube  : channel ID starting with "UC" (22 chars). Never changes.
  Telegram : numeric chat ID from Bot API. Username is NOT permanent.
             URL-only extraction yields the username at confidence=0.5.
  TikTok   : numeric uid from page source. Username is NOT permanent.
             URL-only extraction yields the username at confidence=0.6.
"""

import datetime
import json
import re
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import parse_qs, urlparse


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------

@dataclass
class OffenderIdentity:
    platform: str            # e.g. "facebook"
    permanent_id: str        # platform-issued permanent identifier (never changes)
    display_name: str        # name at time of resolution (informational only)
    profile_url: str         # canonical profile/channel URL
    resolved_at: str         # ISO 8601 timestamp of last resolution
    confidence: float        # 0.0–1.0 confidence that this ID is correct
    metadata: dict           # platform-specific extras (follower_count, join_date…)


# ---------------------------------------------------------------------------
# PageFetcher protocol
# ---------------------------------------------------------------------------

class PageFetcher(Protocol):
    """
    Minimal interface for fetching a URL's HTML content.

    Implement with Playwright (headless, handles JS) or httpx (lightweight).
    Tests inject a _FakeFetcher that returns pre-baked HTML strings.
    """

    def fetch(self, url: str) -> str:
        """Return the page HTML as a string. Raise on HTTP errors."""
        ...


class HttpxFetcher:
    """
    Production PageFetcher backed by httpx.

    Uses a browser-like User-Agent. Suitable for pages that don't require JS.
    For JS-rendered pages, use a Playwright-backed fetcher from detection.py.
    """

    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
    }

    def __init__(self, timeout: float = 15.0) -> None:
        self._timeout = timeout

    def fetch(self, url: str) -> str:
        import httpx   # local import — httpx is optional at module load time
        response = httpx.get(url, headers=self._HEADERS, timeout=self._timeout,
                             follow_redirects=True)
        response.raise_for_status()
        return response.text


# ---------------------------------------------------------------------------
# Facebook — pure extraction functions
# ---------------------------------------------------------------------------

# URL path prefixes that identify content (video/post) not a profile
_FB_CONTENT_PREFIXES: frozenset[str] = frozenset({
    "watch", "video", "videos", "reel", "reels",
    "story", "stories", "live", "permalink", "photo", "photos",
})


def _fb_id_from_url(url: str) -> str | None:
    """
    Extract a Facebook numeric entity ID from a profile/group/page URL.

    Handles:
      profile.php?id=<NUM>
      /pages/{category}/{name}/{NUM}/
      /groups/<NUM>/

    Returns None for content URLs (watch, reel, etc.) and username-only
    profile URLs, which require an HTML fetch to resolve.
    """
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    parts = [p for p in parsed.path.strip("/").split("/") if p]

    # profile.php?id=<NUM>
    if parts == ["profile.php"] and "id" in qs:
        val = qs["id"][0]
        if val.isdigit():
            return val

    # No parts → homepage, unresolvable
    if not parts:
        return None

    # Skip content URLs — these carry video/post IDs, not profile IDs
    if parts[0] in _FB_CONTENT_PREFIXES:
        return None

    # /groups/<NUM>/... — numeric group ID is the second segment
    if parts[0] == "groups" and len(parts) >= 2 and parts[1].isdigit():
        return parts[1]

    # /pages/{anything…}/{NUM}/ — scan for a numeric ID ≥ 6 digits
    if parts[0] == "pages":
        for part in reversed(parts[1:]):
            if part.isdigit() and len(part) >= 6:
                return part
        return None   # pages/ without a numeric ID → needs fetch

    # /<username> or similar — no numeric ID extractable from URL alone
    return None


def _fb_id_from_html(html: str) -> str | None:
    """
    Extract a Facebook numeric entity ID from a fetched profile page.

    Checks in priority order:
      1. Android deep-link meta tag: fb://page/123 or fb://profile/123
      2. pageID JSON field embedded in page source
      3. entity_id JSON field embedded in page source
    """
    # 1. al:android:url — most reliable, survives FB redesigns
    m = re.search(r'fb://(?:page|profile|group)/(\d+)', html)
    if m:
        return m.group(1)

    # 2. pageID in page JSON
    m = re.search(r'"pageID"\s*:\s*"(\d+)"', html)
    if m:
        return m.group(1)

    # 3. entity_id in page JSON
    m = re.search(r'"entity_id"\s*:\s*"(\d+)"', html)
    if m:
        return m.group(1)

    return None


def _fb_display_name_from_html(html: str) -> str:
    """Extract the Facebook page/profile display name from page HTML."""
    m = re.search(r'<meta\s+property="og:title"\s+content="([^"]+)"', html)
    if m:
        return m.group(1)
    m = re.search(r'<title>([^|<]+)', html)
    if m:
        return m.group(1).strip()
    return ""


def _fb_canonical_url(permanent_id: str) -> str:
    """Return the canonical Facebook profile URL for a numeric ID."""
    return f"https://www.facebook.com/profile.php?id={permanent_id}"


# ---------------------------------------------------------------------------
# YouTube — pure extraction functions
# ---------------------------------------------------------------------------

# UC channel IDs are exactly 24 chars: "UC" + 22 base64url chars
_YT_CHANNEL_RE = re.compile(r"UC[a-zA-Z0-9_-]{22}")


def _yt_channel_id_from_url(url: str) -> str | None:
    """
    Extract a YouTube channel ID (UCxxxxxx) from a /channel/UC… URL.

    Returns None for /@Handle and /c/CustomName URLs — those require
    the YouTube Data API to resolve to a channel ID.
    """
    parsed = urlparse(url)
    m = re.match(r"^/channel/(UC[a-zA-Z0-9_-]{22})$", parsed.path)
    return m.group(1) if m else None


def _yt_channel_id_from_html(html: str) -> str | None:
    """
    Extract a YouTube channel ID from a fetched channel page.

    YouTube embeds the channel ID in multiple locations in the page JSON.
    """
    # "channelId":"UCxxxxxx" in ytInitialData or ytInitialPlayerResponse
    m = re.search(r'"channelId"\s*:\s*"(UC[a-zA-Z0-9_-]{22})"', html)
    if m:
        return m.group(1)

    # Canonical link: <link rel="canonical" href="...channel/UCxxxxxx">
    m = re.search(r'/channel/(UC[a-zA-Z0-9_-]{22})', html)
    if m:
        return m.group(1)

    return None


def _yt_display_name_from_html(html: str) -> str:
    """Extract YouTube channel display name from page HTML."""
    # <title>Channel Name - YouTube</title>
    m = re.search(r"<title>(.+?)\s*[-–]\s*YouTube\s*</title>", html)
    if m:
        return m.group(1).strip()
    m = re.search(r'"channelName"\s*:\s*"([^"]+)"', html)
    if m:
        return m.group(1)
    return ""


def _yt_canonical_url(channel_id: str) -> str:
    return f"https://www.youtube.com/channel/{channel_id}"


# ---------------------------------------------------------------------------
# Telegram — pure extraction functions
# ---------------------------------------------------------------------------

_TG_DOMAINS: frozenset[str] = frozenset({"t.me", "telegram.me", "www.t.me"})


def _tg_username_from_url(url: str) -> str | None:
    """
    Extract a Telegram username from a t.me URL.

    Invite links (+hash) and paths with slashes are excluded.
    The returned value is the username string, NOT a numeric chat ID.

    Telegram usernames CAN change — the true permanent identifier is
    the numeric chat ID, which requires a Bot API call to obtain.
    The caller should treat the username as confidence=0.5 until verified.
    """
    parsed = urlparse(url)
    if parsed.netloc not in _TG_DOMAINS:
        return None
    username = parsed.path.strip("/")
    if not username or username.startswith("+") or "/" in username:
        return None
    # Telegram usernames: 5–32 alphanumeric/underscore chars
    if re.fullmatch(r"[a-zA-Z0-9_]{5,32}", username):
        return username
    return None


def _tg_canonical_url(username: str) -> str:
    return f"https://t.me/{username}"


# ---------------------------------------------------------------------------
# TikTok — pure extraction functions
# ---------------------------------------------------------------------------

_TT_DOMAINS: frozenset[str] = frozenset({"www.tiktok.com", "tiktok.com"})


def _tt_username_from_url(url: str) -> str | None:
    """
    Extract a TikTok @username from a profile URL.

    TikTok usernames CAN change — the true permanent identifier is
    the numeric uid, which requires scraping __NEXT_DATA__ from the page.
    The caller should treat the username as confidence=0.6 until uid is fetched.
    """
    parsed = urlparse(url)
    if parsed.netloc not in _TT_DOMAINS:
        return None
    m = re.match(r"^/@([a-zA-Z0-9_.]{2,24})$", parsed.path)
    return m.group(1) if m else None


def _tt_uid_from_html(html: str) -> str | None:
    """
    Extract the TikTok numeric uid from __NEXT_DATA__ JSON in a profile page.

    The uid is the only permanent identifier for TikTok accounts.
    """
    m = re.search(
        r'<script\s+id="__NEXT_DATA__"[^>]*>\s*(\{.*?\})\s*</script>',
        html, re.DOTALL
    )
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
        user = data["props"]["pageProps"]["userInfo"]["user"]
        uid = user.get("id") or user.get("uid")
        return str(uid) if uid else None
    except (KeyError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _tt_display_name_from_html(html: str) -> str:
    """Extract TikTok display name from __NEXT_DATA__ or og:title."""
    m = re.search(
        r'<script\s+id="__NEXT_DATA__"[^>]*>\s*(\{.*?\})\s*</script>',
        html, re.DOTALL
    )
    if m:
        try:
            data = json.loads(m.group(1))
            user = data["props"]["pageProps"]["userInfo"]["user"]
            name = user.get("nickname") or user.get("uniqueId", "")
            if name:
                return name
        except (KeyError, json.JSONDecodeError, TypeError):
            pass
    m = re.search(r'<meta\s+property="og:title"\s+content="([^"]+)"', html)
    return m.group(1) if m else ""


def _tt_canonical_url(username: str) -> str:
    return f"https://www.tiktok.com/@{username}"


# ---------------------------------------------------------------------------
# IdentityResolver
# ---------------------------------------------------------------------------

class IdentityResolver:
    """
    Resolves a scraped profile URL to a stable OffenderIdentity.

    Resolution is two-phase per platform:
      Phase 1 — URL parsing (no I/O, confidence 1.0 for numeric IDs).
      Phase 2 — HTML scraping via PageFetcher (confidence 0.9).
                Only runs if Phase 1 fails and a fetcher is configured.

    Results are cached by (platform, normalized_url) to avoid redundant
    fetches within a pipeline run.

    Args:
        fetcher: PageFetcher for HTML scraping. None = Phase 2 disabled.
        cache:   injected dict for testing (default: fresh dict per instance).
    """

    def __init__(
        self,
        fetcher: PageFetcher | None = None,
        cache: dict[str, OffenderIdentity] | None = None,
    ) -> None:
        self._fetcher = fetcher
        self._cache: dict[str, OffenderIdentity] = {} if cache is None else cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self,
        platform: str,
        raw_url: str,
        display_name_hint: str = "",
    ) -> OffenderIdentity | None:
        """
        Resolve a raw scraped profile URL to a permanent OffenderIdentity.

        Args:
            platform: platform slug ("facebook", "youtube", "telegram", "tiktok")
            raw_url: profile/channel URL as returned by the crawler
            display_name_hint: display name from the crawler, used as fallback
                               when HTML is not fetched

        Returns:
            OffenderIdentity if resolved, None if unresolvable or unsupported platform.

        Results are cached by (platform, raw_url). Calling resolve() twice
        with the same arguments returns the cached result without a second fetch.
        """
        cache_key = f"{platform}:{raw_url}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        resolver = {
            "facebook": self._resolve_facebook,
            "youtube":  self._resolve_youtube,
            "telegram": self._resolve_telegram,
            "tiktok":   self._resolve_tiktok,
        }.get(platform)

        if resolver is None:
            return None

        identity = resolver(raw_url, display_name_hint)
        if identity is not None:
            self._cache[cache_key] = identity
        return identity

    # ------------------------------------------------------------------
    # Per-platform resolvers
    # ------------------------------------------------------------------

    def _resolve_facebook(
        self, url: str, display_name_hint: str
    ) -> OffenderIdentity | None:
        """Two-phase Facebook resolution: URL pattern → HTML scrape."""
        pid = _fb_id_from_url(url)
        display_name = display_name_hint
        confidence = 1.0

        if pid is None:
            if self._fetcher is None:
                return None
            html = self._fetcher.fetch(url)
            pid = _fb_id_from_html(html)
            if pid is None:
                return None
            display_name = _fb_display_name_from_html(html) or display_name_hint
            confidence = 0.9

        return OffenderIdentity(
            platform="facebook",
            permanent_id=pid,
            display_name=display_name,
            profile_url=_fb_canonical_url(pid),
            resolved_at=_utc_now(),
            confidence=confidence,
            metadata={},
        )

    def _resolve_youtube(
        self, url: str, display_name_hint: str
    ) -> OffenderIdentity | None:
        """Two-phase YouTube resolution: /channel/UC… URL → HTML scrape."""
        channel_id = _yt_channel_id_from_url(url)
        display_name = display_name_hint
        confidence = 1.0

        if channel_id is None:
            if self._fetcher is None:
                return None
            html = self._fetcher.fetch(url)
            channel_id = _yt_channel_id_from_html(html)
            if channel_id is None:
                return None
            display_name = _yt_display_name_from_html(html) or display_name_hint
            confidence = 0.9

        return OffenderIdentity(
            platform="youtube",
            permanent_id=channel_id,
            display_name=display_name,
            profile_url=_yt_canonical_url(channel_id),
            resolved_at=_utc_now(),
            confidence=confidence,
            metadata={},
        )

    def _resolve_telegram(
        self, url: str, display_name_hint: str
    ) -> OffenderIdentity | None:
        """
        Telegram resolution: extract username from t.me URL.

        Returns confidence=0.5 because Telegram usernames can change.
        The true permanent ID is the numeric chat ID, obtainable only via
        the Bot API (requires TG_BOT_TOKEN env var — not yet integrated in v1).
        """
        username = _tg_username_from_url(url)
        if username is None:
            return None

        return OffenderIdentity(
            platform="telegram",
            permanent_id=username,      # NOT permanent — see docstring
            display_name=display_name_hint,
            profile_url=_tg_canonical_url(username),
            resolved_at=_utc_now(),
            confidence=0.5,             # username can change; Bot API needed for 1.0
            metadata={"id_type": "username", "numeric_id_pending": True},
        )

    def _resolve_tiktok(
        self, url: str, display_name_hint: str
    ) -> OffenderIdentity | None:
        """
        Two-phase TikTok resolution: username from URL → uid from __NEXT_DATA__.

        Phase 1: extract username (confidence=0.6 — usernames can change).
        Phase 2: scrape __NEXT_DATA__ for numeric uid (confidence=0.95).
        """
        username = _tt_username_from_url(url)
        if username is None:
            return None

        # Phase 2: try to get the stable numeric uid from page source
        if self._fetcher is not None:
            try:
                html = self._fetcher.fetch(url)
                uid = _tt_uid_from_html(html)
                if uid:
                    display_name = _tt_display_name_from_html(html) or display_name_hint
                    return OffenderIdentity(
                        platform="tiktok",
                        permanent_id=uid,
                        display_name=display_name,
                        profile_url=_tt_canonical_url(username),
                        resolved_at=_utc_now(),
                        confidence=0.95,
                        metadata={"username": username},
                    )
            except Exception:
                pass   # fall through to Phase 1 result

        # Phase 1 fallback: username only
        return OffenderIdentity(
            platform="tiktok",
            permanent_id=username,      # NOT permanent — see docstring
            display_name=display_name_hint,
            profile_url=_tt_canonical_url(username),
            resolved_at=_utc_now(),
            confidence=0.6,             # username can change; uid needed for higher confidence
            metadata={"id_type": "username", "numeric_uid_pending": True},
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def is_same_identity(a: OffenderIdentity, b: OffenderIdentity) -> bool:
    """
    Return True if two OffenderIdentity objects refer to the same real-world entity.

    Compares platform + permanent_id. Cross-platform identity linking
    (e.g. same pirate on FB and TikTok) is out of scope for v1.
    """
    return a.platform == b.platform and a.permanent_id == b.permanent_id


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
