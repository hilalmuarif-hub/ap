"""
detection.py — Multi-platform piracy crawler/scraper.

Starts with a single platform (Facebook), designed for easy extension.
Each platform returns a list of RawDetection objects for downstream processing.

Architecture: two clear layers.
  Pure functions (module-level, no I/O):
    URL building, HTML extraction, result parsing — all fully unit-testable.

  I/O layer (injectable protocol):
    BrowserFactory / PageDriver — Playwright in production, fakes in tests.
    FacebookCrawler orchestrates the two layers.

Adding a new platform:
  1. Subclass BasePlatformCrawler.
  2. Implement _build_search_url, _parse_results.
  3. Register in _CRAWLER_REGISTRY.
"""

import datetime
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Generator, Protocol
from urllib.parse import parse_qs, quote_plus, urlparse


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------

@dataclass
class RawDetection:
    platform: str          # e.g. "facebook", "telegram", "youtube"
    url: str               # canonical URL of the infringing content
    title: str             # page/post/video title as-scraped
    channel_id: str        # platform-permanent ID (not display name)
    channel_name: str      # display name at time of scrape
    snapshot_html: str     # raw HTML snippet around the result (evidence)
    detected_at: str       # ISO 8601 UTC timestamp
    query_used: str        # search query that surfaced this result
    extra: dict = field(default_factory=dict)   # platform-specific extras


# ---------------------------------------------------------------------------
# PageDriver / BrowserFactory protocols
# ---------------------------------------------------------------------------

class PageDriver(Protocol):
    """Minimal browser page interface. Implemented by PlaywrightPageDriver and fakes."""

    def goto(self, url: str) -> None:
        """Navigate to `url` and wait for the page to settle."""
        ...

    def content(self) -> str:
        """Return the current page's full HTML content."""
        ...

    def scroll_down(self, pixels: int = 1000) -> None:
        """Scroll the page down by `pixels` to trigger lazy-loaded content."""
        ...

    def evaluate(self, expression: str) -> object:
        """Evaluate JavaScript expression in the page context."""
        ...

    def wait_for_timeout(self, millis: int) -> None:
        """Wait for the given number of milliseconds."""
        ...

    def close(self) -> None:
        """Release the page and its resources."""
        ...


class BrowserFactory(Protocol):
    """Produces PageDriver instances. One factory per crawler session."""

    def new_page(self) -> PageDriver:
        """Open a new browser tab and return its driver."""
        ...

    def close(self) -> None:
        """Shut down the browser and release all resources."""
        ...


# ---------------------------------------------------------------------------
# Production Playwright adapters
# ---------------------------------------------------------------------------

class PlaywrightPageDriver:
    """
    Production PageDriver backed by a Playwright sync Page.

    Wraps playwright.sync_api.Page so that callers never import playwright
    directly — only this adapter touches the Playwright API.
    """

    def __init__(self, page) -> None:   # page: playwright.sync_api.Page
        self._page = page
        # Bound all page operations to 5s. Without this, evaluate() and
        # content() can hang indefinitely if the browser connection breaks.
        # Keep it short: these ops should be instant; 5s is generous.
        self._page.set_default_timeout(5_000)

    def goto(self, url: str) -> None:
        # "domcontentloaded" instead of "networkidle" — Facebook never reaches
        # networkidle because it continuously polls; domcontentloaded fires as
        # soon as the HTML is parsed, which is enough to get search results.
        self._page.goto(url, wait_until="domcontentloaded", timeout=60_000)

    def content(self) -> str:
        return self._page.content()

    def scroll_down(self, pixels: int = 1000) -> None:
        try:
            self._page.evaluate(f"window.scrollBy(0, {pixels})")
        except Exception:
            pass   # scroll failure is non-critical, continue with current HTML

    def evaluate(self, expression: str) -> object:
        """Run JavaScript in the page context. Returns the result."""
        try:
            return self._page.evaluate(expression)
        except Exception:
            return None

    def wait_for_timeout(self, millis: int) -> None:
        """Wait for the given number of milliseconds."""
        self._page.wait_for_timeout(millis)

    def close(self) -> None:
        self._page.close()


class PlaywrightBrowserFactory:
    """
    Production BrowserFactory backed by Playwright Chromium.

    Usage:
        factory = PlaywrightBrowserFactory(headless=True, cookies_path="cookies.json")
        factory.start()
        crawler = FacebookCrawler(browser_factory=factory)
        ...
        factory.close()

    Or as a context manager:
        with PlaywrightBrowserFactory() as factory:
            ...
    """

    def __init__(
        self,
        headless: bool = True,
        cookies_path: str | None = None,
    ) -> None:
        self.headless = headless
        self.cookies_path = cookies_path
        self._pw = None
        self._browser = None

    def start(self) -> None:
        """Initialize Playwright. Must be called before new_page()."""
        import json
        from playwright.sync_api import sync_playwright   # local import
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        self._cookies: list[dict] = []
        if self.cookies_path:
            with open(self.cookies_path) as f:
                self._cookies = json.load(f)

    def new_page(self) -> PlaywrightPageDriver:
        context = self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )
        if self._cookies:
            context.add_cookies(self._cookies)
        page = context.new_page()
        return PlaywrightPageDriver(page)

    def close(self) -> None:
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._pw:
            self._pw.stop()
            self._pw = None

    def __enter__(self) -> "PlaywrightBrowserFactory":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Facebook — pure extraction functions (no I/O, fully unit-testable)
# ---------------------------------------------------------------------------

# Path prefixes that indicate content (video/post) rather than a profile page
_FB_CONTENT_PREFIXES: frozenset[str] = frozenset({
    "watch", "video", "videos", "reel", "reels",
    "live", "stories", "story", "photo", "photos",
    "permalink",
})

# Path prefixes for page/profile URLs we want to skip during channel extraction
_FB_SYSTEM_PATHS: frozenset[str] = frozenset({
    "watch", "video", "videos", "reel", "reels", "live",
    "stories", "story", "search", "marketplace", "groups",
    "pages", "events", "gaming", "photo", "photos",
    "about", "friends", "timeline", "help", "settings",
})


def _fb_search_url(query: str) -> str:
    """Build a Facebook video-search URL for a given query string."""
    return f"https://www.facebook.com/search/videos/?q={quote_plus(query)}"


def _normalize_fb_video_url(url: str) -> str | None:
    """
    Normalize a raw Facebook video URL to a canonical, dedup-safe form.

    Handled patterns:
      /watch?v=<NUM>            → https://www.facebook.com/watch?v=<NUM>
      /videos/<NUM>/            → https://www.facebook.com/videos/<NUM>
      /reel/<NUM>/              → https://www.facebook.com/reel/<NUM>
      /<page>/videos/<NUM>/     → https://www.facebook.com/<page>/videos/<NUM>

    Returns None for non-video URLs or unrecognised patterns.
    """
    if not url:
        return None

    # Make absolute
    if url.startswith("/"):
        url = f"https://www.facebook.com{url}"

    parsed = urlparse(url)
    if parsed.netloc and "facebook.com" not in parsed.netloc:
        return None

    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if not parts:
        return None

    # /watch?v=<NUM>
    if parts[0] == "watch":
        qs = parse_qs(parsed.query)
        vid = qs.get("v", [""])[0]
        return f"https://www.facebook.com/watch?v={vid}" if vid.isdigit() else None

    # /videos/<NUM>  /reel/<NUM>  /reels/<NUM>
    if parts[0] in ("videos", "reel", "reels") and len(parts) >= 2 and parts[1].isdigit():
        return f"https://www.facebook.com/{parts[0]}/{parts[1]}"

    # /groups/<GID>/videos/<VID>  or  /groups/<GID>/live/<VID>
    if (
        parts[0] == "groups"
        and len(parts) >= 4
        and parts[1].isdigit()
        and parts[2] in ("videos", "video", "live")
        and parts[3].isdigit()
    ):
        return f"https://www.facebook.com/groups/{parts[1]}/{parts[2]}/{parts[3]}"

    # /<page>/videos/<NUM>   /<page>/live/<NUM>
    if (
        len(parts) >= 3
        and parts[1] in ("videos", "video", "live")
        and parts[2].isdigit()
    ):
        return f"https://www.facebook.com/{parts[0]}/{parts[1]}/{parts[2]}"

    # Last path segment is numeric ID under a known content prefix
    if parts[0] in _FB_CONTENT_PREFIXES and parts[-1].isdigit():
        return f"https://www.facebook.com/{'/'.join(parts)}"

    return None


def _extract_video_urls_from_html(html: str) -> list[str]:
    """
    Extract unique canonical Facebook video/reel/stream URLs from HTML.

    Scans all href attributes for durability across FB HTML structure changes.
    Returns URLs in order of first appearance, deduplicated.
    """
    seen: set[str] = set()
    results: list[str] = []

    for raw_href in re.findall(r'href="([^"]+)"', html):
        href = raw_href.replace("&amp;", "&")
        norm = _normalize_fb_video_url(href)
        if norm and norm not in seen:
            seen.add(norm)
            results.append(norm)

    return results


def _extract_page_url_near_video(html: str, video_url: str) -> str | None:
    """
    Find the posting page/profile URL in the HTML near a video URL.

    Facebook renders the page name adjacent to the video link in search results.
    Scans a 2 000-character window around the video URL position.

    Priority order:
      1. Numeric profile/group/pages URLs (permanent IDs available from URL)
      2. Username-style profile URLs (impermanent — needs identity resolution)
    """
    idx = _find_url_in_html(html, video_url)
    if idx == -1:
        return None

    # Forward-only window: channel links always appear AFTER the video link in
    # FB search result HTML. Any look-back risks picking up data from the
    # previous result card. We start at idx (the video URL position itself).
    window = html[idx: idx + 1000]

    for pattern in (
        # Numeric profile — highest confidence
        r'href="((?:https://www\.facebook\.com)?/profile\.php\?id=\d+)"',
        # Pages URL — appears within the same result card as the video
        r'href="((?:https://www\.facebook\.com)?/pages/[^"?]{5,})"',
        # Group profile — strict pattern excludes /groups/<ID>/videos/<VID>
        r'href="((?:https://www\.facebook\.com)?/groups/\d+/?)"',
    ):
        m = re.search(pattern, window)
        if m:
            raw = m.group(1)
            return raw if raw.startswith("http") else f"https://www.facebook.com{raw}"

    # Username fallback — skip system/content paths
    m = re.search(r'href="/([A-Za-z0-9][A-Za-z0-9._-]{2,})/?(?:\?[^"]*)??"', window)
    if m:
        username = m.group(1).split("?")[0]
        if username not in _FB_SYSTEM_PATHS:
            return f"https://www.facebook.com/{username}"

    return None


def _extract_title_near_video(html: str, video_url: str) -> str:
    """
    Extract the video title from HTML near a video URL.

    Tries (in order): aria-label attribute, title attribute, nearby text content.
    Returns empty string when nothing can be extracted.
    """
    idx = _find_url_in_html(html, video_url)
    if idx == -1:
        return ""

    # Forward-only: titles (aria-label, title attr, text) always come after the
    # href value in FB HTML. Starting at idx prevents bleeding into prior cards.
    window = html[idx: idx + 400]

    # Find the FIRST/closest title-like attribute in the forward window,
    # regardless of whether it's aria-label or title.
    # This avoids hardcoding priority between the two: the attribute closest
    # to the video URL belongs to the current result card, not a later one.
    m = re.search(r'(?:aria-label|title)="([^"]{5,200})"', window)
    if m:
        return m.group(1).strip()

    chunks = [c.strip() for c in re.findall(r">([^<]{5,200})<", window) if c.strip()]
    return chunks[0] if chunks else ""


def _extract_view_count_near_video(html: str, video_url: str) -> int | None:
    """
    Extract view count near a video URL.

    Strategy A — HTML proximity: look for "X views" / "X penonton" text.
    Strategy B — FB JSON: look for view_count fields in the JSON data block
                 keyed by video ID (e.g. "play_count":{"count":12345}).

    Returns None when not found or unparseable.
    """
    idx = _find_url_in_html(html, video_url)
    if idx == -1:
        return None

    window = html[idx: idx + 1000]

    # Strategy A1: English pattern — "2K views", "1.2M views", "500 views"
    m = re.search(
        r'([\d,]+(?:\.\d+)?)\s*([KkMm])?\s*(?:views?|penonton|ditonton)',
        window,
    )
    if m:
        try:
            num = float(m.group(1).replace(",", ""))
            suffix = (m.group(2) or "").lower()
            if suffix == "k":
                num *= 1_000
            elif suffix == "m":
                num *= 1_000_000
            return int(num)
        except ValueError:
            pass

    # Strategy A2: Indonesian reel pattern — "6,8 rb Tayangan", "199 Tayangan"
    # Comma is decimal separator in Indonesian locale (6,8 = 6.8)
    m = re.search(r'([\d]+(?:[,.]\d+)?)\s*(rb|jt)?\s*Tayangan', window)
    if m:
        try:
            num = float(m.group(1).replace(",", "."))
            suffix = (m.group(2) or "").lower()
            if suffix == "rb":
                num *= 1_000
            elif suffix == "jt":
                num *= 1_000_000
            return int(num)
        except ValueError:
            pass

    # Strategy B: FB JSON numeric fields — "play_count" or "view_count"
    for field in (r'"play_count"\s*:\s*\{"count"\s*:\s*(\d+)',
                  r'"video_view_count"\s*:\s*\{"count"\s*:\s*(\d+)',
                  r'"view_count"\s*:\s*(\d+)'):
        j = re.search(field, window)
        if j:
            try:
                return int(j.group(1))
            except ValueError:
                pass

    return None


def _find_url_in_html(html: str, url: str) -> int:
    """
    Find the position of a URL in HTML, trying multiple strategies.

    FB search results render links as /watch/?ref=search&v=ID&external_log_id=...
    The normalized URL we store (/watch?v=ID) never appears literally.
    Strategy 3 finds the JSON data block that contains the video — this is
    the best anchor for title/channel extraction since the data is there.

    Returns -1 if not found.
    """
    # Phase 1: exact full URL
    idx = html.find(url)
    if idx != -1:
        return idx

    # Phase 2: relative path + query (no scheme/host)
    parsed = urlparse(url)
    path_q = parsed.path + (f"?{parsed.query}" if parsed.query else "")
    idx = html.find(path_q)
    if idx != -1:
        return idx

    # Phase 3: find video/reel ID in JSON data block.
    # Watch videos: "id":"VIDEO_ID" in FB's Relay JSON.
    # Reels: "open_video_uri":"\/reel\/REEL_ID" in click_metadata_model.
    vid = _fb_video_id_from_url(url)
    if vid:
        for pattern in (
            rf'"id"\s*:\s*"{re.escape(vid)}"',           # watch videos
            rf'\\\/reel\\\/{re.escape(vid)}',             # reels in JSON
            rf'\\\/videos\\\/{re.escape(vid)}',           # videos in JSON
        ):
            m = re.search(pattern, html)
            if m:
                return m.start()

    return -1


def _fb_video_id_from_url(url: str) -> str | None:
    """
    Extract the numeric video/reel ID from a normalized FB content URL.

    Handles:
      /watch?v=ID          → ID from query param
      /reel/ID             → ID from path (FB now shows reels in search)
      /reels/ID            → same
      /videos/ID           → same
      /groups/GID/videos/VID → VID
    """
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")

    # /watch?v=ID
    if path == "/watch":
        vid = parse_qs(parsed.query).get("v", [None])[0]
        return vid if vid and vid.isdigit() else None

    # /reel/ID, /reels/ID, /videos/ID
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 2 and parts[0] in ("reel", "reels", "videos") and parts[1].isdigit():
        return parts[1]

    # /<page>/videos/<VID>  (e.g. /bangkapos/videos/283466043640108)
    if (len(parts) >= 3 and parts[1] in ("videos", "video", "live")
            and parts[2].isdigit()):
        return parts[2]

    # /groups/GID/videos/VID
    if (len(parts) >= 4 and parts[0] == "groups" and parts[1].isdigit()
            and parts[2] in ("videos", "video", "live") and parts[3].isdigit()):
        return parts[3]

    return None


def _extract_fb_title_from_json(html: str, video_id: str) -> str:
    """
    Extract video title from FB's embedded JSON data by video ID.

    Watch videos: looks for "id":"VIDEO_ID" → finds "title":"..." nearby.
    Reels: looks for /reel/REEL_ID in click_metadata_model JSON.
    """
    m = None
    for pattern in (
        rf'"id"\s*:\s*"{re.escape(video_id)}"',
        rf'\\\/reel\\\/{re.escape(video_id)}',
        rf'\\\/videos\\\/{re.escape(video_id)}',
    ):
        m = re.search(pattern, html)
        if m:
            break
    if not m:
        return ""
    window = html[m.start(): m.start() + 2000]

    for field in (r'"title"\s*:\s*"([^"]{5,400})"',
                  r'"save_description"\s*:\s*"([^"]{5,400})"',
                  r'"accessibility_label"\s*:\s*"([^"]{5,400})"'):
        t = re.search(field, window)
        if t:
            # Unescape JSON unicode escapes (e.g. ⚽ -> ⚽)
            try:
                import json as _json
                return _json.loads(f'"{t.group(1)}"')
            except Exception:
                return t.group(1)
    return ""


def _extract_fb_channel_from_json(html: str, video_id: str) -> tuple[str, str]:
    """
    Extract (channel_id, channel_name) from FB's embedded JSON near a video ID.

    Works for both watch videos ("id":"ID") and reels (/reel/ID).
    The owner's numeric ID is the next distinct numeric ID after the video ID.
    """
    m = None
    for pattern in (
        rf'"id"\s*:\s*"{re.escape(video_id)}"',
        rf'\\\/reel\\\/{re.escape(video_id)}',
        rf'\\\/videos\\\/{re.escape(video_id)}',
    ):
        m = re.search(pattern, html)
        if m:
            break
    if not m:
        return ("unknown", "")
    window = html[m.start(): m.start() + 3000]

    # All numeric IDs in window — second one (after video ID itself) is owner
    all_ids = re.findall(r'"id"\s*:\s*"(\d{5,20})"', window)
    owner_id = next((i for i in all_ids if i != video_id), None)

    # Owner name: first "name":"..." in window
    name_m = re.search(r'"name"\s*:\s*"([^"]{2,80})"', window)
    owner_name = name_m.group(1) if name_m else ""

    return (owner_id or "unknown", owner_name)


# ---------------------------------------------------------------------------
# Crawler base class
# ---------------------------------------------------------------------------

class BasePlatformCrawler:
    """Abstract base for all platform crawlers."""

    platform_name: str = ""

    def search(self, query: str, max_results: int = 50) -> list[RawDetection]:
        """
        Search the platform for infringing content matching `query`.

        Args:
            query: normalized search string from normalize_query.py
            max_results: hard cap on returned results per query

        Returns:
            List of RawDetection; empty list if nothing found or on error.
        """
        raise NotImplementedError

    def _build_search_url(self, query: str) -> str:
        """Construct the platform-specific search URL for a query."""
        raise NotImplementedError

    def _parse_results(self, raw_html: str, query: str) -> list[RawDetection]:
        """Parse raw page HTML into RawDetection objects."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# FacebookCrawler
# ---------------------------------------------------------------------------

class FacebookCrawler(BasePlatformCrawler):
    """
    Crawler for Facebook public pages, groups, and Reels.

    Uses a Playwright headless browser (via BrowserFactory) to handle
    JS-rendered search results. Cookie-based session support for content
    that requires login.

    Args:
        headless: run browser without a visible window (default True)
        cookies_path: path to a Playwright-format cookies JSON file
        browser_factory: pre-configured factory; created lazily if None
        scroll_count: number of scroll events to trigger lazy loading
        scroll_pause_secs: seconds to wait between scrolls
        min_delay_secs: minimum seconds between consecutive searches
                        (rate limiting — be a polite crawler)
    """

    platform_name = "facebook"

    def __init__(
        self,
        headless: bool = True,
        cookies_path: str | None = None,
        browser_factory: BrowserFactory | None = None,
        scroll_count: int = 3,
        scroll_pause_secs: float = 1.0,
        min_delay_secs: float = 2.0,
    ) -> None:
        self.headless = headless
        self.cookies_path = cookies_path
        self._browser_factory = browser_factory
        self._scroll_count = scroll_count
        self._scroll_pause_secs = scroll_pause_secs
        self._min_delay_secs = min_delay_secs
        self._last_search_at: float = 0.0

    def search(self, query: str, max_results: int = 50) -> list[RawDetection]:
        """
        Search Facebook for infringing live streams or VOD uploads.

        Flow per query:
          1. Enforce rate limit (min_delay_secs between calls)
          2. Open browser page → navigate to search URL
          3. Scroll to trigger lazy-loaded results
          4. Capture full page HTML
          5. Parse with _parse_results
          6. Close page (always, even on error)

        Returns empty list on any browser or parsing error (logged to stderr).
        """
        self._enforce_rate_limit()

        page = self._get_factory().new_page()
        try:
            page.goto(self._build_search_url(query))
            for _ in range(self._scroll_count):
                page.scroll_down(2000)
                time.sleep(self._scroll_pause_secs)
            html = page.content()
            results = self._parse_results(html, query)
        except Exception as exc:
            print(
                f"[FacebookCrawler] error during search({query!r}): {exc}",
                file=sys.stderr,
            )
            results = []
        finally:
            page.close()
            self._last_search_at = time.monotonic()

        return results[:max_results]

    def _build_search_url(self, query: str) -> str:
        return _fb_search_url(query)

    def _parse_results(self, raw_html: str, query: str) -> list[RawDetection]:
        """
        Parse Facebook search result HTML into RawDetection objects.

        Extraction strategy (two-layer):
          1. Proximity-based: find title/channel from HTML near the video URL.
          2. JSON-based fallback: FB embeds structured data in <script> JSON.
             When proximity fails, extract from "title", "name", numeric ID
             fields in the JSON block keyed by video ID.
        """
        now = _utc_now()
        results: list[RawDetection] = []

        for video_url in _extract_video_urls_from_html(raw_html):
            video_id = _fb_video_id_from_url(video_url)

            # --- Title ---
            title = _extract_title_near_video(raw_html, video_url)
            if not title and video_id:
                title = _extract_fb_title_from_json(raw_html, video_id)

            # --- Channel ---
            page_url = _extract_page_url_near_video(raw_html, video_url)
            if page_url:
                channel_id, channel_name = self._channel_from_page_url(page_url)
            elif video_id:
                channel_id, channel_name = _extract_fb_channel_from_json(
                    raw_html, video_id
                )
            else:
                channel_id, channel_name = ("unknown", "")

            # --- View count and snapshot ---
            view_count = _extract_view_count_near_video(raw_html, video_url)
            idx = _find_url_in_html(raw_html, video_url)
            snapshot = (
                raw_html[max(0, idx - 200): idx + 1500] if idx != -1 else ""
            )

            extra: dict = {}
            if view_count is not None:
                extra["view_count"] = view_count

            results.append(RawDetection(
                platform=self.platform_name,
                url=video_url,
                title=title,
                channel_id=channel_id,
                channel_name=channel_name,
                snapshot_html=snapshot,
                detected_at=now,
                query_used=query,
                extra=extra,
            ))

        return results

    def _resolve_page_id(self, page_url: str) -> str:
        """
        Resolve a Facebook page URL to its numeric permanent ID.

        Phase 1: URL pattern extraction (no fetch).
        Phase 2: Fetch the page and scrape the ID from HTML.
        Fallback: username from URL path (impermanent — confidence lower).
        """
        from identity import _fb_id_from_url, _fb_id_from_html

        pid = _fb_id_from_url(page_url)
        if pid:
            return pid

        page = self._get_factory().new_page()
        try:
            page.goto(page_url)
            html = page.content()
            pid = _fb_id_from_html(html)
            if pid:
                return pid
        finally:
            page.close()

        parts = [p for p in urlparse(page_url).path.strip("/").split("/") if p]
        return parts[0] if parts else "unknown"

    def close(self) -> None:
        """Shut down the browser factory and release Playwright resources."""
        if self._browser_factory is not None:
            self._browser_factory.close()
            self._browser_factory = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_factory(self) -> BrowserFactory:
        """Lazily initialize PlaywrightBrowserFactory on first use."""
        if self._browser_factory is None:
            factory = PlaywrightBrowserFactory(
                headless=self.headless,
                cookies_path=self.cookies_path,
            )
            factory.start()
            self._browser_factory = factory
        return self._browser_factory

    def _channel_from_page_url(
        self, page_url: str | None
    ) -> tuple[str, str]:
        """
        Extract (channel_id, channel_name) from a page URL without fetching.

        channel_id is a numeric permanent ID if extractable, otherwise the
        username from the URL path (impermanent, flagged in extra by the caller).
        channel_name is left empty — identity.py fills it during resolution.
        """
        if not page_url:
            return ("unknown", "")

        from identity import _fb_id_from_url

        pid = _fb_id_from_url(page_url)
        if pid:
            return (pid, "")

        parts = [
            p for p in urlparse(page_url).path.strip("/").split("/") if p
        ]
        return (parts[0] if parts else "unknown", "")

    def _enforce_rate_limit(self) -> None:
        """Sleep if the last search was too recent."""
        elapsed = time.monotonic() - self._last_search_at
        if elapsed < self._min_delay_secs:
            time.sleep(self._min_delay_secs - elapsed)


# ---------------------------------------------------------------------------
# Crawler registry and orchestrator
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# BilibiliCrawler
# ---------------------------------------------------------------------------

class BilibiliCrawler(BasePlatformCrawler):
    """
    Crawler for Bilibili TV (bilibili.tv/id) — user-uploaded piracy videos.

    Bilibili TV is a heavy React SPA. Standard HTML scraping doesn't work
    because search results are loaded client-side via API after the page mounts.

    Strategy:
      1. Navigate to Bilibili TV homepage
      2. Type the query into the search box
      3. Press Enter → wait 15s for results to load
      4. Query DOM directly for video cards (JavaScript evaluation)
      5. Filter to /video/{id} URLs only — /play/{id} = official series, skip

    No cookies needed — Bilibili TV search is publicly accessible.
    """

    platform_name = "bilibili"
    _HOMEPAGE = "https://www.bilibili.tv/id"

    def __init__(
        self,
        browser_factory: BrowserFactory | None = None,
        results_wait_secs: float = 15.0,
        min_delay_secs: float = 3.0,
    ) -> None:
        self._browser_factory = browser_factory
        self._results_wait_secs = results_wait_secs
        self._min_delay_secs = min_delay_secs
        self._last_search_at: float = 0.0
        self._page: "PageDriver | None" = None   # persistent page reused across queries

    def search(self, query: str, max_results: int = 50) -> list[RawDetection]:
        """
        Search Bilibili TV for user-uploaded infringing content.

        Uses a persistent browser page across all queries so the homepage is
        loaded only ONCE per BilibiliCrawler instance. Each subsequent query
        reuses the same page (types in search box, waits, extracts results).
        This avoids 50 homepage navigations × 5-60s each.
        """
        self._enforce_rate_limit()
        try:
            page = self._get_or_create_page()
            query_js = query.replace("'", "\\'")
            page.evaluate(
                f"""
                (function() {{
                    var inputs = document.querySelectorAll('input[type="text"]');
                    for (var i = 0; i < inputs.length; i++) {{
                        inputs[i].focus();
                        inputs[i].value = '{query_js}';
                        inputs[i].dispatchEvent(new Event('input', {{bubbles: true}}));
                        inputs[i].dispatchEvent(new KeyboardEvent('keydown',
                            {{key: 'Enter', keyCode: 13, bubbles: true}}));
                        return true;
                    }}
                }})();
                """
            )
            page.wait_for_timeout(int(self._results_wait_secs * 1000))
            results = self._parse_results_from_dom(page, query)
        except Exception as exc:
            print(f"[BilibiliCrawler] error during search({query!r}): {exc}", file=sys.stderr)
            results = []
        finally:
            self._last_search_at = time.monotonic()

        return results[:max_results]

    def _get_or_create_page(self) -> "PageDriver":
        """Return the persistent page, creating and loading homepage if needed."""
        if self._page is None:
            self._page = self._get_factory().new_page()
            self._page.goto(self._HOMEPAGE)
            self._page.wait_for_timeout(3000)
            # Dismiss cookie banner if present
            self._page.evaluate(
                "document.querySelector('[class*=\"cookie\"] button') && "
                "document.querySelector('[class*=\"cookie\"] button').click()"
            )
        return self._page

    def _parse_results_from_dom(self, page: "PageDriver", query: str) -> list[RawDetection]:
        """Extract video cards from Bilibili TV search results via DOM query."""
        now = _utc_now()
        try:
            items = page.evaluate("""
                () => {
                    const seen = new Set();
                    const results = [];
                    const links = Array.from(document.querySelectorAll('a[href]'));
                    for (const a of links) {
                        const href = a.href || '';
                        // Only user-uploaded videos (/video/ID) — skip official series (/play/ID)
                        if (!href.includes('/video/') || !href.includes('bilibili.tv')) continue;
                        // Extract numeric video ID (12+ digits)
                        const m = href.match(/[/]video[/]([0-9]{10,})/);
                        if (!m) continue;
                        const videoId = m[1];
                        if (seen.has(videoId)) continue;
                        seen.add(videoId);

                        // Canonical URL
                        const url = 'https://www.bilibili.tv/video/' + videoId;

                        // Card container — use bstar-video-card class (Bilibili TV naming convention)
                        const card = a.closest('[class*="bstar-video-card"]') || a.parentElement || a;

                        // Title: img alt attribute is the most reliable source on Bilibili TV.
                        // The cover link wraps a <picture><img alt="Video Title"> — always present.
                        let title = '';
                        const coverImg = a.querySelector('img');
                        if (coverImg) title = coverImg.getAttribute('alt') || '';

                        // Fallback: bstar title element in info section
                        if (!title) {
                            const titleEl = card.querySelector(
                                '[class*="bstar-video-card__title"], [class*="bstar-video-card__name"]'
                            );
                            if (titleEl) title = titleEl.textContent.trim();
                        }
                        title = title.substring(0, 200);

                        // Channel: try every known Bilibili TV uploader link pattern.
                        // Also collect all <a> hrefs in card for debug when none match.
                        const chanSelectors = [
                            '[class*="bstar-video-card__up"]',
                            '[class*="up-name"]',
                            '[class*="uploader"]',
                            '[class*="author"]',
                            '[class*="owner"]',
                            '[class*="creator"]',
                            '[class*="user-name"]',
                            '[class*="username"]',
                            'a[href*="/space/"]',
                            'a[href*="/@"]',
                            'a[href*="/user/"]',
                        ];
                        let chanLink = null;
                        for (const sel of chanSelectors) {
                            chanLink = card.querySelector(sel);
                            if (chanLink) break;
                        }
                        const chanUrl = chanLink ? chanLink.href : '';
                        const chanIdM = chanUrl.match(/\/space\/([\w.-]+)/) ||
                                        chanUrl.match(/\/@([\w.-]+)/) ||
                                        chanUrl.match(/\/user\/([\w.-]+)/);
                        const chanId = chanIdM ? chanIdM[1] : ('bilibili_' + videoId);
                        const chanName = chanLink ? chanLink.textContent.trim() : '';

                        // Debug: for first card only, dump full outerHTML + all link hrefs
                        let debugCardHtml = '';
                        let debugAllLinks = '';
                        if (results.length === 0) {
                            debugCardHtml = card.outerHTML.substring(0, 1200).replace(/\s+/g, ' ');
                            debugAllLinks = Array.from(card.querySelectorAll('a[href]'))
                                .map(x => x.href + '|' + x.className + '|' + x.textContent.trim().substring(0,30))
                                .join(' ;; ');
                        }

                        // View count
                        const viewEl = card.querySelector(
                            '[class*="bstar-video-card__stat"], [class*="view"], [class*="play"], [class*="count"]'
                        );
                        const views = viewEl ? viewEl.textContent.trim() : '';

                        // Duration: shown as overlay on thumbnail, e.g. "1:23:45" or "23:45"
                        const durEl = card.querySelector(
                            '[class*="duration"], [class*="bstar-video-card__duration"], [class*="bstar-play-progress"]'
                        );
                        const duration = durEl ? durEl.textContent.trim() : '';

                        results.push({url, videoId, title, chanId, chanName, views, duration,
                                      debugCardHtml, debugAllLinks});
                    }
                    return results;
                }
            """)
        except Exception as exc:
            print(f"[BilibiliCrawler] DOM query failed: {exc}", file=sys.stderr)
            return []

        detections: list[RawDetection] = []
        no_title = 0
        no_chan = 0
        skipped_short = 0
        for item in (items or []):
            # Filter: skip videos shorter than 10 minutes.
            # Full-length piracy (movies/series) is always > 10 min; short clips are noise.
            # If duration is unknown (empty string), include the video — don't drop unknowns.
            dur_secs = _parse_duration_secs(item.get("duration", ""))
            if dur_secs is not None and dur_secs < 600:
                skipped_short += 1
                continue

            extra: dict = {}
            if dur_secs is not None:
                extra["duration_secs"] = dur_secs

            # Parse view count (e.g. "12.5K Putar", "1.2 rb")
            view_str = item.get("views", "")
            view_m = re.search(r'([\d,.]+)\s*(K|M|rb|jt)?', view_str)
            if view_m:
                try:
                    num = float(view_m.group(1).replace(",", "."))
                    suf = (view_m.group(2) or "").lower()
                    if suf == "k":  num *= 1_000
                    elif suf == "m": num *= 1_000_000
                    elif suf == "rb": num *= 1_000
                    elif suf == "jt": num *= 1_000_000
                    extra["view_count"] = int(num)
                except Exception:
                    pass

            title = item.get("title", "")
            chan_name = item.get("chanName", "")
            if not title:
                no_title += 1
            if not chan_name:
                no_chan += 1

            detections.append(RawDetection(
                platform=self.platform_name,
                url=item["url"],
                title=title,
                channel_id=item.get("chanId", "unknown"),
                channel_name=chan_name,
                snapshot_html="",
                detected_at=now,
                query_used=query,
                extra=extra,
            ))

        total = len(detections)
        raw_total = total + skipped_short
        print(
            f"[BilibiliCrawler] query={query!r} raw={raw_total} kept={total} "
            f"skipped_short={skipped_short} no_title={no_title} no_chan_name={no_chan}"
            + (f" sample_title={detections[0].title!r}" if detections else ""),
            file=sys.stderr,
        )
        # One-time debug per query: dump card HTML + all links to diagnose channel selector
        if items:
            first = items[0]
            if first.get("debugCardHtml"):
                print(f"[BilibiliCrawler][DEBUG] card_html={first['debugCardHtml']!r}",
                      file=sys.stderr)
            if first.get("debugAllLinks"):
                print(f"[BilibiliCrawler][DEBUG] card_links={first['debugAllLinks']!r}",
                      file=sys.stderr)
        return detections

    def _build_search_url(self, query: str) -> str:
        from urllib.parse import quote_plus
        return f"{self._HOMEPAGE}/search?keyword={quote_plus(query)}&type=VIDEO"

    def _parse_results(self, raw_html: str, query: str) -> list[RawDetection]:
        # Not used — Bilibili uses DOM-based extraction
        return []

    def close(self) -> None:
        if self._page is not None:
            try:
                self._page.close()
            except Exception:
                pass
            self._page = None
        if self._browser_factory is not None:
            self._browser_factory.close()
            self._browser_factory = None

    def _get_factory(self) -> BrowserFactory:
        if self._browser_factory is None:
            factory = PlaywrightBrowserFactory(headless=True)
            factory.start()
            self._browser_factory = factory
        return self._browser_factory

    def _enforce_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_search_at
        if elapsed < self._min_delay_secs:
            time.sleep(self._min_delay_secs - elapsed)


# Maps platform_name → crawler class. Add entries here to support new platforms.
_CRAWLER_REGISTRY: dict[str, type[BasePlatformCrawler]] = {
    "facebook": FacebookCrawler,
    "bilibili": BilibiliCrawler,
}


def run_all_crawlers(
    queries: list[str],
    platforms: list[str] | None = None,
    crawlers: list[BasePlatformCrawler] | None = None,
) -> Generator[RawDetection, None, None]:
    """
    Orchestrate crawlers across enabled platforms, yielding results as found.

    Args:
        queries: list of normalized search queries (from normalize_query.py)
        platforms: whitelist of platform slugs; None = all registered platforms
        crawlers: pre-configured crawler instances (for testing or custom setup);
                  if provided, `platforms` is ignored

    Yields:
        RawDetection items in (crawler × query) order.

    Error handling:
        Exceptions from a single (crawler, query) pair are caught, logged to
        stderr, and the loop continues. A broken query never aborts the run.
    """
    active = crawlers if crawlers is not None else _build_crawlers(platforms)

    for crawler in active:
        for query in queries:
            try:
                yield from crawler.search(query)
            except Exception as exc:
                print(
                    f"[run_all_crawlers] {crawler.platform_name} "
                    f"query={query!r}: {exc}",
                    file=sys.stderr,
                )


def _build_crawlers(
    platforms: list[str] | None,
) -> list[BasePlatformCrawler]:
    """
    Instantiate crawlers for the requested platforms using environment config.

    Reads from environment:
      ENABLED_PLATFORMS  — comma-separated list (default: "facebook")
      FB_HEADLESS        — "false" to show the browser window
      FB_COOKIE_FILE     — path to cookies JSON for Facebook login
    """
    import os

    env_platforms = os.environ.get("ENABLED_PLATFORMS", "facebook").split(",")
    enabled = platforms if platforms is not None else env_platforms

    crawlers: list[BasePlatformCrawler] = []
    for slug in enabled:
        slug = slug.strip()
        cls = _CRAWLER_REGISTRY.get(slug)
        if cls is None:
            print(f"[_build_crawlers] unknown platform {slug!r} - skipped", file=sys.stderr)
            continue
        if slug == "facebook":
            crawlers.append(FacebookCrawler(
                headless=os.environ.get("FB_HEADLESS", "true").lower() != "false",
                cookies_path=os.environ.get("FB_COOKIE_FILE"),
            ))
        elif slug == "bilibili":
            crawlers.append(BilibiliCrawler())
        else:
            crawlers.append(cls())   # type: ignore[abstract]
    return crawlers


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_duration_secs(duration_str: str) -> int | None:
    """
    Parse a duration string like "1:23:45" or "23:45" into total seconds.

    Returns None if the string is empty or unparseable — callers treat None
    as "unknown duration" and do not filter on it.
    """
    if not duration_str:
        return None
    parts = duration_str.strip().split(":")
    try:
        parts_int = [int(p) for p in parts]
    except ValueError:
        return None
    if len(parts_int) == 3:   # H:MM:SS
        return parts_int[0] * 3600 + parts_int[1] * 60 + parts_int[2]
    if len(parts_int) == 2:   # MM:SS
        return parts_int[0] * 60 + parts_int[1]
    return None
