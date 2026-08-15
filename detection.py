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

    def goto(self, url: str) -> None:
        # "domcontentloaded" instead of "networkidle" — Facebook never reaches
        # networkidle because it continuously polls; domcontentloaded fires as
        # soon as the HTML is parsed, which is enough to get search results.
        self._page.goto(url, wait_until="domcontentloaded", timeout=60_000)

    def content(self) -> str:
        return self._page.content()

    def scroll_down(self, pixels: int = 1000) -> None:
        # 5s timeout — evaluate() can hang on broken pages without a timeout
        try:
            self._page.evaluate(f"window.scrollBy(0, {pixels})",
                                timeout=5000)
        except Exception:
            pass   # scroll failure is non-critical, continue with current HTML

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

    # Strategy A: human-readable text (e.g. "2,5 jt views", "500 penonton")
    m = re.search(
        r'([\d,]+(?:\.\d+)?)\s*([KkMm])?\s*'
        r'(?:views?|penonton|ditonton)',
        window,
    )
    if m:
        try:
            num = float(m.group(1).replace(",", ""))
            suffix = m.group(2) or ""
            if suffix in ("K", "k"):
                num *= 1_000
            elif suffix in ("M", "m"):
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

    # Phase 3: for FB watch URLs — find video ID in JSON data block.
    # FB embeds {"video":{"id":"VIDEO_ID"},...,"title":"..."} in the page.
    # This position is the best anchor for JSON-based title/channel extraction.
    if parsed.path.rstrip("/") == "/watch" and parsed.query:
        vid = parse_qs(parsed.query).get("v", [""])[0]
        if vid and vid.isdigit():
            m = re.search(rf'"id"\s*:\s*"{re.escape(vid)}"', html)
            if m:
                return m.start()

    return -1


def _fb_video_id_from_url(url: str) -> str | None:
    """Extract the numeric video ID from a normalized FB watch URL."""
    parsed = urlparse(url)
    if parsed.path.rstrip("/") != "/watch":
        return None
    vid = parse_qs(parsed.query).get("v", [None])[0]
    return vid if vid and vid.isdigit() else None


def _extract_fb_title_from_json(html: str, video_id: str) -> str:
    """
    Extract video title from FB's embedded JSON data by video ID.

    FB embeds search result data as JSON in <script> tags. The title appears
    as "title":"..." or "save_description":"..." near the video id field.
    """
    m = re.search(rf'"id"\s*:\s*"{re.escape(video_id)}"', html)
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

    The owner's numeric ID is the second distinct numeric ID that appears in
    the JSON block after the video ID. The owner name is in a "name" field.
    """
    m = re.search(rf'"id"\s*:\s*"{re.escape(video_id)}"', html)
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
        scroll_count: int = 1,
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

# Maps platform_name → crawler class. Add entries here to support new platforms.
_CRAWLER_REGISTRY: dict[str, type[BasePlatformCrawler]] = {
    "facebook": FacebookCrawler,
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
        else:
            crawlers.append(cls())   # type: ignore[abstract]
    return crawlers


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
