"""Tests for detection.py — pure functions and crawler orchestration."""

import pytest
from detection import (
    RawDetection,
    BasePlatformCrawler,
    FacebookCrawler,
    _fb_search_url,
    _normalize_fb_video_url,
    _extract_video_urls_from_html,
    _extract_page_url_near_video,
    _extract_title_near_video,
    _extract_view_count_near_video,
    _find_url_in_html,
    run_all_crawlers,
)


# ---------------------------------------------------------------------------
# Fake browser infrastructure (no Playwright needed in tests)
# ---------------------------------------------------------------------------

class _FakePageDriver:
    """
    Fake PageDriver that returns static HTML without hitting a real browser.
    Records all method calls for assertion.
    """

    def __init__(self, html_map: dict[str, str]) -> None:
        self._map = html_map
        self.visited: list[str] = []
        self.scroll_calls: int = 0
        self.closed: bool = False
        self._current_url: str = ""

    def goto(self, url: str) -> None:
        self._current_url = url
        self.visited.append(url)

    def content(self) -> str:
        return self._map.get(self._current_url, "<html></html>")

    def scroll_down(self, pixels: int = 1000) -> None:
        self.scroll_calls += 1

    def close(self) -> None:
        self.closed = True


class _FakeBrowserFactory:
    def __init__(self, html_map: dict[str, str]) -> None:
        self._map = html_map
        self.pages_created: int = 0
        self.closed: bool = False

    def new_page(self) -> _FakePageDriver:
        self.pages_created += 1
        return _FakePageDriver(self._map)

    def close(self) -> None:
        self.closed = True


class _FakeCrawler(BasePlatformCrawler):
    """A fake crawler that returns pre-baked detections for testing run_all_crawlers."""

    platform_name = "fake"

    def __init__(self, detections: list[RawDetection], raises: bool = False) -> None:
        self._detections = detections
        self._raises = raises
        self.search_calls: list[str] = []

    def search(self, query: str, max_results: int = 50) -> list[RawDetection]:
        self.search_calls.append(query)
        if self._raises:
            raise RuntimeError("crawler exploded")
        return self._detections[:max_results]

    def _build_search_url(self, query: str) -> str:
        return f"https://fake.com/search?q={query}"

    def _parse_results(self, raw_html: str, query: str) -> list[RawDetection]:
        return []


def _make_detection(**kwargs) -> RawDetection:
    defaults = dict(
        platform="facebook", url="https://fb.com/watch?v=1",
        title="Test", channel_id="123", channel_name="TestCh",
        snapshot_html="<html/>", detected_at="2025-01-15T08:00:00Z",
        query_used="test", extra={},
    )
    defaults.update(kwargs)
    return RawDetection(**defaults)


# ---------------------------------------------------------------------------
# Minimal HTML fixture used across multiple tests
# ---------------------------------------------------------------------------

_FB_SEARCH_HTML = """
<html><body>
<div>
  <a href="/watch?v=111222333&amp;_rdc=1">
    <span aria-label="Liga Champions Vidio 2025">Liga Champions Vidio 2025</span>
  </a>
  <a href="/profile.php?id=987654321">PirateChannel</a>
  <span>2K views</span>
</div>
<div>
  <a href="/reel/444555666/">Stream Sinetron</a>
  <a href="/pages/Entertainment/StreamPirate/123456789/">StreamPirate</a>
  <span title="Stream Sinetron Bajakan Full">5.5K views</span>
</div>
<div>
  <a href="/groups/111222333/videos/999888777/">Liga Live Group</a>
  <a href="/groups/111222333/">Pirate Group</a>
  <span aria-label="Liga Live Group Stream">Liga Live Group Stream</span>
  <span>100 penonton</span>
</div>
</body></html>
"""


# ---------------------------------------------------------------------------
# _fb_search_url
# ---------------------------------------------------------------------------

class TestFbSearchUrl:
    def test_basic(self):
        url = _fb_search_url("liga champions")
        assert url == "https://www.facebook.com/search/videos/?q=liga+champions"

    def test_special_chars_encoded(self):
        url = _fb_search_url("v1d10 & stream")
        assert "v1d10" in url
        assert "&" not in url.split("?q=")[1].replace("&amp;", "")

    def test_empty_query(self):
        url = _fb_search_url("")
        assert url.startswith("https://www.facebook.com/search/videos/")

    def test_unicode_encoded(self):
        url = _fb_search_url("sinetron nusantara")
        assert "sinetron" in url


# ---------------------------------------------------------------------------
# _normalize_fb_video_url
# ---------------------------------------------------------------------------

class TestNormalizeFbVideoUrl:
    @pytest.mark.parametrize("raw,expected", [
        # /watch?v= forms
        ("https://www.facebook.com/watch?v=111222333",
         "https://www.facebook.com/watch?v=111222333"),
        # with extra params
        ("https://www.facebook.com/watch?v=111222333&_rdc=1&fbclid=xxx",
         "https://www.facebook.com/watch?v=111222333"),
        # relative /watch
        ("/watch?v=111222333",
         "https://www.facebook.com/watch?v=111222333"),
        # /reel/<NUM>
        ("https://www.facebook.com/reel/444555666/",
         "https://www.facebook.com/reel/444555666"),
        # /videos/<NUM>
        ("https://www.facebook.com/videos/777888999/",
         "https://www.facebook.com/videos/777888999"),
        # /<page>/videos/<NUM>
        ("https://www.facebook.com/PiratePage/videos/123456789/",
         "https://www.facebook.com/PiratePage/videos/123456789"),
    ])
    def test_normalizes(self, raw, expected):
        assert _normalize_fb_video_url(raw) == expected

    @pytest.mark.parametrize("url", [
        "https://www.facebook.com/PiratePage",           # profile, not video
        "https://www.facebook.com/",                     # homepage
        "https://www.youtube.com/watch?v=abc",           # wrong domain
        "https://www.facebook.com/watch?v=notanumber",   # non-numeric video ID
        "",                                              # empty
    ])
    def test_returns_none(self, url):
        assert _normalize_fb_video_url(url) is None

    def test_dedup_equivalent_urls(self):
        a = _normalize_fb_video_url("https://www.facebook.com/watch?v=111222333&fbclid=x")
        b = _normalize_fb_video_url("https://www.facebook.com/watch?v=111222333")
        assert a == b


# ---------------------------------------------------------------------------
# _extract_video_urls_from_html
# ---------------------------------------------------------------------------

class TestExtractVideoUrls:
    def test_extracts_watch_url(self):
        urls = _extract_video_urls_from_html(_FB_SEARCH_HTML)
        assert "https://www.facebook.com/watch?v=111222333" in urls

    def test_extracts_reel_url(self):
        urls = _extract_video_urls_from_html(_FB_SEARCH_HTML)
        assert "https://www.facebook.com/reel/444555666" in urls

    def test_extracts_group_video_url(self):
        urls = _extract_video_urls_from_html(_FB_SEARCH_HTML)
        # /groups/<GID>/videos/<VID>
        assert any("999888777" in u for u in urls)

    def test_no_duplicate_urls(self):
        html = """
        <a href="/watch?v=111">video</a>
        <a href="/watch?v=111&_rdc=1">same video</a>
        """
        urls = _extract_video_urls_from_html(html)
        assert len(urls) == 1

    def test_empty_html_returns_empty(self):
        assert _extract_video_urls_from_html("") == []

    def test_no_video_urls_returns_empty(self):
        html = '<a href="/profile.php?id=123">Profile</a>'
        assert _extract_video_urls_from_html(html) == []

    def test_html_entity_amp_decoded(self):
        # &amp; in href should be treated as &
        html = '<a href="/watch?v=999888777&amp;_rdc=1">video</a>'
        urls = _extract_video_urls_from_html(html)
        assert "https://www.facebook.com/watch?v=999888777" in urls

    def test_preserves_order(self):
        html = """
        <a href="/reel/111/">first</a>
        <a href="/reel/222/">second</a>
        <a href="/reel/333/">third</a>
        """
        urls = _extract_video_urls_from_html(html)
        ids = [u.split("/")[-1] for u in urls]
        assert ids == ["111", "222", "333"]


# ---------------------------------------------------------------------------
# _extract_page_url_near_video
# ---------------------------------------------------------------------------

class TestExtractPageUrlNearVideo:
    def test_finds_numeric_profile(self):
        result = _extract_page_url_near_video(
            _FB_SEARCH_HTML, "https://www.facebook.com/watch?v=111222333"
        )
        assert result is not None
        assert "987654321" in result

    def test_finds_pages_url(self):
        result = _extract_page_url_near_video(
            _FB_SEARCH_HTML, "https://www.facebook.com/reel/444555666"
        )
        assert result is not None
        assert "123456789" in result or "StreamPirate" in result

    def test_finds_group_url(self):
        result = _extract_page_url_near_video(
            _FB_SEARCH_HTML, "https://www.facebook.com/groups/111222333/videos/999888777"
        )
        assert result is not None
        assert "111222333" in result

    def test_video_not_in_html_returns_none(self):
        result = _extract_page_url_near_video(
            _FB_SEARCH_HTML, "https://www.facebook.com/watch?v=nonexistent"
        )
        assert result is None

    def test_result_is_absolute_url(self):
        result = _extract_page_url_near_video(
            _FB_SEARCH_HTML, "https://www.facebook.com/watch?v=111222333"
        )
        assert result is None or result.startswith("https://")


# ---------------------------------------------------------------------------
# _extract_title_near_video
# ---------------------------------------------------------------------------

class TestExtractTitleNearVideo:
    def test_extracts_aria_label(self):
        title = _extract_title_near_video(
            _FB_SEARCH_HTML, "https://www.facebook.com/watch?v=111222333"
        )
        assert "Liga Champions" in title

    def test_extracts_title_attribute(self):
        title = _extract_title_near_video(
            _FB_SEARCH_HTML, "https://www.facebook.com/reel/444555666"
        )
        assert "Stream Sinetron" in title

    def test_extracts_aria_label_for_group_video(self):
        title = _extract_title_near_video(
            _FB_SEARCH_HTML,
            "https://www.facebook.com/groups/111222333/videos/999888777",
        )
        assert "Liga" in title or title != ""

    def test_video_not_in_html_returns_empty(self):
        title = _extract_title_near_video(
            _FB_SEARCH_HTML, "https://www.facebook.com/watch?v=99999"
        )
        assert title == ""

    def test_custom_html_aria_label(self):
        html = '<a href="/watch?v=123"><span aria-label="Custom Video Title">.</span></a>'
        title = _extract_title_near_video(html, "https://www.facebook.com/watch?v=123")
        assert title == "Custom Video Title"


# ---------------------------------------------------------------------------
# _extract_view_count_near_video
# ---------------------------------------------------------------------------

class TestExtractViewCount:
    def test_k_suffix(self):
        count = _extract_view_count_near_video(
            _FB_SEARCH_HTML, "https://www.facebook.com/watch?v=111222333"
        )
        assert count == 2000

    def test_decimal_k_suffix(self):
        count = _extract_view_count_near_video(
            _FB_SEARCH_HTML, "https://www.facebook.com/reel/444555666"
        )
        assert count == 5500

    def test_penonton_indonesian(self):
        count = _extract_view_count_near_video(
            _FB_SEARCH_HTML,
            "https://www.facebook.com/groups/111222333/videos/999888777",
        )
        assert count == 100

    def test_no_view_count_returns_none(self):
        html = '<a href="/watch?v=555">video</a><span>No engagement data</span>'
        count = _extract_view_count_near_video(html, "https://www.facebook.com/watch?v=555")
        assert count is None

    def test_million_suffix(self):
        html = '<a href="/watch?v=777">video</a><span>1.2M views</span>'
        count = _extract_view_count_near_video(html, "https://www.facebook.com/watch?v=777")
        assert count == 1_200_000

    def test_plain_number(self):
        html = '<a href="/watch?v=888">video</a><span>500 views</span>'
        count = _extract_view_count_near_video(html, "https://www.facebook.com/watch?v=888")
        assert count == 500


# ---------------------------------------------------------------------------
# _find_url_in_html
# ---------------------------------------------------------------------------

class TestFindUrlInHtml:
    def test_finds_full_url(self):
        html = 'some text https://www.facebook.com/watch?v=123 more text'
        assert _find_url_in_html(html, "https://www.facebook.com/watch?v=123") != -1

    def test_finds_path_form(self):
        html = 'href="/watch?v=456" text'
        assert _find_url_in_html(html, "https://www.facebook.com/watch?v=456") != -1

    def test_not_found_returns_minus_one(self):
        assert _find_url_in_html("<html></html>", "https://fb.com/watch?v=999") == -1


# ---------------------------------------------------------------------------
# FacebookCrawler._build_search_url
# ---------------------------------------------------------------------------

class TestFacebookCrawlerBuildUrl:
    def test_delegates_to_pure_function(self):
        crawler = FacebookCrawler(browser_factory=_FakeBrowserFactory({}))
        url = crawler._build_search_url("liga champions vidio")
        assert url == _fb_search_url("liga champions vidio")


# ---------------------------------------------------------------------------
# FacebookCrawler._channel_from_page_url
# ---------------------------------------------------------------------------

class TestChannelFromPageUrl:
    def setup_method(self):
        self.crawler = FacebookCrawler(browser_factory=_FakeBrowserFactory({}))

    def test_numeric_id_from_profile_php(self):
        cid, name = self.crawler._channel_from_page_url(
            "https://www.facebook.com/profile.php?id=123456789"
        )
        assert cid == "123456789"
        assert name == ""

    def test_numeric_id_from_pages_url(self):
        cid, _ = self.crawler._channel_from_page_url(
            "https://www.facebook.com/pages/Cat/Name/987654321/"
        )
        assert cid == "987654321"

    def test_username_fallback(self):
        cid, _ = self.crawler._channel_from_page_url(
            "https://www.facebook.com/PiratePage"
        )
        assert cid == "PiratePage"

    def test_none_returns_unknown(self):
        cid, name = self.crawler._channel_from_page_url(None)
        assert cid == "unknown"
        assert name == ""

    def test_empty_path_returns_unknown(self):
        cid, _ = self.crawler._channel_from_page_url("https://www.facebook.com/")
        assert cid == "unknown"


# ---------------------------------------------------------------------------
# FacebookCrawler._parse_results
# ---------------------------------------------------------------------------

class TestFacebookCrawlerParseResults:
    def setup_method(self):
        self.crawler = FacebookCrawler(browser_factory=_FakeBrowserFactory({}))

    def test_returns_list(self):
        results = self.crawler._parse_results(_FB_SEARCH_HTML, "test query")
        assert isinstance(results, list)

    def test_finds_multiple_detections(self):
        results = self.crawler._parse_results(_FB_SEARCH_HTML, "test query")
        assert len(results) >= 2

    def test_platform_is_facebook(self):
        results = self.crawler._parse_results(_FB_SEARCH_HTML, "test query")
        assert all(r.platform == "facebook" for r in results)

    def test_query_used_set(self):
        results = self.crawler._parse_results(_FB_SEARCH_HTML, "liga champions")
        assert all(r.query_used == "liga champions" for r in results)

    def test_detected_at_is_iso8601(self):
        results = self.crawler._parse_results(_FB_SEARCH_HTML, "test")
        for r in results:
            assert "T" in r.detected_at
            assert r.detected_at.endswith("Z")

    def test_first_result_url_is_watch(self):
        results = self.crawler._parse_results(_FB_SEARCH_HTML, "test")
        assert results[0].url == "https://www.facebook.com/watch?v=111222333"

    def test_channel_id_from_numeric_profile(self):
        results = self.crawler._parse_results(_FB_SEARCH_HTML, "test")
        first = results[0]
        assert first.channel_id == "987654321"

    def test_view_count_in_extra(self):
        results = self.crawler._parse_results(_FB_SEARCH_HTML, "test")
        first = results[0]
        assert first.extra.get("view_count") == 2000

    def test_snapshot_html_nonempty(self):
        results = self.crawler._parse_results(_FB_SEARCH_HTML, "test")
        assert all(len(r.snapshot_html) > 0 for r in results)

    def test_empty_html_returns_empty_list(self):
        results = self.crawler._parse_results("<html></html>", "test")
        assert results == []

    def test_title_extracted(self):
        results = self.crawler._parse_results(_FB_SEARCH_HTML, "test")
        titles = [r.title for r in results]
        assert any("Liga Champions" in t or "Stream" in t for t in titles)


# ---------------------------------------------------------------------------
# FacebookCrawler.search (with fake browser)
# ---------------------------------------------------------------------------

class TestFacebookCrawlerSearch:
    def test_returns_detections(self):
        factory = _FakeBrowserFactory({
            _fb_search_url("liga champions"): _FB_SEARCH_HTML
        })
        crawler = FacebookCrawler(
            browser_factory=factory,
            scroll_count=0,     # no scrolling in tests
            scroll_pause_secs=0,
            min_delay_secs=0,
        )
        results = crawler.search("liga champions")
        assert len(results) >= 1

    def test_max_results_respected(self):
        factory = _FakeBrowserFactory({
            _fb_search_url("liga"): _FB_SEARCH_HTML
        })
        crawler = FacebookCrawler(
            browser_factory=factory, scroll_count=0,
            scroll_pause_secs=0, min_delay_secs=0,
        )
        results = crawler.search("liga", max_results=1)
        assert len(results) <= 1

    def test_page_always_closed(self):
        factory = _FakeBrowserFactory({
            _fb_search_url("test"): _FB_SEARCH_HTML
        })
        pages: list[_FakePageDriver] = []
        original_new_page = factory.new_page

        def tracking_new_page():
            page = original_new_page()
            pages.append(page)
            return page

        factory.new_page = tracking_new_page
        crawler = FacebookCrawler(
            browser_factory=factory, scroll_count=0,
            scroll_pause_secs=0, min_delay_secs=0,
        )
        crawler.search("test")
        assert all(p.closed for p in pages)

    def test_page_closed_on_error(self):
        """Browser errors must not leak open pages."""
        class _ErrorFactory:
            def __init__(self):
                self.page = None

            def new_page(self):
                p = _FakePageDriver({})
                self.page = p
                return p

            def close(self):
                pass

        class _ErrorPage(_FakePageDriver):
            def goto(self, url):
                raise RuntimeError("network failure")

        factory = _ErrorFactory()
        orig = factory.new_page

        def make_error_page():
            p = _ErrorPage({})
            factory.page = p
            return p

        factory.new_page = make_error_page
        crawler = FacebookCrawler(
            browser_factory=factory, scroll_count=0,
            scroll_pause_secs=0, min_delay_secs=0,
        )
        results = crawler.search("test")
        assert results == []
        assert factory.page.closed

    def test_empty_html_returns_empty_list(self):
        factory = _FakeBrowserFactory({
            _fb_search_url("nothing"): "<html></html>"
        })
        crawler = FacebookCrawler(
            browser_factory=factory, scroll_count=0,
            scroll_pause_secs=0, min_delay_secs=0,
        )
        assert crawler.search("nothing") == []

    def test_scrolls_configured_times(self):
        factory = _FakeBrowserFactory({
            _fb_search_url("test"): "<html></html>"
        })
        pages: list[_FakePageDriver] = []
        orig = factory.new_page

        def tracking():
            p = orig()
            pages.append(p)
            return p

        factory.new_page = tracking
        crawler = FacebookCrawler(
            browser_factory=factory, scroll_count=2,
            scroll_pause_secs=0, min_delay_secs=0,
        )
        crawler.search("test")
        assert pages[0].scroll_calls == 2


# ---------------------------------------------------------------------------
# FacebookCrawler.close
# ---------------------------------------------------------------------------

class TestFacebookCrawlerClose:
    def test_close_shuts_down_factory(self):
        factory = _FakeBrowserFactory({})
        crawler = FacebookCrawler(browser_factory=factory)
        crawler.close()
        assert factory.closed

    def test_close_without_factory_no_error(self):
        crawler = FacebookCrawler(browser_factory=None)
        crawler.close()   # should not raise


# ---------------------------------------------------------------------------
# run_all_crawlers
# ---------------------------------------------------------------------------

class TestRunAllCrawlers:
    def _det(self, url: str) -> RawDetection:
        return _make_detection(url=url)

    def test_yields_detections_from_crawler(self):
        d = self._det("https://fb.com/watch?v=1")
        crawler = _FakeCrawler([d])
        results = list(run_all_crawlers(["q1"], crawlers=[crawler]))
        assert len(results) == 1
        assert results[0] is d

    def test_yields_for_each_query(self):
        d = self._det("https://fb.com/watch?v=1")
        crawler = _FakeCrawler([d])
        results = list(run_all_crawlers(["q1", "q2", "q3"], crawlers=[crawler]))
        assert len(results) == 3
        assert crawler.search_calls == ["q1", "q2", "q3"]

    def test_multiple_crawlers(self):
        c1 = _FakeCrawler([self._det("https://fb.com/watch?v=1")])
        c2 = _FakeCrawler([self._det("https://fb.com/watch?v=2")])
        results = list(run_all_crawlers(["q1"], crawlers=[c1, c2]))
        assert len(results) == 2

    def test_exception_in_one_query_continues(self):
        good = _FakeCrawler([self._det("https://fb.com/watch?v=1")])
        broken = _FakeCrawler([], raises=True)
        results = list(run_all_crawlers(["q1"], crawlers=[good, broken]))
        assert len(results) == 1   # broken crawler didn't kill the run

    def test_empty_queries_yields_nothing(self):
        crawler = _FakeCrawler([self._det("https://fb.com/watch?v=1")])
        results = list(run_all_crawlers([], crawlers=[crawler]))
        assert results == []
        assert crawler.search_calls == []

    def test_empty_crawler_list_yields_nothing(self):
        results = list(run_all_crawlers(["q1"], crawlers=[]))
        assert results == []

    def test_results_are_raw_detections(self):
        d = self._det("https://fb.com/watch?v=1")
        crawler = _FakeCrawler([d])
        results = list(run_all_crawlers(["q"], crawlers=[crawler]))
        assert all(isinstance(r, RawDetection) for r in results)
