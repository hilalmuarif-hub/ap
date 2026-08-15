"""Tests for identity.py."""

import pytest
from identity import (
    OffenderIdentity,
    IdentityResolver,
    _fb_id_from_url,
    _fb_id_from_html,
    _fb_display_name_from_html,
    _fb_canonical_url,
    _yt_channel_id_from_url,
    _yt_channel_id_from_html,
    _yt_display_name_from_html,
    _tg_username_from_url,
    _tt_username_from_url,
    _tt_uid_from_html,
    _tt_display_name_from_html,
    is_same_identity,
)


# ---------------------------------------------------------------------------
# Fake fetcher for injecting HTML without real HTTP calls
# ---------------------------------------------------------------------------

class _FakeFetcher:
    def __init__(self, html_map: dict[str, str]) -> None:
        self._map = html_map
        self.calls: list[str] = []

    def fetch(self, url: str) -> str:
        self.calls.append(url)
        if url not in self._map:
            raise ValueError(f"No fake HTML registered for {url!r}")
        return self._map[url]


# ---------------------------------------------------------------------------
# Facebook — URL parsing
# ---------------------------------------------------------------------------

class TestFbIdFromUrl:
    @pytest.mark.parametrize("url,expected", [
        # profile.php pattern
        ("https://www.facebook.com/profile.php?id=123456789", "123456789"),
        ("https://m.facebook.com/profile.php?id=987654321",  "987654321"),
        # pages with numeric ID at end
        ("https://www.facebook.com/pages/Category/PageName/111222333", "111222333"),
        ("https://www.facebook.com/pages/SomePage/444555666/",         "444555666"),
        # groups with numeric ID
        ("https://www.facebook.com/groups/123456789",         "123456789"),
        ("https://www.facebook.com/groups/123456789/",        "123456789"),
        ("https://www.facebook.com/groups/123456789/about",   "123456789"),
    ])
    def test_extracts_id(self, url, expected):
        assert _fb_id_from_url(url) == expected

    @pytest.mark.parametrize("url", [
        # Content URLs — must not return a video/post ID as channel ID
        "https://www.facebook.com/watch?v=999888777",
        "https://www.facebook.com/reel/555444333",
        "https://www.facebook.com/video/123456789",
        "https://www.facebook.com/stories/user/123",
        # Username-only profile (no numeric ID in URL)
        "https://www.facebook.com/SomePiratePage",
        "https://www.facebook.com/pirate.channel.id",
        # Empty / homepage
        "https://www.facebook.com/",
        "https://www.facebook.com",
        # pages/ without numeric ID
        "https://www.facebook.com/pages/Category/PageName",
    ])
    def test_returns_none(self, url):
        assert _fb_id_from_url(url) is None

    def test_profile_php_non_numeric_id_ignored(self):
        assert _fb_id_from_url("https://www.facebook.com/profile.php?id=abc") is None

    def test_short_numeric_not_treated_as_page_id(self):
        # Short numeric paths (< 6 digits) should not be mistaken for page IDs
        result = _fb_id_from_url("https://www.facebook.com/pages/Cat/Name/123")
        assert result is None or (result is not None and len(result) >= 6)


# ---------------------------------------------------------------------------
# Facebook — HTML extraction
# ---------------------------------------------------------------------------

class TestFbIdFromHtml:
    def test_android_meta_page(self):
        html = '<meta property="al:android:url" content="fb://page/123456789" />'
        assert _fb_id_from_html(html) == "123456789"

    def test_android_meta_profile(self):
        html = 'fb://profile/987654321 some text'
        assert _fb_id_from_html(html) == "987654321"

    def test_android_meta_group(self):
        html = 'href="fb://group/555444333"'
        assert _fb_id_from_html(html) == "555444333"

    def test_page_id_json(self):
        html = '{"pageID":"111222333","name":"Pirate Page"}'
        assert _fb_id_from_html(html) == "111222333"

    def test_entity_id_json(self):
        html = '{"entity_id":"444555666"}'
        assert _fb_id_from_html(html) == "444555666"

    def test_android_meta_preferred_over_json(self):
        # android meta should match first
        html = '{"pageID":"AAAAA"} fb://page/999888777'
        assert _fb_id_from_html(html) == "999888777"

    def test_no_id_returns_none(self):
        html = "<html><body>No ID here</body></html>"
        assert _fb_id_from_html(html) is None


class TestFbDisplayNameFromHtml:
    def test_og_title(self):
        html = '<meta property="og:title" content="Pirate Page Name" />'
        assert _fb_display_name_from_html(html) == "Pirate Page Name"

    def test_title_fallback(self):
        html = "<title>Pirate Page | Facebook</title>"
        name = _fb_display_name_from_html(html)
        assert "Pirate Page" in name

    def test_empty_html(self):
        assert _fb_display_name_from_html("") == ""


class TestFbCanonicalUrl:
    def test_format(self):
        url = _fb_canonical_url("123456789")
        assert url == "https://www.facebook.com/profile.php?id=123456789"


# ---------------------------------------------------------------------------
# YouTube — URL parsing
# ---------------------------------------------------------------------------

class TestYtChannelIdFromUrl:
    @pytest.mark.parametrize("url,expected", [
        ("https://www.youtube.com/channel/UCddiUEpeqJcYeBxX1IVBKvQ",
         "UCddiUEpeqJcYeBxX1IVBKvQ"),
        ("https://youtube.com/channel/UC_x5XG1OV2P6uZZ5FSM9Ttw",
         "UC_x5XG1OV2P6uZZ5FSM9Ttw"),
        ("https://www.youtube.com/channel/UC-lHJZR3Gqxm24_Vd_AJ37g",
         "UC-lHJZR3Gqxm24_Vd_AJ37g"),
    ])
    def test_extracts_channel_id(self, url, expected):
        assert _yt_channel_id_from_url(url) == expected

    @pytest.mark.parametrize("url", [
        "https://www.youtube.com/@SomeHandle",        # requires API
        "https://www.youtube.com/c/CustomName",        # requires API
        "https://www.youtube.com/user/OldUsername",    # requires API
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ", # video URL
        "https://www.youtube.com/",
    ])
    def test_returns_none(self, url):
        assert _yt_channel_id_from_url(url) is None

    def test_wrong_length_returns_none(self):
        # UC + 21 chars (not 22) → no match
        assert _yt_channel_id_from_url("https://www.youtube.com/channel/UCshort") is None


# ---------------------------------------------------------------------------
# YouTube — HTML extraction
# ---------------------------------------------------------------------------

class TestYtChannelIdFromHtml:
    def test_channel_id_json(self):
        html = '"channelId":"UCddiUEpeqJcYeBxX1IVBKvQ"'
        assert _yt_channel_id_from_html(html) == "UCddiUEpeqJcYeBxX1IVBKvQ"

    def test_canonical_link(self):
        html = '<link rel="canonical" href="https://www.youtube.com/channel/UC_x5XG1OV2P6uZZ5FSM9Ttw">'
        assert _yt_channel_id_from_html(html) == "UC_x5XG1OV2P6uZZ5FSM9Ttw"

    def test_json_preferred_over_canonical(self):
        html = (
            '"channelId":"UCddiUEpeqJcYeBxX1IVBKvQ"'
            ' <link rel="canonical" href="/channel/UC_x5XG1OV2P6uZZ5FSM9Ttw">'
        )
        assert _yt_channel_id_from_html(html) == "UCddiUEpeqJcYeBxX1IVBKvQ"

    def test_no_id_returns_none(self):
        assert _yt_channel_id_from_html("<html>no id</html>") is None


class TestYtDisplayNameFromHtml:
    def test_title_tag(self):
        html = "<title>Pirate Channel - YouTube</title>"
        assert _yt_display_name_from_html(html) == "Pirate Channel"

    def test_title_with_em_dash(self):
        html = "<title>Pirate Channel – YouTube</title>"
        assert _yt_display_name_from_html(html) == "Pirate Channel"

    def test_channel_name_json(self):
        html = '"channelName":"StreamPirate"'
        assert _yt_display_name_from_html(html) == "StreamPirate"

    def test_empty_html(self):
        assert _yt_display_name_from_html("") == ""


# ---------------------------------------------------------------------------
# Telegram — URL parsing
# ---------------------------------------------------------------------------

class TestTgUsernameFromUrl:
    @pytest.mark.parametrize("url,expected", [
        ("https://t.me/piratechannel",        "piratechannel"),
        ("https://t.me/pirate_channel",       "pirate_channel"),
        ("https://telegram.me/streampirate",  "streampirate"),
        ("https://t.me/UPPERCASE_NAME",       "UPPERCASE_NAME"),
    ])
    def test_extracts_username(self, url, expected):
        assert _tg_username_from_url(url) == expected

    @pytest.mark.parametrize("url", [
        "https://t.me/+AbCdEfGhIjKlMnOp",    # invite link
        "https://t.me/joinchat/AbC",           # old invite format
        "https://t.me/pirate/123",             # message link (has slash)
        "https://t.me/ab",                     # too short (< 5 chars)
        "https://t.me/",                       # empty path
        "https://www.facebook.com/page",       # wrong domain
        "https://t.me/user@name",              # invalid char
    ])
    def test_returns_none(self, url):
        assert _tg_username_from_url(url) is None


# ---------------------------------------------------------------------------
# TikTok — URL parsing and HTML extraction
# ---------------------------------------------------------------------------

class TestTtUsernameFromUrl:
    @pytest.mark.parametrize("url,expected", [
        ("https://www.tiktok.com/@pirate123",        "pirate123"),
        ("https://www.tiktok.com/@pirate.chan",       "pirate.chan"),
        ("https://www.tiktok.com/@pirate_stream",     "pirate_stream"),
    ])
    def test_extracts_username(self, url, expected):
        assert _tt_username_from_url(url) == expected

    @pytest.mark.parametrize("url", [
        "https://www.tiktok.com/",                         # no username
        "https://www.tiktok.com/@pirate/video/123",        # video link
        "https://www.tiktok.com/tag/piracy",               # hashtag
        "https://www.facebook.com/@pirate",                # wrong domain
        "https://www.tiktok.com/@a",                       # too short (< 2 chars)
    ])
    def test_returns_none(self, url):
        assert _tt_username_from_url(url) is None


class TestTtUidFromHtml:
    def _make_next_data(self, uid: str, username: str = "pirate") -> str:
        import json
        data = {
            "props": {
                "pageProps": {
                    "userInfo": {
                        "user": {"id": uid, "uniqueId": username, "nickname": "Pirate"}
                    }
                }
            }
        }
        return f'<script id="__NEXT_DATA__" type="application/json">{json.dumps(data)}</script>'

    def test_extracts_uid(self):
        html = self._make_next_data("9876543210123456789")
        assert _tt_uid_from_html(html) == "9876543210123456789"

    def test_no_next_data_returns_none(self):
        assert _tt_uid_from_html("<html>no script</html>") is None

    def test_malformed_json_returns_none(self):
        html = '<script id="__NEXT_DATA__">{broken json</script>'
        assert _tt_uid_from_html(html) is None

    def test_missing_uid_returns_none(self):
        import json
        data = {"props": {"pageProps": {"userInfo": {"user": {"nickname": "pirate"}}}}}
        html = f'<script id="__NEXT_DATA__">{json.dumps(data)}</script>'
        assert _tt_uid_from_html(html) is None


class TestTtDisplayNameFromHtml:
    def _make_next_data(self, nickname: str) -> str:
        import json
        data = {
            "props": {
                "pageProps": {
                    "userInfo": {
                        "user": {"id": "123", "uniqueId": "pirate", "nickname": nickname}
                    }
                }
            }
        }
        return f'<script id="__NEXT_DATA__">{json.dumps(data)}</script>'

    def test_extracts_nickname(self):
        assert _tt_display_name_from_html(self._make_next_data("Stream Pirate")) == "Stream Pirate"

    def test_og_title_fallback(self):
        html = '<meta property="og:title" content="Pirate OG" />'
        assert _tt_display_name_from_html(html) == "Pirate OG"

    def test_empty_html(self):
        assert _tt_display_name_from_html("") == ""


# ---------------------------------------------------------------------------
# IdentityResolver — dispatch and cache
# ---------------------------------------------------------------------------

class TestIdentityResolverDispatch:
    def test_unknown_platform_returns_none(self):
        resolver = IdentityResolver()
        result = resolver.resolve("instagram", "https://instagram.com/pirate")
        assert result is None

    def test_facebook_url_resolves_without_fetcher(self):
        resolver = IdentityResolver(fetcher=None)
        result = resolver.resolve(
            "facebook",
            "https://www.facebook.com/profile.php?id=123456789",
        )
        assert result is not None
        assert result.permanent_id == "123456789"
        assert result.platform == "facebook"

    def test_youtube_channel_url_resolves_without_fetcher(self):
        resolver = IdentityResolver(fetcher=None)
        result = resolver.resolve(
            "youtube",
            "https://www.youtube.com/channel/UCddiUEpeqJcYeBxX1IVBKvQ",
        )
        assert result is not None
        assert result.permanent_id == "UCddiUEpeqJcYeBxX1IVBKvQ"

    def test_telegram_resolves_to_username(self):
        resolver = IdentityResolver(fetcher=None)
        result = resolver.resolve("telegram", "https://t.me/piratechannel")
        assert result is not None
        assert result.permanent_id == "piratechannel"
        assert result.confidence == 0.5

    def test_tiktok_fallback_to_username_without_fetcher(self):
        resolver = IdentityResolver(fetcher=None)
        result = resolver.resolve("tiktok", "https://www.tiktok.com/@pirate123")
        assert result is not None
        assert result.permanent_id == "pirate123"
        assert result.confidence == 0.6

    def test_facebook_username_url_needs_fetcher(self):
        # Username-only URL → Phase 1 fails → no fetcher → None
        resolver = IdentityResolver(fetcher=None)
        result = resolver.resolve("facebook", "https://www.facebook.com/PiratePage")
        assert result is None

    def test_display_name_hint_used_when_no_fetch(self):
        resolver = IdentityResolver(fetcher=None)
        result = resolver.resolve(
            "facebook",
            "https://www.facebook.com/profile.php?id=111222333",
            display_name_hint="Evil Pirate Inc",
        )
        assert result.display_name == "Evil Pirate Inc"

    def test_resolved_at_is_iso8601(self):
        resolver = IdentityResolver(fetcher=None)
        result = resolver.resolve(
            "facebook",
            "https://www.facebook.com/profile.php?id=123",
        )
        assert result.resolved_at.endswith("Z")
        assert "T" in result.resolved_at


# ---------------------------------------------------------------------------
# IdentityResolver — confidence values
# ---------------------------------------------------------------------------

class TestConfidenceLevels:
    def test_facebook_url_parse_confidence_1(self):
        r = IdentityResolver().resolve(
            "facebook", "https://www.facebook.com/profile.php?id=123456789"
        )
        assert r.confidence == 1.0

    def test_youtube_url_parse_confidence_1(self):
        r = IdentityResolver().resolve(
            "youtube", "https://www.youtube.com/channel/UCddiUEpeqJcYeBxX1IVBKvQ"
        )
        assert r.confidence == 1.0

    def test_telegram_username_confidence_05(self):
        r = IdentityResolver().resolve("telegram", "https://t.me/piratechan")
        assert r.confidence == 0.5

    def test_tiktok_username_fallback_confidence_06(self):
        r = IdentityResolver(fetcher=None).resolve(
            "tiktok", "https://www.tiktok.com/@pirate123"
        )
        assert r.confidence == 0.6


# ---------------------------------------------------------------------------
# IdentityResolver — Phase 2 (with fetcher)
# ---------------------------------------------------------------------------

class TestIdentityResolverWithFetcher:
    def test_facebook_html_scrape(self):
        html = 'fb://page/999888777 <title>Scraped Page | Facebook</title>'
        fetcher = _FakeFetcher({"https://www.facebook.com/PiratePage": html})
        resolver = IdentityResolver(fetcher=fetcher)
        result = resolver.resolve("facebook", "https://www.facebook.com/PiratePage")
        assert result is not None
        assert result.permanent_id == "999888777"
        assert result.confidence == 0.9

    def test_facebook_html_scrape_extracts_display_name(self):
        html = (
            'fb://page/999888777 '
            '<meta property="og:title" content="Scraped Pirate Page" />'
        )
        fetcher = _FakeFetcher({"https://www.facebook.com/PiratePage": html})
        resolver = IdentityResolver(fetcher=fetcher)
        result = resolver.resolve("facebook", "https://www.facebook.com/PiratePage")
        assert result.display_name == "Scraped Pirate Page"

    def test_facebook_html_scrape_no_id_returns_none(self):
        html = "<html>no ID here</html>"
        fetcher = _FakeFetcher({"https://www.facebook.com/PiratePage": html})
        resolver = IdentityResolver(fetcher=fetcher)
        result = resolver.resolve("facebook", "https://www.facebook.com/PiratePage")
        assert result is None

    def test_youtube_handle_scrape(self):
        html = '"channelId":"UCddiUEpeqJcYeBxX1IVBKvQ"'
        fetcher = _FakeFetcher({"https://www.youtube.com/@PirateHandle": html})
        resolver = IdentityResolver(fetcher=fetcher)
        result = resolver.resolve("youtube", "https://www.youtube.com/@PirateHandle")
        assert result is not None
        assert result.permanent_id == "UCddiUEpeqJcYeBxX1IVBKvQ"
        assert result.confidence == 0.9

    def test_tiktok_uid_from_html(self):
        import json
        data = {
            "props": {"pageProps": {"userInfo": {
                "user": {"id": "9876543210", "uniqueId": "pirate", "nickname": "Pirate"}
            }}}
        }
        html = f'<script id="__NEXT_DATA__">{json.dumps(data)}</script>'
        fetcher = _FakeFetcher({"https://www.tiktok.com/@pirate": html})
        resolver = IdentityResolver(fetcher=fetcher)
        result = resolver.resolve("tiktok", "https://www.tiktok.com/@pirate")
        assert result is not None
        assert result.permanent_id == "9876543210"
        assert result.confidence == 0.95
        assert result.metadata["username"] == "pirate"

    def test_tiktok_fetcher_error_falls_back_to_username(self):
        class _ErrorFetcher:
            def fetch(self, url: str) -> str:
                raise ConnectionError("network down")

        resolver = IdentityResolver(fetcher=_ErrorFetcher())
        result = resolver.resolve("tiktok", "https://www.tiktok.com/@pirate123")
        # Should fall back to username with confidence 0.6
        assert result is not None
        assert result.permanent_id == "pirate123"
        assert result.confidence == 0.6

    def test_fetcher_not_called_when_url_parse_succeeds(self):
        fetcher = _FakeFetcher({})  # no registered URLs → fetch would raise
        resolver = IdentityResolver(fetcher=fetcher)
        # profile.php?id= → Phase 1 succeeds → fetcher not called
        result = resolver.resolve(
            "facebook",
            "https://www.facebook.com/profile.php?id=123456789",
        )
        assert result is not None
        assert fetcher.calls == []


# ---------------------------------------------------------------------------
# IdentityResolver — caching
# ---------------------------------------------------------------------------

class TestIdentityResolverCache:
    def test_cache_hit_avoids_second_fetch(self):
        html = 'fb://page/111222333'
        fetcher = _FakeFetcher({"https://www.facebook.com/PiratePage": html})
        resolver = IdentityResolver(fetcher=fetcher)
        r1 = resolver.resolve("facebook", "https://www.facebook.com/PiratePage")
        r2 = resolver.resolve("facebook", "https://www.facebook.com/PiratePage")
        assert r1 is r2              # same object from cache
        assert len(fetcher.calls) == 1  # fetched only once

    def test_different_urls_cached_independently(self):
        fetcher = _FakeFetcher({
            "https://www.facebook.com/PageA": 'fb://page/111',
            "https://www.facebook.com/PageB": 'fb://page/222',
        })
        resolver = IdentityResolver(fetcher=fetcher)
        r_a = resolver.resolve("facebook", "https://www.facebook.com/PageA")
        r_b = resolver.resolve("facebook", "https://www.facebook.com/PageB")
        assert r_a.permanent_id == "111"
        assert r_b.permanent_id == "222"

    def test_none_result_not_cached(self):
        html_with_id  = 'fb://page/999888777'
        html_no_id    = '<html>nothing</html>'
        fetcher = _FakeFetcher({
            "https://www.facebook.com/PiratePage": html_no_id,
        })
        resolver = IdentityResolver(fetcher=fetcher)
        # First call: HTML has no ID → returns None
        r1 = resolver.resolve("facebook", "https://www.facebook.com/PiratePage")
        assert r1 is None

        # Update the HTML to now have an ID (page was re-scraped with more data)
        fetcher._map["https://www.facebook.com/PiratePage"] = html_with_id
        r2 = resolver.resolve("facebook", "https://www.facebook.com/PiratePage")
        # None is not cached, so second call fetches again and succeeds
        assert r2 is not None
        assert r2.permanent_id == "999888777"

    def test_injected_cache_is_used(self):
        prefilled_id = OffenderIdentity(
            platform="facebook", permanent_id="PRE_CACHED_ID",
            display_name="Pre-cached", profile_url="https://fb.com/pre",
            resolved_at="2025-01-01T00:00:00Z", confidence=1.0, metadata={},
        )
        cache = {"facebook:https://www.facebook.com/PiratePage": prefilled_id}
        resolver = IdentityResolver(cache=cache)
        result = resolver.resolve("facebook", "https://www.facebook.com/PiratePage")
        assert result is prefilled_id


# ---------------------------------------------------------------------------
# is_same_identity
# ---------------------------------------------------------------------------

class TestIsSameIdentity:
    def _make(self, platform: str, pid: str) -> OffenderIdentity:
        return OffenderIdentity(
            platform=platform, permanent_id=pid,
            display_name="", profile_url="", resolved_at="",
            confidence=1.0, metadata={},
        )

    def test_same_platform_same_id(self):
        a = self._make("facebook", "123")
        b = self._make("facebook", "123")
        assert is_same_identity(a, b) is True

    def test_same_platform_different_id(self):
        a = self._make("facebook", "123")
        b = self._make("facebook", "456")
        assert is_same_identity(a, b) is False

    def test_different_platform_same_id(self):
        a = self._make("facebook", "123")
        b = self._make("youtube",  "123")
        assert is_same_identity(a, b) is False

    def test_reflexive(self):
        a = self._make("telegram", "piratechan")
        assert is_same_identity(a, a) is True

    def test_symmetric(self):
        a = self._make("tiktok", "uid_001")
        b = self._make("tiktok", "uid_001")
        assert is_same_identity(a, b) == is_same_identity(b, a)
