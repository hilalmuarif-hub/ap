"""Tests for normalize_query.py."""

import pytest
from normalize_query import normalize, normalize_url, expand_to_queries, similarity


class TestNormalize:
    def test_lowercase(self):
        assert normalize("VIDIO Premium") == "vidio premium"

    def test_leet_at_sign(self):
        assert normalize("v@dio") == "vadio"

    def test_leet_zero_and_one(self):
        # "v1d10" → v + i + d + i + o → "vidio"
        assert normalize("v1d10") == "vidio"

    def test_leet_dollar(self):
        assert normalize("$treaming") == ""  # $→s, "streaming" is a stopword → stripped
        assert normalize("$ecret") == "secret"

    def test_stripwords_quality(self):
        assert normalize("Liga Champions 1080p HD") == "liga champions"

    def test_stripwords_noise(self):
        assert normalize("Nonton Sinetron Gratis Live") == "sinetron"

    def test_unicode_nfkc(self):
        # fullwidth chars collapse to ASCII
        assert normalize("ＶＩＤＩＯ") == "vidio"

    def test_emoji_stripped(self):
        assert normalize("Vidio 🔥 Premium") == "vidio premium"

    def test_empty_string(self):
        assert normalize("") == ""

    def test_whitespace_only(self):
        assert normalize("   ") == ""

    def test_punctuation_stripped(self):
        # . and / → space (not concatenated); ! → i via leet (LEET_MAP["!"] = "i")
        assert normalize("vidio.com/premium!") == "vidio com premiumi"

    def test_collapse_whitespace(self):
        assert normalize("vidio   premium") == "vidio premium"

    def test_leet_pipe(self):
        # | → l via leet, then stopword removal applies to the decoded form
        assert normalize("|ive") == ""    # |→l → "live" is a stopword → stripped
        assert normalize("|ink") == ""    # |→l → "link" is a stopword → stripped
        assert normalize("|iga") == "liga"  # "liga" is not a stopword → kept

    def test_720p_stripped(self):
        # "720p" → leet "t2op" → in STOPWORDS
        assert normalize("Sinetron 720p") == "sinetron"

    def test_480p_stripped(self):
        assert normalize("Sinetron 480p") == "sinetron"


class TestNormalizeUrl:
    def test_strips_utm(self):
        url = "https://www.facebook.com/video/123?utm_source=ig&utm_medium=social"
        assert "utm_source" not in normalize_url(url)
        assert "utm_medium" not in normalize_url(url)

    def test_strips_fbclid(self):
        url = "https://www.facebook.com/watch?v=123&fbclid=abc"
        assert "fbclid" not in normalize_url(url)
        assert "v=123" in normalize_url(url)

    def test_mobile_fb_to_canonical(self):
        url = "https://m.facebook.com/video/123"
        assert "www.facebook.com" in normalize_url(url)
        assert "m.facebook.com" not in normalize_url(url)

    def test_fb_com_to_canonical(self):
        url = "https://fb.com/watch/123"
        assert "www.facebook.com" in normalize_url(url)

    def test_trailing_slash_stripped(self):
        url = "https://www.facebook.com/page/videos/"
        result = normalize_url(url)
        assert not result.endswith("/videos/")
        assert result.endswith("/videos")

    def test_preserves_meaningful_params(self):
        url = "https://www.facebook.com/watch?v=987654321"
        assert "v=987654321" in normalize_url(url)

    def test_lowercase_host(self):
        url = "https://WWW.FACEBOOK.COM/video"
        assert "www.facebook.com" in normalize_url(url)

    def test_strips_ref(self):
        url = "https://www.facebook.com/video/123?ref=page_internal"
        assert "ref" not in normalize_url(url)


class TestExpandToQueries:
    def test_generates_brand_query(self):
        queries = expand_to_queries(["Sinetron Indah"])
        assert "sinetron indah vidio" in queries

    def test_generates_plain_query(self):
        queries = expand_to_queries(["Sinetron Indah"])
        assert "sinetron indah" in queries

    def test_generates_evasion_variants(self):
        queries = expand_to_queries(["Sinetron Indah"])
        # at least one evasion variant for "vidio" should be present
        evasion_found = any("v1d" in q or "vid1" in q for q in queries)
        assert evasion_found

    def test_no_duplicates(self):
        queries = expand_to_queries(["Sinetron Indah", "Sinetron Indah"])
        assert len(queries) == len(set(queries))

    def test_cap_respected(self):
        titles = [f"Title {i}" for i in range(100)]
        queries = expand_to_queries(titles, max_queries=10)
        assert len(queries) <= 10

    def test_brand_queries_before_evasion(self):
        queries = expand_to_queries(["Liga Champions"])
        brand_idx = next(i for i, q in enumerate(queries) if q == "liga champions vidio")
        evasion_idx = next(i for i, q in enumerate(queries) if "v1d" in q or "vid1" in q)
        assert brand_idx < evasion_idx

    def test_custom_brand_terms(self):
        queries = expand_to_queries(["Sinetron X"], brand_terms=["mybrand"])
        assert "sinetron x mybrand" in queries

    def test_empty_titles(self):
        assert expand_to_queries([]) == []

    def test_blank_title_skipped(self):
        queries = expand_to_queries(["", "   ", "Sinetron X"])
        assert all("sinetron x" in q for q in queries)


class TestSimilarity:
    def test_identical(self):
        assert similarity("Sinetron Indah", "Sinetron Indah") == 1.0

    def test_both_empty(self):
        assert similarity("", "") == 1.0

    def test_one_empty(self):
        assert similarity("Sinetron", "") == 0.0

    def test_order_insensitive(self):
        # token-set ratio should treat word order as irrelevant
        score = similarity("Liga Champions Vidio", "Vidio Liga Champions")
        assert score >= 0.9

    def test_leet_similarity(self):
        # "v1d10" normalizes to "vidio", should be highly similar to "vidio"
        score = similarity("Sinetron v1d10", "Sinetron vidio")
        assert score >= 0.9

    def test_completely_different(self):
        score = similarity("Sinetron Indonesia", "Berita Olahraga")
        assert score < 0.5

    def test_returns_float(self):
        result = similarity("foo", "bar")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
