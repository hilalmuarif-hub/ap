"""Tests for dedupe_cluster.py."""

import pytest
from detection import RawDetection
from dedupe_cluster import (
    DetectionCluster,
    _UnionFind,
    cluster_id_for,
    deduplicate,
    merge_clusters,
    pick_canonical,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def det(
    url: str,
    title: str = "Sinetron Vidio",
    channel_id: str = "ch_001",
    platform: str = "facebook",
    detected_at: str = "2025-01-15T08:00:00Z",
    snapshot_html: str = "<html>evidence</html>",
) -> RawDetection:
    return RawDetection(
        platform=platform,
        url=url,
        title=title,
        channel_id=channel_id,
        channel_name="Test Channel",
        snapshot_html=snapshot_html,
        detected_at=detected_at,
        query_used="test query",
    )


# ---------------------------------------------------------------------------
# _UnionFind
# ---------------------------------------------------------------------------

class TestUnionFind:
    def test_initial_each_own_group(self):
        uf = _UnionFind(3)
        assert uf.find(0) != uf.find(1)
        assert uf.find(1) != uf.find(2)

    def test_union_merges(self):
        uf = _UnionFind(3)
        uf.union(0, 1)
        assert uf.find(0) == uf.find(1)
        assert uf.find(0) != uf.find(2)

    def test_transitivity(self):
        uf = _UnionFind(3)
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.find(0) == uf.find(1) == uf.find(2)

    def test_groups_correct(self):
        uf = _UnionFind(4)
        uf.union(0, 1)
        uf.union(2, 3)
        groups = uf.groups()
        assert len(groups) == 2
        sizes = sorted(len(v) for v in groups.values())
        assert sizes == [2, 2]

    def test_idempotent_union(self):
        uf = _UnionFind(2)
        uf.union(0, 1)
        uf.union(0, 1)  # second call should be a no-op
        assert len(uf.groups()) == 1


# ---------------------------------------------------------------------------
# pick_canonical
# ---------------------------------------------------------------------------

class TestPickCanonical:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            pick_canonical([])

    def test_single_returns_itself(self):
        d = det("https://fb.com/v/1")
        assert pick_canonical([d]) is d

    def test_prefers_longer_snapshot(self):
        short = det("https://fb.com/v/1", snapshot_html="<html>x</html>")
        long_ = det("https://fb.com/v/2", snapshot_html="<html>" + "x" * 500 + "</html>")
        assert pick_canonical([short, long_]) is long_

    def test_tiebreak_earlier_date(self):
        early = det("https://fb.com/v/1", detected_at="2025-01-01T00:00:00Z")
        late  = det("https://fb.com/v/2", detected_at="2025-06-01T00:00:00Z")
        # Same snapshot length → earlier wins
        assert pick_canonical([late, early]) is early

    def test_snapshot_beats_date(self):
        rich_late  = det("https://fb.com/v/1", snapshot_html="x" * 1000,
                         detected_at="2025-12-01T00:00:00Z")
        poor_early = det("https://fb.com/v/2", snapshot_html="x",
                         detected_at="2025-01-01T00:00:00Z")
        assert pick_canonical([poor_early, rich_late]) is rich_late


# ---------------------------------------------------------------------------
# cluster_id_for
# ---------------------------------------------------------------------------

class TestClusterIdFor:
    def test_deterministic(self):
        d = det("https://www.facebook.com/v/123")
        assert cluster_id_for(d) == cluster_id_for(d)

    def test_length_16_hex(self):
        d = det("https://www.facebook.com/v/123")
        cid = cluster_id_for(d)
        assert len(cid) == 16
        assert all(c in "0123456789abcdef" for c in cid)

    def test_different_platform(self):
        a = det("https://www.facebook.com/v/1", platform="facebook")
        b = det("https://www.facebook.com/v/1", platform="youtube")
        assert cluster_id_for(a) != cluster_id_for(b)

    def test_different_channel(self):
        a = det("https://fb.com/v/1", channel_id="ch_001")
        b = det("https://fb.com/v/1", channel_id="ch_002")
        assert cluster_id_for(a) != cluster_id_for(b)

    def test_utm_stripped_same_id(self):
        # UTM params should not change the cluster ID
        base = det("https://www.facebook.com/video/123")
        with_utm = det("https://www.facebook.com/video/123?utm_source=ig&fbclid=xyz")
        assert cluster_id_for(base) == cluster_id_for(with_utm)

    def test_mobile_same_as_desktop(self):
        desktop = det("https://www.facebook.com/watch?v=456")
        mobile  = det("https://m.facebook.com/watch?v=456")
        assert cluster_id_for(desktop) == cluster_id_for(mobile)


# ---------------------------------------------------------------------------
# deduplicate — pass 1 (exact URL)
# ---------------------------------------------------------------------------

class TestDeduplicateExactUrl:
    def test_empty(self):
        assert deduplicate([]) == []

    def test_single(self):
        result = deduplicate([det("https://fb.com/v/1")])
        assert len(result) == 1
        assert result[0].cluster_size == 1
        assert result[0].duplicates == []

    def test_exact_url_collapse(self):
        a = det("https://fb.com/v/1", title="Title A", detected_at="2025-01-01T00:00:00Z")
        b = det("https://fb.com/v/1", title="Title B", detected_at="2025-01-02T00:00:00Z")
        result = deduplicate([a, b])
        assert len(result) == 1
        assert result[0].cluster_size == 2

    def test_utm_variants_same_cluster(self):
        a = det("https://www.facebook.com/watch?v=99")
        b = det("https://www.facebook.com/watch?v=99&utm_source=ig")
        result = deduplicate([a, b])
        assert len(result) == 1

    def test_mobile_and_desktop_same_cluster(self):
        a = det("https://www.facebook.com/watch?v=99")
        b = det("https://m.facebook.com/watch?v=99")
        result = deduplicate([a, b])
        assert len(result) == 1

    def test_different_urls_stay_separate(self):
        # Different channel_ids so Pass 3 (fuzzy, within-channel) doesn't compare them.
        # This test isolates the URL pass: two genuinely unrelated detections stay separate.
        a = det("https://fb.com/v/1", title="Liga Champions", channel_id="ch_A")
        b = det("https://fb.com/v/2", title="Sinetron Cinta", channel_id="ch_B")
        result = deduplicate([a, b])
        assert len(result) == 2

    def test_url_exact_disabled(self):
        a = det("https://fb.com/v/1")
        b = det("https://fb.com/v/1")
        result = deduplicate([a, b], url_exact=False, fuzzy_title=False)
        # Without any dedup, same URL on different channel (both ch_001 here)
        # would still merge on pass 2 (same channel + same title). Use different titles.
        a2 = det("https://fb.com/v/1", title="Title A", channel_id="ch_A")
        b2 = det("https://fb.com/v/1", title="Title B", channel_id="ch_B")
        result2 = deduplicate([a2, b2], url_exact=False, fuzzy_title=False)
        assert len(result2) == 2


# ---------------------------------------------------------------------------
# deduplicate — pass 2 (same channel + exact title)
# ---------------------------------------------------------------------------

class TestDeduplicateSameChannelTitle:
    def test_same_channel_same_title_different_url(self):
        # Content re-posted to a new URL on the same channel
        a = det("https://fb.com/v/1", title="Liga Champions Vidio", channel_id="ch_X")
        b = det("https://fb.com/v/2", title="Liga Champions Vidio", channel_id="ch_X")
        result = deduplicate([a, b])
        assert len(result) == 1
        assert result[0].cluster_size == 2

    def test_different_channels_same_title_stay_separate(self):
        a = det("https://fb.com/v/1", title="Liga Champions", channel_id="ch_A")
        b = det("https://fb.com/v/2", title="Liga Champions", channel_id="ch_B")
        result = deduplicate([a, b])
        assert len(result) == 2

    def test_same_channel_different_title_stay_separate(self):
        a = det("https://fb.com/v/1", title="Liga Champions",    channel_id="ch_X")
        b = det("https://fb.com/v/2", title="Sinetron Nusantara", channel_id="ch_X")
        # titles are very different → fuzzy won't merge at 0.85
        result = deduplicate([a, b])
        assert len(result) == 2

    def test_title_normalized_before_compare(self):
        # These titles should normalize to the same string
        a = det("https://fb.com/v/1", title="Liga Champions 1080p HD",  channel_id="ch_X")
        b = det("https://fb.com/v/2", title="LIGA CHAMPIONS 720p",      channel_id="ch_X")
        # Both normalize to "liga champions" → same title key
        result = deduplicate([a, b])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# deduplicate — pass 3 (fuzzy title within same channel)
# ---------------------------------------------------------------------------

class TestDeduplicateFuzzyTitle:
    def test_fuzzy_same_channel(self):
        a = det("https://fb.com/v/1", title="Liga Champions Vidio 2025",   channel_id="ch_X")
        b = det("https://fb.com/v/2", title="Liga Champions 2025 Vidio",   channel_id="ch_X")
        result = deduplicate([a, b], fuzzy_threshold=0.85)
        assert len(result) == 1

    def test_fuzzy_disabled(self):
        a = det("https://fb.com/v/1", title="Liga Champions Vidio 2025",   channel_id="ch_X")
        b = det("https://fb.com/v/2", title="Liga Champions 2025 Vidio",   channel_id="ch_X")
        result = deduplicate([a, b], fuzzy_title=False)
        # Without fuzzy, these only merge on pass 2 (exact normalize).
        # "liga champions vidio 2025" != "liga champions 2025 vidio" in normalized form
        # but wait: normalize strips stopwords only, word order stays.
        # So "liga champions vidio 2025" and "liga champions 2025 vidio" differ in order.
        # They should NOT merge on pass 2 (exact). Assert they stay separate.
        assert len(result) == 2

    def test_fuzzy_different_channels_no_merge(self):
        a = det("https://fb.com/v/1", title="Liga Champions Vidio 2025", channel_id="ch_A")
        b = det("https://fb.com/v/2", title="Liga Champions Vidio 2025", channel_id="ch_B")
        # Same title, but different channels: pass 3 groups by channel first
        # so these should only merge if pass 2 catches them (it won't, diff channel_id)
        result = deduplicate([a, b], fuzzy_title=True)
        # Pass 2 groups by (platform, channel_id, title) — different channels → 2 clusters
        assert len(result) == 2

    def test_fuzzy_threshold_respected(self):
        # Very different titles should not merge
        a = det("https://fb.com/v/1", title="Liga Champions Eropa", channel_id="ch_X")
        b = det("https://fb.com/v/2", title="Sinetron Cinta Abadi", channel_id="ch_X")
        result = deduplicate([a, b], fuzzy_threshold=0.85)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# deduplicate — transitivity
# ---------------------------------------------------------------------------

class TestDeduplicateTransitivity:
    def test_three_way_chain(self):
        # A and B share same URL → same cluster (pass 1)
        # B and C share same channel + title → same cluster (pass 2)
        # Union-find transitivity: A, B, C all in one cluster
        a = det("https://fb.com/v/1", title="Foo", channel_id="ch_X")
        b = det("https://fb.com/v/1", title="Foo", channel_id="ch_X",  # same URL as A
                detected_at="2025-01-02T00:00:00Z")
        c = det("https://fb.com/v/2", title="Foo", channel_id="ch_X",  # same title as B
                detected_at="2025-01-03T00:00:00Z")
        result = deduplicate([a, b, c])
        assert len(result) == 1
        assert result[0].cluster_size == 3


# ---------------------------------------------------------------------------
# deduplicate — cluster structure
# ---------------------------------------------------------------------------

class TestDeduplicateClusterStructure:
    def test_canonical_not_in_duplicates(self):
        # Give different detected_at so the two objects are distinguishable by value.
        # RawDetection is a dataclass: `in` uses field equality, not identity.
        # With identical fields both objects would compare equal, making the
        # assertion trivially fail even though the implementation is correct.
        a = det("https://fb.com/v/1", detected_at="2025-01-01T00:00:00Z")
        b = det("https://fb.com/v/1", detected_at="2025-06-01T00:00:00Z")
        result = deduplicate([a, b])
        cluster = result[0]
        assert cluster.canonical not in cluster.duplicates

    def test_cluster_size_matches(self):
        a = det("https://fb.com/v/1")
        b = det("https://fb.com/v/1")
        c = det("https://fb.com/v/1")
        result = deduplicate([a, b, c])
        assert result[0].cluster_size == 3
        assert len(result[0].duplicates) == 2

    def test_sorted_by_detected_at(self):
        # Different channel_ids isolate Pass 3; genuinely different content stays in two clusters.
        a = det("https://fb.com/v/1", title="Liga Champions",
                channel_id="ch_A", detected_at="2025-06-01T00:00:00Z")
        b = det("https://fb.com/v/2", title="Sinetron Cinta",
                channel_id="ch_B", detected_at="2025-01-01T00:00:00Z")
        result = deduplicate([a, b])
        assert len(result) == 2
        assert result[0].canonical.detected_at < result[1].canonical.detected_at

    def test_cluster_id_is_string(self):
        result = deduplicate([det("https://fb.com/v/1")])
        assert isinstance(result[0].cluster_id, str)
        assert len(result[0].cluster_id) > 0

    def test_canonical_is_richest_snapshot(self):
        poor  = det("https://fb.com/v/1", snapshot_html="x")
        rich  = det("https://fb.com/v/1", snapshot_html="x" * 1000)
        result = deduplicate([poor, rich])
        assert result[0].canonical is rich


# ---------------------------------------------------------------------------
# merge_clusters
# ---------------------------------------------------------------------------

class TestMergeClusters:
    def _make_cluster(self, url: str, title: str = "T", channel_id: str = "ch",
                      snapshot: str = "snap") -> DetectionCluster:
        d = det(url, title=title, channel_id=channel_id, snapshot_html=snapshot)
        return DetectionCluster(
            canonical=d,
            duplicates=[],
            cluster_id=cluster_id_for(d),
            cluster_size=1,
        )

    def test_new_cluster_appended(self):
        c1 = self._make_cluster("https://fb.com/v/1")
        c2 = self._make_cluster("https://fb.com/v/2")
        result = merge_clusters([c1], [c2])
        assert len(result) == 2

    def test_existing_cluster_merged(self):
        c1 = self._make_cluster("https://fb.com/v/1")
        # Same cluster_id: same platform + channel_id + url
        d_extra = det("https://fb.com/v/1", snapshot_html="extra evidence " * 10)
        c1_update = DetectionCluster(
            canonical=d_extra,
            duplicates=[],
            cluster_id=c1.cluster_id,
            cluster_size=1,
        )
        result = merge_clusters([c1], [c1_update])
        assert len(result) == 1
        # After merge, size should reflect the pooled unique URLs
        # (both have same URL after normalize, so dedup keeps 1 unique)
        assert result[0].cluster_size == 1

    def test_merge_grows_size(self):
        d1 = det("https://fb.com/v/1", snapshot_html="short")
        d2 = det("https://fb.com/v/2", snapshot_html="short", channel_id="ch_001")
        c1 = DetectionCluster(canonical=d1, duplicates=[], cluster_id="aabbccdd11223344", cluster_size=1)
        c2 = DetectionCluster(canonical=d2, duplicates=[], cluster_id="aabbccdd11223344", cluster_size=1)
        result = merge_clusters([c1], [c2])
        assert result[0].cluster_size == 2

    def test_merge_no_url_double_count(self):
        # Replaying the same run should not grow cluster size
        d = det("https://fb.com/v/1")
        c = DetectionCluster(canonical=d, duplicates=[], cluster_id=cluster_id_for(d), cluster_size=1)
        result = merge_clusters([c], [c])
        assert result[0].cluster_size == 1

    def test_merge_sorted_by_detected_at(self):
        c1 = self._make_cluster("https://fb.com/v/1")
        c2 = self._make_cluster("https://fb.com/v/2", channel_id="ch_002")
        # c1 detected_at = "2025-01-15T08:00:00Z" (default in det())
        # c2 same — just check output is a sorted list
        result = merge_clusters([c1], [c2])
        dates = [r.canonical.detected_at for r in result]
        assert dates == sorted(dates)

    def test_empty_existing(self):
        c = self._make_cluster("https://fb.com/v/1")
        result = merge_clusters([], [c])
        assert len(result) == 1

    def test_empty_incoming(self):
        c = self._make_cluster("https://fb.com/v/1")
        result = merge_clusters([c], [])
        assert len(result) == 1
