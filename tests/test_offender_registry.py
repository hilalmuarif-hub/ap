"""Tests for offender_registry.py."""

import pytest
from detection import RawDetection
from identity import OffenderIdentity
from evidence_scorer import ScoredEvidence
from offender_registry import (
    TIER_THRESHOLDS,
    VALID_STATUSES,
    OffenderRecord,
    OffenderRegistry,
    RegistryBackend,
)


# ---------------------------------------------------------------------------
# Test backend (implements RegistryBackend protocol)
# ---------------------------------------------------------------------------

class _InMemoryBackend:
    """Minimal in-memory backend for testing. Satisfies RegistryBackend protocol."""

    def __init__(self) -> None:
        self._store: dict[int, OffenderRecord] = {}
        self._next_row: int = 1
        self.save_calls: int = 0
        self.update_calls: int = 0
        self.load_calls: int = 0

    def save_record(self, record: OffenderRecord) -> int:
        row = self._next_row
        self._next_row += 1
        self._store[row] = record
        self.save_calls += 1
        return row

    def update_record(self, row_id: int, record: OffenderRecord) -> None:
        self._store[row_id] = record
        self.update_calls += 1

    def load_all_records(self) -> list[OffenderRecord]:
        self.load_calls += 1
        return list(self._store.values())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_evidence(
    permanent_id: str = "pid_001",
    platform: str = "facebook",
    display_name: str = "Pirate Channel",
    profile_url: str = "https://fb.com/pirate",
    detected_at: str = "2025-01-15T08:00:00Z",
    url: str = "https://fb.com/watch?v=123",
    score: int = 85,
    verdict: str = "confirmed",
) -> ScoredEvidence:
    detection = RawDetection(
        platform=platform,
        url=url,
        title="Liga Champions Vidio",
        channel_id=permanent_id,
        channel_name=display_name,
        snapshot_html="<html>evidence</html>",
        detected_at=detected_at,
        query_used="liga champions",
        extra={},
    )
    identity = OffenderIdentity(
        platform=platform,
        permanent_id=permanent_id,
        display_name=display_name,
        profile_url=profile_url,
        resolved_at=detected_at,
        confidence=0.95,
        metadata={},
    )
    return ScoredEvidence(
        detection=detection,
        identity=identity,
        score=score,
        breakdown={},
        verdict=verdict,
        scored_at=detected_at,
    )


def make_registry(backend: _InMemoryBackend | None = None) -> OffenderRegistry:
    return OffenderRegistry(sheet_id="test_sheet", backend=backend)


# ---------------------------------------------------------------------------
# _calculate_tier
# ---------------------------------------------------------------------------

class TestCalculateTier:
    def setup_method(self):
        self.reg = make_registry()

    @pytest.mark.parametrize("violations,expected", [
        (1, "tier3"),
        (2, "tier2"),
        (3, "tier2"),
        (4, "tier2"),
        (5, "tier1"),
        (6, "tier1"),
        (100, "tier1"),
    ])
    def test_tiers(self, violations, expected):
        assert self.reg._calculate_tier(violations) == expected

    def test_tier2_boundary(self):
        assert self.reg._calculate_tier(TIER_THRESHOLDS["tier2"]) == "tier2"
        assert self.reg._calculate_tier(TIER_THRESHOLDS["tier2"] - 1) == "tier3"

    def test_tier1_boundary(self):
        assert self.reg._calculate_tier(TIER_THRESHOLDS["tier1"]) == "tier1"
        assert self.reg._calculate_tier(TIER_THRESHOLDS["tier1"] - 1) == "tier2"


# ---------------------------------------------------------------------------
# upsert — new offender
# ---------------------------------------------------------------------------

class TestUpsertNew:
    def test_creates_record(self):
        reg = make_registry()
        ev = make_evidence()
        record = reg.upsert(ev)
        assert record.permanent_id == "pid_001"
        assert record.platform == "facebook"
        assert record.violation_count == 1
        assert record.status == "active"
        assert record.tier == "tier3"

    def test_first_and_last_seen_set(self):
        reg = make_registry()
        ev = make_evidence(detected_at="2025-03-01T10:00:00Z")
        record = reg.upsert(ev)
        assert record.first_seen == "2025-03-01T10:00:00Z"
        assert record.last_seen == "2025-03-01T10:00:00Z"

    def test_evidence_url_added(self):
        reg = make_registry()
        ev = make_evidence(url="https://fb.com/watch?v=999")
        record = reg.upsert(ev)
        assert "https://fb.com/watch?v=999" in record.evidence_urls

    def test_display_name_stored(self):
        reg = make_registry()
        ev = make_evidence(display_name="Evil Pirate Inc")
        record = reg.upsert(ev)
        assert record.display_name == "Evil Pirate Inc"

    def test_stored_in_cache(self):
        reg = make_registry()
        ev = make_evidence()
        reg.upsert(ev)
        assert reg.lookup("pid_001", "facebook") is not None

    def test_backend_save_called(self):
        backend = _InMemoryBackend()
        reg = make_registry(backend)
        reg.upsert(make_evidence())
        assert backend.save_calls == 1
        assert backend.update_calls == 0

    def test_sheet_row_assigned_from_backend(self):
        backend = _InMemoryBackend()
        reg = make_registry(backend)
        record = reg.upsert(make_evidence())
        assert record.sheet_row == 1   # first row assigned by backend

    def test_no_backend_sheet_row_is_none(self):
        reg = make_registry(backend=None)
        record = reg.upsert(make_evidence())
        assert record.sheet_row is None


# ---------------------------------------------------------------------------
# upsert — existing offender
# ---------------------------------------------------------------------------

class TestUpsertExisting:
    def test_violation_count_increments(self):
        reg = make_registry()
        reg.upsert(make_evidence(url="https://fb.com/v/1"))
        record = reg.upsert(make_evidence(url="https://fb.com/v/2"))
        assert record.violation_count == 2

    def test_first_seen_preserved(self):
        reg = make_registry()
        reg.upsert(make_evidence(detected_at="2025-01-01T00:00:00Z"))
        record = reg.upsert(make_evidence(detected_at="2025-06-01T00:00:00Z"))
        assert record.first_seen == "2025-01-01T00:00:00Z"

    def test_last_seen_advances(self):
        reg = make_registry()
        reg.upsert(make_evidence(detected_at="2025-01-01T00:00:00Z"))
        record = reg.upsert(make_evidence(detected_at="2025-06-01T00:00:00Z"))
        assert record.last_seen == "2025-06-01T00:00:00Z"

    def test_last_seen_does_not_go_backward(self):
        reg = make_registry()
        reg.upsert(make_evidence(detected_at="2025-06-01T00:00:00Z"))
        record = reg.upsert(make_evidence(detected_at="2025-01-01T00:00:00Z"))
        assert record.last_seen == "2025-06-01T00:00:00Z"

    def test_tier_upgrades_at_threshold(self):
        reg = make_registry()
        for i in range(4):
            reg.upsert(make_evidence(url=f"https://fb.com/v/{i}"))
        # 4 violations → still tier2
        record = reg.lookup("pid_001", "facebook")
        assert record.tier == "tier2"
        # 5th violation → tier1
        record = reg.upsert(make_evidence(url="https://fb.com/v/4"))
        assert record.tier == "tier1"

    def test_evidence_url_appended(self):
        reg = make_registry()
        reg.upsert(make_evidence(url="https://fb.com/v/1"))
        record = reg.upsert(make_evidence(url="https://fb.com/v/2"))
        assert "https://fb.com/v/1" in record.evidence_urls
        assert "https://fb.com/v/2" in record.evidence_urls

    def test_evidence_url_not_duplicated(self):
        reg = make_registry()
        reg.upsert(make_evidence(url="https://fb.com/v/1"))
        reg.upsert(make_evidence(url="https://fb.com/v/1"))
        record = reg.lookup("pid_001", "facebook")
        assert record.evidence_urls.count("https://fb.com/v/1") == 1

    def test_display_name_refreshed(self):
        reg = make_registry()
        reg.upsert(make_evidence(display_name="Old Name"))
        record = reg.upsert(make_evidence(display_name="New Name"))
        assert record.display_name == "New Name"

    def test_backend_update_called_not_save(self):
        backend = _InMemoryBackend()
        reg = make_registry(backend)
        reg.upsert(make_evidence(url="https://fb.com/v/1"))
        reg.upsert(make_evidence(url="https://fb.com/v/2"))
        assert backend.save_calls == 1
        assert backend.update_calls == 1

    def test_same_object_returned_from_cache(self):
        reg = make_registry()
        r1 = reg.upsert(make_evidence(url="https://fb.com/v/1"))
        r2 = reg.upsert(make_evidence(url="https://fb.com/v/2"))
        assert r1 is r2   # same object mutated in place


# ---------------------------------------------------------------------------
# lookup
# ---------------------------------------------------------------------------

class TestLookup:
    def test_hit_returns_record(self):
        reg = make_registry()
        reg.upsert(make_evidence())
        result = reg.lookup("pid_001", "facebook")
        assert result is not None
        assert result.permanent_id == "pid_001"

    def test_miss_returns_none(self):
        reg = make_registry()
        assert reg.lookup("nonexistent", "facebook") is None

    def test_platform_isolation(self):
        reg = make_registry()
        reg.upsert(make_evidence(platform="facebook"))
        # Same permanent_id but different platform → miss
        assert reg.lookup("pid_001", "youtube") is None

    def test_implicit_load_on_empty_cache(self):
        backend = _InMemoryBackend()
        reg = make_registry(backend)
        # Seed backend directly (simulating a prior run's data)
        record = OffenderRecord(
            permanent_id="pid_seed",
            platform="facebook",
            display_name="Seeded",
            profile_url="https://fb.com/seeded",
            first_seen="2025-01-01T00:00:00Z",
            last_seen="2025-01-01T00:00:00Z",
            violation_count=3,
            status="active",
            tier="tier2",
            sheet_row=42,
        )
        backend._store[42] = record
        backend._next_row = 43

        # Cache is empty — lookup triggers one implicit full load
        result = reg.lookup("pid_seed", "facebook")
        assert result is not None
        assert result.violation_count == 3
        assert backend.load_calls == 1

    def test_no_implicit_load_when_cache_has_entries(self):
        backend = _InMemoryBackend()
        reg = make_registry(backend)
        reg.upsert(make_evidence(permanent_id="pid_A"))
        # upsert calls lookup internally on an empty cache, potentially triggering
        # one implicit load. Snapshot load_calls *after* upsert to isolate the
        # assertion: a subsequent lookup on a non-empty cache must NOT call the backend.
        load_calls_after_upsert = backend.load_calls
        reg.lookup("pid_unknown", "facebook")
        assert backend.load_calls == load_calls_after_upsert


# ---------------------------------------------------------------------------
# update_status
# ---------------------------------------------------------------------------

class TestUpdateStatus:
    def setup_method(self):
        self.reg = make_registry()
        self.reg.upsert(make_evidence())

    def test_status_updated(self):
        self.reg.update_status("pid_001", "facebook", "suspended")
        record = self.reg.lookup("pid_001", "facebook")
        assert record.status == "suspended"

    def test_audit_trail_appended(self):
        self.reg.update_status("pid_001", "facebook", "suspended")
        record = self.reg.lookup("pid_001", "facebook")
        assert "active → suspended" in record.notes

    def test_audit_trail_accumulates(self):
        self.reg.update_status("pid_001", "facebook", "suspended")
        self.reg.update_status("pid_001", "facebook", "removed")
        record = self.reg.lookup("pid_001", "facebook")
        assert "active → suspended" in record.notes
        assert "suspended → removed" in record.notes

    def test_notes_start_clean(self):
        # No prior notes — transition becomes the entire notes string
        record = self.reg.lookup("pid_001", "facebook")
        assert record.notes == ""
        self.reg.update_status("pid_001", "facebook", "suspended")
        record = self.reg.lookup("pid_001", "facebook")
        assert "|" not in record.notes or record.notes.startswith("[")

    def test_invalid_status_raises(self):
        with pytest.raises(ValueError, match="Invalid status"):
            self.reg.update_status("pid_001", "facebook", "banned")

    def test_unknown_offender_raises(self):
        with pytest.raises(KeyError):
            self.reg.update_status("nonexistent", "facebook", "suspended")

    def test_same_status_noop(self):
        self.reg.update_status("pid_001", "facebook", "active")   # already active
        record = self.reg.lookup("pid_001", "facebook")
        assert record.notes == ""   # no audit entry for no-op

    def test_backend_update_called(self):
        backend = _InMemoryBackend()
        reg = make_registry(backend)
        reg.upsert(make_evidence())
        initial_updates = backend.update_calls
        reg.update_status("pid_001", "facebook", "suspended")
        assert backend.update_calls == initial_updates + 1

    @pytest.mark.parametrize("status", list(VALID_STATUSES))
    def test_all_valid_statuses_accepted(self, status):
        reg = make_registry()
        reg.upsert(make_evidence())
        # Set to a known different status first if needed
        try:
            reg.update_status("pid_001", "facebook", status)
        except ValueError:
            pytest.fail(f"Valid status {status!r} raised ValueError")


# ---------------------------------------------------------------------------
# list_active
# ---------------------------------------------------------------------------

class TestListActive:
    def test_returns_only_active(self):
        reg = make_registry()
        reg.upsert(make_evidence(permanent_id="pid_A", platform="facebook"))
        reg.upsert(make_evidence(permanent_id="pid_B", platform="facebook"))
        reg.update_status("pid_B", "facebook", "suspended")
        active = reg.list_active()
        assert len(active) == 1
        assert active[0].permanent_id == "pid_A"

    def test_platform_filter(self):
        reg = make_registry()
        reg.upsert(make_evidence(permanent_id="pid_fb", platform="facebook"))
        reg.upsert(make_evidence(permanent_id="pid_yt", platform="youtube"))
        fb_active = reg.list_active(platform="facebook")
        assert len(fb_active) == 1
        assert fb_active[0].platform == "facebook"

    def test_sorted_by_last_seen_descending(self):
        reg = make_registry()
        reg.upsert(make_evidence(permanent_id="pid_A", detected_at="2025-01-01T00:00:00Z"))
        reg.upsert(make_evidence(permanent_id="pid_B", detected_at="2025-06-01T00:00:00Z"))
        active = reg.list_active()
        assert active[0].last_seen > active[1].last_seen

    def test_empty_registry_returns_empty(self):
        reg = make_registry()
        assert reg.list_active() == []

    def test_no_match_for_platform_returns_empty(self):
        reg = make_registry()
        reg.upsert(make_evidence(platform="facebook"))
        assert reg.list_active(platform="youtube") == []


# ---------------------------------------------------------------------------
# list_active — violation_count after multiple upserts
# ---------------------------------------------------------------------------

class TestListActiveViolationCount:
    def test_violation_count_correct_after_upserts(self):
        reg = make_registry()
        reg.upsert(make_evidence(url="https://fb.com/v/1"))
        reg.upsert(make_evidence(url="https://fb.com/v/2"))
        reg.upsert(make_evidence(url="https://fb.com/v/3"))
        active = reg.list_active()
        assert active[0].violation_count == 3


# ---------------------------------------------------------------------------
# export_for_legal
# ---------------------------------------------------------------------------

class TestExportForLegal:
    def setup_method(self):
        self.reg = make_registry()
        self.reg.upsert(make_evidence(
            url="https://fb.com/v/1",
            display_name="Pirate Inc",
            profile_url="https://fb.com/pirate",
        ))

    def test_returns_dict(self):
        result = self.reg.export_for_legal("pid_001")
        assert isinstance(result, dict)

    def test_required_fields_present(self):
        result = self.reg.export_for_legal("pid_001")
        required = {
            "permanent_id", "platform", "display_name", "profile_url",
            "first_seen", "last_seen", "violation_count", "tier",
            "status", "evidence_urls", "notes", "exported_at",
        }
        assert required.issubset(result.keys())

    def test_evidence_urls_is_copy(self):
        result = self.reg.export_for_legal("pid_001")
        result["evidence_urls"].append("injected_url")
        # Mutation of the export dict must not affect the registry
        record = self.reg.lookup("pid_001", "facebook")
        assert "injected_url" not in record.evidence_urls

    def test_exported_at_is_iso8601(self):
        result = self.reg.export_for_legal("pid_001")
        assert "T" in result["exported_at"]
        assert result["exported_at"].endswith("Z")

    def test_unknown_raises_key_error(self):
        with pytest.raises(KeyError):
            self.reg.export_for_legal("nonexistent_pid")

    def test_values_match_record(self):
        result = self.reg.export_for_legal("pid_001")
        record = self.reg.lookup("pid_001", "facebook")
        assert result["violation_count"] == record.violation_count
        assert result["tier"] == record.tier
        assert result["status"] == record.status


# ---------------------------------------------------------------------------
# refresh_cache
# ---------------------------------------------------------------------------

class TestRefreshCache:
    def test_loads_from_backend(self):
        backend = _InMemoryBackend()
        reg = make_registry(backend)
        reg.upsert(make_evidence())

        # New registry instance with the same backend but a fresh, empty cache.
        # We verify _cache state directly rather than via lookup(), because lookup()
        # on an empty cache triggers an implicit full load before refresh_cache() runs.
        reg2 = OffenderRegistry(sheet_id="test_sheet", backend=backend)
        assert len(reg2._cache) == 0   # cache starts empty

        reg2.refresh_cache()

        assert len(reg2._cache) > 0
        assert reg2.lookup("pid_001", "facebook") is not None

    def test_clears_stale_cache_entries(self):
        backend = _InMemoryBackend()
        reg = make_registry(backend)
        reg.upsert(make_evidence(permanent_id="pid_stale"))

        # Remove from backend (simulating a manual delete)
        backend._store.clear()

        reg.refresh_cache()
        assert reg.lookup("pid_stale", "facebook") is None

    def test_noop_without_backend(self):
        reg = make_registry(backend=None)
        reg.upsert(make_evidence())
        reg.refresh_cache()   # should not raise or clear the cache
        assert reg.lookup("pid_001", "facebook") is not None

    def test_load_calls_backend_once(self):
        backend = _InMemoryBackend()
        reg = make_registry(backend)
        reg.refresh_cache()
        assert backend.load_calls == 1


# ---------------------------------------------------------------------------
# cache_key
# ---------------------------------------------------------------------------

class TestCacheKey:
    def test_platform_prefix(self):
        key = OffenderRegistry._cache_key("facebook", "123")
        assert key.startswith("facebook:")

    def test_cross_platform_different_keys(self):
        fb  = OffenderRegistry._cache_key("facebook", "123")
        yt  = OffenderRegistry._cache_key("youtube", "123")
        assert fb != yt

    def test_same_platform_same_id_equal(self):
        k1 = OffenderRegistry._cache_key("facebook", "abc")
        k2 = OffenderRegistry._cache_key("facebook", "abc")
        assert k1 == k2
