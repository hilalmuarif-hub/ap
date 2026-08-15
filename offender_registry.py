"""
offender_registry.py — Verified offender database backed by Google Sheets.

The registry is the single source of truth for confirmed piracy offenders.
It stores permanent IDs only — display names are informational and may change.

Architecture:
  - In-memory dict (_cache) is the working copy during a pipeline run.
  - RegistryBackend is an injected interface for persistence (Sheets, SQLite, etc.).
  - refresh_cache() rebuilds the cache from the backend at the start of each run.
  - All mutations go to cache first, then backend (if configured).
  - Tests inject _InMemoryBackend; production injects a Sheets adapter.
"""

import datetime
from dataclasses import dataclass, field
from typing import Protocol

from evidence_scorer import ScoredEvidence


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_STATUSES: frozenset[str] = frozenset({"active", "suspended", "removed", "appealing"})

# Tier thresholds (inclusive lower bound on violation_count)
TIER_THRESHOLDS: dict[str, int] = {
    "tier1": 5,   # habitual offender
    "tier2": 2,   # repeat offender
    # tier3: anything below tier2 threshold (1 violation)
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class OffenderRecord:
    permanent_id: str           # platform-permanent identifier (never changes)
    platform: str               # e.g. "facebook"
    display_name: str           # last known display name (informational only)
    profile_url: str            # canonical profile/channel URL
    first_seen: str             # ISO 8601 — earliest known violation
    last_seen: str              # ISO 8601 — most recent violation
    violation_count: int        # total confirmed violations
    status: str                 # "active" | "suspended" | "removed" | "appealing"
    tier: str                   # "tier1" | "tier2" | "tier3"
    evidence_urls: list[str] = field(default_factory=list)   # per-violation evidence links
    notes: str = ""             # ops free-text + automated audit trail
    sheet_row: int | None = None  # backend row ID for in-place updates


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

class RegistryBackend(Protocol):
    """
    Persistence interface for OffenderRegistry.

    Implement this protocol to connect to Google Sheets (via sheet_writer.py),
    SQLite, or any other store. The registry only calls these three methods.

    All implementations must be idempotent on update_record — calling it
    twice with the same data must not create duplicate entries.
    """

    def save_record(self, record: OffenderRecord) -> int:
        """
        Persist a new offender record.

        Returns:
            Backend-assigned row/ID (stored as record.sheet_row for future updates).
        """
        ...

    def update_record(self, row_id: int, record: OffenderRecord) -> None:
        """Update the record identified by row_id in place."""
        ...

    def load_all_records(self) -> list[OffenderRecord]:
        """Load every record from the backend (called during refresh_cache)."""
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class OffenderRegistry:
    """
    CRUD interface for the verified offender registry.

    Usage pattern for daily pipeline:
        registry = OffenderRegistry(sheet_id=..., backend=SheetsBackend(...))
        registry.refresh_cache()          # load all existing offenders at run start
        for evidence in confirmed_hits:
            registry.upsert(evidence)     # insert or increment
        count = registry.lookup(pid, platform).violation_count
    """

    def __init__(
        self,
        sheet_id: str,
        cache_db_path: str | None = None,
        backend: RegistryBackend | None = None,
    ) -> None:
        self.sheet_id = sheet_id
        self.cache_db_path = cache_db_path   # reserved for future SQLite cache
        self._backend = backend
        self._cache: dict[str, OffenderRecord] = {}   # cache_key → record

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert(self, evidence: ScoredEvidence) -> OffenderRecord:
        """
        Insert a new offender or increment violation_count for an existing one.

        Call this for every detection whose verdict is "confirmed" or "likely".
        The caller (daily.sh orchestrator) decides which verdicts warrant registry
        entries — the registry itself does not filter by verdict.

        Behaviour:
          - New offender: inserts with violation_count=1, status="active".
          - Existing offender: increments violation_count, updates last_seen,
            re-calculates tier, appends evidence URL (no URL duplicates).
          - display_name and profile_url are always refreshed to the latest values.
          - last_seen only moves forward (protects against out-of-order evidence).

        Returns:
            The inserted or updated OffenderRecord (same object as in cache).
        """
        pid = evidence.identity.permanent_id
        platform = evidence.identity.platform
        evidence_ts = evidence.detection.detected_at
        evidence_url = evidence.detection.url

        existing = self.lookup(pid, platform)

        if existing is None:
            record = OffenderRecord(
                permanent_id=pid,
                platform=platform,
                display_name=evidence.identity.display_name,
                profile_url=evidence.identity.profile_url,
                first_seen=evidence_ts,
                last_seen=evidence_ts,
                violation_count=1,
                status="active",
                tier=self._calculate_tier(1),
                evidence_urls=[evidence_url],
            )
            if self._backend is not None:
                record.sheet_row = self._backend.save_record(record)
        else:
            existing.violation_count += 1
            # ISO 8601 strings sort lexicographically — max() is correct
            existing.last_seen = max(existing.last_seen, evidence_ts)
            existing.display_name = evidence.identity.display_name
            existing.profile_url = evidence.identity.profile_url
            existing.tier = self._calculate_tier(existing.violation_count)
            if evidence_url not in existing.evidence_urls:
                existing.evidence_urls.append(evidence_url)
            record = existing
            if self._backend is not None and record.sheet_row is not None:
                self._backend.update_record(record.sheet_row, record)

        self._cache[self._cache_key(platform, pid)] = record
        return record

    def lookup(self, permanent_id: str, platform: str) -> OffenderRecord | None:
        """
        Look up an offender by permanent platform ID.

        Cache-first: if the cache is empty and a backend is configured, triggers
        one full load (equivalent to refresh_cache). After that first load, only
        the cache is consulted — call refresh_cache() explicitly at run start to
        ensure the cache reflects the latest backend state.

        Returns:
            OffenderRecord, or None if this offender is not in the registry.
        """
        key = self._cache_key(platform, permanent_id)

        if key in self._cache:
            return self._cache[key]

        # Implicit full load only if the cache is completely empty.
        # If the cache has entries but is missing this key, the offender is new.
        if not self._cache and self._backend is not None:
            self.refresh_cache()
            return self._cache.get(key)

        return None

    def list_active(self, platform: str | None = None) -> list[OffenderRecord]:
        """
        Return all active offenders from the cache, sorted by last_seen descending.

        Args:
            platform: if given, filter to this platform only.

        Note: reflects the in-memory cache state. Call refresh_cache() first
        to ensure up-to-date results.
        """
        records = [r for r in self._cache.values() if r.status == "active"]
        if platform is not None:
            records = [r for r in records if r.platform == platform]
        return sorted(records, key=lambda r: r.last_seen, reverse=True)

    def update_status(
        self, permanent_id: str, platform: str, new_status: str
    ) -> None:
        """
        Transition an offender's status and append an audit entry to notes.

        Args:
            permanent_id: offender's permanent platform ID
            platform: platform slug
            new_status: one of VALID_STATUSES

        Raises:
            ValueError: if new_status is not a valid status string
            KeyError: if the offender is not in the cache (call refresh_cache first)
        """
        if new_status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status {new_status!r}. Must be one of {sorted(VALID_STATUSES)}"
            )

        key = self._cache_key(platform, permanent_id)
        record = self._cache.get(key)
        if record is None:
            raise KeyError(
                f"Offender not in cache: platform={platform!r}, "
                f"permanent_id={permanent_id!r}. Call refresh_cache() first."
            )

        old_status = record.status
        if old_status == new_status:
            return   # idempotent — no audit entry for no-op transitions

        record.status = new_status
        ts = _utc_now()
        transition = f"[{ts}] {old_status} → {new_status}"
        record.notes = f"{record.notes} | {transition}" if record.notes else transition

        if self._backend is not None and record.sheet_row is not None:
            self._backend.update_record(record.sheet_row, record)

    def export_for_legal(self, permanent_id: str) -> dict:
        """
        Export a complete evidence package for one offender for legal action.

        Searches the cache across all platforms for the given permanent_id.
        Returns a JSON-serializable dict.

        Raises:
            KeyError: if no record with this permanent_id exists in the cache.
        """
        matches = [r for r in self._cache.values() if r.permanent_id == permanent_id]
        if not matches:
            raise KeyError(
                f"No registry entry for permanent_id={permanent_id!r}. "
                "Call refresh_cache() or upsert() first."
            )
        record = matches[0]

        return {
            "permanent_id": record.permanent_id,
            "platform": record.platform,
            "display_name": record.display_name,
            "profile_url": record.profile_url,
            "first_seen": record.first_seen,
            "last_seen": record.last_seen,
            "violation_count": record.violation_count,
            "tier": record.tier,
            "status": record.status,
            "evidence_urls": list(record.evidence_urls),   # copy — don't expose mutable ref
            "notes": record.notes,
            "exported_at": _utc_now(),
        }

    def refresh_cache(self) -> None:
        """
        Reload the in-memory cache from the backend.

        Replaces all cached entries — any in-flight changes not yet written to
        the backend will be lost. Call at the start of each daily pipeline run,
        before any lookups or upserts.

        No-op if no backend is configured.
        """
        if self._backend is None:
            return
        self._cache.clear()
        for record in self._backend.load_all_records():
            self._cache[self._cache_key(record.platform, record.permanent_id)] = record

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(platform: str, permanent_id: str) -> str:
        """
        Composite cache key prevents collision between platforms that could
        share a numeric permanent_id by coincidence (e.g. FB numeric ID = TG chat ID).
        """
        return f"{platform}:{permanent_id}"

    def _calculate_tier(self, violation_count: int) -> str:
        """
        Assign tier based on violation count.

        Tier boundaries (inclusive lower bound):
          tier1 ≥ 5 violations  — habitual: priority takedown target
          tier2 ≥ 2 violations  — repeat: elevated scrutiny
          tier3 = 1 violation   — first offence: standard workflow
        """
        if violation_count >= TIER_THRESHOLDS["tier1"]:
            return "tier1"
        if violation_count >= TIER_THRESHOLDS["tier2"]:
            return "tier2"
        return "tier3"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
