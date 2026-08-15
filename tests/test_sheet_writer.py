"""Tests for sheet_writer.py."""

import pytest
from detection import RawDetection
from identity import OffenderIdentity
from evidence_scorer import ScoredEvidence
from offender_registry import OffenderRecord
from sheet_writer import (
    DETECTIONS_COLUMNS,
    REGISTRY_COLUMNS,
    RUNLOG_COLUMNS,
    SheetConfig,
    SheetWriter,
)


# ---------------------------------------------------------------------------
# In-memory backend (implements SheetsBackend protocol)
# ---------------------------------------------------------------------------

class _InMemorySheetsBackend:
    """
    Simulates a Google Spreadsheet with multiple named sheets.
    Each sheet is a list of rows; row 0 is always the header.
    """

    def __init__(self, sheet_names: list[str]) -> None:
        # Initialise with a header row for each sheet
        headers = {
            "Detections": [DETECTIONS_COLUMNS],
            "Registry":   [REGISTRY_COLUMNS],
            "RunLog":      [RUNLOG_COLUMNS],
        }
        self._sheets: dict[str, list[list[str]]] = {}
        for name in sheet_names:
            self._sheets[name] = [list(headers.get(name, ["header"]))]

    # SheetsBackend interface -------------------------------------------------

    def get_col_values(self, sheet_name: str, col: int) -> list[str]:
        rows = self._sheets[sheet_name]
        col_idx = col - 1   # 1-based → 0-based
        return [row[col_idx] if col_idx < len(row) else "" for row in rows]

    def get_all_rows(self, sheet_name: str) -> list[list[str]]:
        return [list(r) for r in self._sheets[sheet_name]]   # defensive copy

    def append_row(self, sheet_name: str, values: list) -> int:
        str_values = [str(v) for v in values]
        self._sheets[sheet_name].append(str_values)
        return len(self._sheets[sheet_name])   # 1-based row number

    def update_row(self, sheet_name: str, row: int, values: list) -> None:
        self._sheets[sheet_name][row - 1] = [str(v) for v in values]

    def update_cell(self, sheet_name: str, row: int, col: int, value: str) -> None:
        target_row = self._sheets[sheet_name][row - 1]
        col_idx = col - 1
        while len(target_row) <= col_idx:
            target_row.append("")
        target_row[col_idx] = str(value)

    # Inspection helpers ------------------------------------------------------

    def data_rows(self, sheet_name: str) -> list[list[str]]:
        """Return data rows only (skip header at index 0)."""
        return self._sheets[sheet_name][1:]

    def row_count(self, sheet_name: str) -> int:
        """Number of data rows (excludes header)."""
        return len(self._sheets[sheet_name]) - 1


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SHEET_NAMES = ["Detections", "Registry", "RunLog"]


def make_backend() -> _InMemorySheetsBackend:
    return _InMemorySheetsBackend(SHEET_NAMES)


def make_writer(backend: _InMemorySheetsBackend | None = None) -> SheetWriter:
    config = SheetConfig(spreadsheet_id="test_sheet_id")
    return SheetWriter(config, backend=backend or make_backend())


def make_evidence(
    permanent_id: str = "pid_001",
    platform: str = "facebook",
    score: int = 85,
    verdict: str = "confirmed",
    url: str = "https://fb.com/watch?v=123",
    title: str = "Liga Champions Vidio",
    detected_at: str = "2025-01-15T08:00:00Z",
) -> ScoredEvidence:
    detection = RawDetection(
        platform=platform, url=url, title=title,
        channel_id=permanent_id, channel_name="Pirate",
        snapshot_html="<html>evidence</html>",
        detected_at=detected_at, query_used="liga champions",
        extra={},
    )
    identity = OffenderIdentity(
        platform=platform, permanent_id=permanent_id,
        display_name="Pirate Channel", profile_url=f"https://fb.com/{permanent_id}",
        resolved_at=detected_at, confidence=0.95, metadata={},
    )
    return ScoredEvidence(
        detection=detection, identity=identity,
        score=score, breakdown={}, verdict=verdict, scored_at=detected_at,
    )


def make_record(
    permanent_id: str = "pid_001",
    platform: str = "facebook",
    violation_count: int = 1,
    tier: str = "tier3",
    status: str = "active",
    sheet_row: int | None = None,
) -> OffenderRecord:
    return OffenderRecord(
        permanent_id=permanent_id, platform=platform,
        display_name="Pirate Channel", profile_url=f"https://fb.com/{permanent_id}",
        first_seen="2025-01-15T08:00:00Z", last_seen="2025-01-15T08:00:00Z",
        violation_count=violation_count, status=status, tier=tier,
        sheet_row=sheet_row,
    )


# ---------------------------------------------------------------------------
# Column layout invariants
# ---------------------------------------------------------------------------

class TestColumnLayouts:
    def test_detections_has_cluster_id_first(self):
        assert DETECTIONS_COLUMNS[0] == "cluster_id"

    def test_registry_has_permanent_id_first(self):
        assert REGISTRY_COLUMNS[0] == "permanent_id"

    def test_registry_has_platform_second(self):
        assert REGISTRY_COLUMNS[1] == "platform"

    def test_no_duplicate_detection_cols(self):
        assert len(DETECTIONS_COLUMNS) == len(set(DETECTIONS_COLUMNS))

    def test_no_duplicate_registry_cols(self):
        assert len(REGISTRY_COLUMNS) == len(set(REGISTRY_COLUMNS))

    def test_detection_row_length_matches_columns(self):
        writer = make_writer()
        ev = make_evidence()
        values = writer._row_to_detection_values("aabbccdd11223344", ev, make_record())
        assert len(values) == len(DETECTIONS_COLUMNS)

    def test_registry_row_length_matches_columns(self):
        writer = make_writer()
        values = writer._row_to_registry_values(make_record())
        assert len(values) == len(REGISTRY_COLUMNS)


# ---------------------------------------------------------------------------
# _row_to_detection_values
# ---------------------------------------------------------------------------

class TestRowToDetectionValues:
    def setup_method(self):
        self.writer = make_writer()
        self.ev = make_evidence(score=90, verdict="confirmed")
        self.rec = make_record(violation_count=3, tier="tier2")
        self.cid = "aabbccdd11223344"
        self.values = self.writer._row_to_detection_values(self.cid, self.ev, self.rec)

    def test_cluster_id_at_position_0(self):
        assert self.values[DETECTIONS_COLUMNS.index("cluster_id")] == self.cid

    def test_score_correct(self):
        assert self.values[DETECTIONS_COLUMNS.index("score")] == 90

    def test_verdict_correct(self):
        assert self.values[DETECTIONS_COLUMNS.index("verdict")] == "confirmed"

    def test_status_is_new_on_insert(self):
        assert self.values[DETECTIONS_COLUMNS.index("status")] == "new"

    def test_notes_empty_on_insert(self):
        assert self.values[DETECTIONS_COLUMNS.index("notes")] == ""

    def test_violation_count_from_record(self):
        assert self.values[DETECTIONS_COLUMNS.index("violation_count")] == 3

    def test_tier_from_record(self):
        assert self.values[DETECTIONS_COLUMNS.index("tier")] == "tier2"

    def test_no_record_blanks_violation_fields(self):
        values = self.writer._row_to_detection_values(self.cid, self.ev, None)
        assert values[DETECTIONS_COLUMNS.index("violation_count")] == ""
        assert values[DETECTIONS_COLUMNS.index("tier")] == ""

    def test_first_and_last_detected_same_on_insert(self):
        first = self.values[DETECTIONS_COLUMNS.index("first_detected")]
        last  = self.values[DETECTIONS_COLUMNS.index("last_detected")]
        assert first == last == "2025-01-15T08:00:00Z"

    def test_platform_correct(self):
        assert self.values[DETECTIONS_COLUMNS.index("platform")] == "facebook"

    def test_content_url_correct(self):
        assert self.values[DETECTIONS_COLUMNS.index("content_url")] == "https://fb.com/watch?v=123"


# ---------------------------------------------------------------------------
# _row_to_registry_values
# ---------------------------------------------------------------------------

class TestRowToRegistryValues:
    def setup_method(self):
        self.writer = make_writer()
        self.rec = make_record(violation_count=5, tier="tier1", status="suspended")
        self.values = self.writer._row_to_registry_values(self.rec)

    def test_permanent_id_at_0(self):
        assert self.values[REGISTRY_COLUMNS.index("permanent_id")] == "pid_001"

    def test_platform_at_1(self):
        assert self.values[REGISTRY_COLUMNS.index("platform")] == "facebook"

    def test_violation_count(self):
        assert self.values[REGISTRY_COLUMNS.index("violation_count")] == 5

    def test_tier(self):
        assert self.values[REGISTRY_COLUMNS.index("tier")] == "tier1"

    def test_status(self):
        assert self.values[REGISTRY_COLUMNS.index("status")] == "suspended"


# ---------------------------------------------------------------------------
# write_detection — insert
# ---------------------------------------------------------------------------

class TestWriteDetectionInsert:
    def test_appends_row_to_sheet(self):
        backend = make_backend()
        writer = make_writer(backend)
        writer.write_detection("cid_001", make_evidence(), make_record())
        assert backend.row_count("Detections") == 1

    def test_returns_row_number(self):
        writer = make_writer()
        row = writer.write_detection("cid_001", make_evidence(), make_record())
        assert row == 2   # row 1 = header, row 2 = first data row

    def test_cluster_id_written_in_col_a(self):
        backend = make_backend()
        writer = make_writer(backend)
        writer.write_detection("cid_abc123", make_evidence(), make_record())
        col_a = backend.get_col_values("Detections", 1)
        assert "cid_abc123" in col_a

    def test_score_written(self):
        backend = make_backend()
        writer = make_writer(backend)
        ev = make_evidence(score=77, verdict="likely")
        writer.write_detection("cid_001", ev, make_record())
        row = backend.data_rows("Detections")[0]
        assert row[DETECTIONS_COLUMNS.index("score")] == "77"

    def test_status_is_new(self):
        backend = make_backend()
        writer = make_writer(backend)
        writer.write_detection("cid_001", make_evidence(), make_record())
        row = backend.data_rows("Detections")[0]
        assert row[DETECTIONS_COLUMNS.index("status")] == "new"


# ---------------------------------------------------------------------------
# write_detection — idempotency (re-run)
# ---------------------------------------------------------------------------

class TestWriteDetectionIdempotent:
    def test_second_write_does_not_duplicate(self):
        backend = make_backend()
        writer = make_writer(backend)
        writer.write_detection("cid_001", make_evidence(score=70), make_record())
        writer.write_detection("cid_001", make_evidence(score=85), make_record())
        assert backend.row_count("Detections") == 1

    def test_second_write_returns_same_row(self):
        writer = make_writer()
        r1 = writer.write_detection("cid_001", make_evidence(), make_record())
        r2 = writer.write_detection("cid_001", make_evidence(), make_record())
        assert r1 == r2

    def test_score_updated_on_re_run(self):
        backend = make_backend()
        writer = make_writer(backend)
        writer.write_detection("cid_001", make_evidence(score=70), make_record())
        writer.write_detection("cid_001", make_evidence(score=90), make_record())
        row = backend.data_rows("Detections")[0]
        assert row[DETECTIONS_COLUMNS.index("score")] == "90"

    def test_status_not_overwritten_on_re_run(self):
        backend = make_backend()
        writer = make_writer(backend)
        writer.write_detection("cid_001", make_evidence(), make_record())
        # Simulate ops changing the status to "takedown_sent"
        status_col = DETECTIONS_COLUMNS.index("status") + 1
        backend.update_cell("Detections", 2, status_col, "takedown_sent")
        # Re-run pipeline
        writer.write_detection("cid_001", make_evidence(score=90), make_record())
        row = backend.data_rows("Detections")[0]
        assert row[DETECTIONS_COLUMNS.index("status")] == "takedown_sent"

    def test_notes_not_overwritten_on_re_run(self):
        backend = make_backend()
        writer = make_writer(backend)
        writer.write_detection("cid_001", make_evidence(), make_record())
        notes_col = DETECTIONS_COLUMNS.index("notes") + 1
        backend.update_cell("Detections", 2, notes_col, "ops note here")
        writer.write_detection("cid_001", make_evidence(), make_record())
        row = backend.data_rows("Detections")[0]
        assert row[DETECTIONS_COLUMNS.index("notes")] == "ops note here"

    def test_different_cluster_ids_stay_separate(self):
        backend = make_backend()
        writer = make_writer(backend)
        writer.write_detection("cid_001", make_evidence(), make_record())
        writer.write_detection("cid_002", make_evidence(), make_record())
        assert backend.row_count("Detections") == 2


# ---------------------------------------------------------------------------
# write_offender / RegistryBackend
# ---------------------------------------------------------------------------

class TestWriteOffender:
    def test_appends_new_record(self):
        backend = make_backend()
        writer = make_writer(backend)
        writer.write_offender(make_record())
        assert backend.row_count("Registry") == 1

    def test_returns_row_number(self):
        writer = make_writer()
        row = writer.write_offender(make_record())
        assert row == 2

    def test_sets_sheet_row_on_record(self):
        writer = make_writer()
        rec = make_record()
        writer.write_offender(rec)
        assert rec.sheet_row == 2

    def test_upsert_does_not_duplicate(self):
        backend = make_backend()
        writer = make_writer(backend)
        writer.write_offender(make_record(violation_count=1))
        writer.write_offender(make_record(violation_count=2))
        assert backend.row_count("Registry") == 1

    def test_upsert_updates_violation_count(self):
        backend = make_backend()
        writer = make_writer(backend)
        writer.write_offender(make_record(violation_count=1))
        writer.write_offender(make_record(violation_count=3))
        row = backend.data_rows("Registry")[0]
        assert row[REGISTRY_COLUMNS.index("violation_count")] == "3"

    def test_different_platforms_stay_separate(self):
        backend = make_backend()
        writer = make_writer(backend)
        writer.write_offender(make_record(platform="facebook"))
        writer.write_offender(make_record(platform="youtube"))
        assert backend.row_count("Registry") == 2

    def test_save_record_delegates_to_write_offender(self):
        backend = make_backend()
        writer = make_writer(backend)
        rec = make_record()
        row = writer.save_record(rec)
        assert row == 2
        assert backend.row_count("Registry") == 1

    def test_update_record_overwrites_row(self):
        backend = make_backend()
        writer = make_writer(backend)
        rec = make_record(violation_count=1)
        writer.write_offender(rec)
        rec.violation_count = 5
        writer.update_record(rec.sheet_row, rec)
        row = backend.data_rows("Registry")[0]
        assert row[REGISTRY_COLUMNS.index("violation_count")] == "5"


# ---------------------------------------------------------------------------
# load_all_records
# ---------------------------------------------------------------------------

class TestLoadAllRecords:
    def test_empty_sheet_returns_empty(self):
        writer = make_writer()
        assert writer.load_all_records() == []

    def test_loads_written_records(self):
        writer = make_writer()
        writer.write_offender(make_record(permanent_id="pid_A"))
        writer.write_offender(make_record(permanent_id="pid_B", platform="youtube"))
        records = writer.load_all_records()
        assert len(records) == 2

    def test_sheet_row_set_correctly(self):
        writer = make_writer()
        writer.write_offender(make_record())
        records = writer.load_all_records()
        assert records[0].sheet_row == 2   # header=row1, data=row2

    def test_violation_count_parsed_as_int(self):
        writer = make_writer()
        writer.write_offender(make_record(violation_count=7))
        records = writer.load_all_records()
        assert records[0].violation_count == 7
        assert isinstance(records[0].violation_count, int)

    def test_permanent_id_preserved(self):
        writer = make_writer()
        writer.write_offender(make_record(permanent_id="unique_pid_xyz"))
        records = writer.load_all_records()
        assert records[0].permanent_id == "unique_pid_xyz"

    def test_roundtrip_tier_and_status(self):
        writer = make_writer()
        writer.write_offender(make_record(tier="tier1", status="suspended"))
        records = writer.load_all_records()
        assert records[0].tier == "tier1"
        assert records[0].status == "suspended"


# ---------------------------------------------------------------------------
# write_run_log
# ---------------------------------------------------------------------------

class TestWriteRunLog:
    def test_appends_row(self):
        backend = make_backend()
        writer = make_writer(backend)
        writer.write_run_log(
            "run_001", "2025-01-15T08:00:00Z", "2025-01-15T08:30:00Z",
            {"confirmed_count": 5, "detections_found": 100},
        )
        assert backend.row_count("RunLog") == 1

    def test_run_id_written(self):
        backend = make_backend()
        writer = make_writer(backend)
        writer.write_run_log("run_abc", "2025-01-15T08:00:00Z", "2025-01-15T09:00:00Z", {})
        row = backend.data_rows("RunLog")[0]
        assert row[RUNLOG_COLUMNS.index("run_id")] == "run_abc"

    def test_duration_calculated(self):
        backend = make_backend()
        writer = make_writer(backend)
        writer.write_run_log(
            "run_x", "2025-01-15T08:00:00Z", "2025-01-15T08:01:30Z", {}
        )
        row = backend.data_rows("RunLog")[0]
        assert row[RUNLOG_COLUMNS.index("duration_seconds")] == "90.0"

    def test_missing_stats_written_as_empty(self):
        backend = make_backend()
        writer = make_writer(backend)
        writer.write_run_log("run_x", "2025-01-15T08:00:00Z", "2025-01-15T08:00:00Z", {})
        row = backend.data_rows("RunLog")[0]
        assert row[RUNLOG_COLUMNS.index("confirmed_count")] == ""

    def test_row_length_matches_columns(self):
        backend = make_backend()
        writer = make_writer(backend)
        writer.write_run_log("run_x", "2025-01-15T08:00:00Z", "2025-01-15T08:00:00Z",
                             {"errors": "none"})
        row = backend.data_rows("RunLog")[0]
        assert len(row) == len(RUNLOG_COLUMNS)


# ---------------------------------------------------------------------------
# update_status
# ---------------------------------------------------------------------------

class TestUpdateStatus:
    def test_status_updated(self):
        backend = make_backend()
        writer = make_writer(backend)
        writer.write_detection("cid_001", make_evidence(), make_record())
        writer.update_status("cid_001", "takedown_sent")
        row = backend.data_rows("Detections")[0]
        assert row[DETECTIONS_COLUMNS.index("status")] == "takedown_sent"

    def test_note_written_when_provided(self):
        backend = make_backend()
        writer = make_writer(backend)
        writer.write_detection("cid_001", make_evidence(), make_record())
        writer.update_status("cid_001", "reviewed", note="Confirmed by ops team")
        row = backend.data_rows("Detections")[0]
        assert row[DETECTIONS_COLUMNS.index("notes")] == "Confirmed by ops team"

    def test_no_note_does_not_clear_notes(self):
        backend = make_backend()
        writer = make_writer(backend)
        writer.write_detection("cid_001", make_evidence(), make_record())
        notes_col = DETECTIONS_COLUMNS.index("notes") + 1
        backend.update_cell("Detections", 2, notes_col, "existing ops note")
        writer.update_status("cid_001", "reviewed")   # no note= provided
        row = backend.data_rows("Detections")[0]
        assert row[DETECTIONS_COLUMNS.index("notes")] == "existing ops note"

    def test_unknown_cluster_raises_key_error(self):
        writer = make_writer()
        with pytest.raises(KeyError, match="nonexistent_cid"):
            writer.update_status("nonexistent_cid", "reviewed")


# ---------------------------------------------------------------------------
# Integration: SheetWriter as RegistryBackend
# ---------------------------------------------------------------------------

class TestSheetWriterAsRegistryBackend:
    def test_satisfies_registry_backend_protocol(self):
        from offender_registry import OffenderRegistry, RegistryBackend
        writer = make_writer()
        # Protocol check via isinstance with runtime_checkable
        # (RegistryBackend is defined as Protocol, not runtime_checkable,
        # so we check duck-typing by calling the required methods directly)
        assert hasattr(writer, "save_record")
        assert hasattr(writer, "update_record")
        assert hasattr(writer, "load_all_records")

    def test_registry_upsert_writes_to_sheet(self):
        from offender_registry import OffenderRegistry
        from evidence_scorer import ScoredEvidence
        backend = make_backend()
        writer = make_writer(backend)
        registry = OffenderRegistry(
            sheet_id="test_sheet_id",
            backend=writer,
        )

        ev = make_evidence()
        registry.upsert(ev)

        assert backend.row_count("Registry") == 1
        rows = backend.data_rows("Registry")
        assert rows[0][REGISTRY_COLUMNS.index("permanent_id")] == "pid_001"

    def test_registry_refresh_cache_reads_from_sheet(self):
        from offender_registry import OffenderRegistry
        backend = make_backend()
        writer = make_writer(backend)

        # Write directly to sheet (simulating prior run)
        writer.write_offender(make_record(permanent_id="pid_prior", violation_count=3))

        # New registry backed by same writer — refresh should load that record
        registry = OffenderRegistry(sheet_id="test_sheet_id", backend=writer)
        registry.refresh_cache()

        record = registry.lookup("pid_prior", "facebook")
        assert record is not None
        assert record.violation_count == 3
