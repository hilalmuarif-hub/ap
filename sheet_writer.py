"""
sheet_writer.py — Write pipeline results to Google Sheets.

Google Sheets is the operational dashboard for the anti-piracy team.
This module handles all writes: new detections, score updates, status changes.
All writes are idempotent — re-running the pipeline will not create duplicate rows.

Architecture:
  - SheetsBackend protocol abstracts over gspread (injected at construction).
  - GspreadBackend is the production adapter (wraps gspread, not imported by default).
  - SheetWriter implements RegistryBackend (from offender_registry.py) so that
    OffenderRegistry(backend=sheet_writer) works directly.
  - Tests inject _InMemorySheetsBackend without needing Google credentials.

Header convention:
  - Row 1 of every sheet is a header row (column names).
  - Data rows start at row 2. All row indices returned/stored are 1-based Sheets rows.
"""

import datetime
from dataclasses import dataclass
from typing import Protocol

from evidence_scorer import ScoredEvidence
from offender_registry import OffenderRecord


# ---------------------------------------------------------------------------
# Column layouts — order is the physical column order in Google Sheets.
# Changing order here changes the on-disk format; bump a migration note.
# ---------------------------------------------------------------------------

DETECTIONS_COLUMNS: list[str] = [
    "cluster_id",       # A — primary key for idempotency lookups
    "platform",
    "permanent_id",
    "display_name",
    "profile_url",
    "content_url",
    "content_title",
    "score",
    "verdict",
    "status",           # ops-owned: "new"|"reviewed"|"takedown_sent"|"resolved"
    "first_detected",
    "last_detected",
    "violation_count",  # from OffenderRecord at write time
    "tier",             # from OffenderRecord at write time
    "snapshot_url",     # Google Drive link to evidence HTML (filled later)
    "notes",            # ops-owned free text
]

REGISTRY_COLUMNS: list[str] = [
    "permanent_id",     # A — primary key (with platform)
    "platform",         # B — secondary key
    "display_name",
    "profile_url",
    "tier",
    "status",
    "violation_count",
    "first_seen",
    "last_seen",
    "notes",
]

RUNLOG_COLUMNS: list[str] = [
    "run_id",
    "started_at",
    "finished_at",
    "duration_seconds",
    "detections_found",
    "clusters_after_dedup",
    "confirmed_count",
    "likely_count",
    "possible_count",
    "ignored_count",
    "platforms_crawled",
    "errors",
]

# Columns that are updated in-place when a detection is re-seen on a later run.
# Status and notes are excluded — they are ops-owned and must not be overwritten.
_DETECTION_UPDATE_COLS: frozenset[str] = frozenset({
    "score", "verdict", "last_detected", "violation_count", "tier",
    "display_name", "content_title",   # backfill empty values from earlier runs
})


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SheetConfig:
    spreadsheet_id: str
    detections_sheet_name: str = "Detections"
    registry_sheet_name: str = "Registry"
    log_sheet_name: str = "RunLog"
    service_account_path: str = "service_account.json"


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

class SheetsBackend(Protocol):
    """
    Minimal interface over a Google Spreadsheet required by SheetWriter.

    All row/col indices are 1-based (Sheets convention).
    Row 1 is the header row; data starts at row 2.
    """

    def get_col_values(self, sheet_name: str, col: int) -> list[str]:
        """Return all values in `col` (1-based), including the header at index 0."""
        ...

    def get_all_rows(self, sheet_name: str) -> list[list[str]]:
        """Return every row as a list of strings, including the header at index 0."""
        ...

    def append_row(self, sheet_name: str, values: list) -> int:
        """Append `values` as a new row. Returns the 1-based row number written."""
        ...

    def update_row(self, sheet_name: str, row: int, values: list) -> None:
        """Overwrite the entire row at `row` (1-based) with `values`."""
        ...

    def update_cell(self, sheet_name: str, row: int, col: int, value: str) -> None:
        """Overwrite a single cell at (row, col), both 1-based."""
        ...


# ---------------------------------------------------------------------------
# Production gspread adapter
# ---------------------------------------------------------------------------

class GspreadBackend:
    """
    Production SheetsBackend backed by gspread.

    Supports two credential modes, auto-detected from the credential file:
      - Service account  : file has {"type": "service_account", ...}
      - OAuth installed  : file has {"installed": {...}}
                           Requires authorized_user_path with a saved refresh token.
                           Run once locally with gspread.oauth() to generate the token.

    Not imported at module load — gspread is optional at import time.
    Construct this only when a real Sheets connection is needed.
    """

    def __init__(
        self,
        service_account_path: str,
        spreadsheet_id: str,
        authorized_user_path: str | None = None,
    ) -> None:
        import json
        import gspread   # local import — keeps module importable without gspread

        with open(service_account_path) as f:
            creds = json.load(f)

        if creds.get("type") == "service_account":
            gc = gspread.service_account(filename=service_account_path)
        else:
            # OAuth: load refresh token directly — no browser needed (CI-safe).
            # gspread.oauth() opens a LocalServer flow when the token is absent,
            # which hangs in headless CI. Using Credentials directly bypasses that.
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request

            token_path = authorized_user_path or "authorized_user.json"
            with open(token_path) as tf:
                user_info = json.load(tf)

            google_creds = Credentials(
                token=None,
                refresh_token=user_info["refresh_token"],
                token_uri=user_info.get("token_uri", "https://oauth2.googleapis.com/token"),
                client_id=user_info["client_id"],
                client_secret=user_info["client_secret"],
                scopes=user_info.get("scopes", [
                    "https://www.googleapis.com/auth/spreadsheets",
                    "https://www.googleapis.com/auth/drive",
                ]),
            )
            google_creds.refresh(Request())
            gc = gspread.authorize(google_creds)

        self._spreadsheet = gc.open_by_key(spreadsheet_id)

    def _ws(self, sheet_name: str):
        return self._spreadsheet.worksheet(sheet_name)

    def get_col_values(self, sheet_name: str, col: int) -> list[str]:
        print(f"[SA_READ] get_col_values({sheet_name!r}, col={col})", flush=True)
        return self._retry(self._ws(sheet_name).col_values, col)

    def get_all_rows(self, sheet_name: str) -> list[list[str]]:
        print(f"[SA_READ] get_all_rows({sheet_name!r})", flush=True)
        return self._retry(self._ws(sheet_name).get_all_values)

    def append_row(self, sheet_name: str, values: list) -> int:
        ws = self._ws(sheet_name)
        print(f"[SA_READ] append_row pre-read({sheet_name!r})", flush=True)
        existing = self._retry(ws.get_all_values)
        self._retry(ws.append_row, [str(v) for v in values],
                    value_input_option="USER_ENTERED")
        return len(existing) + 1

    def append_rows_batch(self, sheet_name: str, rows: list[list]) -> None:
        """Append multiple rows in a single API call (no row-count read needed)."""
        ws = self._ws(sheet_name)
        str_rows = [[str(v) for v in row] for row in rows]
        print(f"[SA_WRITE] append_rows_batch({sheet_name!r}, {len(rows)} rows)", flush=True)
        self._retry(ws.append_rows, str_rows, value_input_option="USER_ENTERED")

    @staticmethod
    def _retry(fn, *args, **kwargs):
        """
        Call fn with args, retrying up to 3x on 429 quota errors.

        Initial delay is 65s — just over 1 minute — to reliably clear the
        Sheets API 'Read requests per minute per user' quota window.
        """
        import time
        delay = 65   # must exceed the 60s quota window
        last_exc = None
        for attempt in range(3):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                exc_str = str(exc)
                if ("429" in exc_str or "quota" in exc_str.lower()) and attempt < 2:
                    time.sleep(delay)
                    delay = min(delay * 2, 120)
                else:
                    raise
        raise last_exc   # all retries exhausted

    def update_row(self, sheet_name: str, row: int, values: list) -> None:
        self._retry(self._ws(sheet_name).update,
                    f"A{row}", [[str(v) for v in values]])

    def update_cell(self, sheet_name: str, row: int, col: int, value: str) -> None:
        self._retry(self._ws(sheet_name).update_cell, row, col, str(value))

    def update_cells_batch(
        self, sheet_name: str, updates: list[tuple[int, int, str]]
    ) -> None:
        """
        Batch update multiple cells in ONE Sheets API call.

        Args:
            updates: list of (row, col, value) — all 1-based.

        Each cell is sent as its own range entry in a single values.batchUpdate
        request, eliminating the N-per-row API calls that blow the quota.
        """
        if not updates:
            return
        import gspread.utils as gu
        ws = self._ws(sheet_name)
        data = [
            {"range": gu.rowcol_to_a1(r, c), "values": [[str(v)]]}
            for r, c, v in updates
        ]
        print(f"[SA_WRITE] update_cells_batch({sheet_name!r}, {len(data)} cells)", flush=True)
        self._retry(ws.batch_update, data, value_input_option="USER_ENTERED")


# ---------------------------------------------------------------------------
# SheetWriter
# ---------------------------------------------------------------------------

class SheetWriter:
    """
    Idempotent writer for anti-piracy Google Sheets.

    Also implements the RegistryBackend protocol so it can be passed directly to
    OffenderRegistry as its persistence backend:

        writer = SheetWriter(config, backend=GspreadBackend(...))
        registry = OffenderRegistry(sheet_id=config.spreadsheet_id, backend=writer)

    All public methods are safe to call multiple times with the same arguments.
    """

    def __init__(self, config: SheetConfig, backend: SheetsBackend) -> None:
        self.config = config
        self._backend = backend

    # ------------------------------------------------------------------
    # Detections sheet
    # ------------------------------------------------------------------

    def write_detection(
        self,
        cluster_id: str,
        evidence: ScoredEvidence,
        record: OffenderRecord | None = None,
    ) -> int:
        """
        Append a scored detection to the Detections sheet, or update it in-place
        if `cluster_id` already exists (idempotent on re-runs).

        On insert: all columns are written; `status` is set to "new".
        On update: only score, verdict, last_detected, violation_count, tier
                   are refreshed. Status and notes are never overwritten
                   (ops may have edited them since the last run).

        Args:
            cluster_id: stable ID from dedupe_cluster.cluster_id_for()
            evidence: scored detection from evidence_scorer
            record: matching OffenderRecord (supplies violation_count and tier);
                    columns are left blank if None

        Returns:
            1-based Sheets row number of the written or existing row.
        """
        existing_row = self._find_detection_row(cluster_id)

        if existing_row is not None:
            self._update_detection_in_place(existing_row, evidence, record)
            return existing_row

        values = self._row_to_detection_values(cluster_id, evidence, record)
        return self._backend.append_row(self.config.detections_sheet_name, values)

    def write_detections_batch(
        self,
        items: list[tuple[str, "ScoredEvidence", "OffenderRecord | None"]],
    ) -> int:
        """
        Write multiple detections in as few API calls as possible.

        Algorithm:
          1. Read all existing cluster_ids once (1 read API call)
          2. Split items into: new (append) and existing (update in-place)
          3. Append all new rows in one batch API call
          4. Update existing rows individually (unavoidable per-row calls)

        This reduces API calls from O(2N) to O(N_existing + 2) for typical
        first-run scenarios where most detections are new.

        Returns:
            Number of rows written (new inserts only).
        """
        if not items:
            return 0

        # Build cluster_id → row_number mapping in ONE read call.
        # Previously _find_detection_row() was called per existing detection
        # (N additional reads), blowing past the 60 reads/min quota.
        id_to_row = self._get_cluster_id_to_row_map()
        new_rows: list[list] = []
        cell_updates: list[tuple[int, int, str]] = []   # (row, col, value)

        for cluster_id, evidence, record in items:
            if cluster_id in id_to_row:
                # Collect updates for existing row — written in one batch call below
                cell_updates.extend(
                    self._detection_cell_updates(id_to_row[cluster_id], evidence, record)
                )
            else:
                new_rows.append(
                    self._row_to_detection_values(cluster_id, evidence, record)
                )

        # Flush existing-row updates in ONE API call (avoids N×7 individual calls)
        if cell_updates:
            if hasattr(self._backend, "update_cells_batch"):
                self._backend.update_cells_batch(
                    self.config.detections_sheet_name, cell_updates
                )
            else:
                sheet = self.config.detections_sheet_name
                for row, col_idx, value in cell_updates:
                    self._backend.update_cell(sheet, row, col_idx, value)

        # Append new rows in ONE API call
        if new_rows:
            if hasattr(self._backend, "append_rows_batch"):
                self._backend.append_rows_batch(
                    self.config.detections_sheet_name, new_rows
                )
            else:
                for row in new_rows:
                    self._backend.append_row(self.config.detections_sheet_name, row)

        return len(new_rows)

    def _detection_cell_updates(
        self,
        row: int,
        evidence: ScoredEvidence,
        record: OffenderRecord | None,
    ) -> list[tuple[int, int, str]]:
        """
        Return (row, col, value) tuples for mutable columns of an existing detection.

        Does NOT call the backend — the caller collects these and sends them as
        a single batch_update to avoid N×7 individual API calls blowing the quota.
        """
        updates: dict[str, str] = {
            "score":           str(evidence.score),
            "verdict":         evidence.verdict,
            "last_detected":   evidence.detection.detected_at,
            "violation_count": str(record.violation_count) if record else "",
            "tier":            record.tier if record else "",
        }
        # Backfill display_name and content_title when newly available
        if evidence.identity.display_name:
            updates["display_name"] = evidence.identity.display_name
        if evidence.detection.title:
            updates["content_title"] = evidence.detection.title

        return [
            (row, DETECTIONS_COLUMNS.index(col_name) + 1, value)
            for col_name, value in updates.items()
        ]

    # ------------------------------------------------------------------
    # Registry sheet — implements RegistryBackend protocol
    # ------------------------------------------------------------------

    def write_offender(self, record: OffenderRecord) -> int:
        """
        Upsert an offender record to the Registry sheet.

        Matches on (permanent_id, platform). If found, updates mutable fields
        in-place. If not found, appends a new row.

        Returns:
            1-based Sheets row of the written or existing row.
            Also sets record.sheet_row to this value.
        """
        existing_row = self._find_registry_row(record.permanent_id, record.platform)

        if existing_row is not None:
            self._backend.update_row(
                self.config.registry_sheet_name,
                existing_row,
                self._row_to_registry_values(record),
            )
            record.sheet_row = existing_row
            return existing_row

        row = self._backend.append_row(
            self.config.registry_sheet_name,
            self._row_to_registry_values(record),
        )
        record.sheet_row = row
        return row

    # RegistryBackend.save_record
    def save_record(self, record: OffenderRecord) -> int:
        return self.write_offender(record)

    # RegistryBackend.update_record
    def update_record(self, row_id: int, record: OffenderRecord) -> None:
        self._backend.update_row(
            self.config.registry_sheet_name,
            row_id,
            self._row_to_registry_values(record),
        )

    # RegistryBackend.load_all_records
    def load_all_records(self) -> list[OffenderRecord]:
        """
        Deserialize all rows from the Registry sheet into OffenderRecord objects.

        Row 1 (header) is skipped. Empty rows are skipped.
        `sheet_row` on each record is set to its 1-based Sheets row number.
        """
        rows = self._backend.get_all_rows(self.config.registry_sheet_name)
        records: list[OffenderRecord] = []
        for sheet_row_idx, row in enumerate(rows):
            if sheet_row_idx == 0:
                continue   # skip header
            if not any(row):
                continue   # skip empty rows
            records.append(self._record_from_registry_row(row, sheet_row=sheet_row_idx + 1))
        return records

    # ------------------------------------------------------------------
    # RunLog sheet
    # ------------------------------------------------------------------

    def write_run_log(
        self,
        run_id: str,
        started_at: str,
        finished_at: str,
        stats: dict,
    ) -> None:
        """
        Append a summary row to the RunLog sheet.

        Expected stats keys: detections_found, clusters_after_dedup,
        confirmed_count, likely_count, possible_count, ignored_count,
        platforms_crawled, errors. Missing keys are written as empty string.
        """
        try:
            started = datetime.datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            finished = datetime.datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
            duration = round((finished - started).total_seconds(), 1)
        except (ValueError, AttributeError):
            duration = ""

        values = [
            run_id,
            started_at,
            finished_at,
            duration,
            stats.get("detections_found", ""),
            stats.get("clusters_after_dedup", ""),
            stats.get("confirmed_count", ""),
            stats.get("likely_count", ""),
            stats.get("possible_count", ""),
            stats.get("ignored_count", ""),
            stats.get("platforms_crawled", ""),
            stats.get("errors", ""),
        ]
        assert len(values) == len(RUNLOG_COLUMNS)
        self._backend.append_row(self.config.log_sheet_name, values)

    # ------------------------------------------------------------------
    # Detection status updates (ops workflow)
    # ------------------------------------------------------------------

    def update_status(
        self,
        cluster_id: str,
        new_status: str,
        note: str = "",
    ) -> None:
        """
        Update the ops-owned status column for a detection row.

        Also updates the notes column if `note` is provided.

        Raises:
            KeyError: if cluster_id is not found in the Detections sheet.
        """
        row = self._find_detection_row(cluster_id)
        if row is None:
            raise KeyError(
                f"cluster_id {cluster_id!r} not found in "
                f"'{self.config.detections_sheet_name}' sheet."
            )

        sheet = self.config.detections_sheet_name
        status_col = DETECTIONS_COLUMNS.index("status") + 1
        self._backend.update_cell(sheet, row, status_col, new_status)

        if note:
            notes_col = DETECTIONS_COLUMNS.index("notes") + 1
            self._backend.update_cell(sheet, row, notes_col, note)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_detection_row(self, cluster_id: str) -> int | None:
        """
        Return the 1-based Sheets row containing `cluster_id`, or None.

        Reads column A (cluster_id). Header at index 0 maps to row 1;
        data at index k maps to row k+1.
        """
        col_a = self._backend.get_col_values(
            self.config.detections_sheet_name,
            col=DETECTIONS_COLUMNS.index("cluster_id") + 1,
        )
        try:
            return col_a.index(cluster_id) + 1   # 0-based → 1-based
        except ValueError:
            return None

    def _find_registry_row(self, permanent_id: str, platform: str) -> int | None:
        """
        Return the 1-based Sheets row matching (permanent_id, platform), or None.

        Reads columns A (permanent_id) and B (platform) to build the composite key.
        """
        rows = self._backend.get_all_rows(self.config.registry_sheet_name)
        pid_col = REGISTRY_COLUMNS.index("permanent_id")   # 0-based
        plat_col = REGISTRY_COLUMNS.index("platform")       # 0-based
        for sheet_row_idx, row in enumerate(rows):
            if sheet_row_idx == 0:
                continue   # skip header
            if len(row) > max(pid_col, plat_col):
                if row[pid_col] == permanent_id and row[plat_col] == platform:
                    return sheet_row_idx + 1   # 0-based → 1-based
        return None

    def _row_to_detection_values(
        self,
        cluster_id: str,
        evidence: ScoredEvidence,
        record: OffenderRecord | None,
    ) -> list:
        """Map arguments to an ordered list matching DETECTIONS_COLUMNS."""
        return [
            cluster_id,
            evidence.detection.platform,
            evidence.identity.permanent_id,
            evidence.identity.display_name,
            evidence.identity.profile_url,
            evidence.detection.url,
            evidence.detection.title,
            evidence.score,
            evidence.verdict,
            "new",                                              # status — ops-owned, start as "new"
            evidence.detection.detected_at,                     # first_detected
            evidence.detection.detected_at,                     # last_detected — same on insert
            record.violation_count if record else "",
            record.tier if record else "",
            "",                                                 # snapshot_url — filled later
            "",                                                 # notes — ops-owned, start blank
        ]

    def _row_to_registry_values(self, record: OffenderRecord) -> list:
        """Map an OffenderRecord to an ordered list matching REGISTRY_COLUMNS."""
        return [
            record.permanent_id,
            record.platform,
            record.display_name,
            record.profile_url,
            record.tier,
            record.status,
            record.violation_count,
            record.first_seen,
            record.last_seen,
            record.notes,
        ]

    def _record_from_registry_row(self, row: list[str], sheet_row: int) -> OffenderRecord:
        """
        Deserialize one Registry sheet row into an OffenderRecord.

        Pads short rows with empty strings so index access never raises.
        """
        padded = list(row) + [""] * len(REGISTRY_COLUMNS)   # ensure enough length

        def col(name: str) -> str:
            return padded[REGISTRY_COLUMNS.index(name)]

        return OffenderRecord(
            permanent_id=col("permanent_id"),
            platform=col("platform"),
            display_name=col("display_name"),
            profile_url=col("profile_url"),
            tier=col("tier"),
            status=col("status") or "active",
            violation_count=int(col("violation_count") or 0),
            first_seen=col("first_seen"),
            last_seen=col("last_seen"),
            notes=col("notes"),
            sheet_row=sheet_row,
        )

    def _get_all_cluster_ids(self) -> set[str]:
        """Fetch set of existing cluster_ids (1 read call). Header skipped."""
        col_a = self._backend.get_col_values(
            self.config.detections_sheet_name,
            col=DETECTIONS_COLUMNS.index("cluster_id") + 1,
        )
        return set(col_a[1:])

    def _get_cluster_id_to_row_map(self) -> dict[str, int]:
        """
        Fetch mapping {cluster_id: 1-based-row-number} in ONE read API call.

        Eliminates the N additional _find_detection_row() calls that
        write_detections_batch() previously made for existing detections,
        reducing API reads from O(N_existing + 1) to O(1).
        """
        col_a = self._backend.get_col_values(
            self.config.detections_sheet_name,
            col=DETECTIONS_COLUMNS.index("cluster_id") + 1,
        )
        # col_a[0] = header, col_a[k] = data at Sheets row k+1
        return {
            cid: idx + 1
            for idx, cid in enumerate(col_a)
            if idx > 0 and cid   # skip header and empty cells
        }
