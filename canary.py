"""
canary.py — Health check and monitoring for the anti-piracy pipeline.

Runs before and after each pipeline execution. Canary failures do NOT stop
the pipeline — they alert the operator and let the run continue or abort
based on severity as determined by the caller (daily.sh).

Check severity:
  "ok"       — within expected bounds
  "degraded" — anomalous but not data-loss level; ops should investigate
  "critical" — data loss or crawler silence; must page immediately
  "skipped"  — insufficient data to evaluate (e.g. no detections yet)
"""

import datetime
import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class HealthReport:
    run_id: str
    timestamp: str                       # ISO 8601 UTC
    checks: list[dict] = field(default_factory=list)
    overall_status: str = "unknown"      # "healthy" | "degraded" | "critical"
    alerts_sent: list[str] = field(default_factory=list)


@dataclass
class PipelineStats:
    """Collected statistics from a single pipeline run."""
    detections_raw: int = 0
    clusters_after_dedup: int = 0
    confirmed_count: int = 0
    likely_count: int = 0
    possible_count: int = 0
    ignored_count: int = 0
    errors: list[str] = field(default_factory=list)
    platforms_crawled: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    sheet_write_errors: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Severity order for computing overall_status
_SEVERITY: dict[str, int] = {
    "ok": 0, "skipped": 0, "healthy": 0,
    "degraded": 1,
    "critical": 2,
}

# Emoji prefixes used in alert messages
_STATUS_EMOJI: dict[str, str] = {
    "ok": "✅",
    "skipped": "⏭️",
    "degraded": "⚠️",
    "critical": "❌",
}

# Slack attachment colours
_ALERT_COLOUR: dict[str, str] = {
    "critical": "#CC0000",
    "degraded": "#E8A000",
    "healthy": "#36A64F",
}


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _worst_status(checks: list[dict]) -> str:
    """Return the most severe overall status across all check results."""
    worst = max((_SEVERITY.get(c.get("status", "ok"), 0) for c in checks), default=0)
    if worst >= 2:
        return "critical"
    if worst >= 1:
        return "degraded"
    return "healthy"


def _check_result(
    name: str,
    status: str,
    message: str,
    detail: dict | None = None,
) -> dict:
    return {"name": name, "status": status, "message": message, "detail": detail or {}}


# ---------------------------------------------------------------------------
# CanaryChecker
# ---------------------------------------------------------------------------

class CanaryChecker:
    """
    Runs a battery of health checks against PipelineStats.

    All threshold parameters are injectable so tests don't need to monkeypatch
    environment variables. Production callers can read thresholds from env and
    pass them in.

    Args:
        alert_webhook_url: Slack incoming webhook URL. No alert sent if None.
        runtime_threshold_secs: DEGRADED if run exceeds this (default 3600 = 1 hour).
        dedup_low: DEGRADED if dedup ratio < this (default 0.10 = 10%).
        dedup_high: DEGRADED if dedup ratio > this (default 0.60 = 60%).
        _dispatcher: optional callable(payload: dict) for alert delivery.
                     Production uses urllib; tests inject a capturing function.
    """

    def __init__(
        self,
        alert_webhook_url: str | None = None,
        runtime_threshold_secs: float = 3600.0,
        dedup_low: float = 0.10,
        dedup_high: float = 0.70,   # raised from 0.60 — piracy search legitimately dedupes 60-70%
        _dispatcher: Callable[[dict], None] | None = None,
    ) -> None:
        self.alert_webhook_url = alert_webhook_url
        self._runtime_threshold = runtime_threshold_secs
        self._dedup_low = dedup_low
        self._dedup_high = dedup_high
        self._dispatcher = _dispatcher

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(self, stats: PipelineStats, run_id: str) -> HealthReport:
        """
        Execute all health checks, aggregate results, and send alerts if needed.

        Checks run in this order (most critical first):
          1. zero_detections    — critical
          2. sheet_write_errors — critical
          3. dedup_ratio        — degraded
          4. score_distribution — degraded
          5. runtime            — degraded

        Returns:
            HealthReport with all check results and overall_status set.
        """
        report = HealthReport(run_id=run_id, timestamp=_utc_now())

        report.checks = [
            self._check_zero_detections(stats),
            self._check_sheet_write_errors(stats),
            self._check_dedup_ratio(stats),
            self._check_score_distribution(stats),
            self._check_runtime(stats),
        ]

        report.overall_status = _worst_status(report.checks)

        if report.overall_status != "healthy":
            destination = self._send_alert(report)
            if destination:
                report.alerts_sent.append(destination)

        return report

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_zero_detections(self, stats: PipelineStats) -> dict:
        """
        CRITICAL if detections_raw == 0.

        Zero detections almost always means the crawler is blocked or broken,
        not that all piracy has stopped. Silent failures are the worst kind.

        Note: two-consecutive-zero detection (stronger signal) requires storing
        previous-run state, which is out of scope for v1. Track manually via RunLog.
        """
        if stats.detections_raw == 0:
            return _check_result(
                name="zero_detections",
                status="critical",
                message="No detections found - crawler may be blocked or broken.",
                detail={"detections_raw": 0},
            )
        return _check_result(
            name="zero_detections",
            status="ok",
            message=f"{stats.detections_raw} detection(s) found.",
            detail={"detections_raw": stats.detections_raw},
        )

    def _check_sheet_write_errors(self, stats: PipelineStats) -> dict:
        """
        CRITICAL if sheet_write_errors > 0.

        Write failures mean detections are silently discarded. Any non-zero
        value here represents lost evidence and must page immediately.
        """
        if stats.sheet_write_errors > 0:
            return _check_result(
                name="sheet_write_errors",
                status="critical",
                message=f"{stats.sheet_write_errors} Sheets write error(s) — data may be lost.",
                detail={"sheet_write_errors": stats.sheet_write_errors},
            )
        return _check_result(
            name="sheet_write_errors",
            status="ok",
            message="All Sheets writes succeeded.",
            detail={"sheet_write_errors": 0},
        )

    def _check_dedup_ratio(self, stats: PipelineStats) -> dict:
        """
        DEGRADED if dedup ratio is outside [dedup_low, dedup_high].

        dedup_ratio = 1 - (clusters_after_dedup / detections_raw)

        Low ratio (<10%): almost nothing deduped → queries are too broad and
          returning noisy, unrelated results.
        High ratio (>60%): most results collapsed → crawler may be stuck in a
          loop returning the same content repeatedly.
        """
        if stats.detections_raw == 0:
            return _check_result(
                name="dedup_ratio",
                status="skipped",
                message="No detections to evaluate.",
                detail={},
            )

        # Compute from integer difference first to avoid FP loss:
        # 1.0 - (90/100) = 0.09999...98, but (100-90)/100 = 10/100 = 0.1 exactly.
        deduped = stats.detections_raw - stats.clusters_after_dedup
        ratio = deduped / stats.detections_raw
        pct = round(ratio * 100, 1)

        if ratio < self._dedup_low:
            return _check_result(
                name="dedup_ratio",
                status="degraded",
                message=(
                    f"Dedup ratio {pct}% is below {self._dedup_low * 100:.0f}% threshold — "
                    "queries may be too broad."
                ),
                detail={
                    "detections_raw": stats.detections_raw,
                    "clusters_after_dedup": stats.clusters_after_dedup,
                    "dedup_ratio_pct": pct,
                },
            )

        if ratio > self._dedup_high:
            return _check_result(
                name="dedup_ratio",
                status="degraded",
                message=(
                    f"Dedup ratio {pct}% exceeds {self._dedup_high * 100:.0f}% threshold — "
                    "crawler may be looping."
                ),
                detail={
                    "detections_raw": stats.detections_raw,
                    "clusters_after_dedup": stats.clusters_after_dedup,
                    "dedup_ratio_pct": pct,
                },
            )

        return _check_result(
            name="dedup_ratio",
            status="ok",
            message=f"Dedup ratio {pct}% is within expected range.",
            detail={
                "detections_raw": stats.detections_raw,
                "clusters_after_dedup": stats.clusters_after_dedup,
                "dedup_ratio_pct": pct,
            },
        )

    def _check_score_distribution(self, stats: PipelineStats) -> dict:
        """
        DEGRADED if every scored detection is below the "possible" threshold (all ignored).

        Indicates scoring signals may be broken (content catalog out of date,
        brand markers changed, fingerprint DB unavailable).

        Skipped when there are no detections to score.
        """
        if stats.detections_raw == 0:
            return _check_result(
                name="score_distribution",
                status="skipped",
                message="No detections to score.",
                detail={},
            )

        actionable = stats.confirmed_count + stats.likely_count + stats.possible_count

        if actionable == 0:
            return _check_result(
                name="score_distribution",
                status="degraded",
                message=(
                    "All detections scored below threshold (all ignored). "
                    "Scoring signals or content catalog may need updating."
                ),
                detail={
                    "confirmed": stats.confirmed_count,
                    "likely": stats.likely_count,
                    "possible": stats.possible_count,
                    "ignored": stats.ignored_count,
                },
            )

        return _check_result(
            name="score_distribution",
            status="ok",
            message=(
                f"{actionable} actionable detection(s): "
                f"{stats.confirmed_count} confirmed, "
                f"{stats.likely_count} likely, "
                f"{stats.possible_count} possible."
            ),
            detail={
                "confirmed": stats.confirmed_count,
                "likely": stats.likely_count,
                "possible": stats.possible_count,
                "ignored": stats.ignored_count,
            },
        )

    def _check_runtime(self, stats: PipelineStats) -> dict:
        """
        DEGRADED if duration_seconds exceeds the configured threshold (default 1 hour).

        Slow runs suggest crawler rate limiting, network timeouts, or a hung
        Sheets write. The pipeline itself completes, but ops should investigate.
        """
        threshold = self._runtime_threshold
        duration = stats.duration_seconds

        if duration > threshold:
            return _check_result(
                name="runtime",
                status="degraded",
                message=(
                    f"Run took {duration:.0f}s, exceeding {threshold:.0f}s threshold. "
                    "Check crawler rate limits or Sheets write latency."
                ),
                detail={"duration_seconds": duration, "threshold_seconds": threshold},
            )

        return _check_result(
            name="runtime",
            status="ok",
            message=f"Run completed in {duration:.0f}s.",
            detail={"duration_seconds": duration, "threshold_seconds": threshold},
        )

    # ------------------------------------------------------------------
    # Alert delivery
    # ------------------------------------------------------------------

    def _send_alert(self, report: HealthReport) -> str | None:
        """
        Send a summary alert to the configured webhook (Google Chat or Slack).

        Payload format is auto-detected from the URL:
          chat.googleapis.com → Google Chat simple text format
          everything else     → Slack attachment format

        Returns the destination string on success, None if no webhook configured.
        Alert delivery failures are swallowed — a broken webhook must never
        crash the pipeline or shadow the actual health report.
        """
        if not self.alert_webhook_url and self._dispatcher is None:
            return None

        if self.alert_webhook_url and "chat.googleapis.com" in self.alert_webhook_url:
            payload = _build_gchat_payload(report)
        else:
            payload = _build_slack_payload(report)

        if self._dispatcher is not None:
            try:
                self._dispatcher(payload)
            except Exception:
                pass
            return "dispatcher"

        # POST to webhook via urllib (no extra deps)
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.alert_webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=10)
            return self.alert_webhook_url
        except Exception:
            # Swallow delivery errors — do not propagate
            return None


# ---------------------------------------------------------------------------
# Alert payload builder (pure function, easily unit-tested)
# ---------------------------------------------------------------------------

def _build_gchat_payload(report: HealthReport) -> dict:
    """
    Build a Google Chat incoming webhook payload from a HealthReport.

    Google Chat uses {"text": "..."} with *bold* and `code` markdown.
    """
    status = report.overall_status
    lines = [
        f"*AP Army Pipeline — {status.upper()}*",
        f"Run: `{report.run_id}`",
        f"Time: {report.timestamp}",
        "",
    ]
    for check in report.checks:
        emoji = _STATUS_EMOJI.get(check["status"], "❓")
        lines.append(f"{emoji} *{check['name']}*: {check['message']}")
    return {"text": "\n".join(lines)}


def _build_slack_payload(report: HealthReport) -> dict:
    """
    Build a Slack incoming webhook JSON payload from a HealthReport.

    Uses simple attachment format (not Block Kit) for broad compatibility.
    """
    status = report.overall_status
    colour = _ALERT_COLOUR.get(status, "#AAAAAA")
    title = f"AP Army Pipeline — {status.upper()}"

    lines = [f"*Run ID:* `{report.run_id}`", f"*Time:* {report.timestamp}", ""]
    for check in report.checks:
        emoji = _STATUS_EMOJI.get(check["status"], "❓")
        lines.append(f"{emoji} *{check['name']}*: {check['message']}")

    return {
        "text": title,
        "attachments": [
            {
                "color": colour,
                "text": "\n".join(lines),
                "mrkdwn_in": ["text"],
            }
        ],
    }


# ---------------------------------------------------------------------------
# Pre-flight check
# ---------------------------------------------------------------------------

def pre_flight_check(
    env: dict | None = None,
    _isfile: Callable[[str], bool] | None = None,
) -> tuple[bool, list[dict]]:
    """
    Verify all required environment variables and credentials exist before run.

    Critical checks (returning False if any fail):
      - GOOGLE_SHEETS_ID env var is set
      - service_account.json file exists (path from GOOGLE_SERVICE_ACCOUNT or default)
      - FB_COOKIE_FILE env var is set and the file exists

    Warning-only (degraded but not critical):
      - GCHAT_WEBHOOK_URL or SLACK_WEBHOOK_URL env var is set (either one suffices)

    Args:
        env: environment dict (defaults to os.environ). Override in tests.
        _isfile: file-existence checker (defaults to os.path.isfile). Override in tests.

    Returns:
        (passed: bool, results: list[dict])
        passed is True only if all critical checks are "ok".
    """
    env = env if env is not None else os.environ
    isfile = _isfile if _isfile is not None else os.path.isfile
    results: list[dict] = []

    # Critical: GOOGLE_SHEETS_ID
    sheets_id = env.get("GOOGLE_SHEETS_ID", "")
    results.append(_check_result(
        name="GOOGLE_SHEETS_ID",
        status="ok" if sheets_id else "critical",
        message="Set." if sheets_id else "GOOGLE_SHEETS_ID env var is not set.",
        detail={"value_present": bool(sheets_id)},
    ))

    # Critical: service_account.json
    sa_path = env.get("GOOGLE_SERVICE_ACCOUNT", "service_account.json")
    sa_found = isfile(sa_path)
    results.append(_check_result(
        name="service_account",
        status="ok" if sa_found else "critical",
        message=f"Found at {sa_path!r}." if sa_found else f"File not found: {sa_path!r}.",
        detail={"path": sa_path, "found": sa_found},
    ))

    # Critical: FB_COOKIE_FILE
    fb_cookie = env.get("FB_COOKIE_FILE", "")
    if not fb_cookie:
        fb_status, fb_msg = "critical", "FB_COOKIE_FILE env var is not set."
    elif not isfile(fb_cookie):
        fb_status, fb_msg = "critical", f"Cookie file not found: {fb_cookie!r}."
    else:
        fb_status, fb_msg = "ok", f"Found at {fb_cookie!r}."
    results.append(_check_result(
        name="FB_COOKIE_FILE",
        status=fb_status,
        message=fb_msg,
        detail={"path": fb_cookie, "found": fb_status == "ok"},
    ))

    # Warning-only: at least one alert webhook must be configured (GChat or Slack)
    gchat_url = env.get("GCHAT_WEBHOOK_URL", "")
    slack_url  = env.get("SLACK_WEBHOOK_URL", "")
    webhook_ok = bool(gchat_url or slack_url)
    webhook_which = "GChat" if gchat_url else ("Slack" if slack_url else "")
    results.append(_check_result(
        name="alert_webhook",
        status="ok" if webhook_ok else "degraded",
        message=(
            f"Set ({webhook_which})." if webhook_ok
            else "No alert webhook set (GCHAT_WEBHOOK_URL or SLACK_WEBHOOK_URL) — alerts will not be sent."
        ),
        detail={"gchat_present": bool(gchat_url), "slack_present": bool(slack_url)},
    ))

    # Critical check passes only if all non-warning checks are "ok"
    passed = all(
        r["status"] in ("ok", "degraded") for r in results
    )
    return passed, results
