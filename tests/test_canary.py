"""Tests for canary.py."""

import pytest
from canary import (
    CanaryChecker,
    HealthReport,
    PipelineStats,
    _build_slack_payload,
    _worst_status,
    pre_flight_check,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_stats(**kwargs) -> PipelineStats:
    defaults = dict(
        detections_raw=100,
        clusters_after_dedup=70,
        confirmed_count=10,
        likely_count=20,
        possible_count=15,
        ignored_count=25,
        errors=[],
        platforms_crawled=["facebook"],
        duration_seconds=900.0,
        sheet_write_errors=0,
    )
    defaults.update(kwargs)
    return PipelineStats(**defaults)


def make_checker(**kwargs) -> CanaryChecker:
    """Checker with no webhook — alerts go nowhere unless _dispatcher is set."""
    return CanaryChecker(**kwargs)


def capturing_checker() -> tuple[CanaryChecker, list[dict]]:
    """Returns a checker and a list that accumulates dispatched alert payloads."""
    received: list[dict] = []
    checker = CanaryChecker(
        alert_webhook_url="https://hooks.slack.com/fake",
        _dispatcher=received.append,
    )
    return checker, received


# ---------------------------------------------------------------------------
# _worst_status
# ---------------------------------------------------------------------------

class TestWorstStatus:
    def test_all_ok_is_healthy(self):
        checks = [{"status": "ok"}, {"status": "ok"}]
        assert _worst_status(checks) == "healthy"

    def test_skipped_counts_as_ok(self):
        checks = [{"status": "ok"}, {"status": "skipped"}]
        assert _worst_status(checks) == "healthy"

    def test_degraded_gives_degraded(self):
        checks = [{"status": "ok"}, {"status": "degraded"}]
        assert _worst_status(checks) == "degraded"

    def test_critical_beats_degraded(self):
        checks = [{"status": "degraded"}, {"status": "critical"}]
        assert _worst_status(checks) == "critical"

    def test_empty_list_is_healthy(self):
        assert _worst_status([]) == "healthy"

    def test_single_critical(self):
        assert _worst_status([{"status": "critical"}]) == "critical"


# ---------------------------------------------------------------------------
# Check: zero_detections
# ---------------------------------------------------------------------------

class TestCheckZeroDetections:
    def setup_method(self):
        self.checker = make_checker()

    def test_zero_is_critical(self):
        result = self.checker._check_zero_detections(make_stats(detections_raw=0))
        assert result["status"] == "critical"

    def test_nonzero_is_ok(self):
        result = self.checker._check_zero_detections(make_stats(detections_raw=1))
        assert result["status"] == "ok"

    def test_name_is_set(self):
        result = self.checker._check_zero_detections(make_stats(detections_raw=0))
        assert result["name"] == "zero_detections"

    def test_detail_contains_count(self):
        result = self.checker._check_zero_detections(make_stats(detections_raw=42))
        assert result["detail"]["detections_raw"] == 42

    def test_message_non_empty(self):
        result = self.checker._check_zero_detections(make_stats(detections_raw=0))
        assert len(result["message"]) > 0


# ---------------------------------------------------------------------------
# Check: sheet_write_errors
# ---------------------------------------------------------------------------

class TestCheckSheetWriteErrors:
    def setup_method(self):
        self.checker = make_checker()

    def test_zero_errors_is_ok(self):
        result = self.checker._check_sheet_write_errors(make_stats(sheet_write_errors=0))
        assert result["status"] == "ok"

    def test_one_error_is_critical(self):
        result = self.checker._check_sheet_write_errors(make_stats(sheet_write_errors=1))
        assert result["status"] == "critical"

    def test_many_errors_is_critical(self):
        result = self.checker._check_sheet_write_errors(make_stats(sheet_write_errors=99))
        assert result["status"] == "critical"

    def test_detail_contains_error_count(self):
        result = self.checker._check_sheet_write_errors(make_stats(sheet_write_errors=3))
        assert result["detail"]["sheet_write_errors"] == 3


# ---------------------------------------------------------------------------
# Check: dedup_ratio
# ---------------------------------------------------------------------------

class TestCheckDedupRatio:
    def setup_method(self):
        self.checker = make_checker(dedup_low=0.10, dedup_high=0.60)

    def test_no_detections_is_skipped(self):
        result = self.checker._check_dedup_ratio(make_stats(detections_raw=0))
        assert result["status"] == "skipped"

    def test_zero_dedup_is_degraded(self):
        # 100 raw, 100 clusters → 0% dedup → below 10% threshold
        result = self.checker._check_dedup_ratio(
            make_stats(detections_raw=100, clusters_after_dedup=100)
        )
        assert result["status"] == "degraded"

    def test_below_low_threshold_is_degraded(self):
        # 100 raw, 95 clusters → 5% dedup → below 10%
        result = self.checker._check_dedup_ratio(
            make_stats(detections_raw=100, clusters_after_dedup=95)
        )
        assert result["status"] == "degraded"

    def test_above_high_threshold_is_degraded(self):
        # 100 raw, 30 clusters → 70% dedup → above 60%
        result = self.checker._check_dedup_ratio(
            make_stats(detections_raw=100, clusters_after_dedup=30)
        )
        assert result["status"] == "degraded"

    def test_exactly_at_low_boundary_is_ok(self):
        # 100 raw, 90 clusters → 10% dedup → exactly at lower bound
        result = self.checker._check_dedup_ratio(
            make_stats(detections_raw=100, clusters_after_dedup=90)
        )
        assert result["status"] == "ok"

    def test_exactly_at_high_boundary_is_ok(self):
        # 100 raw, 40 clusters → 60% dedup → exactly at upper bound
        result = self.checker._check_dedup_ratio(
            make_stats(detections_raw=100, clusters_after_dedup=40)
        )
        assert result["status"] == "ok"

    def test_within_range_is_ok(self):
        # 100 raw, 70 clusters → 30% dedup → within [10%, 60%]
        result = self.checker._check_dedup_ratio(
            make_stats(detections_raw=100, clusters_after_dedup=70)
        )
        assert result["status"] == "ok"

    def test_detail_contains_ratio(self):
        result = self.checker._check_dedup_ratio(
            make_stats(detections_raw=100, clusters_after_dedup=70)
        )
        assert "dedup_ratio_pct" in result["detail"]
        assert result["detail"]["dedup_ratio_pct"] == 30.0

    def test_custom_thresholds_respected(self):
        checker = make_checker(dedup_low=0.05, dedup_high=0.80)
        # 0% dedup — below 5% low threshold → still degraded
        result = checker._check_dedup_ratio(
            make_stats(detections_raw=100, clusters_after_dedup=100)
        )
        assert result["status"] == "degraded"
        # 70% dedup — below 80% high threshold → ok with custom high
        result2 = checker._check_dedup_ratio(
            make_stats(detections_raw=100, clusters_after_dedup=30)
        )
        assert result2["status"] == "ok"


# ---------------------------------------------------------------------------
# Check: score_distribution
# ---------------------------------------------------------------------------

class TestCheckScoreDistribution:
    def setup_method(self):
        self.checker = make_checker()

    def test_no_detections_is_skipped(self):
        result = self.checker._check_score_distribution(make_stats(detections_raw=0))
        assert result["status"] == "skipped"

    def test_all_ignored_is_degraded(self):
        result = self.checker._check_score_distribution(make_stats(
            detections_raw=50,
            confirmed_count=0, likely_count=0, possible_count=0, ignored_count=50,
        ))
        assert result["status"] == "degraded"

    def test_some_actionable_is_ok(self):
        result = self.checker._check_score_distribution(make_stats(
            detections_raw=50, confirmed_count=5, likely_count=0, possible_count=0,
        ))
        assert result["status"] == "ok"

    def test_only_possible_is_ok(self):
        result = self.checker._check_score_distribution(make_stats(
            detections_raw=10, confirmed_count=0, likely_count=0, possible_count=3,
        ))
        assert result["status"] == "ok"

    def test_detail_contains_counts(self):
        result = self.checker._check_score_distribution(make_stats(
            detections_raw=10, confirmed_count=2, likely_count=3,
            possible_count=1, ignored_count=4,
        ))
        assert result["detail"]["confirmed"] == 2
        assert result["detail"]["likely"] == 3
        assert result["detail"]["possible"] == 1
        assert result["detail"]["ignored"] == 4


# ---------------------------------------------------------------------------
# Check: runtime
# ---------------------------------------------------------------------------

class TestCheckRuntime:
    def setup_method(self):
        self.checker = make_checker(runtime_threshold_secs=3600.0)

    def test_within_threshold_is_ok(self):
        result = self.checker._check_runtime(make_stats(duration_seconds=3600.0))
        assert result["status"] == "ok"

    def test_over_threshold_is_degraded(self):
        result = self.checker._check_runtime(make_stats(duration_seconds=3601.0))
        assert result["status"] == "degraded"

    def test_zero_seconds_is_ok(self):
        result = self.checker._check_runtime(make_stats(duration_seconds=0.0))
        assert result["status"] == "ok"

    def test_detail_contains_duration(self):
        result = self.checker._check_runtime(make_stats(duration_seconds=500.0))
        assert result["detail"]["duration_seconds"] == 500.0
        assert result["detail"]["threshold_seconds"] == 3600.0

    def test_custom_threshold_respected(self):
        checker = make_checker(runtime_threshold_secs=60.0)
        result = checker._check_runtime(make_stats(duration_seconds=61.0))
        assert result["status"] == "degraded"
        result2 = checker._check_runtime(make_stats(duration_seconds=60.0))
        assert result2["status"] == "ok"


# ---------------------------------------------------------------------------
# run_all — integration
# ---------------------------------------------------------------------------

class TestRunAll:
    def test_returns_health_report(self):
        checker = make_checker()
        report = checker.run_all(make_stats(), run_id="run_001")
        assert isinstance(report, HealthReport)

    def test_run_id_set(self):
        checker = make_checker()
        report = checker.run_all(make_stats(), run_id="run_xyz")
        assert report.run_id == "run_xyz"

    def test_timestamp_set(self):
        checker = make_checker()
        report = checker.run_all(make_stats(), run_id="run_001")
        assert "T" in report.timestamp and report.timestamp.endswith("Z")

    def test_five_checks_run(self):
        checker = make_checker()
        report = checker.run_all(make_stats(), run_id="run_001")
        assert len(report.checks) == 5

    def test_all_checks_have_required_keys(self):
        checker = make_checker()
        report = checker.run_all(make_stats(), run_id="run_001")
        for check in report.checks:
            assert "name" in check
            assert "status" in check
            assert "message" in check
            assert "detail" in check

    def test_healthy_when_all_ok(self):
        checker = make_checker()
        report = checker.run_all(make_stats(), run_id="run_001")
        assert report.overall_status == "healthy"

    def test_critical_on_zero_detections(self):
        checker = make_checker()
        report = checker.run_all(make_stats(detections_raw=0), run_id="run_001")
        assert report.overall_status == "critical"

    def test_degraded_on_slow_runtime(self):
        checker = make_checker(runtime_threshold_secs=100.0)
        report = checker.run_all(make_stats(duration_seconds=200.0), run_id="run_001")
        assert report.overall_status == "degraded"

    def test_critical_overrides_degraded(self):
        checker = make_checker(runtime_threshold_secs=100.0)
        # Both runtime (degraded) and sheet_write_errors (critical) fire
        report = checker.run_all(
            make_stats(duration_seconds=200.0, sheet_write_errors=1),
            run_id="run_001",
        )
        assert report.overall_status == "critical"

    def test_no_alert_when_healthy(self):
        checker, received = capturing_checker()
        checker.run_all(make_stats(), run_id="run_001")
        assert len(received) == 0

    def test_alert_sent_when_degraded(self):
        checker, received = capturing_checker()
        checker.run_all(make_stats(duration_seconds=9999.0), run_id="run_001")
        assert len(received) == 1

    def test_alert_sent_when_critical(self):
        checker, received = capturing_checker()
        checker.run_all(make_stats(detections_raw=0), run_id="run_001")
        assert len(received) == 1

    def test_alerts_sent_list_populated(self):
        checker, received = capturing_checker()
        report = checker.run_all(make_stats(detections_raw=0), run_id="run_001")
        assert len(report.alerts_sent) == 1
        assert report.alerts_sent[0] == "dispatcher"

    def test_alerts_sent_empty_when_no_issues(self):
        checker, received = capturing_checker()
        report = checker.run_all(make_stats(), run_id="run_001")
        assert report.alerts_sent == []

    def test_dispatcher_failure_does_not_raise(self):
        def failing_dispatcher(payload):
            raise RuntimeError("network error")

        checker = CanaryChecker(
            alert_webhook_url="https://fake",
            _dispatcher=failing_dispatcher,
        )
        # Should not raise even when dispatcher fails
        report = checker.run_all(make_stats(detections_raw=0), run_id="run_001")
        assert report.overall_status == "critical"

    def test_no_webhook_no_dispatcher_no_alert(self):
        checker = CanaryChecker(alert_webhook_url=None, _dispatcher=None)
        report = checker.run_all(make_stats(detections_raw=0), run_id="run_001")
        assert report.alerts_sent == []


# ---------------------------------------------------------------------------
# _build_slack_payload
# ---------------------------------------------------------------------------

class TestBuildSlackPayload:
    def _make_report(self, status: str = "critical") -> HealthReport:
        return HealthReport(
            run_id="run_test",
            timestamp="2025-01-15T08:00:00Z",
            checks=[
                {"name": "zero_detections", "status": "critical",
                 "message": "No detections.", "detail": {}},
                {"name": "runtime", "status": "ok",
                 "message": "Fast.", "detail": {}},
            ],
            overall_status=status,
        )

    def test_returns_dict(self):
        payload = _build_slack_payload(self._make_report())
        assert isinstance(payload, dict)

    def test_has_text_key(self):
        payload = _build_slack_payload(self._make_report())
        assert "text" in payload

    def test_status_in_text(self):
        payload = _build_slack_payload(self._make_report("critical"))
        assert "CRITICAL" in payload["text"]

    def test_has_attachments(self):
        payload = _build_slack_payload(self._make_report())
        assert "attachments" in payload
        assert len(payload["attachments"]) > 0

    def test_attachment_has_colour(self):
        payload = _build_slack_payload(self._make_report("critical"))
        assert "color" in payload["attachments"][0]

    def test_run_id_in_body(self):
        payload = _build_slack_payload(self._make_report())
        body = payload["attachments"][0]["text"]
        assert "run_test" in body

    def test_check_names_in_body(self):
        payload = _build_slack_payload(self._make_report())
        body = payload["attachments"][0]["text"]
        assert "zero_detections" in body
        assert "runtime" in body

    def test_degraded_colour_different_from_critical(self):
        crit = _build_slack_payload(self._make_report("critical"))["attachments"][0]["color"]
        deg  = _build_slack_payload(self._make_report("degraded"))["attachments"][0]["color"]
        assert crit != deg


# ---------------------------------------------------------------------------
# _build_gchat_payload
# ---------------------------------------------------------------------------

class TestBuildGchatPayload:
    def _make_report(self, status: str = "critical") -> HealthReport:
        return HealthReport(
            run_id="run_test",
            timestamp="2025-01-15T08:00:00Z",
            checks=[
                {"name": "zero_detections", "status": "critical",
                 "message": "No detections.", "detail": {}},
                {"name": "runtime", "status": "ok",
                 "message": "Fast.", "detail": {}},
            ],
            overall_status=status,
        )

    def test_returns_dict_with_text_key(self):
        from canary import _build_gchat_payload
        payload = _build_gchat_payload(self._make_report())
        assert isinstance(payload, dict)
        assert "text" in payload
        assert "attachments" not in payload   # GChat uses plain text, not attachments

    def test_status_in_text(self):
        from canary import _build_gchat_payload
        payload = _build_gchat_payload(self._make_report("critical"))
        assert "CRITICAL" in payload["text"]

    def test_run_id_in_text(self):
        from canary import _build_gchat_payload
        payload = _build_gchat_payload(self._make_report())
        assert "run_test" in payload["text"]

    def test_check_names_in_text(self):
        from canary import _build_gchat_payload
        payload = _build_gchat_payload(self._make_report())
        assert "zero_detections" in payload["text"]
        assert "runtime" in payload["text"]

    def test_gchat_url_triggers_gchat_payload(self):
        """CanaryChecker._send_alert uses GChat format when URL contains chat.googleapis.com."""
        received: list[dict] = []
        checker = CanaryChecker(
            alert_webhook_url="https://chat.googleapis.com/v1/spaces/FAKE/messages?key=x",
            _dispatcher=received.append,
        )
        report = HealthReport(
            run_id="r1", timestamp="2025-01-01T00:00:00Z",
            checks=[{"name": "test", "status": "critical", "message": "bad", "detail": {}}],
            overall_status="critical",
        )
        checker._send_alert(report)
        assert len(received) == 1
        assert "text" in received[0]
        assert "attachments" not in received[0]   # GChat format, not Slack


# ---------------------------------------------------------------------------
# pre_flight_check
# ---------------------------------------------------------------------------

class TestPreFlightCheck:
    def _all_present_env(self, cookie_path: str = "/tmp/fb_cookies.txt") -> dict:
        return {
            "GOOGLE_SHEETS_ID": "sheet_abc123",
            "GOOGLE_SERVICE_ACCOUNT": "service_account.json",
            "FB_COOKIE_FILE": cookie_path,
            "GCHAT_WEBHOOK_URL": "https://chat.googleapis.com/v1/spaces/FAKE/messages?key=x",
        }

    def _isfile_all_exist(self, path: str) -> bool:
        return True

    def test_all_present_returns_true(self):
        passed, _ = pre_flight_check(
            env=self._all_present_env(),
            _isfile=self._isfile_all_exist,
        )
        assert passed is True

    def test_missing_sheets_id_returns_false(self):
        env = self._all_present_env()
        del env["GOOGLE_SHEETS_ID"]
        passed, _ = pre_flight_check(env=env, _isfile=self._isfile_all_exist)
        assert passed is False

    def test_missing_service_account_file_returns_false(self):
        passed, _ = pre_flight_check(
            env=self._all_present_env(),
            _isfile=lambda path: path != "service_account.json",
        )
        assert passed is False

    def test_missing_fb_cookie_env_returns_false(self):
        env = self._all_present_env()
        del env["FB_COOKIE_FILE"]
        passed, _ = pre_flight_check(env=env, _isfile=self._isfile_all_exist)
        assert passed is False

    def test_fb_cookie_file_not_found_returns_false(self):
        passed, _ = pre_flight_check(
            env=self._all_present_env(cookie_path="/nonexistent/cookies.txt"),
            _isfile=lambda path: False,
        )
        assert passed is False

    def test_missing_alert_webhook_still_passes(self):
        # Neither GCHAT_WEBHOOK_URL nor SLACK_WEBHOOK_URL — warning-only, not critical
        env = self._all_present_env()
        del env["GCHAT_WEBHOOK_URL"]
        passed, results = pre_flight_check(env=env, _isfile=self._isfile_all_exist)
        assert passed is True
        webhook_check = next(r for r in results if r["name"] == "alert_webhook")
        assert webhook_check["status"] == "degraded"

    def test_gchat_webhook_satisfies_alert_check(self):
        env = self._all_present_env()   # already has GCHAT_WEBHOOK_URL
        passed, results = pre_flight_check(env=env, _isfile=self._isfile_all_exist)
        webhook_check = next(r for r in results if r["name"] == "alert_webhook")
        assert webhook_check["status"] == "ok"
        assert "GChat" in webhook_check["message"]

    def test_slack_webhook_also_satisfies_alert_check(self):
        env = self._all_present_env()
        del env["GCHAT_WEBHOOK_URL"]
        env["SLACK_WEBHOOK_URL"] = "https://hooks.slack.com/test"
        passed, results = pre_flight_check(env=env, _isfile=self._isfile_all_exist)
        webhook_check = next(r for r in results if r["name"] == "alert_webhook")
        assert webhook_check["status"] == "ok"
        assert "Slack" in webhook_check["message"]

    def test_returns_four_check_results(self):
        _, results = pre_flight_check(
            env=self._all_present_env(),
            _isfile=self._isfile_all_exist,
        )
        assert len(results) == 4

    def test_all_results_have_required_keys(self):
        _, results = pre_flight_check(
            env=self._all_present_env(),
            _isfile=self._isfile_all_exist,
        )
        for r in results:
            assert "name" in r
            assert "status" in r
            assert "message" in r

    def test_custom_service_account_path(self):
        env = self._all_present_env()
        env["GOOGLE_SERVICE_ACCOUNT"] = "/etc/secrets/sa.json"
        # File exists at custom path
        passed, results = pre_flight_check(
            env=env,
            _isfile=lambda p: p == "/etc/secrets/sa.json",
        )
        sa_check = next(r for r in results if r["name"] == "service_account")
        assert sa_check["status"] == "ok"
