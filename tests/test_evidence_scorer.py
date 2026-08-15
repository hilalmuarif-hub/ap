"""Tests for evidence_scorer.py."""

import pytest
from detection import RawDetection
from identity import OffenderIdentity
from evidence_scorer import (
    SIGNAL_WEIGHTS,
    VERDICT_THRESHOLDS,
    EvidenceScorer,
    ScoredEvidence,
    batch_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_detection(
    title: str = "Liga Champions Vidio",
    snapshot_html: str = "<html>test</html>",
    extra: dict | None = None,
    query_used: str = "liga champions vidio",
) -> RawDetection:
    return RawDetection(
        platform="facebook",
        url="https://www.facebook.com/watch?v=123",
        title=title,
        channel_id="ch_001",
        channel_name="Test Pirate",
        snapshot_html=snapshot_html,
        detected_at="2025-01-15T08:00:00Z",
        query_used=query_used,
        extra=extra or {},
    )


def make_identity(permanent_id: str = "pid_001") -> OffenderIdentity:
    return OffenderIdentity(
        platform="facebook",
        permanent_id=permanent_id,
        display_name="Test Pirate",
        profile_url="https://www.facebook.com/testpirate",
        resolved_at="2025-01-15T08:00:00Z",
        confidence=0.95,
        metadata={},
    )


CATALOG = ["Liga Champions 2025", "Sinetron Nusantara", "Film Horor Indonesia"]


# ---------------------------------------------------------------------------
# Module-level invariants
# ---------------------------------------------------------------------------

class TestModuleInvariants:
    def test_signal_weights_sum_to_100(self):
        assert sum(SIGNAL_WEIGHTS.values()) == 100

    def test_all_weights_positive(self):
        assert all(v > 0 for v in SIGNAL_WEIGHTS.values())

    def test_verdict_thresholds_ordered(self):
        assert VERDICT_THRESHOLDS["confirmed"] > VERDICT_THRESHOLDS["likely"]
        assert VERDICT_THRESHOLDS["likely"] > VERDICT_THRESHOLDS["possible"]


# ---------------------------------------------------------------------------
# verdict_from_score
# ---------------------------------------------------------------------------

class TestVerdictFromScore:
    @pytest.mark.parametrize("score,expected", [
        (100, "confirmed"),
        (80,  "confirmed"),
        (79,  "likely"),
        (60,  "likely"),
        (59,  "possible"),
        (40,  "possible"),
        (25,  "possible"),   # new lower boundary for "possible"
        (24,  "ignore"),
        (1,   "ignore"),
        (0,   "ignore"),
    ])
    def test_thresholds(self, score, expected):
        assert EvidenceScorer.verdict_from_score(score) == expected

    def test_exactly_at_confirmed_boundary(self):
        assert EvidenceScorer.verdict_from_score(VERDICT_THRESHOLDS["confirmed"]) == "confirmed"

    def test_one_below_confirmed(self):
        assert EvidenceScorer.verdict_from_score(VERDICT_THRESHOLDS["confirmed"] - 1) == "likely"


# ---------------------------------------------------------------------------
# _score_title_keyword_match
# ---------------------------------------------------------------------------

class TestTitleKeywordMatch:
    def setup_method(self):
        self.scorer = EvidenceScorer(content_catalog=CATALOG)

    def test_near_exact_match(self):
        # "Liga Champions 2025" vs "Liga Champions 2025" — 100% similarity
        det = make_detection(title="Liga Champions 2025")
        score = self.scorer._score_title_keyword_match(det)
        assert score == SIGNAL_WEIGHTS["title_keyword_match"]   # 25

    def test_strong_fuzzy_match(self):
        # token_set_ratio gives 1.0 when one string's tokens are a superset of
        # the other's — "Vidio 2025 Liga Champions" contains all tokens from
        # catalog "Liga Champions 2025", so it scores perfect (25), not 15.
        # This is the correct behavior: extra tokens don't penalize a match.
        det = make_detection(title="Vidio 2025 Liga Champions")
        score = self.scorer._score_title_keyword_match(det)
        assert score == 25

    def test_partial_match_score(self):
        # "Liga Champions" overlaps but adds unrelated tokens that lower similarity
        det = make_detection(title="Liga Champions Eropa Paling Seru Terbaik Sepanjang Masa")
        score = self.scorer._score_title_keyword_match(det)
        # Expect either partial (5) or strong (15) — not zero
        assert score > 0

    def test_no_catalog_returns_zero(self):
        scorer = EvidenceScorer(content_catalog=[])
        det = make_detection(title="Liga Champions 2025")
        assert scorer._score_title_keyword_match(det) == 0

    def test_empty_title_uses_query_used_fallback(self):
        # Empty title → scorer falls back to query_used.
        # query_used="liga champions vidio" matches catalog "Liga Champions 2025" → >0
        det = make_detection(title="   ", query_used="liga champions vidio")
        assert self.scorer._score_title_keyword_match(det) > 0

    def test_empty_title_and_unrelated_query_returns_zero(self):
        det = make_detection(title="   ", query_used="resep masak ayam")
        assert self.scorer._score_title_keyword_match(det) == 0

    def test_unrelated_title_returns_zero(self):
        det = make_detection(title="Resep Masak Ayam Goreng Enak")
        assert self.scorer._score_title_keyword_match(det) == 0

    def test_best_catalog_match_used(self):
        # Detection matches "Film Horor Indonesia" better than other catalog titles
        det = make_detection(title="Film Horor Indonesia 2025 Full")
        score = self.scorer._score_title_keyword_match(det)
        assert score >= 15   # should match at strong level or better


# ---------------------------------------------------------------------------
# _score_offender_repeat_history
# ---------------------------------------------------------------------------

class TestOffenderRepeatHistory:
    def setup_method(self):
        self.scorer = EvidenceScorer()

    @pytest.mark.parametrize("violations,expected", [
        (0,  0),
        (1,  5),
        (2,  10),
        (3,  10),
        (4,  15),
        (5,  15),
        (100, 15),
    ])
    def test_tiers(self, violations, expected):
        assert self.scorer._score_offender_repeat_history(violations) == expected

    def test_max_is_signal_weight(self):
        max_possible = self.scorer._score_offender_repeat_history(999)
        assert max_possible == SIGNAL_WEIGHTS["offender_repeat_history"]


# ---------------------------------------------------------------------------
# _score_engagement_anomaly
# ---------------------------------------------------------------------------

class TestEngagementAnomaly:
    def setup_method(self):
        self.scorer = EvidenceScorer()

    def test_missing_view_count_returns_zero(self):
        det = make_detection(extra={"account_age_days": 10})
        assert self.scorer._score_engagement_anomaly(det) == 0

    def test_missing_account_age_returns_zero(self):
        det = make_detection(extra={"view_count": 100_000})
        assert self.scorer._score_engagement_anomaly(det) == 0

    def test_both_missing_returns_zero(self):
        det = make_detection(extra={})
        assert self.scorer._score_engagement_anomaly(det) == 0

    @pytest.mark.parametrize("views,age_days,expected", [
        (100_000, 1,   10),   # 100k/day → max tier
        (10_000,  1,   10),   # 10k/day → max tier boundary
        (9_999,   1,   7),    # just below max tier → second tier
        (1_000,   1,   7),    # 1k/day → second tier boundary
        (999,     1,   4),    # just below → third tier
        (100,     1,   4),    # 100/day → third tier boundary
        (99,      1,   0),    # below threshold
        (10_000,  10,  7),    # 1k/day → second tier
        (10_000,  100, 4),    # 100/day → third tier
        (50,      1,   0),    # low views → 0
    ])
    def test_scoring_tiers(self, views, age_days, expected):
        det = make_detection(extra={"view_count": views, "account_age_days": age_days})
        assert self.scorer._score_engagement_anomaly(det) == expected

    def test_zero_account_age_no_division_error(self):
        # account_age_days=0 is treated as 1 to prevent ZeroDivisionError
        det = make_detection(extra={"view_count": 50_000, "account_age_days": 0})
        score = self.scorer._score_engagement_anomaly(det)
        assert score == SIGNAL_WEIGHTS["engagement_anomaly"]   # 10


# ---------------------------------------------------------------------------
# _score_content_id_watermark
# ---------------------------------------------------------------------------

class TestContentIdWatermark:
    def setup_method(self):
        self.scorer = EvidenceScorer()

    def test_content_id_flag_full_score(self):
        det = make_detection(extra={"content_id_match": True})
        assert self.scorer._score_content_id_watermark(det) == SIGNAL_WEIGHTS["content_id_watermark"]

    def test_content_id_flag_false_uses_html(self):
        # flag=False → fall back to HTML marker scanning
        det = make_detection(
            snapshot_html="<html>vidio.com player</html>",
            extra={"content_id_match": False},
        )
        score = self.scorer._score_content_id_watermark(det)
        assert 0 < score < SIGNAL_WEIGHTS["content_id_watermark"]

    def test_three_markers_half_score(self):
        html = "vidio.com content.vidio.com vidio-player"
        det = make_detection(snapshot_html=html)
        assert self.scorer._score_content_id_watermark(det) == 15   # 30 * 0.5

    def test_one_marker_quarter_score(self):
        det = make_detection(snapshot_html="<html>vidio.com</html>")
        assert self.scorer._score_content_id_watermark(det) == 7    # floor(30 * 0.25)

    def test_no_markers_zero(self):
        det = make_detection(snapshot_html="<html>random content</html>")
        assert self.scorer._score_content_id_watermark(det) == 0

    def test_case_insensitive_marker(self):
        det = make_detection(snapshot_html="<html>VIDIO.COM</html>")
        assert self.scorer._score_content_id_watermark(det) > 0

    def test_custom_markers(self):
        scorer = EvidenceScorer(brand_markers=["custommark"])
        det = make_detection(snapshot_html="<html>custommark</html>")
        assert scorer._score_content_id_watermark(det) > 0


# ---------------------------------------------------------------------------
# _score_stream_key_fingerprint
# ---------------------------------------------------------------------------

class TestStreamKeyFingerprint:
    def setup_method(self):
        self.scorer = EvidenceScorer()

    def test_confirmed_full_score(self):
        det = make_detection(extra={"fingerprint_match_level": "confirmed"})
        assert self.scorer._score_stream_key_fingerprint(det) == SIGNAL_WEIGHTS["stream_key_fingerprint"]

    def test_likely_half_score(self):
        det = make_detection(extra={"fingerprint_match_level": "likely"})
        assert self.scorer._score_stream_key_fingerprint(det) == SIGNAL_WEIGHTS["stream_key_fingerprint"] // 2

    def test_no_flag_zero(self):
        det = make_detection(extra={})
        assert self.scorer._score_stream_key_fingerprint(det) == 0

    def test_unknown_level_zero(self):
        det = make_detection(extra={"fingerprint_match_level": "unknown_value"})
        assert self.scorer._score_stream_key_fingerprint(det) == 0


# ---------------------------------------------------------------------------
# EvidenceScorer.score — full integration
# ---------------------------------------------------------------------------

class TestScorerIntegration:
    def setup_method(self):
        self.scorer = EvidenceScorer(content_catalog=CATALOG)
        self.identity = make_identity()

    def test_returns_scored_evidence(self):
        det = make_detection()
        result = self.scorer.score(det, self.identity)
        assert isinstance(result, ScoredEvidence)

    def test_score_in_range(self):
        det = make_detection()
        result = self.scorer.score(det, self.identity)
        assert 0 <= result.score <= 100

    def test_breakdown_keys_match_signal_weights(self):
        det = make_detection()
        result = self.scorer.score(det, self.identity)
        assert set(result.breakdown.keys()) == set(SIGNAL_WEIGHTS.keys())

    def test_breakdown_values_sum_to_score(self):
        det = make_detection()
        result = self.scorer.score(det, self.identity)
        assert sum(result.breakdown.values()) == result.score

    def test_breakdown_values_non_negative(self):
        det = make_detection()
        result = self.scorer.score(det, self.identity)
        assert all(v >= 0 for v in result.breakdown.values())

    def test_breakdown_values_capped_at_signal_max(self):
        det = make_detection()
        result = self.scorer.score(det, self.identity)
        for signal, value in result.breakdown.items():
            assert value <= SIGNAL_WEIGHTS[signal]

    def test_verdict_matches_score(self):
        det = make_detection()
        result = self.scorer.score(det, self.identity)
        assert result.verdict == EvidenceScorer.verdict_from_score(result.score)

    def test_deterministic(self):
        det = make_detection(
            title="Liga Champions 2025",
            extra={"view_count": 50_000, "account_age_days": 1},
        )
        r1 = self.scorer.score(det, self.identity, prior_violations=2)
        r2 = self.scorer.score(det, self.identity, prior_violations=2)
        assert r1.score == r2.score
        assert r1.breakdown == r2.breakdown
        assert r1.verdict == r2.verdict

    def test_high_confidence_detection(self):
        # All signals firing — should reach "confirmed"
        det = make_detection(
            title="Liga Champions 2025",
            snapshot_html="vidio.com content.vidio.com vidio-player player.vidio.com",
            extra={
                "content_id_match": True,
                "fingerprint_match_level": "confirmed",
                "view_count": 100_000,
                "account_age_days": 1,
            },
        )
        result = self.scorer.score(det, self.identity, prior_violations=5)
        assert result.verdict == "confirmed"
        assert result.score == 100

    def test_zero_signal_detection(self):
        det = make_detection(title="Resep Masak Ayam", snapshot_html="<html></html>")
        result = self.scorer.score(det, self.identity, prior_violations=0)
        assert result.verdict == "ignore"
        assert result.score == 0

    def test_prior_violations_affect_score(self):
        det = make_detection()
        r_clean = self.scorer.score(det, self.identity, prior_violations=0)
        r_repeat = self.scorer.score(det, self.identity, prior_violations=4)
        assert r_repeat.score > r_clean.score
        assert r_repeat.breakdown["offender_repeat_history"] == 15
        assert r_clean.breakdown["offender_repeat_history"] == 0

    def test_scored_at_is_iso8601(self):
        det = make_detection()
        result = self.scorer.score(det, self.identity)
        # Basic ISO 8601 UTC check: "YYYY-MM-DDTHH:MM:SSZ"
        assert "T" in result.scored_at
        assert result.scored_at.endswith("Z")


# ---------------------------------------------------------------------------
# Clamping: sub-method returning over-max doesn't break total
# ---------------------------------------------------------------------------

class TestClamping:
    def test_total_never_exceeds_100(self):
        # Even with all signals at max, total must be exactly 100 (not more)
        scorer = EvidenceScorer(content_catalog=CATALOG)
        det = make_detection(
            title="Liga Champions 2025",
            snapshot_html=" ".join(["vidio.com"] * 5),
            extra={
                "content_id_match": True,
                "fingerprint_match_level": "confirmed",
                "view_count": 1_000_000,
                "account_age_days": 0,
            },
        )
        result = scorer.score(det, make_identity(), prior_violations=100)
        assert result.score <= 100

    def test_total_never_below_zero(self):
        scorer = EvidenceScorer()
        det = make_detection()
        result = scorer.score(det, make_identity(), prior_violations=0)
        assert result.score >= 0


# ---------------------------------------------------------------------------
# batch_score
# ---------------------------------------------------------------------------

class TestBatchScore:
    def test_empty_returns_empty(self):
        assert batch_score([], {}) == []

    def test_sorted_descending(self):
        scorer = EvidenceScorer(content_catalog=CATALOG)

        det_high = make_detection(
            title="Liga Champions 2025",
            extra={"fingerprint_match_level": "confirmed"},
        )
        det_low = make_detection(title="Resep Masak Ayam")
        id_high = make_identity("pid_A")
        id_low  = make_identity("pid_B")

        results = batch_score(
            [(det_high, id_high), (det_low, id_low)],
            prior_violations_map={},
            scorer=scorer,
        )
        assert results[0].score >= results[1].score

    def test_prior_violations_map_applied(self):
        scorer = EvidenceScorer()
        det = make_detection()
        identity = make_identity("pid_repeat")

        results = batch_score(
            [(det, identity)],
            prior_violations_map={"pid_repeat": 4},
            scorer=scorer,
        )
        assert results[0].breakdown["offender_repeat_history"] == 15

    def test_unknown_id_defaults_to_zero_violations(self):
        scorer = EvidenceScorer()
        det = make_detection()
        identity = make_identity("pid_unknown")

        results = batch_score(
            [(det, identity)],
            prior_violations_map={},   # pid_unknown not in map
            scorer=scorer,
        )
        assert results[0].breakdown["offender_repeat_history"] == 0

    def test_default_scorer_created_when_none(self):
        det = make_detection()
        results = batch_score([(det, make_identity())], {}, scorer=None)
        assert len(results) == 1
        # Default scorer has no catalog → title signal = 0
        assert results[0].breakdown["title_keyword_match"] == 0
