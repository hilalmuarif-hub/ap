"""
evidence_scorer.py — Deterministic 0-100 evidence scoring engine.

Score represents confidence that a detection is genuine piracy of Vidio content.
All scoring rules must be deterministic and auditable — no ML, no randomness.
Same input always produces the same score.

Signals that require external tooling (image hashing, audio fingerprint) use
passthrough flags from RawDetection.extra set by the crawler. The scorer reads
those flags but never calls external APIs — separation of concerns is strict.
"""

import datetime
from dataclasses import dataclass

from detection import RawDetection
from identity import OffenderIdentity
from normalize_query import similarity


@dataclass
class ScoredEvidence:
    detection: RawDetection
    identity: OffenderIdentity
    score: int                  # 0–100
    breakdown: dict[str, int]   # per-signal scores; values sum to `score`
    verdict: str                # "confirmed" | "likely" | "possible" | "ignore"
    scored_at: str              # ISO 8601 UTC timestamp of when scoring ran


# Verdict thresholds — tune these as false-positive rates are measured in prod
VERDICT_THRESHOLDS: dict[str, int] = {
    "confirmed": 80,
    "likely":    60,
    "possible":  25,   # lowered from 40 — title match alone (25pts) is enough to queue
    # below 25 → "ignore"
}

# Per-signal max contributions — must always sum to 100
SIGNAL_WEIGHTS: dict[str, int] = {
    "title_keyword_match":    25,   # Vidio content title found in post title
    "content_id_watermark":   30,   # Vidio branding / Content ID match in media
    "stream_key_fingerprint": 20,   # re-stream fingerprint match
    "offender_repeat_history": 15,  # prior confirmed violations by same offender
    "engagement_anomaly":     10,   # abnormally high views for account age
}

assert sum(SIGNAL_WEIGHTS.values()) == 100, "SIGNAL_WEIGHTS must sum to 100"

# Vidio brand strings expected in HTML snapshots of legitimately Vidio-hosted content.
# Pirates who re-embed without stripping metadata leave these behind.
_DEFAULT_BRAND_MARKERS: list[str] = [
    "vidio.com",
    "content.vidio.com",
    "vidio-player",
    "player.vidio.com",
    "vidio premium",
]


class EvidenceScorer:
    """
    Score a RawDetection + OffenderIdentity pair against the signal matrix.

    All scoring is deterministic: same (detection, identity, prior_violations)
    always produces the same (score, breakdown, verdict). The only non-deterministic
    field is `scored_at` (wall-clock timestamp).

    Args:
        content_catalog: list of canonical Vidio content titles used for title
                         matching. Pass an empty list to disable that signal.
        brand_markers: HTML strings that indicate Vidio-originated content.
                       Defaults to _DEFAULT_BRAND_MARKERS if not provided.
    """

    def __init__(
        self,
        content_catalog: list[str] | None = None,
        brand_markers: list[str] | None = None,
    ) -> None:
        self._content_catalog: list[str] = content_catalog or []
        # Pre-lowercase markers so html.lower() comparison is O(n) without per-call work
        self._brand_markers: list[str] = [
            m.lower() for m in (brand_markers if brand_markers is not None else _DEFAULT_BRAND_MARKERS)
        ]

    def score(
        self,
        detection: RawDetection,
        identity: OffenderIdentity,
        prior_violations: int = 0,
    ) -> ScoredEvidence:
        """
        Compute a deterministic 0-100 score for a detection.

        Each signal is capped at its SIGNAL_WEIGHTS max before summing,
        so bugs in individual sub-methods cannot push the total above 100.

        Args:
            detection: raw detection from a crawler
            identity: resolved permanent identity
            prior_violations: confirmed violation count for this offender
                              from offender_registry.py

        Returns:
            ScoredEvidence with full per-signal breakdown and verdict.
        """
        raw: dict[str, int] = {
            "title_keyword_match":    self._score_title_keyword_match(detection),
            "content_id_watermark":   self._score_content_id_watermark(detection),
            "stream_key_fingerprint": self._score_stream_key_fingerprint(detection),
            "offender_repeat_history": self._score_offender_repeat_history(prior_violations),
            "engagement_anomaly":     self._score_engagement_anomaly(detection),
        }

        # Clamp each signal to [0, its max weight] — defensive against sub-method bugs
        breakdown: dict[str, int] = {
            sig: max(0, min(SIGNAL_WEIGHTS[sig], raw[sig]))
            for sig in SIGNAL_WEIGHTS
        }

        total = min(100, max(0, sum(breakdown.values())))

        return ScoredEvidence(
            detection=detection,
            identity=identity,
            score=total,
            breakdown=breakdown,
            verdict=self.verdict_from_score(total),
            scored_at=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    def _score_title_keyword_match(self, detection: RawDetection) -> int:
        """
        Score 0–25 based on similarity between the post title and any title
        in the Vidio content catalog.

        Uses token-set ratio (order-insensitive) so that "Liga Champions Vidio 2025"
        matches "Vidio Liga Champions" at high similarity.

        Tiers:
          sim ≥ 0.95 → 25 (near-exact match)
          sim ≥ 0.80 → 15 (strong fuzzy match)
          sim ≥ 0.60 → 5  (partial match — warrants manual review)
          sim <  0.60 → 0
        """
        if not self._content_catalog or not detection.title.strip():
            return 0

        best_sim = max(
            similarity(detection.title, catalog_title)
            for catalog_title in self._content_catalog
        )

        max_w = SIGNAL_WEIGHTS["title_keyword_match"]  # 25
        if best_sim >= 0.95:
            return max_w
        if best_sim >= 0.80:
            return int(max_w * 0.60)   # 15
        if best_sim >= 0.60:
            return int(max_w * 0.20)   # 5
        return 0

    def _score_content_id_watermark(self, detection: RawDetection) -> int:
        """
        Score 0–30 based on Vidio branding evidence in the detection.

        Two sources checked in order:
          1. `detection.extra["content_id_match"] == True` — set by the crawler
             when it confirmed a Content ID hit via API. Full score (30).
          2. Vidio brand marker strings found in snapshot_html — proxy for
             re-uploads that left Vidio's player embed code intact.
             3+ markers → 15, 1–2 markers → 7, 0 → 0.

        Note: full image/audio Content ID matching requires external tooling.
        The `content_id_match` flag is the hook for that integration.
        """
        max_w = SIGNAL_WEIGHTS["content_id_watermark"]  # 30

        if detection.extra.get("content_id_match"):
            return max_w

        html_lower = detection.snapshot_html.lower()
        marker_hits = sum(1 for m in self._brand_markers if m in html_lower)

        if marker_hits >= 3:
            return int(max_w * 0.50)   # 15
        if marker_hits >= 1:
            return int(max_w * 0.25)   # 7  (floor of 7.5)
        return 0

    def _score_stream_key_fingerprint(self, detection: RawDetection) -> int:
        """
        Score 0–20 based on audio/video fingerprint match.

        Reads `detection.extra["fingerprint_match_level"]` set by the crawler
        or a pre-processing step that ran fingerprint analysis:
          "confirmed" → 20 (fingerprint database hit with high confidence)
          "likely"    → 10 (candidate match below confirmation threshold)
          absent/other → 0

        Full fingerprint extraction (Acoustid / in-house DB) is out of scope
        for the scorer — the crawler owns that integration.
        """
        max_w = SIGNAL_WEIGHTS["stream_key_fingerprint"]  # 20
        level = detection.extra.get("fingerprint_match_level", "none")

        if level == "confirmed":
            return max_w
        if level == "likely":
            return max_w // 2   # 10
        return 0

    def _score_offender_repeat_history(self, prior_violations: int) -> int:
        """
        Score 0–15 based on how many prior confirmed violations this offender has.

        A habitual offender is significantly more likely to be pirating again.
        Tiers: 0 → 0, 1 → 5, 2–3 → 10, 4+ → 15.
        """
        if prior_violations >= 4:
            return SIGNAL_WEIGHTS["offender_repeat_history"]   # 15
        if prior_violations >= 2:
            return 10
        if prior_violations == 1:
            return 5
        return 0

    def _score_engagement_anomaly(self, detection: RawDetection) -> int:
        """
        Score 0–10 based on views-per-day relative to account age.

        Piracy accounts typically accumulate massive views on young accounts.
        Reads `detection.extra["view_count"]` and `extra["account_age_days"]`.
        Returns 0 silently if either field is absent — we don't penalize
        detections where the crawler couldn't retrieve engagement metrics.

        Tiers (views / max(account_age_days, 1)):
          ≥ 10 000 → 10
          ≥  1 000 →  7
          ≥    100 →  4
          <    100 →  0
        """
        view_count = detection.extra.get("view_count")
        account_age_days = detection.extra.get("account_age_days")

        if view_count is None or account_age_days is None:
            return 0

        views_per_day = view_count / max(int(account_age_days), 1)

        max_w = SIGNAL_WEIGHTS["engagement_anomaly"]  # 10
        if views_per_day >= 10_000:
            return max_w
        if views_per_day >= 1_000:
            return int(max_w * 0.7)   # 7
        if views_per_day >= 100:
            return int(max_w * 0.4)   # 4
        return 0

    @staticmethod
    def verdict_from_score(score: int) -> str:
        """
        Map a 0-100 score to a verdict string.

        Thresholds (inclusive lower bound):
          80+ → "confirmed"   auto-flagged for takedown
          60+ → "likely"      manual review queue
          40+ → "possible"    low-priority queue
           0+ → "ignore"      discarded, not written to Sheets
        """
        if score >= VERDICT_THRESHOLDS["confirmed"]:
            return "confirmed"
        if score >= VERDICT_THRESHOLDS["likely"]:
            return "likely"
        if score >= VERDICT_THRESHOLDS["possible"]:
            return "possible"
        return "ignore"


def batch_score(
    detections: list[tuple[RawDetection, OffenderIdentity]],
    prior_violations_map: dict[str, int],
    scorer: EvidenceScorer | None = None,
) -> list[ScoredEvidence]:
    """
    Score a batch of (detection, identity) pairs.

    Args:
        detections: pairs produced by the detection + identity pipeline
        prior_violations_map: permanent_id → prior confirmed violation count
                              (from offender_registry.py)
        scorer: pre-configured EvidenceScorer; creates one with empty catalog
                if not provided (title_keyword_match signal will score 0)

    Returns:
        List of ScoredEvidence sorted by score descending (highest confidence first).
    """
    if scorer is None:
        scorer = EvidenceScorer()

    results: list[ScoredEvidence] = []
    for detection, identity in detections:
        prior = prior_violations_map.get(identity.permanent_id, 0)
        results.append(scorer.score(detection, identity, prior_violations=prior))

    results.sort(key=lambda e: e.score, reverse=True)
    return results
