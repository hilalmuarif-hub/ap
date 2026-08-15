"""
dedupe_cluster.py — Deduplication and clustering of raw detections.

Multiple crawlers and queries may surface the same infringing content.
This module collapses duplicates before scoring and registry writes,
preventing double-counting of violations.
"""

import hashlib
from collections import defaultdict
from dataclasses import dataclass

from detection import RawDetection
from normalize_query import normalize, normalize_url, similarity


@dataclass
class DetectionCluster:
    canonical: RawDetection          # best representative of the cluster
    duplicates: list[RawDetection]   # other detections collapsed into this cluster
    cluster_id: str                  # stable hash-based ID for this cluster
    cluster_size: int                # total detections (canonical + duplicates)


class _UnionFind:
    """Path-compressed, union-by-rank union-find over integer indices 0..n-1."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def groups(self) -> dict[int, list[int]]:
        """Return {root_index: [member_indices]} for all clusters."""
        result: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            result[self.find(i)].append(i)
        return dict(result)


def deduplicate(
    detections: list[RawDetection],
    url_exact: bool = True,
    fuzzy_title: bool = True,
    fuzzy_threshold: float = 0.85,
) -> list[DetectionCluster]:
    """
    Deduplicate a flat list of RawDetections into clusters.

    Three passes applied in order; union-find ensures transitivity
    (if A=B and B=C by any rule, all three land in the same cluster):

      Pass 1 — exact normalized URL:
        Same URL seen from multiple queries → one cluster.
        Normalization strips UTM params, FB mobile domain variants, etc.

      Pass 2 — same channel_id + exact normalized title:
        Same channel posting the same content under the same title.

      Pass 3 — same channel_id + fuzzy title similarity:
        Same channel, slightly different title wording.
        Only runs within a channel group, so it never merges
        across different offenders.

    Args:
        detections: raw output from run_all_crawlers()
        url_exact: enable Pass 1 (always recommended)
        fuzzy_title: enable Pass 3
        fuzzy_threshold: similarity cutoff for Pass 3 (0.0–1.0)

    Returns:
        List of DetectionCluster sorted by canonical.detected_at ascending.
        Empty input returns empty list.
    """
    if not detections:
        return []

    n = len(detections)
    uf = _UnionFind(n)

    # --- Pass 1: exact normalized URL ---
    if url_exact:
        url_to_first: dict[str, int] = {}
        for i, det in enumerate(detections):
            norm = normalize_url(det.url)
            if norm in url_to_first:
                uf.union(i, url_to_first[norm])
            else:
                url_to_first[norm] = i

    # --- Pass 2: same channel + exact normalized title ---
    channel_title_to_first: dict[tuple, int] = {}
    for i, det in enumerate(detections):
        key = (det.platform, det.channel_id, normalize(det.title))
        if key in channel_title_to_first:
            uf.union(i, channel_title_to_first[key])
        else:
            channel_title_to_first[key] = i

    # --- Pass 3: same channel + fuzzy title ---
    if fuzzy_title:
        channel_to_idxs: dict[tuple, list[int]] = defaultdict(list)
        for i, det in enumerate(detections):
            channel_to_idxs[(det.platform, det.channel_id)].append(i)

        for idxs in channel_to_idxs.values():
            if len(idxs) < 2:
                continue
            # Pre-normalize once; similarity() is idempotent on normalized input
            norm_titles = [(i, normalize(detections[i].title)) for i in idxs]
            for a_pos in range(len(norm_titles)):
                for b_pos in range(a_pos + 1, len(norm_titles)):
                    idx_a, title_a = norm_titles[a_pos]
                    idx_b, title_b = norm_titles[b_pos]
                    if uf.find(idx_a) == uf.find(idx_b):
                        continue  # already merged — skip the similarity call
                    if similarity(title_a, title_b) >= fuzzy_threshold:
                        uf.union(idx_a, idx_b)

    # Build DetectionCluster objects from union-find groups
    clusters: list[DetectionCluster] = []
    for member_idxs in uf.groups().values():
        members = [detections[i] for i in member_idxs]
        canonical = pick_canonical(members)
        clusters.append(DetectionCluster(
            canonical=canonical,
            duplicates=[d for d in members if d is not canonical],
            cluster_id=cluster_id_for(canonical),
            cluster_size=len(members),
        ))

    # ISO 8601 strings sort lexicographically in chronological order
    clusters.sort(key=lambda c: c.canonical.detected_at)
    return clusters


def cluster_id_for(detection: RawDetection) -> str:
    """
    Compute a stable deterministic cluster ID from a detection's key fields.

    Hashes (platform, channel_id, normalize_url(url)) so that:
      - The same FB video URL with/without UTM params → same ID
      - FB mobile URL and desktop URL for the same video → same ID
      - Different videos on the same channel → different IDs

    Returns the first 16 hex characters of SHA-256 (64-bit collision space,
    sufficient for the expected registry size of thousands of entries).
    Null-byte delimiters prevent "fbvideo" + "123" ≠ "fb" + "video123" collisions.
    """
    key = "\x00".join([
        detection.platform,
        detection.channel_id,
        normalize_url(detection.url),
    ])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def merge_clusters(
    existing: list[DetectionCluster],
    incoming: list[DetectionCluster],
) -> list[DetectionCluster]:
    """
    Merge a new batch of clusters into an existing set (e.g. across daily runs).

    Matching is by cluster_id. When IDs collide:
      - All members (canonical + duplicates) are pooled
      - URL-level dedup removes exact duplicates from the pool
      - pick_canonical() re-selects the best representative
      - cluster_size is updated to the merged count

    Clusters in `incoming` that have no match in `existing` are appended as-is.
    Output is sorted by canonical.detected_at ascending.
    """
    by_id: dict[str, DetectionCluster] = {c.cluster_id: c for c in existing}

    for inc in incoming:
        if inc.cluster_id not in by_id:
            by_id[inc.cluster_id] = inc
            continue

        ex = by_id[inc.cluster_id]
        all_members = [ex.canonical, *ex.duplicates, inc.canonical, *inc.duplicates]

        # URL-level dedup within the merged pool — prevents double-counting
        # when the same run is replayed or overlapping date ranges are merged
        seen_urls: set[str] = set()
        unique_members: list[RawDetection] = []
        for d in all_members:
            norm = normalize_url(d.url)
            if norm not in seen_urls:
                seen_urls.add(norm)
                unique_members.append(d)

        canonical = pick_canonical(unique_members)
        by_id[inc.cluster_id] = DetectionCluster(
            canonical=canonical,
            duplicates=[d for d in unique_members if d is not canonical],
            cluster_id=inc.cluster_id,
            cluster_size=len(unique_members),
        )

    return sorted(by_id.values(), key=lambda c: c.canonical.detected_at)


def pick_canonical(candidates: list[RawDetection]) -> RawDetection:
    """
    From a group of duplicate detections, select the single best representative.

    Selection criteria (applied as a sort key, first criterion wins):
      1. Longest snapshot_html  — more HTML means more captured evidence
      2. Earliest detected_at   — first-seen wins on ties (ISO 8601 lexicographic sort)

    Args:
        candidates: non-empty list of RawDetection from the same cluster

    Returns:
        The single best representative detection.

    Raises:
        ValueError: if candidates is empty
    """
    if not candidates:
        raise ValueError("Cannot pick canonical from empty candidate list")
    return sorted(
        candidates,
        key=lambda d: (-len(d.snapshot_html), d.detected_at),
    )[0]
