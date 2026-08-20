#!/usr/bin/env python3
"""
pipeline.py — Single CLI entrypoint for the AP Army anti-piracy pipeline.

Each subcommand maps to one logical phase in daily.sh.
All business logic lives in the individual modules; this file only wires them.

Usage:
  python pipeline.py preflight            # pre-flight env check
  python pipeline.py run [OPTIONS]        # full detection → scoring → write run
  python pipeline.py canary --stats FILE  # post-run health check + alert
"""

import dataclasses
import datetime
import json
import os
import sys
import time

import click

from canary import CanaryChecker, PipelineStats, pre_flight_check


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """AP Army anti-piracy pipeline."""


# ---------------------------------------------------------------------------
# preflight — step 1
# ---------------------------------------------------------------------------

@cli.command("preflight")
def cmd_preflight() -> None:
    """
    Verify all required environment variables and credentials exist.

    Exits 0 if all critical checks pass, 1 if any critical check fails.
    Warning-only checks (GCHAT_WEBHOOK_URL / SLACK_WEBHOOK_URL) do not cause a non-zero exit.
    """
    passed, results = pre_flight_check()
    for r in results:
        icon = "[OK]" if r["status"] == "ok" else ("[??]" if r["status"] == "degraded" else "[!!]")
        click.echo(f"  {icon} {r['name']}: {r['message']}")
    sys.exit(0 if passed else 1)


# ---------------------------------------------------------------------------
# run — steps 2-8
# ---------------------------------------------------------------------------

@cli.command("run")
@click.option("--run-id",       required=True, help="Unique run identifier (e.g. run_20250115T080000Z)")
@click.option("--titles",       "titles_path",  default="content_titles.txt", show_default=True,
              help="Path to content titles file (one title per line, # for comments)")
@click.option("--stats-output", "stats_path",   default="stats.json",         show_default=True,
              help="Path to write run stats JSON (read by canary subcommand)")
@click.option("--dry-run",      is_flag=True,
              help="Skip registry update and Google Sheets writes; crawl and score only")
@click.option("--platform",     multiple=True,
              help="Platforms to crawl; may be repeated. Default: all registered platforms")
def cmd_run(
    run_id: str,
    titles_path: str,
    stats_path: str,
    dry_run: bool,
    platform: tuple[str, ...],
) -> None:
    """
    Run the full detection -> identity -> dedup -> score -> write pipeline.

    Critical steps (detect, dedupe, score) abort on failure.
    Non-critical steps (registry, sheet write) log errors and continue.
    Stats are written to --stats-output for the canary subcommand.
    """
    stats = PipelineStats()
    t_start = time.monotonic()
    sheet_write_error_msgs: list[str] = []

    # ---- Step 2: Load content titles ----------------------------------------
    try:
        with open(titles_path) as f:
            content_titles = [
                line.strip() for line in f
                if line.strip() and not line.startswith("#")
            ]
    except FileNotFoundError:
        _abort(f"titles file not found: {titles_path!r}")
        return

    # ---- Step 3: Normalize queries ------------------------------------------
    from normalize_query import expand_to_queries
    queries = expand_to_queries(content_titles)
    _step("normalize", f"{len(queries)} queries from {len(content_titles)} titles")

    # ---- Step 4: Detect -----------------------------------------------------
    from detection import run_all_crawlers
    platforms = list(platform) or None
    stats.platforms_crawled = platforms or ["facebook"]

    try:
        raw_detections = list(run_all_crawlers(queries, platforms=platforms))
    except Exception as exc:
        _abort(f"detection failed: {exc}")
        return

    stats.detections_raw = len(raw_detections)
    _step("detect", f"{len(raw_detections)} raw detections")

    if not raw_detections:
        _warn("detect", "zero detections - crawler may be blocked. Continuing to canary.")

    # ---- Step 4b: Apply channel whitelist -----------------------------------
    whitelist = _load_channel_whitelist(titles_path)
    if whitelist:
        before = len(raw_detections)
        raw_detections = [d for d in raw_detections if d.channel_id not in whitelist]
        removed = before - len(raw_detections)
        if removed:
            _step("whitelist", f"excluded {removed} detections from {removed} whitelisted channels")

    # ---- Step 5: Resolve identities -----------------------------------------
    from identity import IdentityResolver, OffenderIdentity

    resolver = IdentityResolver()   # no fetcher: URL-only resolution (fast, no I/O)
    url_to_identity: dict[str, OffenderIdentity] = {}

    for det in raw_detections:
        profile_url = _fb_profile_url(det.channel_id) if det.platform == "facebook" else det.url
        identity = resolver.resolve(
            det.platform,
            profile_url,
            display_name_hint=det.channel_name,
        )
        if identity is None:
            # Fallback: wrap the channel_id the crawler already extracted
            identity = OffenderIdentity(
                platform=det.platform,
                permanent_id=det.channel_id,
                display_name=det.channel_name,
                profile_url=profile_url,
                resolved_at=_utc_now(),
                confidence=0.7,   # crawler-extracted ID, not API-confirmed
                metadata={},
            )
        url_to_identity[det.url] = identity

    _step("identify", f"{len(url_to_identity)} identities resolved")

    # ---- Step 6: Deduplicate -------------------------------------------------
    from dedupe_cluster import deduplicate, cluster_id_for

    try:
        clusters = deduplicate(raw_detections)
    except Exception as exc:
        _abort(f"deduplication failed: {exc}")
        return

    stats.clusters_after_dedup = len(clusters)
    _step("dedupe", f"{len(clusters)} clusters (dedup ratio "
          f"{100 * (1 - len(clusters) / max(len(raw_detections), 1)):.0f}%)")

    # ---- Step 7: Score evidence ----------------------------------------------
    from evidence_scorer import EvidenceScorer, ScoredEvidence

    scorer = EvidenceScorer(content_catalog=content_titles)
    scored_items: list[tuple[str, ScoredEvidence]] = []   # (cluster_id, evidence)

    # Build prior violations lookup (populated from registry after refresh below)
    prior_violations: dict[str, int] = {}

    # Initialise Sheets clients early so registry can be refreshed before scoring
    writer, registry = None, None
    if not dry_run:
        writer, registry = _init_sheets(sheet_write_error_msgs)
        if registry is not None:
            registry.refresh_cache()
            for record in registry.list_active():
                key = f"{record.platform}:{record.permanent_id}"
                prior_violations[key] = record.violation_count

    try:
        for cluster in clusters:
            canonical = cluster.canonical
            identity = url_to_identity.get(canonical.url)
            if identity is None:
                continue
            pv_key = f"{identity.platform}:{identity.permanent_id}"
            pv = prior_violations.get(pv_key, 0)
            evidence = scorer.score(canonical, identity, prior_violations=pv)
            scored_items.append((cluster_id_for(canonical), evidence))
    except Exception as exc:
        _abort(f"scoring failed: {exc}")
        return

    stats.confirmed_count = sum(1 for _, e in scored_items if e.verdict == "confirmed")
    stats.likely_count    = sum(1 for _, e in scored_items if e.verdict == "likely")
    stats.possible_count  = sum(1 for _, e in scored_items if e.verdict == "possible")
    stats.ignored_count   = sum(1 for _, e in scored_items if e.verdict == "ignore")
    _step("score",
          f"confirmed={stats.confirmed_count} likely={stats.likely_count} "
          f"possible={stats.possible_count} ignored={stats.ignored_count}")

    # ---- Step 8: Registry update (non-critical) -----------------------------
    if not dry_run and registry is not None:
        try:
            upserted = 0
            for _, evidence in scored_items:
                if evidence.verdict in ("confirmed", "likely"):
                    registry.upsert(evidence)
                    upserted += 1
            _step("registry", f"upserted {upserted} offenders")
        except Exception as exc:
            stats.sheet_write_errors += 1
            msg = f"registry: {exc}"
            sheet_write_error_msgs.append(msg)
            _warn("registry", msg)

    # ---- Step 9: Write to Google Sheets (non-critical) ----------------------
    if not dry_run and writer is not None:
        try:
            # Batch write: 1 read + 1 batch-append instead of N reads + N appends
            batch = [
                (
                    cid,
                    evidence,
                    registry.lookup(evidence.identity.permanent_id,
                                    evidence.identity.platform) if registry else None,
                )
                for cid, evidence in scored_items
                if evidence.verdict != "ignore"
            ]
            written = writer.write_detections_batch(batch)
            _step("sheets", f"wrote {written} detections")
        except Exception as exc:
            stats.sheet_write_errors += 1
            msg = f"sheets-detections: {exc}"
            sheet_write_error_msgs.append(msg)
            _warn("sheets", msg)

        # Run log
        try:
            writer.write_run_log(
                run_id=run_id,
                started_at=_utc_now(),
                finished_at=_utc_now(),
                stats={
                    "detections_found":    stats.detections_raw,
                    "clusters_after_dedup": stats.clusters_after_dedup,
                    "confirmed_count":     stats.confirmed_count,
                    "likely_count":        stats.likely_count,
                    "possible_count":      stats.possible_count,
                    "ignored_count":       stats.ignored_count,
                    "errors":              len(sheet_write_error_msgs),
                    "platforms_crawled":   ",".join(stats.platforms_crawled),
                },
            )
        except Exception as exc:
            _warn("sheets-runlog", str(exc))

    if dry_run:
        _step("dry-run", "skipped registry + sheets writes")

    # ---- Finalise stats ------------------------------------------------------
    stats.duration_seconds = time.monotonic() - t_start
    stats.errors = sheet_write_error_msgs

    _write_stats(stats, stats_path)
    _step("complete", f"run {run_id} finished in {stats.duration_seconds:.0f}s")


# ---------------------------------------------------------------------------
# canary — step 9
# ---------------------------------------------------------------------------

@cli.command("canary")
@click.option("--run-id", required=True, help="Run ID to include in alert messages")
@click.option("--stats",  "stats_path", required=True,
              help="Path to stats.json written by the run subcommand")
def cmd_canary(run_id: str, stats_path: str) -> None:
    """
    Run post-run health checks and send Slack alert if anything is degraded.

    Exits 0 unless the overall status is CRITICAL (to allow CI to fail loudly).
    """
    try:
        with open(stats_path) as f:
            stats_dict = json.load(f)
    except FileNotFoundError:
        click.echo(f"[canary] stats file not found: {stats_path!r}", err=True)
        sys.exit(1)

    # Reconstruct PipelineStats from dict (only known fields)
    known_fields = {f.name for f in dataclasses.fields(PipelineStats)}
    stats = PipelineStats(**{k: v for k, v in stats_dict.items() if k in known_fields})

    checker = CanaryChecker(
        # GChat takes priority; fall back to Slack if set
        alert_webhook_url=(
            os.environ.get("GCHAT_WEBHOOK_URL") or os.environ.get("SLACK_WEBHOOK_URL")
        ),
        runtime_threshold_secs=float(os.environ.get("CANARY_RUNTIME_THRESHOLD_SECS", "3600")),
        dedup_low=float(os.environ.get("CANARY_DEDUP_LOW", "0.10")),
        dedup_high=float(os.environ.get("CANARY_DEDUP_HIGH", "0.70")),
    )
    report = checker.run_all(stats, run_id)

    icon_map = {"ok": "[OK]", "skipped": "[--]", "degraded": "[??]", "critical": "[!!]"}
    click.echo(f"[canary] overall={report.overall_status.upper()}")
    for check in report.checks:
        icon = icon_map.get(check["status"], "[?]")
        click.echo(f"  {icon} {check['name']}: {check['message']}")

    if report.alerts_sent:
        click.echo(f"[canary] alert sent to: {report.alerts_sent}")

    sys.exit(1 if report.overall_status == "critical" else 0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _step(name: str, msg: str) -> None:
    click.echo(f"[{name}] {msg}")


def _warn(name: str, msg: str) -> None:
    click.echo(f"[{name}] WARNING: {msg}", err=True)


def _abort(msg: str) -> None:
    click.echo(f"[ABORT] {msg}", err=True)
    sys.exit(1)


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_channel_whitelist(titles_path: str) -> set[str]:
    """
    Load channel IDs to exclude from detection results.

    Reads channel_whitelist.txt from the same directory as titles_path.
    Lines starting with # are comments. Empty lines are ignored.
    Returns a set of channel_id strings (numeric IDs or usernames).
    """
    import pathlib
    whitelist_path = pathlib.Path(titles_path).parent / "channel_whitelist.txt"
    if not whitelist_path.exists():
        return set()
    ids: set[str] = set()
    for line in whitelist_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.add(line)
    return ids


def _fb_profile_url(channel_id: str) -> str:
    """Construct a Facebook profile URL from a channel_id (numeric or username)."""
    if channel_id.isdigit():
        return f"https://www.facebook.com/profile.php?id={channel_id}"
    return f"https://www.facebook.com/{channel_id}"


def _init_sheets(error_log: list[str]):
    """
    Initialise SheetWriter and OffenderRegistry clients.

    Returns (writer, registry) tuple. Either may be None if init fails
    (missing env vars, invalid credentials). Errors are appended to error_log.
    """
    from offender_registry import OffenderRegistry
    from sheet_writer import GspreadBackend, SheetConfig, SheetWriter

    sheets_id = os.environ.get("GOOGLE_SHEETS_ID", "")
    sa_path = os.environ.get("GOOGLE_SERVICE_ACCOUNT", "service_account.json")
    auth_user_path = os.environ.get("GOOGLE_AUTHORIZED_USER", "authorized_user.json")

    if not sheets_id:
        error_log.append("GOOGLE_SHEETS_ID not set - sheet writes skipped")
        _warn("sheets-init", "GOOGLE_SHEETS_ID not set - sheet writes skipped")
        return None, None

    try:
        config = SheetConfig(
            spreadsheet_id=sheets_id,
            service_account_path=sa_path,
        )
        backend = GspreadBackend(
            service_account_path=sa_path,
            spreadsheet_id=sheets_id,
            authorized_user_path=auth_user_path,
        )
        writer = SheetWriter(config, backend=backend)
        registry = OffenderRegistry(sheet_id=sheets_id, backend=writer)
        return writer, registry
    except Exception as exc:
        msg = f"sheets init: {exc}"
        error_log.append(msg)
        _warn("sheets-init", msg)
        return None, None


def _write_stats(stats: PipelineStats, path: str) -> None:
    """Serialise PipelineStats to JSON and write to path."""
    try:
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(stats), f, indent=2)
    except Exception as exc:
        _warn("stats-write", str(exc))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
