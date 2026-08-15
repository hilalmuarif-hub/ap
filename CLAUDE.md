# CLAUDE.md — AP Army Pipeline

## Project context

This is Vidio.com's internal anti-piracy detection pipeline. It automatically discovers unauthorized streams / re-uploads of Vidio-licensed content on social platforms, scores the evidence, and maintains a verified offender registry in Google Sheets for the anti-piracy (AP) team to act on.

**Owner:** AP Army team (hilal.muarif@vidio.com)
**Trigger:** GitHub Actions cron, once per day
**Dashboard:** Google Sheets (spreadsheet ID in env)

## Current scope (v1)

- Platform: Facebook only (Pages, Groups, Reels, public profiles)
- Content: Live streams and VOD re-uploads
- Action: Detection + scoring + registry; takedown requests are manual

## Design principles

1. **Deterministic scoring** — `evidence_scorer.py` must produce the same score for the same input, always. No ML, no randomness. Ops must be able to explain every score to legal.
2. **Permanent IDs only** — never store display names as primary keys. Display names change; numeric platform IDs do not.
3. **Idempotent writes** — re-running the pipeline must not duplicate rows in Sheets. Check cluster_id / permanent_id before every append.
4. **Canary-first** — every pipeline run starts and ends with a health check. Silent failures are worse than noisy ones.
5. **Separation of concerns** — crawlers produce `RawDetection`, scorer consumes it, writer consumes scored output. No module skips the chain.

## Module responsibilities (strict)

| Module | Owns | Does NOT own |
|--------|------|-------------|
| `detection.py` | raw HTML capture, channel_id extraction | identity resolution, scoring |
| `identity.py` | permanent ID resolution | scraping, scoring |
| `normalize_query.py` | text normalization, similarity | query dispatch, platform logic |
| `evidence_scorer.py` | deterministic 0-100 scoring | registry writes, sheet writes |
| `offender_registry.py` | offender lifecycle (upsert, status) | detection, scoring |
| `dedupe_cluster.py` | dedup logic | identity resolution, scoring |
| `sheet_writer.py` | Google Sheets I/O | business logic |
| `canary.py` | health checks, alerting | pipeline logic |

## Key data contracts

- `RawDetection` — output of crawlers; `channel_id` must be a permanent platform ID (not display name)
- `OffenderIdentity` — output of identity resolver; `permanent_id` is the canonical key
- `ScoredEvidence` — output of scorer; `score` is 0-100 int, `verdict` is one of: confirmed / likely / possible / ignore
- `OffenderRecord` — registry entry; `status` is one of: active / suspended / removed / appealing

## Environment variables required

```
GOOGLE_SHEETS_ID          # spreadsheet ID for Sheets writes
GOOGLE_SERVICE_ACCOUNT    # path to service_account.json
FB_COOKIE_FILE            # path to exported Facebook cookies for Playwright
SLACK_WEBHOOK_URL         # canary alert destination
LOG_LEVEL                 # DEBUG | INFO | WARNING (default: INFO)
```

## Coding conventions

- All timestamps: ISO 8601 UTC strings (`2025-01-15T08:00:00Z`)
- All IDs: strings, not integers (platform IDs can exceed int32)
- Logging: use `structlog` with key=value pairs, not f-strings
- No bare `except:` — always catch specific exceptions
- New platforms: subclass `BasePlatformCrawler`, do not fork `detection.py`
- Tests: place in `tests/` directory, one test file per module

## What NOT to do

- Do not store credentials in code — use `.env` / GitHub Secrets
- Do not use display names as join keys — they change silently
- Do not skip `canary.py` even during manual runs
- Do not write to Sheets outside of `sheet_writer.py`
- Do not add ML/probabilistic scoring — legal requires auditability
