# AP Army Pipeline

Anti-piracy detection pipeline for Vidio.com. Crawls platforms for unauthorized streams, scores evidence, and maintains a verified offender registry in Google Sheets.

## Architecture

```
normalize_query.py   → build search queries from content titles
        ↓
detection.py         → crawl platforms (Facebook first), emit RawDetection
        ↓
identity.py          → resolve display names → permanent IDs
        ↓
dedupe_cluster.py    → collapse duplicate detections into clusters
        ↓
evidence_scorer.py   → score 0–100, assign verdict (confirmed/likely/possible)
        ↓
offender_registry.py → upsert to verified offender database
        ↓
sheet_writer.py      → write results to Google Sheets dashboard
        ↓
canary.py            → health check + alert on anomalies
```

Orchestrated daily via `daily.sh`, triggered by GitHub Actions.

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt
playwright install chromium

# 2. Configure credentials
cp .env.example .env
# Fill in: GOOGLE_SHEETS_ID, FB_COOKIE_FILE, SLACK_WEBHOOK_URL

# 3. Add service account
# Place service_account.json in project root (from Google Cloud Console)

# 4. Run pipeline
./daily.sh
```

## Key Files

| File | Purpose |
|------|---------|
| `detection.py` | Platform crawlers (Playwright) |
| `identity.py` | Permanent ID resolution |
| `evidence_scorer.py` | 0-100 deterministic scoring |
| `offender_registry.py` | Verified offender CRUD |
| `dedupe_cluster.py` | Deduplication |
| `normalize_query.py` | Text normalization + fuzzy match |
| `sheet_writer.py` | Google Sheets writer |
| `canary.py` | Health checks + alerts |
| `daily.sh` | Pipeline orchestrator |

## Verdict Thresholds

| Score | Verdict | Action |
|-------|---------|--------|
| 80–100 | confirmed | Auto-flag for takedown |
| 60–79 | likely | Manual review |
| 40–59 | possible | Low priority queue |
| 0–39 | ignore | Discarded |

## Adding a New Platform

1. Subclass `BasePlatformCrawler` in `detection.py`
2. Add a `_resolve_<platform>` method in `identity.py`
3. Register the crawler in `run_all_crawlers()`
4. Add pre-flight check in `canary.py`
