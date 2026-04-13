# Agent Coordination

## Current State
- Canonical working clone: `fresh-clone/`
- Default branch: `main`
- `main` includes:
  - core MLB model/server stack
  - DFS optimizer stack (`dfs_optimizer.py`, `showdown_optimizer.py`, `outcome_distribution.py`, `projection_overlay.py`)
  - backtesting module under `backtesting/`
- Experimental branch still separate: `origin/feature/9-event-model`
- As of 2026-04-07, no open GitHub PRs were found for `aiwrangler1/mlb-predictions-model`

## Branch Map
- `main`
  - Base branch for all active work
  - Contains the recent merge commit `621bfc9` for backtesting and optimization docs
- `feature/odds-tracking`
  - Odds snapshot DB, line movement tracking, CLV analysis, auto_pipeline
  - New files: schema_v2.sql, odds_tracker.py, auto_pipeline.py
  - Updated: analytics.py (CLV + daily breakdown)
  - Ready for review/merge
- `feature/9-event-model`
  - Planning-only / incomplete experimental work
  - Do not merge into `main` until implementation is complete and validated

## Confirmed Issues
1. Backtesting DB initialization is broken in a fresh clone.
   - File: `backtesting/results_db.py`
   - Problem: DB path points to `data/mlb_backtest.db`, but `data/` is not created automatically.
   - Current behavior: `sqlite3.OperationalError: unable to open database file`

2. Backtesting logging in the API is wired to the wrong payload shape.
   - File: `server.py`
   - Problem: `lineup_source`, `umpire_name`, and market odds are read from top-level keys that are not actually present in the returned response payload.
   - Consequence: prediction rows are logged with missing market/context fields, which breaks downstream edge and ROI analytics.

## Active Plan
1. Fix DB initialization in `backtesting/results_db.py`
   - Ensure parent directory exists before opening SQLite
   - Keep schema bootstrap idempotent

2. Fix API-to-backtesting logging integration in `server.py`
   - Read market odds from `results["market_odds"]["consensus"]`
   - Read context fields from `results["adjustments"]`
   - Preserve non-blocking behavior for background logging

3. Add minimal regression coverage
   - Validate DB bootstrap in a fresh checkout
   - Validate prediction logging with representative nested payload data

4. Reassess whether any follow-up cleanup is needed
   - `.gitignore` for generated local artifacts
   - lightweight docs for running backtesting locally

## Ownership
- Primary owner from this point: Codex in this thread
- If another agent contributes:
  - they must not modify the same files without explicit coordination
  - they should treat `main` as the source of truth
  - they should avoid merging `feature/9-event-model`

## Coordination Rules
- Before changing scope, update this file.
- Record only concrete decisions and verified defects here.
- Do not use ZIP snapshots as the source of truth when a live clone exists.
- Prefer narrow fixes over broad refactors until the recent backtesting merge is stable.

## Files Expected To Change Next
- `server.py`
- `backtesting/results_db.py`
- possibly a small test or validation helper under `backtesting/` or repo root
