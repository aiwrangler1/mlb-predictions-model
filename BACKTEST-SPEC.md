# Backtesting Module — Architecture Spec

## Goals
1. Log every model prediction at simulation time (prediction record)
2. At end of each day, fetch actual results from MLB Stats API and record them
3. Compute calibration, accuracy, and ROI metrics over time
4. Expose analytics via /api/backtest/* endpoints for dashboard use

## File Structure
```
mlb-model/
  backtesting/
    __init__.py
    results_db.py      # SQLite storage: predictions + actuals tables
    backtester.py      # log_prediction(), record_actuals(), edge helpers
    analytics.py       # calibration, ATS, ROI, accuracy, trend analysis
    daily_runner.py    # fetch yesterday's scores, close out open predictions
```

## SQLite Schema: mlb_backtest.db

### Table: predictions
Every time /api/simulate is called, one row is inserted per game.
```sql
CREATE TABLE predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_ts   TEXT NOT NULL,        -- ISO datetime when model ran
    game_date       TEXT NOT NULL,        -- YYYY-MM-DD
    game_id         INTEGER NOT NULL,
    away_team       TEXT NOT NULL,
    home_team       TEXT NOT NULL,
    away_pitcher    TEXT,
    home_pitcher    TEXT,

    -- Model win probabilities
    model_away_win  REAL NOT NULL,        -- e.g. 0.484
    model_home_win  REAL NOT NULL,
    model_total     REAL NOT NULL,        -- projected runs
    model_away_runs REAL,
    model_home_runs REAL,
    model_f5_away   REAL,
    model_f5_home   REAL,
    model_f5_draw   REAL,

    -- Market odds at prediction time
    mkt_away_ml     INTEGER,             -- American odds e.g. +105
    mkt_home_ml     INTEGER,
    mkt_total_line  REAL,               -- e.g. 7.5
    mkt_total_dir   TEXT,               -- 'over' or 'under' (model lean)
    mkt_run_line    REAL,               -- usually 1.5

    -- Computed edges
    edge_away       REAL,               -- model_away_win - mkt_away_implied (decimal)
    edge_home       REAL,

    -- Context
    lineup_source   TEXT,               -- 'actual' or 'projected'
    park_factor     REAL,
    weather_temp    INTEGER,
    umpire_name     TEXT,
    n_sims          INTEGER DEFAULT 5000,

    UNIQUE(game_date, game_id)          -- one prediction row per game per day
);
```

### Table: actuals
Populated by daily_runner.py after games complete.
```sql
CREATE TABLE actuals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    game_date       TEXT NOT NULL,
    game_id         INTEGER NOT NULL UNIQUE,
    away_team       TEXT NOT NULL,
    home_team       TEXT NOT NULL,

    -- Actual game outcome
    away_score      INTEGER NOT NULL,
    home_score      INTEGER NOT NULL,
    winner          TEXT NOT NULL,       -- 'away' or 'home'
    total_runs      INTEGER NOT NULL,
    innings         INTEGER DEFAULT 9,
    game_status     TEXT DEFAULT 'Final',

    -- First 5 innings
    f5_away_score   INTEGER,
    f5_home_score   INTEGER,
    f5_winner       TEXT,               -- 'away', 'home', or 'draw'

    recorded_ts     TEXT NOT NULL        -- when this was fetched
);
```

### Table: bet_log (optional, for tracking hypothetical bets)
```sql
CREATE TABLE bet_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    game_date       TEXT NOT NULL,
    game_id         INTEGER NOT NULL,
    bet_type        TEXT NOT NULL,       -- 'ml_away', 'ml_home', 'over', 'under', 'f5_away', etc.
    model_prob      REAL NOT NULL,
    market_odds     INTEGER NOT NULL,    -- American odds
    edge            REAL NOT NULL,       -- model_prob - implied
    kelly_fraction  REAL,               -- recommended bet size as fraction of bankroll
    hypothetical_units REAL,            -- units bet at half-Kelly on $500 bankroll
    result          TEXT,               -- 'win', 'loss', 'push' (filled by daily_runner)
    profit_units    REAL                 -- profit/loss in units
);
```

## Key Functions

### results_db.py
- `init_db(db_path)` — create tables if not exist
- `upsert_prediction(conn, pred_dict)` — insert/replace prediction
- `upsert_actual(conn, actual_dict)` — insert/replace actual result
- `get_open_predictions(conn, date)` — predictions without actuals yet
- `get_predictions_range(conn, start, end)` — for analytics window

### backtester.py
- `log_prediction(game_id, game_date, sim_result, market_odds)` — called from server.py after each simulation
- `record_actuals_for_date(game_date)` — fetch MLB Stats API, update actuals table
- `compute_bet_log(game_date)` — for each prediction with actuals, compute win/loss

### analytics.py
- `calibration_report(df)` — bucket model probs into deciles, compare to actual win rates
- `ml_accuracy(df)` — % of games where model favorite won
- `ats_record(df)` — model predicted spread vs actual margin
- `total_accuracy(df)` — model total vs actual, over/under hit rates  
- `roi_report(df, kelly_fraction=0.25)` — hypothetical P&L at half-Kelly
- `edge_accuracy(df, min_edge=0.03)` — accuracy only on "high confidence" predictions
- `summary_report(start_date, end_date)` — full printable summary

### daily_runner.py
- `run_daily_close(game_date)` — main function: fetch scores, record actuals, compute bets, print summary
- Designed to be called by a cron/scheduler at 11:59 PM or next morning
- Idempotent: safe to run multiple times

## Integration with server.py
After `return JSONResponse(...)` in api_simulate:
- Call `log_prediction(game_id, ...)` asynchronously (background task, non-blocking)
- Add endpoints:
  - GET /api/backtest/summary?days=30
  - GET /api/backtest/calibration?days=30
  - GET /api/backtest/games?start=YYYY-MM-DD&end=YYYY-MM-DD
  - POST /api/backtest/record-actuals?date=YYYY-MM-DD  (manual trigger)

## Database Location
`/home/user/workspace/mlb-model/data/mlb_backtest.db`
- gitignored (it's data)
- But schema SQL and all Python code are committed
- Include a `backtest_schema.sql` file so the DB can be recreated from scratch

## MLB Stats API — fetching actuals
```python
# Get final score for a game
url = f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
# Fields: liveData.boxscore, gameData.status.detailedState == 'Final'
# Score: liveData.linescore.teams.away.runs / home.runs
# Innings: liveData.linescore.currentInning
# F5: sum liveData.linescore.innings[0:5]
```
