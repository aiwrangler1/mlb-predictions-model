-- MLB Model Backtesting Schema v2
-- Adds: odds_snapshots (timestamped line tracking), line_movement view
-- Migration: run after schema.sql — all IF NOT EXISTS, safe to re-run

-- ─── Odds Snapshots ──────────────────────────────────────────
-- One row per game per snapshot. Captures DK lines at a point in time.
-- "opening" = first snapshot captured for a game
-- "closing" = last snapshot before commence_time
CREATE TABLE IF NOT EXISTS odds_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_ts     TEXT NOT NULL,           -- ISO 8601 UTC timestamp of capture
    game_date       TEXT NOT NULL,           -- YYYY-MM-DD
    game_id         TEXT NOT NULL,           -- Odds API event ID (string)
    mlb_game_id     INTEGER,                 -- MLB Stats API gamePk (nullable, linked later)
    away_team       TEXT NOT NULL,
    home_team       TEXT NOT NULL,
    commence_time   TEXT,                    -- ISO 8601 UTC game start
    -- Moneyline
    away_ml         INTEGER,
    home_ml         INTEGER,
    -- Totals
    total_line      REAL,
    over_price      INTEGER,
    under_price     INTEGER,
    -- Spreads / Run line
    away_spread     REAL,
    away_spread_price INTEGER,
    home_spread     REAL,
    home_spread_price INTEGER,
    -- Meta
    snapshot_type   TEXT DEFAULT 'scheduled', -- 'opening', 'closing', 'scheduled', 'manual'
    bookmaker       TEXT DEFAULT 'draftkings',
    UNIQUE(game_id, captured_ts, bookmaker)
);

CREATE INDEX IF NOT EXISTS idx_odds_game_date  ON odds_snapshots(game_date);
CREATE INDEX IF NOT EXISTS idx_odds_game_id    ON odds_snapshots(game_id);
CREATE INDEX IF NOT EXISTS idx_odds_type       ON odds_snapshots(snapshot_type);

-- ─── Upgrade predictions table ───────────────────────────────
-- Add columns for odds snapshot linkage (safe if already exist)
-- These link predictions to the exact odds snapshot they were evaluated against

-- SQLite doesn't support ADD COLUMN IF NOT EXISTS, so we handle in Python

-- ─── Line Movement View ──────────────────────────────────────
-- Convenient view: opening line, closing line, and movement per game
CREATE VIEW IF NOT EXISTS v_line_movement AS
SELECT
    o.game_date,
    o.game_id,
    o.away_team,
    o.home_team,
    o.commence_time,
    -- Opening lines (first snapshot)
    o.away_ml  AS open_away_ml,
    o.home_ml  AS open_home_ml,
    o.total_line AS open_total,
    -- Closing lines (last snapshot)
    c.away_ml  AS close_away_ml,
    c.home_ml  AS close_home_ml,
    c.total_line AS close_total,
    -- Movement
    (c.away_ml - o.away_ml) AS ml_movement_away,
    (c.total_line - o.total_line) AS total_movement
FROM odds_snapshots o
JOIN odds_snapshots c ON o.game_id = c.game_id
WHERE o.id = (
    SELECT id FROM odds_snapshots s
    WHERE s.game_id = o.game_id
    ORDER BY s.captured_ts ASC LIMIT 1
)
AND c.id = (
    SELECT id FROM odds_snapshots s
    WHERE s.game_id = o.game_id
    ORDER BY s.captured_ts DESC LIMIT 1
)
AND o.id != c.id;

-- ─── Prediction-Odds Linkage Table ──────────────────────────
-- Links each prediction to the odds snapshot it was evaluated against
-- and the closing snapshot for CLV calculation
CREATE TABLE IF NOT EXISTS prediction_odds (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id       INTEGER NOT NULL,       -- predictions.id
    odds_snapshot_id    INTEGER NOT NULL,        -- odds_snapshots.id at time of prediction
    closing_snapshot_id INTEGER,                 -- odds_snapshots.id for closing line (filled later)
    -- Cached edge calculations at prediction time
    edge_away_at_pred   REAL,
    edge_home_at_pred   REAL,
    edge_total_at_pred  REAL,
    -- CLV (closing line value) — filled after closing snapshot
    clv_away            REAL,                   -- model_prob - closing_implied_prob
    clv_home            REAL,
    clv_total           REAL,
    UNIQUE(prediction_id),
    FOREIGN KEY (prediction_id) REFERENCES predictions(id),
    FOREIGN KEY (odds_snapshot_id) REFERENCES odds_snapshots(id),
    FOREIGN KEY (closing_snapshot_id) REFERENCES odds_snapshots(id)
);

CREATE INDEX IF NOT EXISTS idx_predodds_pred ON prediction_odds(prediction_id);
