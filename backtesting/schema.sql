-- MLB Model Backtesting Schema
-- Run: sqlite3 data/mlb_backtest.db < backtesting/schema.sql

CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_ts   TEXT NOT NULL,
    game_date       TEXT NOT NULL,
    game_id         INTEGER NOT NULL,
    away_team       TEXT NOT NULL,
    home_team       TEXT NOT NULL,
    away_pitcher    TEXT,
    home_pitcher    TEXT,
    model_away_win  REAL NOT NULL,
    model_home_win  REAL NOT NULL,
    model_total     REAL NOT NULL,
    model_away_runs REAL,
    model_home_runs REAL,
    model_f5_away   REAL,
    model_f5_home   REAL,
    model_f5_draw   REAL,
    mkt_away_ml     INTEGER,
    mkt_home_ml     INTEGER,
    mkt_total_line  REAL,
    mkt_total_dir   TEXT,
    edge_away       REAL,
    edge_home       REAL,
    lineup_source   TEXT,
    park_factor     REAL,
    weather_temp    INTEGER,
    umpire_name     TEXT,
    n_sims          INTEGER DEFAULT 5000,
    UNIQUE(game_date, game_id)
);

CREATE TABLE IF NOT EXISTS actuals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    game_date       TEXT NOT NULL,
    game_id         INTEGER NOT NULL UNIQUE,
    away_team       TEXT NOT NULL,
    home_team       TEXT NOT NULL,
    away_score      INTEGER NOT NULL,
    home_score      INTEGER NOT NULL,
    winner          TEXT NOT NULL,
    total_runs      INTEGER NOT NULL,
    innings         INTEGER DEFAULT 9,
    game_status     TEXT DEFAULT 'Final',
    f5_away_score   INTEGER,
    f5_home_score   INTEGER,
    f5_winner       TEXT,
    recorded_ts     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bet_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    game_date           TEXT NOT NULL,
    game_id             INTEGER NOT NULL,
    bet_type            TEXT NOT NULL,
    model_prob          REAL NOT NULL,
    market_odds         INTEGER NOT NULL,
    edge                REAL NOT NULL,
    kelly_fraction      REAL,
    hypothetical_units  REAL,
    result              TEXT,
    profit_units        REAL,
    UNIQUE(game_date, game_id, bet_type)
);

CREATE INDEX IF NOT EXISTS idx_pred_date   ON predictions(game_date);
CREATE INDEX IF NOT EXISTS idx_pred_gameid ON predictions(game_id);
CREATE INDEX IF NOT EXISTS idx_act_date    ON actuals(game_date);
CREATE INDEX IF NOT EXISTS idx_bet_date    ON bet_log(game_date);
