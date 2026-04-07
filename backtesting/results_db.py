import sqlite3, json, os
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent.parent / "data" / "mlb_backtest.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_connection(db_path: str = None) -> sqlite3.Connection:
    """Get SQLite connection, creating DB and tables if needed."""
    path = db_path or str(DB_PATH)
    db_parent = Path(path).expanduser().resolve().parent
    db_parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    # Enable WAL for concurrent reads
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _init_schema(conn)
    return conn


def _init_schema(conn):
    """Create tables if they don't exist."""
    schema = SCHEMA_PATH.read_text()
    conn.executescript(schema)
    conn.commit()


def upsert_prediction(conn, pred: dict) -> int:
    """Insert or replace a prediction. Returns row id."""
    # Uses INSERT OR REPLACE so re-running a sim updates the prediction
    cols = ["prediction_ts", "game_date", "game_id", "away_team", "home_team",
            "away_pitcher", "home_pitcher", "model_away_win", "model_home_win",
            "model_total", "model_away_runs", "model_home_runs",
            "model_f5_away", "model_f5_home", "model_f5_draw",
            "mkt_away_ml", "mkt_home_ml", "mkt_total_line", "mkt_total_dir",
            "edge_away", "edge_home", "lineup_source", "park_factor",
            "weather_temp", "umpire_name", "n_sims"]
    placeholders = ",".join(["?" for _ in cols])
    col_str = ",".join(cols)
    vals = [pred.get(c) for c in cols]
    cursor = conn.execute(
        f"INSERT OR REPLACE INTO predictions ({col_str}) VALUES ({placeholders})",
        vals
    )
    conn.commit()
    return cursor.lastrowid


def upsert_actual(conn, actual: dict) -> int:
    """Insert or replace an actual game result."""
    cols = ["game_date", "game_id", "away_team", "home_team", "away_score", "home_score",
            "winner", "total_runs", "innings", "game_status", "f5_away_score",
            "f5_home_score", "f5_winner", "recorded_ts"]
    placeholders = ",".join(["?" for _ in cols])
    col_str = ",".join(cols)
    vals = [actual.get(c) for c in cols]
    cursor = conn.execute(
        f"INSERT OR REPLACE INTO actuals ({col_str}) VALUES ({placeholders})",
        vals
    )
    conn.commit()
    return cursor.lastrowid


def upsert_bet(conn, bet: dict) -> int:
    """Insert or replace a bet log entry."""
    cols = ["game_date", "game_id", "bet_type", "model_prob", "market_odds", "edge",
            "kelly_fraction", "hypothetical_units", "result", "profit_units"]
    placeholders = ",".join(["?" for _ in cols])
    col_str = ",".join(cols)
    vals = [bet.get(c) for c in cols]
    cursor = conn.execute(
        f"INSERT OR REPLACE INTO bet_log ({col_str}) VALUES ({placeholders})",
        vals
    )
    conn.commit()
    return cursor.lastrowid


def get_open_predictions(conn, game_date: str) -> list[dict]:
    """Return predictions for date that don't have actuals yet."""
    rows = conn.execute("""
        SELECT p.* FROM predictions p
        LEFT JOIN actuals a ON p.game_id = a.game_id
        WHERE p.game_date = ? AND a.id IS NULL
    """, (game_date,)).fetchall()
    return [dict(r) for r in rows]


def get_predictions_with_actuals(conn, start_date: str, end_date: str) -> list[dict]:
    """Return joined predictions + actuals for analytics."""
    rows = conn.execute("""
        SELECT p.*, 
               a.away_score, a.home_score, a.winner, a.total_runs,
               a.f5_away_score, a.f5_home_score, a.f5_winner, a.innings
        FROM predictions p
        JOIN actuals a ON p.game_id = a.game_id
        WHERE p.game_date BETWEEN ? AND ?
        ORDER BY p.game_date
    """, (start_date, end_date)).fetchall()
    return [dict(r) for r in rows]


def get_bet_log(conn, start_date: str, end_date: str) -> list[dict]:
    """Return bet log entries with results."""
    rows = conn.execute("""
        SELECT * FROM bet_log
        WHERE game_date BETWEEN ? AND ?
        ORDER BY game_date
    """, (start_date, end_date)).fetchall()
    return [dict(r) for r in rows]


def get_summary_stats(conn) -> dict:
    """Quick summary: total games tracked, date range, win rate."""
    row = conn.execute("""
        SELECT 
            COUNT(DISTINCT p.game_id) as total_games_predicted,
            COUNT(DISTINCT a.game_id) as total_games_resolved,
            MIN(p.game_date) as earliest_date,
            MAX(p.game_date) as latest_date
        FROM predictions p
        LEFT JOIN actuals a ON p.game_id = a.game_id
    """).fetchone()
    return dict(row) if row else {}


if __name__ == "__main__":
    import tempfile, os

    # Use a temp DB for the test so we don't pollute the real DB
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_db = f.name

    try:
        conn = get_connection(tmp_db)

        # Insert a dummy prediction
        pred = {
            "prediction_ts": "2025-04-06T12:00:00+00:00",
            "game_date": "2025-04-06",
            "game_id": 999999,
            "away_team": "NYY",
            "home_team": "BOS",
            "away_pitcher": "Gerrit Cole",
            "home_pitcher": "Brayan Bello",
            "model_away_win": 0.52,
            "model_home_win": 0.48,
            "model_total": 8.3,
            "model_away_runs": 4.3,
            "model_home_runs": 4.0,
            "model_f5_away": 0.50,
            "model_f5_home": 0.42,
            "model_f5_draw": 0.08,
            "mkt_away_ml": -115,
            "mkt_home_ml": +105,
            "mkt_total_line": 8.0,
            "mkt_total_dir": "over",
            "edge_away": 0.02,
            "edge_home": -0.02,
            "lineup_source": "actual",
            "park_factor": 1.05,
            "weather_temp": 68,
            "umpire_name": "Angel Hernandez",
            "n_sims": 5000,
        }
        pred_id = upsert_prediction(conn, pred)
        assert pred_id > 0, "upsert_prediction should return a positive row id"

        # Insert a dummy actual
        actual = {
            "game_date": "2025-04-06",
            "game_id": 999999,
            "away_team": "NYY",
            "home_team": "BOS",
            "away_score": 5,
            "home_score": 3,
            "winner": "away",
            "total_runs": 8,
            "innings": 9,
            "game_status": "Final",
            "f5_away_score": 3,
            "f5_home_score": 2,
            "f5_winner": "away",
            "recorded_ts": "2025-04-06T23:30:00+00:00",
        }
        actual_id = upsert_actual(conn, actual)
        assert actual_id > 0, "upsert_actual should return a positive row id"

        # Query predictions with actuals
        rows = get_predictions_with_actuals(conn, "2025-04-01", "2025-04-30")
        assert len(rows) == 1, f"Expected 1 joined row, got {len(rows)}"
        row = rows[0]
        assert row["away_team"] == "NYY"
        assert row["away_score"] == 5
        assert row["f5_winner"] == "away"

        # Query open predictions (should be 0 since we have an actual)
        open_preds = get_open_predictions(conn, "2025-04-06")
        assert len(open_preds) == 0, f"Expected 0 open predictions, got {len(open_preds)}"

        # Summary stats
        stats = get_summary_stats(conn)
        assert stats["total_games_predicted"] == 1
        assert stats["total_games_resolved"] == 1
        assert stats["earliest_date"] == "2025-04-06"

        conn.close()
        print("results_db.py OK")

    finally:
        os.unlink(tmp_db)
