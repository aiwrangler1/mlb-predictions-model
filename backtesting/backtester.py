"""
backtester.py - Core backtesting logic for MLB model.

Provides:
  - log_prediction()        : called from server.py after each simulation
  - record_actuals_for_date(): fetch MLB Stats API, write actuals + bet_log
  - compute_bet_log()       : compute win/loss for bets given actuals
"""
import sys
import asyncio
import aiohttp
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtesting.results_db import (
    get_connection,
    upsert_prediction,
    upsert_actual,
    upsert_bet,
    get_open_predictions,
)

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MLB_GAME_FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"

# ─────────────────────────────────────────────────────────────────────────────
# Prediction logging (called from server.py)
# ─────────────────────────────────────────────────────────────────────────────

def log_prediction(game_id: int, game_date: str, sim_result: dict,
                   market_odds: dict, db_path: str = None) -> int:
    """
    Log a prediction to the database after a simulation run.

    sim_result keys expected (from monte_carlo / server.py):
        away_win_pct, home_win_pct, avg_total, avg_away_runs, avg_home_runs,
        f5_away_pct, f5_home_pct, f5_draw_pct,
        away_team, home_team, away_pitcher, home_pitcher,
        lineup_source, park_factor, weather_temp, umpire_name, n_sims

    market_odds keys expected:
        away_ml, home_ml, total_line, total_dir, run_line
    """
    conn = get_connection(db_path)
    try:
        away_win = sim_result.get("away_win_pct", 0.5)
        home_win = sim_result.get("home_win_pct", 0.5)

        # Implied probability from American odds
        away_ml = market_odds.get("away_ml")
        home_ml = market_odds.get("home_ml")
        edge_away = None
        edge_home = None
        if away_ml is not None:
            implied_away = _american_to_implied(away_ml)
            edge_away = round(away_win - implied_away, 4)
        if home_ml is not None:
            implied_home = _american_to_implied(home_ml)
            edge_home = round(home_win - implied_home, 4)

        pred = {
            "prediction_ts": datetime.now(timezone.utc).isoformat(),
            "game_date": game_date,
            "game_id": game_id,
            "away_team": sim_result.get("away_team", ""),
            "home_team": sim_result.get("home_team", ""),
            "away_pitcher": sim_result.get("away_pitcher"),
            "home_pitcher": sim_result.get("home_pitcher"),
            "model_away_win": away_win,
            "model_home_win": home_win,
            "model_total": sim_result.get("avg_total", 0),
            "model_away_runs": sim_result.get("avg_away_runs"),
            "model_home_runs": sim_result.get("avg_home_runs"),
            "model_f5_away": sim_result.get("f5_away_pct"),
            "model_f5_home": sim_result.get("f5_home_pct"),
            "model_f5_draw": sim_result.get("f5_draw_pct"),
            "mkt_away_ml": away_ml,
            "mkt_home_ml": home_ml,
            "mkt_total_line": market_odds.get("total_line"),
            "mkt_total_dir": market_odds.get("total_dir"),
            "mkt_run_line": market_odds.get("run_line"),
            "edge_away": edge_away,
            "edge_home": edge_home,
            "lineup_source": sim_result.get("lineup_source"),
            "park_factor": sim_result.get("park_factor"),
            "weather_temp": sim_result.get("weather_temp"),
            "umpire_name": sim_result.get("umpire_name"),
            "n_sims": sim_result.get("n_sims", 5000),
        }
        row_id = upsert_prediction(conn, pred)
        return row_id
    finally:
        conn.close()


def _american_to_implied(american: int) -> float:
    """Convert American odds to implied probability (no vig)."""
    if american > 0:
        return 100 / (american + 100)
    else:
        return abs(american) / (abs(american) + 100)


# ─────────────────────────────────────────────────────────────────────────────
# Fetch actual scores from MLB Stats API
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_schedule_for_date(game_date: str, session: aiohttp.ClientSession) -> list[int]:
    """Return list of game_ids scheduled on game_date."""
    params = {
        "sportId": 1,
        "date": game_date,
        "gameType": "R,F,D,L,W",  # Regular + postseason
        "fields": "dates,games,gamePk,status,detailedState",
    }
    async with session.get(MLB_SCHEDULE_URL, params=params) as resp:
        if resp.status != 200:
            return []
        data = await resp.json()
    
    game_ids = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            game_ids.append(game["gamePk"])
    return game_ids


async def fetch_game_result(game_id: int, game_date: str,
                            session: aiohttp.ClientSession) -> Optional[dict]:
    """
    Fetch final score for a single game from the MLB live feed.
    Returns an actuals dict or None if game is not final.
    """
    url = MLB_GAME_FEED_URL.format(game_id=game_id)
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
    except Exception:
        return None

    game_data = data.get("gameData", {})
    live_data = data.get("liveData", {})

    status = game_data.get("status", {}).get("detailedState", "")
    if status not in ("Final", "Game Over", "Completed Early"):
        return None

    linescore = live_data.get("linescore", {})
    teams_score = linescore.get("teams", {})
    away_score = teams_score.get("away", {}).get("runs")
    home_score = teams_score.get("home", {}).get("runs")
    if away_score is None or home_score is None:
        return None

    total_runs = away_score + home_score
    winner = "away" if away_score > home_score else "home"
    innings_played = linescore.get("currentInning", 9)

    # Team abbreviations
    away_team = game_data.get("teams", {}).get("away", {}).get("abbreviation", "")
    home_team = game_data.get("teams", {}).get("home", {}).get("abbreviation", "")

    # First 5 innings
    innings_data = linescore.get("innings", [])
    f5_away = sum(inn.get("away", {}).get("runs", 0) or 0 for inn in innings_data[:5])
    f5_home = sum(inn.get("home", {}).get("runs", 0) or 0 for inn in innings_data[:5])
    if len(innings_data) >= 5:
        if f5_away > f5_home:
            f5_winner = "away"
        elif f5_home > f5_away:
            f5_winner = "home"
        else:
            f5_winner = "draw"
    else:
        f5_winner = None

    return {
        "game_date": game_date,
        "game_id": game_id,
        "away_team": away_team,
        "home_team": home_team,
        "away_score": away_score,
        "home_score": home_score,
        "winner": winner,
        "total_runs": total_runs,
        "innings": innings_played,
        "game_status": status,
        "f5_away_score": f5_away if len(innings_data) >= 5 else None,
        "f5_home_score": f5_home if len(innings_data) >= 5 else None,
        "f5_winner": f5_winner,
        "recorded_ts": datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Bet log computation
# ─────────────────────────────────────────────────────────────────────────────

def _kelly_fraction(model_prob: float, decimal_odds: float) -> float:
    """Full Kelly fraction: (bp - q) / b where b = decimal_odds - 1."""
    b = decimal_odds - 1
    q = 1 - model_prob
    kelly = (b * model_prob - q) / b
    return max(kelly, 0.0)


def _american_to_decimal(american: int) -> float:
    if american > 0:
        return american / 100 + 1
    else:
        return 100 / abs(american) + 1


def _compute_bets_for_game(pred: dict, actual: dict) -> list[dict]:
    """
    Given a prediction row and its actual result, compute hypothetical bets.
    Uses half-Kelly sizing. Only bets where model edge > 2%.
    Returns list of bet dicts ready for upsert_bet().
    """
    MIN_EDGE = 0.02
    HALF_KELLY_SCALE = 0.5
    BANKROLL_UNITS = 100  # treat bankroll as 100 units

    bets = []
    game_date = pred["game_date"]
    game_id = pred["game_id"]
    winner = actual["winner"]
    total_runs = actual["total_runs"]
    mkt_total_line = pred.get("mkt_total_line")

    def make_bet(bet_type, model_prob, market_odds, result, profit_units=None):
        decimal_odds = _american_to_decimal(market_odds)
        kelly = _kelly_fraction(model_prob, decimal_odds)
        half_kelly = kelly * HALF_KELLY_SCALE
        units = round(half_kelly * BANKROLL_UNITS, 4)
        if profit_units is None:
            if result == "win":
                profit_units = round(units * (decimal_odds - 1), 4)
            elif result == "loss":
                profit_units = round(-units, 4)
            else:
                profit_units = 0.0
        return {
            "game_date": game_date,
            "game_id": game_id,
            "bet_type": bet_type,
            "model_prob": model_prob,
            "market_odds": market_odds,
            "edge": round(model_prob - _american_to_implied(market_odds), 4),
            "kelly_fraction": round(kelly, 4),
            "hypothetical_units": units,
            "result": result,
            "profit_units": profit_units,
        }

    # ML away bet
    edge_away = pred.get("edge_away") or 0
    away_ml = pred.get("mkt_away_ml")
    if edge_away >= MIN_EDGE and away_ml is not None:
        result = "win" if winner == "away" else "loss"
        bets.append(make_bet("ml_away", pred["model_away_win"], away_ml, result))

    # ML home bet
    edge_home = pred.get("edge_home") or 0
    home_ml = pred.get("mkt_home_ml")
    if edge_home >= MIN_EDGE and home_ml is not None:
        result = "win" if winner == "home" else "loss"
        bets.append(make_bet("ml_home", pred["model_home_win"], home_ml, result))

    # Total over/under bet
    mkt_total_dir = pred.get("mkt_total_dir")
    if mkt_total_dir and mkt_total_line is not None and total_runs is not None:
        # Model leans over/under vs market line → use -110 standard juice
        model_total = pred.get("model_total", 0)
        model_over_prob = None
        if mkt_total_dir == "over":
            # model projects over: estimate prob from how far model_total is from line
            diff = model_total - mkt_total_line
            # Simple heuristic: each 0.5 run ≈ 5% edge
            model_over_prob = min(0.65, max(0.50, 0.50 + diff * 0.05))
            edge = model_over_prob - _american_to_implied(-110)
            if edge >= MIN_EDGE:
                actual_over = total_runs > mkt_total_line
                result = "win" if actual_over else ("push" if total_runs == mkt_total_line else "loss")
                bets.append(make_bet("over", model_over_prob, -110, result))
        elif mkt_total_dir == "under":
            diff = mkt_total_line - model_total
            model_under_prob = min(0.65, max(0.50, 0.50 + diff * 0.05))
            edge = model_under_prob - _american_to_implied(-110)
            if edge >= MIN_EDGE:
                actual_under = total_runs < mkt_total_line
                result = "win" if actual_under else ("push" if total_runs == mkt_total_line else "loss")
                bets.append(make_bet("under", model_under_prob, -110, result))

    return bets


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point: record_actuals_for_date
# ─────────────────────────────────────────────────────────────────────────────

async def record_actuals_for_date(game_date: str, db_path: str = None) -> dict:
    """
    Fetch final scores for game_date from MLB Stats API.
    Records actuals for ALL final games on that date (not just predicted ones).
    Then computes bet_log entries for any games that also have predictions.
    
    Idempotent: safe to run multiple times.
    
    Returns dict with counts.
    """
    conn = get_connection(db_path)

    results_recorded = 0
    games_found = 0
    bets_logged = 0

    try:
        connector = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Step 1: get all game IDs for this date
            game_ids = await fetch_schedule_for_date(game_date, session)
            games_found = len(game_ids)

            if not game_ids:
                return {
                    "game_date": game_date,
                    "games_found": 0,
                    "results_recorded": 0,
                    "bets_logged": 0,
                }

            # Step 2: fetch each game result concurrently
            tasks = [fetch_game_result(gid, game_date, session) for gid in game_ids]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Step 3: store actuals
        actuals_by_game: dict[int, dict] = {}
        for actual in results:
            if isinstance(actual, dict):
                upsert_actual(conn, actual)
                results_recorded += 1
                actuals_by_game[actual["game_id"]] = actual

        # Step 4: compute bet_log for games that have predictions
        open_preds = get_open_predictions(conn, game_date)
        # Also get predictions that now have actuals (open_preds won't include already-resolved)
        # Fetch all predictions for this date that have actuals in actuals_by_game
        all_preds = conn.execute(
            "SELECT * FROM predictions WHERE game_date = ?", (game_date,)
        ).fetchall()

        for pred in all_preds:
            pred_dict = dict(pred)
            gid = pred_dict["game_id"]
            if gid in actuals_by_game:
                actual = actuals_by_game[gid]
                bets = _compute_bets_for_game(pred_dict, actual)
                for bet in bets:
                    upsert_bet(conn, bet)
                    bets_logged += 1

    finally:
        conn.close()

    return {
        "game_date": game_date,
        "games_found": games_found,
        "results_recorded": results_recorded,
        "bets_logged": bets_logged,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, os

    async def _test():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            tmp_db = f.name
        try:
            result = await record_actuals_for_date("2026-04-01", tmp_db)
            print(f"record_actuals_for_date result: {result}")
            assert "game_date" in result
            assert "games_found" in result
            print("backtester.py OK")
        finally:
            os.unlink(tmp_db)

    asyncio.run(_test())
