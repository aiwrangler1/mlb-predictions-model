"""
historical_backfill.py — Backfills historical DK odds from The Odds API.

Fetches opening + closing line snapshots for each game day, stores in odds_snapshots.
Then runs model sims and records actuals for a complete backtest dataset.

Usage:
    python -m backtesting.historical_backfill --start 2026-03-27 --end 2026-04-13
    python -m backtesting.historical_backfill --start 2026-03-27 --end 2026-04-13 --odds-only
    python -m backtesting.historical_backfill --start 2026-03-27 --end 2026-04-13 --sims-only
"""
import os
import sys
import json
import time
import argparse
import sqlite3
from pathlib import Path
from datetime import datetime, timezone, date, timedelta
from typing import Optional

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.results_db import get_connection
from backtesting.odds_tracker import (
    _init_v2_schema, parse_odds_event, store_snapshot, _get_api_key,
    ODDS_API_BASE, SPORT, BOOKMAKER
)

# ─── Historical Odds API ──────────────────────────────────────

def fetch_historical_odds(api_key: str, iso_date: str) -> tuple[dict, dict]:
    """
    Fetch historical DK odds for a specific timestamp.
    
    Args:
        api_key: The Odds API key
        iso_date: ISO 8601 timestamp, e.g. '2026-03-27T14:00:00Z'
    
    Returns (response_data, quota_info)
    Cost: 20 credits per call (h2h + totals, 1 region)
    """
    url = f"{ODDS_API_BASE.replace('/sports', '')}/historical/sports/{SPORT}/odds"
    params = {
        "apiKey": api_key,
        "bookmakers": BOOKMAKER,
        "markets": "h2h,totals",
        "oddsFormat": "american",
        "date": iso_date,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    
    quota = {
        "requests_remaining": resp.headers.get("x-requests-remaining"),
        "requests_used": resp.headers.get("x-requests-used"),
        "last_cost": resp.headers.get("x-requests-last"),
    }
    return resp.json(), quota


def get_game_schedule(game_date: str) -> list[dict]:
    """Get MLB schedule for a date to find game start times."""
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {"sportId": 1, "date": game_date}
    resp = requests.get(url, params=params, timeout=15)
    data = resp.json()
    
    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            games.append({
                "game_pk": g.get("gamePk"),
                "game_date": g.get("gameDate"),  # ISO UTC
                "status": g.get("status", {}).get("detailedState", ""),
            })
    return games


def compute_snapshot_times(game_date: str, schedule: list[dict]) -> tuple[str, str]:
    """
    Compute opening and closing snapshot timestamps for a game day.
    
    Opening: 10:00 AM ET (14:00 UTC) on game day — lines are set by then
    Closing: 30 minutes before earliest game start
    
    Returns (opening_iso, closing_iso)
    """
    # Opening: always 10am ET = 14:00 UTC
    opening = f"{game_date}T14:00:00Z"
    
    # Closing: 30 min before earliest game
    if schedule:
        earliest = min(g["game_date"] for g in schedule if g.get("game_date"))
        try:
            et = datetime.fromisoformat(earliest.replace("Z", "+00:00"))
            closing_dt = et - timedelta(minutes=30)
            closing = closing_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            # Fallback: 6:30 PM ET = 22:30 UTC
            closing = f"{game_date}T22:30:00Z"
    else:
        closing = f"{game_date}T22:30:00Z"
    
    return opening, closing


def backfill_odds_for_date(
    api_key: str, 
    game_date: str, 
    conn: sqlite3.Connection,
    dry_run: bool = False,
) -> dict:
    """
    Fetch opening + closing odds for a single game date.
    
    Returns dict with counts and cost info.
    """
    # Get schedule to compute closing time
    schedule = get_game_schedule(game_date)
    if not schedule:
        return {"game_date": game_date, "status": "no_games", "credits_used": 0}
    
    opening_ts, closing_ts = compute_snapshot_times(game_date, schedule)
    
    result = {
        "game_date": game_date,
        "games_on_schedule": len(schedule),
        "opening_ts": opening_ts,
        "closing_ts": closing_ts,
        "opening_stored": 0,
        "closing_stored": 0,
        "credits_used": 0,
    }
    
    if dry_run:
        result["status"] = "dry_run"
        return result
    
    # Fetch opening odds
    try:
        open_data, open_quota = fetch_historical_odds(api_key, opening_ts)
        open_events = open_data.get("data", [])
        actual_open_ts = open_data.get("timestamp", opening_ts)
        
        for event in open_events:
            snapshot = parse_odds_event(event, actual_open_ts, "opening")
            if snapshot:
                # Override game_date to match our date (API may return next-day games too)
                snapshot["game_date"] = game_date
                row_id = store_snapshot(conn, snapshot)
                if row_id > 0:
                    result["opening_stored"] += 1
        
        result["credits_used"] += int(open_quota.get("last_cost", 20))
        result["quota_remaining"] = open_quota.get("requests_remaining")
        time.sleep(1)  # Rate limit protection
        
    except Exception as e:
        result["opening_error"] = str(e)
    
    # Fetch closing odds
    try:
        close_data, close_quota = fetch_historical_odds(api_key, closing_ts)
        close_events = close_data.get("data", [])
        actual_close_ts = close_data.get("timestamp", closing_ts)
        
        for event in close_events:
            snapshot = parse_odds_event(event, actual_close_ts, "closing")
            if snapshot:
                snapshot["game_date"] = game_date
                row_id = store_snapshot(conn, snapshot)
                if row_id > 0:
                    result["closing_stored"] += 1
        
        result["credits_used"] += int(close_quota.get("last_cost", 20))
        result["quota_remaining"] = close_quota.get("requests_remaining")
        time.sleep(1)
        
    except Exception as e:
        result["closing_error"] = str(e)
    
    result["status"] = "ok"
    return result


# ─── Model Sim Backfill ──────────────────────────────────────

def backfill_sims_for_date(game_date: str, conn: sqlite3.Connection) -> dict:
    """
    Run model simulations for all games on a date via the server API.
    Stores predictions in the DB.
    """
    # Get schedule with pitchers
    url = f"http://localhost:5000/api/schedule?game_date={game_date}"
    try:
        resp = requests.get(url, timeout=30)
        schedule = resp.json().get("games", [])
    except Exception as e:
        return {"game_date": game_date, "error": f"Schedule fetch failed: {e}"}
    
    if not schedule:
        return {"game_date": game_date, "games": 0, "sims": 0}
    
    result = {"game_date": game_date, "games": len(schedule), "sims": 0, "errors": 0}
    
    for g in schedule:
        game_id = g.get("game_id")
        away = g.get("away_team", "?")
        home = g.get("home_team", "?")
        
        # Check if prediction already exists
        existing = conn.execute(
            "SELECT id FROM predictions WHERE game_id = ? AND game_date = ?",
            (game_id, game_date)
        ).fetchone()
        if existing:
            result["sims"] += 1  # Count as done
            continue
        
        try:
            sim_resp = requests.get(
                f"http://localhost:5000/api/simulate/{game_id}",
                params={"n_sims": 5000, "game_date": game_date},
                timeout=120
            )
            if sim_resp.status_code == 200:
                sim = sim_resp.json()
                
                # The server auto-logs predictions, but let's make sure
                # Check if it was logged now
                check = conn.execute(
                    "SELECT id FROM predictions WHERE game_id = ? AND game_date = ?",
                    (game_id, game_date)
                ).fetchone()
                
                if not check:
                    # Manually insert
                    conn.execute("""
                        INSERT OR IGNORE INTO predictions
                        (game_id, game_date, away_team, home_team,
                         model_away_win, model_home_win, model_total,
                         model_away_runs, model_home_runs,
                         away_pitcher, home_pitcher, n_sims)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        game_id, game_date, away, home,
                        sim.get("away_win_pct", 0),
                        sim.get("home_win_pct", 0),
                        sim.get("total_runs_mean", 0),
                        sim.get("away_runs_mean", 0),
                        sim.get("home_runs_mean", 0),
                        g.get("away_pitcher_name", ""),
                        g.get("home_pitcher_name", ""),
                        5000,
                    ))
                    conn.commit()
                
                result["sims"] += 1
            else:
                result["errors"] += 1
        except Exception as e:
            result["errors"] += 1
        
        time.sleep(0.2)  # Don't hammer the server
    
    return result


# ─── Actuals Backfill ─────────────────────────────────────────

def backfill_actuals_for_date(game_date: str, conn: sqlite3.Connection) -> dict:
    """Fetch game results from MLB Stats API and store in actuals table."""
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1,
        "date": game_date,
        "hydrate": "linescore,probablePitcher,team",
    }
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
    except Exception as e:
        return {"game_date": game_date, "error": str(e)}
    
    result = {"game_date": game_date, "games": 0, "stored": 0, "skipped": 0}
    
    for d in data.get("dates", []):
        for g in d.get("games", []):
            result["games"] += 1
            game_pk = g.get("gamePk")
            status = g.get("status", {}).get("detailedState", "")
            
            if status != "Final":
                result["skipped"] += 1
                continue
            
            linescore = g.get("linescore", {})
            away_runs = linescore.get("teams", {}).get("away", {}).get("runs", 0)
            home_runs = linescore.get("teams", {}).get("home", {}).get("runs", 0)
            total_runs = away_runs + home_runs
            
            away_team_data = g["teams"]["away"]["team"]
            home_team_data = g["teams"]["home"]["team"]
            away_abbrev = away_team_data.get("abbreviation", away_team_data.get("name", "?")[:3])
            home_abbrev = home_team_data.get("abbreviation", home_team_data.get("name", "?")[:3])
            
            winner = "away" if away_runs > home_runs else "home"
            
            recorded_ts = datetime.now(timezone.utc).isoformat()
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO actuals
                    (game_id, game_date, away_team, home_team,
                     away_score, home_score, total_runs, winner,
                     game_status, recorded_ts)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    game_pk, game_date, away_abbrev, home_abbrev,
                    away_runs, home_runs, total_runs, winner,
                    "Final", recorded_ts,
                ))
                result["stored"] += 1
            except Exception as e:
                result["skipped"] += 1
    
    conn.commit()
    return result


# ─── Link Predictions to Odds ─────────────────────────────────

def link_predictions_to_odds(conn: sqlite3.Connection) -> dict:
    """
    For each prediction, find matching opening + closing odds snapshots
    and create prediction_odds linkage with edge calculations.
    """
    def american_to_implied(odds):
        if odds is None: return None
        if odds > 0: return 100 / (odds + 100)
        return abs(odds) / (abs(odds) + 100)
    
    preds = conn.execute("""
        SELECT p.* FROM predictions p
        WHERE p.id NOT IN (SELECT prediction_id FROM prediction_odds)
    """).fetchall()
    
    linked = 0
    for pred in preds:
        pred = dict(pred)
        pid = pred["id"]
        gdate = pred["game_date"]
        away = pred["away_team"]
        home = pred["home_team"]
        
        # Find opening snapshot
        opening = conn.execute("""
            SELECT * FROM odds_snapshots
            WHERE game_date = ? AND away_team = ? AND home_team = ?
            AND snapshot_type = 'opening'
            ORDER BY captured_ts ASC LIMIT 1
        """, (gdate, away, home)).fetchone()
        
        # Find closing snapshot
        closing = conn.execute("""
            SELECT * FROM odds_snapshots
            WHERE game_date = ? AND away_team = ? AND home_team = ?
            AND snapshot_type = 'closing'
            ORDER BY captured_ts DESC LIMIT 1
        """, (gdate, away, home)).fetchone()
        
        # Use opening for the prediction linkage, closing for CLV
        snap = opening or closing
        if not snap:
            continue
        
        snap = dict(snap)
        close_dict = dict(closing) if closing else None
        
        # Edge at prediction time (vs opening)
        edge_away = edge_home = None
        if snap.get("away_ml") is not None and pred.get("model_away_win"):
            imp = american_to_implied(snap["away_ml"])
            if imp: edge_away = round(pred["model_away_win"] - imp, 4)
        if snap.get("home_ml") is not None and pred.get("model_home_win"):
            imp = american_to_implied(snap["home_ml"])
            if imp: edge_home = round(pred["model_home_win"] - imp, 4)
        
        # CLV (vs closing)
        clv_away = clv_home = None
        if close_dict and close_dict.get("away_ml") is not None and pred.get("model_away_win"):
            imp = american_to_implied(close_dict["away_ml"])
            if imp: clv_away = round(pred["model_away_win"] - imp, 4)
        if close_dict and close_dict.get("home_ml") is not None and pred.get("model_home_win"):
            imp = american_to_implied(close_dict["home_ml"])
            if imp: clv_home = round(pred["model_home_win"] - imp, 4)
        
        conn.execute("""
            INSERT OR IGNORE INTO prediction_odds
            (prediction_id, odds_snapshot_id, closing_snapshot_id,
             edge_away_at_pred, edge_home_at_pred,
             clv_away, clv_home)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            pid, snap["id"],
            close_dict["id"] if close_dict else None,
            edge_away, edge_home, clv_away, clv_home,
        ))
        
        # Also update predictions table mkt_ fields (use opening odds)
        conn.execute("""
            UPDATE predictions SET mkt_away_ml=?, mkt_home_ml=?, mkt_total_line=?,
            edge_away=?, edge_home=? WHERE id=?
        """, (snap["away_ml"], snap["home_ml"], snap["total_line"],
              edge_away, edge_home, pid))
        
        linked += 1
    
    conn.commit()
    return {"unlinked": len(preds), "linked": linked}


# ─── Full Backfill Pipeline ──────────────────────────────────

def full_backfill(start_date: str, end_date: str,
                  odds: bool = True, sims: bool = True,
                  actuals: bool = True, link: bool = True,
                  dry_run: bool = False) -> dict:
    """
    Run the complete backfill pipeline for a date range.
    """
    api_key = _get_api_key()
    conn = get_connection()
    _init_v2_schema(conn)
    
    d = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    
    total_result = {
        "dates_processed": 0,
        "total_credits": 0,
        "odds_results": [],
        "sim_results": [],
        "actual_results": [],
    }
    
    while d <= end:
        ds = d.isoformat()
        print(f"\n{'='*50}")
        print(f"  {ds}")
        print(f"{'='*50}")
        
        # 1. Odds backfill
        if odds:
            # Check if we already have odds for this date
            existing = conn.execute(
                "SELECT COUNT(*) FROM odds_snapshots WHERE game_date = ?",
                (ds,)
            ).fetchone()[0]
            
            if existing >= 10:  # Assume covered if we have 10+ snapshots
                print(f"  Odds: already have {existing} snapshots, skipping")
                odds_result = {"game_date": ds, "status": "exists", "credits_used": 0}
            else:
                print(f"  Odds: fetching opening + closing...")
                odds_result = backfill_odds_for_date(api_key, ds, conn, dry_run)
                print(f"    Opening: {odds_result.get('opening_stored', 0)} stored")
                print(f"    Closing: {odds_result.get('closing_stored', 0)} stored")
                print(f"    Credits: {odds_result.get('credits_used', 0)}")
                if odds_result.get("quota_remaining"):
                    print(f"    Quota remaining: {odds_result['quota_remaining']}")
            
            total_result["odds_results"].append(odds_result)
            total_result["total_credits"] += odds_result.get("credits_used", 0)
        
        # 2. Model sims
        if sims:
            print(f"  Sims: running model for {ds}...")
            sim_result = backfill_sims_for_date(ds, conn)
            print(f"    {sim_result.get('sims', 0)}/{sim_result.get('games', 0)} games simulated")
            if sim_result.get("errors"):
                print(f"    Errors: {sim_result['errors']}")
            total_result["sim_results"].append(sim_result)
        
        # 3. Actuals
        if actuals:
            print(f"  Actuals: recording results for {ds}...")
            actual_result = backfill_actuals_for_date(ds, conn)
            print(f"    {actual_result.get('stored', 0)}/{actual_result.get('games', 0)} recorded")
            total_result["actual_results"].append(actual_result)
        
        total_result["dates_processed"] += 1
        d += timedelta(days=1)
    
    # 4. Link predictions to odds
    if link:
        print(f"\nLinking predictions to odds snapshots...")
        link_result = link_predictions_to_odds(conn)
        print(f"  Linked {link_result['linked']}/{link_result['unlinked']}")
        total_result["link_result"] = link_result
    
    conn.close()
    
    # Summary
    print(f"\n{'='*50}")
    print(f"  BACKFILL COMPLETE")
    print(f"{'='*50}")
    print(f"  Dates: {total_result['dates_processed']}")
    print(f"  Credits used: {total_result['total_credits']}")
    
    return total_result


# ─── CLI ──────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Historical odds + model backfill")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--odds-only", action="store_true", help="Only fetch odds")
    parser.add_argument("--sims-only", action="store_true", help="Only run sims")
    parser.add_argument("--actuals-only", action="store_true", help="Only record actuals")
    parser.add_argument("--link-only", action="store_true", help="Only link preds to odds")
    parser.add_argument("--dry-run", action="store_true", help="Don't make API calls")
    args = parser.parse_args()
    
    if args.odds_only:
        full_backfill(args.start, args.end, odds=True, sims=False, actuals=False, link=False)
    elif args.sims_only:
        full_backfill(args.start, args.end, odds=False, sims=True, actuals=False, link=False)
    elif args.actuals_only:
        full_backfill(args.start, args.end, odds=False, sims=False, actuals=True, link=False)
    elif args.link_only:
        full_backfill(args.start, args.end, odds=False, sims=False, actuals=False, link=True)
    else:
        full_backfill(args.start, args.end, dry_run=args.dry_run)
