"""
odds_tracker.py — Captures and stores DK odds snapshots from The Odds API.

Usage:
    # Capture current odds for today's games
    python -m backtesting.odds_tracker

    # Capture with specific snapshot type
    python -m backtesting.odds_tracker --type opening
    python -m backtesting.odds_tracker --type closing

    # Programmatic
    from backtesting.odds_tracker import capture_odds_snapshot
    result = capture_odds_snapshot(snapshot_type='opening')
"""
import os
import sys
import json
import argparse
import sqlite3
from pathlib import Path
from datetime import datetime, timezone, date, timedelta
from typing import Optional

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtesting.results_db import get_connection, DB_PATH

# ─── Config ───────────────────────────────────────────────────
ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"
SPORT = "baseball_mlb"
BOOKMAKER = "draftkings"

# Team name mapping: Odds API uses full names, we use abbreviations
ODDS_API_TEAM_MAP = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Oakland Athletics": "ATH",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD", "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
}

def _get_api_key() -> str:
    """Load API key from env or .env file."""
    key = os.environ.get("ODDS_API_KEY")
    if key:
        return key
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("ODDS_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise ValueError("ODDS_API_KEY not found in environment or .env")


def _team_abbrev(full_name: str) -> str:
    """Convert Odds API team name to abbreviation."""
    return ODDS_API_TEAM_MAP.get(full_name, full_name[:3].upper())


def _init_v2_schema(conn: sqlite3.Connection):
    """Run schema_v2.sql to create new tables if needed."""
    schema_path = Path(__file__).parent / "schema_v2.sql"
    if schema_path.exists():
        # Filter out CREATE VIEW since it might already exist
        sql = schema_path.read_text()
        conn.executescript(sql)
        conn.commit()


def fetch_dk_odds(api_key: str) -> tuple[list[dict], dict]:
    """
    Fetch current DK odds from The Odds API.
    Single API call — uses bookmakers=draftkings filter.
    
    Returns (events_list, response_headers) where headers contain quota info.
    """
    url = f"{ODDS_API_BASE}/{SPORT}/odds"
    params = {
        "apiKey": api_key,
        "bookmakers": BOOKMAKER,
        "markets": "h2h,totals,spreads",
        "oddsFormat": "american",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    
    quota = {
        "requests_remaining": resp.headers.get("x-requests-remaining"),
        "requests_used": resp.headers.get("x-requests-used"),
    }
    return resp.json(), quota


def parse_odds_event(event: dict, captured_ts: str, snapshot_type: str) -> Optional[dict]:
    """Parse a single Odds API event into an odds_snapshot row."""
    bookmakers = event.get("bookmakers", [])
    if not bookmakers:
        return None
    
    dk = bookmakers[0]  # We filter to DK only
    
    away_team = _team_abbrev(event.get("away_team", ""))
    home_team = _team_abbrev(event.get("home_team", ""))
    
    # Parse commence time to get game_date
    commence = event.get("commence_time", "")
    if commence:
        try:
            ct = datetime.fromisoformat(commence.replace("Z", "+00:00"))
            # Use ET date (UTC-4) for game_date since MLB uses ET
            from datetime import timedelta as td
            et_time = ct - td(hours=4)
            game_date = et_time.strftime("%Y-%m-%d")
        except Exception:
            game_date = date.today().isoformat()
    else:
        game_date = date.today().isoformat()
    
    row = {
        "captured_ts": captured_ts,
        "game_date": game_date,
        "game_id": event.get("id", ""),
        "mlb_game_id": None,  # Linked later
        "away_team": away_team,
        "home_team": home_team,
        "commence_time": commence,
        "away_ml": None,
        "home_ml": None,
        "total_line": None,
        "over_price": None,
        "under_price": None,
        "away_spread": None,
        "away_spread_price": None,
        "home_spread": None,
        "home_spread_price": None,
        "snapshot_type": snapshot_type,
        "bookmaker": BOOKMAKER,
    }
    
    for market in dk.get("markets", []):
        key = market.get("key")
        outcomes = {o["name"]: o for o in market.get("outcomes", [])}
        
        if key == "h2h":
            away_o = outcomes.get(event.get("away_team", ""), {})
            home_o = outcomes.get(event.get("home_team", ""), {})
            row["away_ml"] = away_o.get("price")
            row["home_ml"] = home_o.get("price")
            
        elif key == "totals":
            over_o = outcomes.get("Over", {})
            under_o = outcomes.get("Under", {})
            row["total_line"] = over_o.get("point")
            row["over_price"] = over_o.get("price")
            row["under_price"] = under_o.get("price")
            
        elif key == "spreads":
            away_o = outcomes.get(event.get("away_team", ""), {})
            home_o = outcomes.get(event.get("home_team", ""), {})
            row["away_spread"] = away_o.get("point")
            row["away_spread_price"] = away_o.get("price")
            row["home_spread"] = home_o.get("point")
            row["home_spread_price"] = home_o.get("price")
    
    return row


def store_snapshot(conn: sqlite3.Connection, snapshot: dict) -> int:
    """Insert a single odds snapshot row. Returns row id."""
    cols = [
        "captured_ts", "game_date", "game_id", "mlb_game_id",
        "away_team", "home_team", "commence_time",
        "away_ml", "home_ml",
        "total_line", "over_price", "under_price",
        "away_spread", "away_spread_price", "home_spread", "home_spread_price",
        "snapshot_type", "bookmaker",
    ]
    placeholders = ",".join(["?" for _ in cols])
    col_str = ",".join(cols)
    vals = [snapshot.get(c) for c in cols]
    
    try:
        cursor = conn.execute(
            f"INSERT OR IGNORE INTO odds_snapshots ({col_str}) VALUES ({placeholders})",
            vals
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        # Duplicate (game_id, captured_ts, bookmaker) — skip
        return 0


def capture_odds_snapshot(
    snapshot_type: str = "scheduled",
    db_path: str = None,
    cache_path: str = None,
) -> dict:
    """
    Main entry point: fetch DK odds and store snapshot.
    
    Args:
        snapshot_type: 'opening', 'closing', 'scheduled', or 'manual'
        db_path: Override DB path
        cache_path: Also save raw JSON to this path (for dashboard use)
    
    Returns dict with counts and quota info.
    """
    api_key = _get_api_key()
    captured_ts = datetime.now(timezone.utc).isoformat()
    
    # Fetch from API
    events, quota = fetch_dk_odds(api_key)
    
    # Store in DB
    conn = get_connection(db_path)
    _init_v2_schema(conn)
    
    stored = 0
    skipped = 0
    games = []
    
    for event in events:
        snapshot = parse_odds_event(event, captured_ts, snapshot_type)
        if snapshot:
            row_id = store_snapshot(conn, snapshot)
            if row_id > 0:
                stored += 1
            else:
                skipped += 1
            games.append(snapshot)
        else:
            skipped += 1
    
    conn.close()
    
    # Optionally cache raw response
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(events, f, indent=2)
    
    return {
        "captured_ts": captured_ts,
        "snapshot_type": snapshot_type,
        "games_found": len(events),
        "stored": stored,
        "skipped_duplicates": skipped,
        "quota": quota,
        "games": games,
    }


def get_latest_odds(conn: sqlite3.Connection, game_date: str) -> list[dict]:
    """Get the most recent odds snapshot for each game on a date."""
    rows = conn.execute("""
        SELECT * FROM odds_snapshots o
        WHERE o.game_date = ?
        AND o.id = (
            SELECT s.id FROM odds_snapshots s 
            WHERE s.game_id = o.game_id 
            ORDER BY s.captured_ts DESC LIMIT 1
        )
        ORDER BY o.commence_time
    """, (game_date,)).fetchall()
    return [dict(r) for r in rows]


def get_opening_odds(conn: sqlite3.Connection, game_date: str) -> list[dict]:
    """Get the first (opening) odds snapshot for each game on a date."""
    rows = conn.execute("""
        SELECT * FROM odds_snapshots o
        WHERE o.game_date = ?
        AND o.id = (
            SELECT s.id FROM odds_snapshots s 
            WHERE s.game_id = o.game_id 
            ORDER BY s.captured_ts ASC LIMIT 1
        )
        ORDER BY o.commence_time
    """, (game_date,)).fetchall()
    return [dict(r) for r in rows]


def get_odds_history(conn: sqlite3.Connection, game_id: str) -> list[dict]:
    """Get all snapshots for a specific game, ordered by time."""
    rows = conn.execute("""
        SELECT * FROM odds_snapshots
        WHERE game_id = ?
        ORDER BY captured_ts
    """, (game_id,)).fetchall()
    return [dict(r) for r in rows]


def store_from_cache(cache_path: str, snapshot_type: str = "manual",
                     db_path: str = None) -> dict:
    """
    Import odds from a previously cached JSON file.
    Handles both raw Odds API format (list of events) and our custom
    dashboard cache format ({timestamp, date, source, games: {matchup: {...}}}).
    """
    with open(cache_path) as f:
        data = json.load(f)
    
    # Use file modification time as captured_ts
    mtime = os.path.getmtime(cache_path)
    captured_ts = datetime.fromtimestamp(mtime, timezone.utc).isoformat()
    
    conn = get_connection(db_path)
    _init_v2_schema(conn)
    
    stored = 0
    total = 0
    
    if isinstance(data, list):
        # Raw Odds API format
        total = len(data)
        for event in data:
            snapshot = parse_odds_event(event, captured_ts, snapshot_type)
            if snapshot:
                row_id = store_snapshot(conn, snapshot)
                if row_id > 0:
                    stored += 1
    elif isinstance(data, dict) and "games" in data:
        # Our dashboard cache format: {timestamp, date, source, games: {"ARI@PHI": {...}}}
        games = data["games"]
        game_date = data.get("date", date.today().isoformat())
        if data.get("timestamp"):
            captured_ts = datetime.fromtimestamp(data["timestamp"], timezone.utc).isoformat()
        
        if isinstance(games, dict):
            total = len(games)
            for matchup_key, g in games.items():
                away_full = g.get("away_team", "")
                home_full = g.get("home_team", "")
                snapshot = {
                    "captured_ts": captured_ts,
                    "game_date": game_date,
                    "game_id": matchup_key,  # Use matchup as ID for cached data
                    "mlb_game_id": None,
                    "away_team": _team_abbrev(away_full),
                    "home_team": _team_abbrev(home_full),
                    "commence_time": g.get("commence"),
                    "away_ml": g.get("away_ml"),
                    "home_ml": g.get("home_ml"),
                    "total_line": g.get("total"),
                    "over_price": g.get("over_price"),
                    "under_price": g.get("under_price"),
                    "away_spread": g.get("away_spread"),
                    "away_spread_price": g.get("away_spread_price"),
                    "home_spread": g.get("home_spread"),
                    "home_spread_price": g.get("home_spread_price"),
                    "snapshot_type": snapshot_type,
                    "bookmaker": data.get("source", BOOKMAKER),
                }
                row_id = store_snapshot(conn, snapshot)
                if row_id > 0:
                    stored += 1
    
    conn.close()
    return {"stored": stored, "total": total, "captured_ts": captured_ts}


# ─── CLI ──────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture DK odds snapshot")
    parser.add_argument("--type", choices=["opening", "closing", "scheduled", "manual"],
                        default="scheduled", help="Snapshot type label")
    parser.add_argument("--cache", type=str, help="Also save raw JSON to this path")
    parser.add_argument("--from-cache", type=str, 
                        help="Import from cached JSON file instead of calling API")
    args = parser.parse_args()
    
    if args.from_cache:
        result = store_from_cache(args.from_cache, snapshot_type=args.type)
        print(f"Imported {result['stored']}/{result['total']} games from cache")
        print(f"  Captured at: {result['captured_ts']}")
    else:
        result = capture_odds_snapshot(
            snapshot_type=args.type,
            cache_path=args.cache or "/tmp/dk_odds_cache.json",
        )
        print(f"Captured {result['stored']} odds snapshots ({result['skipped_duplicates']} dupes)")
        print(f"  Type: {result['snapshot_type']}")
        print(f"  Games: {result['games_found']}")
        print(f"  Quota: {result['quota']}")
