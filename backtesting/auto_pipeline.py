"""
auto_pipeline.py — Single entry point for daily MLB model pipeline.

Workflow:
  1. Capture odds snapshot (opening or closing)
  2. Run model for today's slate (via server API or direct)
  3. Log predictions with odds attached
  4. End of day: record actuals, compute bets, update CLV

Usage:
    # Morning run: capture opening odds
    python -m backtesting.auto_pipeline morning

    # Pre-game run: capture closing odds, link to predictions
    python -m backtesting.auto_pipeline pregame

    # End of day: record results for yesterday (or specific date)
    python -m backtesting.auto_pipeline close
    python -m backtesting.auto_pipeline close --date 2026-04-12

    # Full report
    python -m backtesting.auto_pipeline report --days 30
"""
import sys
import argparse
import asyncio
import json
from pathlib import Path
from datetime import datetime, timezone, date, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.odds_tracker import (
    capture_odds_snapshot, get_latest_odds, get_opening_odds,
    store_from_cache, _init_v2_schema
)
from backtesting.results_db import get_connection
from backtesting.backtester import record_actuals_for_date, _american_to_implied
from backtesting.analytics import summary_report


def _today_et() -> str:
    """Get today's date in ET (UTC-4)."""
    now = datetime.now(timezone.utc) - timedelta(hours=4)
    return now.strftime("%Y-%m-%d")


def _yesterday_et() -> str:
    """Get yesterday's date in ET."""
    now = datetime.now(timezone.utc) - timedelta(hours=4) - timedelta(days=1)
    return now.strftime("%Y-%m-%d")


# ─── Morning Run ──────────────────────────────────────────────

def morning_run(db_path: str = None) -> dict:
    """
    Morning workflow:
    1. Capture opening odds snapshot
    2. Print today's slate with lines
    """
    print("=" * 60)
    print("  MLB Pipeline — Morning Run")
    print(f"  {_today_et()}")
    print("=" * 60)

    # 1. Capture opening odds
    print("\n  Capturing opening odds...")
    result = capture_odds_snapshot(
        snapshot_type="opening",
        db_path=db_path,
        cache_path="/tmp/dk_odds_cache.json",
    )
    print(f"  Stored {result['stored']} games")
    print(f"  API quota: {result['quota']}")

    # 2. Print slate
    conn = get_connection(db_path)
    _init_v2_schema(conn)
    odds = get_latest_odds(conn, _today_et())
    conn.close()

    if odds:
        print(f"\n  Today's Slate ({len(odds)} games):")
        for o in odds:
            ml_str = f"{o['away_team']}({o.get('away_ml','?')}) @ {o['home_team']}({o.get('home_ml','?')})"
            total_str = f"O/U {o.get('total_line','?')}"
            print(f"    {ml_str}  {total_str}")

    return result


# ─── Pre-Game Run ─────────────────────────────────────────────

def pregame_run(db_path: str = None) -> dict:
    """
    Pre-game workflow:
    1. Capture closing odds snapshot
    2. Link closing odds to any predictions that have opening odds
    3. Compute CLV for each prediction
    """
    print("=" * 60)
    print("  MLB Pipeline — Pre-Game (Closing Odds)")
    print(f"  {_today_et()}")
    print("=" * 60)

    # 1. Capture closing odds
    print("\n  Capturing closing odds...")
    result = capture_odds_snapshot(
        snapshot_type="closing",
        db_path=db_path,
        cache_path="/tmp/dk_odds_closing.json",
    )
    print(f"  Stored {result['stored']} games")
    print(f"  API quota: {result['quota']}")

    # 2. Link closing snapshots to predictions and compute CLV
    conn = get_connection(db_path)
    _init_v2_schema(conn)
    _link_closing_odds(conn, _today_et())
    conn.close()

    return result


def _link_closing_odds(conn, game_date: str):
    """
    For each prediction_odds row on game_date that has no closing_snapshot_id,
    find the closing snapshot and compute CLV.
    """
    # Get prediction_odds rows needing closing
    rows = conn.execute("""
        SELECT po.id, po.prediction_id, po.odds_snapshot_id,
               p.model_away_win, p.model_home_win, p.model_total,
               os.game_id
        FROM prediction_odds po
        JOIN predictions p ON p.id = po.prediction_id
        JOIN odds_snapshots os ON os.id = po.odds_snapshot_id
        WHERE p.game_date = ? AND po.closing_snapshot_id IS NULL
    """, (game_date,)).fetchall()

    if not rows:
        print("  No predictions to link closing odds to.")
        return

    updated = 0
    for row in rows:
        row = dict(row)
        # Find closing snapshot for this game
        closing = conn.execute("""
            SELECT * FROM odds_snapshots
            WHERE game_id = ? AND snapshot_type = 'closing'
            ORDER BY captured_ts DESC LIMIT 1
        """, (row["game_id"],)).fetchone()

        if not closing:
            continue

        closing = dict(closing)

        # Compute CLV
        clv_away = clv_home = clv_total = None
        if closing.get("away_ml") is not None and row.get("model_away_win"):
            close_implied_away = _american_to_implied(closing["away_ml"])
            clv_away = round(row["model_away_win"] - close_implied_away, 4)
        if closing.get("home_ml") is not None and row.get("model_home_win"):
            close_implied_home = _american_to_implied(closing["home_ml"])
            clv_home = round(row["model_home_win"] - close_implied_home, 4)

        conn.execute("""
            UPDATE prediction_odds
            SET closing_snapshot_id = ?, clv_away = ?, clv_home = ?, clv_total = ?
            WHERE id = ?
        """, (closing["id"], clv_away, clv_home, clv_total, row["id"]))
        updated += 1

    conn.commit()
    print(f"  Linked closing odds to {updated} predictions.")


# ─── End of Day Close ─────────────────────────────────────────

async def close_run(game_date: str = None, db_path: str = None) -> dict:
    """
    End of day workflow:
    1. Record actuals for game_date
    2. Print summary
    """
    if game_date is None:
        game_date = _yesterday_et()

    print("=" * 60)
    print(f"  MLB Pipeline — Daily Close for {game_date}")
    print("=" * 60)

    result = await record_actuals_for_date(game_date, db_path)
    print(f"\n  Games found:    {result.get('games_found', 0)}")
    print(f"  Results stored: {result.get('results_recorded', 0)}")
    print(f"  Bets logged:    {result.get('bets_logged', 0)}")

    return result


# ─── Prediction Logging with Odds ─────────────────────────────

def log_prediction_with_odds(
    prediction_id: int,
    game_date: str,
    model_away_win: float,
    model_home_win: float,
    model_total: float,
    odds_api_game_id: str = None,
    db_path: str = None,
) -> dict:
    """
    After logging a prediction via backtester.log_prediction(),
    call this to link it to the current odds snapshot.

    If odds_api_game_id is provided, uses that to find the snapshot.
    Otherwise, tries to match by team names and date.
    """
    conn = get_connection(db_path)
    _init_v2_schema(conn)

    # Find the latest odds snapshot for this game
    if odds_api_game_id:
        snap = conn.execute("""
            SELECT * FROM odds_snapshots
            WHERE game_id = ? ORDER BY captured_ts DESC LIMIT 1
        """, (odds_api_game_id,)).fetchone()
    else:
        # Fallback: match by prediction's teams and date
        pred = conn.execute(
            "SELECT away_team, home_team FROM predictions WHERE id = ?",
            (prediction_id,)
        ).fetchone()
        if pred:
            snap = conn.execute("""
                SELECT * FROM odds_snapshots
                WHERE game_date = ? AND away_team = ? AND home_team = ?
                ORDER BY captured_ts DESC LIMIT 1
            """, (game_date, pred["away_team"], pred["home_team"])).fetchone()
        else:
            snap = None

    if not snap:
        conn.close()
        return {"linked": False, "reason": "no matching odds snapshot"}

    snap = dict(snap)

    # Compute edges at prediction time
    edge_away = edge_home = edge_total = None
    if snap.get("away_ml") is not None:
        edge_away = round(model_away_win - _american_to_implied(snap["away_ml"]), 4)
    if snap.get("home_ml") is not None:
        edge_home = round(model_home_win - _american_to_implied(snap["home_ml"]), 4)

    # Insert prediction_odds linkage
    conn.execute("""
        INSERT OR REPLACE INTO prediction_odds
        (prediction_id, odds_snapshot_id, edge_away_at_pred, edge_home_at_pred, edge_total_at_pred)
        VALUES (?, ?, ?, ?, ?)
    """, (prediction_id, snap["id"], edge_away, edge_home, edge_total))
    conn.commit()

    # Also backfill the predictions table mkt_ fields for compatibility
    conn.execute("""
        UPDATE predictions
        SET mkt_away_ml = ?, mkt_home_ml = ?, mkt_total_line = ?,
            edge_away = ?, edge_home = ?
        WHERE id = ?
    """, (snap["away_ml"], snap["home_ml"], snap["total_line"],
          edge_away, edge_home, prediction_id))
    conn.commit()
    conn.close()

    return {
        "linked": True,
        "snapshot_id": snap["id"],
        "away_ml": snap["away_ml"],
        "home_ml": snap["home_ml"],
        "total_line": snap["total_line"],
        "edge_away": edge_away,
        "edge_home": edge_home,
    }


# ─── Backfill Existing Predictions ───────────────────────────

def backfill_predictions_odds(game_date: str = None, db_path: str = None) -> dict:
    """
    For predictions that have no odds linked, try to match them
    to existing odds snapshots by team + date.
    """
    conn = get_connection(db_path)
    _init_v2_schema(conn)

    # Find predictions without prediction_odds entries
    query = "SELECT * FROM predictions WHERE id NOT IN (SELECT prediction_id FROM prediction_odds)"
    if game_date:
        query += f" AND game_date = '{game_date}'"
    
    preds = conn.execute(query).fetchall()
    linked = 0

    for pred in preds:
        pred = dict(pred)
        # Try to find matching odds snapshot
        snap = conn.execute("""
            SELECT * FROM odds_snapshots
            WHERE game_date = ? AND away_team = ? AND home_team = ?
            ORDER BY captured_ts ASC LIMIT 1
        """, (pred["game_date"], pred["away_team"], pred["home_team"])).fetchone()

        if snap:
            snap = dict(snap)
            edge_away = edge_home = None
            if snap.get("away_ml") is not None:
                edge_away = round(pred["model_away_win"] - _american_to_implied(snap["away_ml"]), 4)
            if snap.get("home_ml") is not None:
                edge_home = round(pred["model_home_win"] - _american_to_implied(snap["home_ml"]), 4)

            conn.execute("""
                INSERT OR IGNORE INTO prediction_odds
                (prediction_id, odds_snapshot_id, edge_away_at_pred, edge_home_at_pred)
                VALUES (?, ?, ?, ?)
            """, (pred["id"], snap["id"], edge_away, edge_home))

            # Also update the predictions table
            conn.execute("""
                UPDATE predictions
                SET mkt_away_ml = ?, mkt_home_ml = ?, mkt_total_line = ?,
                    edge_away = ?, edge_home = ?
                WHERE id = ? AND mkt_away_ml IS NULL
            """, (snap["away_ml"], snap["home_ml"], snap["total_line"],
                  edge_away, edge_home, pred["id"]))
            linked += 1

    conn.commit()
    conn.close()
    return {"total_unlinked": len(preds), "linked": linked}


# ─── Report ───────────────────────────────────────────────────

def print_full_report(days: int = 30, db_path: str = None):
    """Extended report including CLV metrics."""
    end = _today_et()
    start_date = date.fromisoformat(end) - timedelta(days=days)
    start = start_date.isoformat()

    print("=" * 60)
    print(f"  MLB Model — Full Report")
    print(f"  {start} → {end}")
    print("=" * 60)

    # Standard report
    report = summary_report(start, end, db_path)
    if "error" in report:
        print(f"\n  {report['error']}")
        return

    ml = report.get("ml_accuracy", {})
    if ml:
        print(f"\n  ML ACCURACY: {ml.get('correct',0)}/{ml.get('total_games',0)} ({ml.get('accuracy',0):.1%})")

    tot = report.get("total_accuracy", {})
    if tot:
        print(f"  O/U ACCURACY: {tot.get('over_under_correct',0)}/{tot.get('total_games',0)} ({tot.get('over_under_accuracy',0):.1%})")
        print(f"  BIAS: {tot.get('avg_bias_runs',0):+.2f} runs/game")

    # CLV report
    conn = get_connection(db_path)
    _init_v2_schema(conn)
    
    clv_rows = conn.execute("""
        SELECT po.*, p.game_date, p.away_team, p.home_team,
               p.model_away_win, p.model_home_win
        FROM prediction_odds po
        JOIN predictions p ON p.id = po.prediction_id
        WHERE p.game_date BETWEEN ? AND ?
        AND (po.clv_away IS NOT NULL OR po.clv_home IS NOT NULL)
    """, (start, end)).fetchall()
    
    if clv_rows:
        clv_data = [dict(r) for r in clv_rows]
        avg_clv_away = sum(r.get("clv_away", 0) or 0 for r in clv_data) / len(clv_data)
        avg_clv_home = sum(r.get("clv_home", 0) or 0 for r in clv_data) / len(clv_data)
        print(f"\n  CLV (Closing Line Value):")
        print(f"    Games with CLV: {len(clv_data)}")
        print(f"    Avg CLV (away): {avg_clv_away:+.3f}")
        print(f"    Avg CLV (home): {avg_clv_home:+.3f}")
        positive_clv = sum(1 for r in clv_data if (r.get("clv_away", 0) or 0) > 0 or (r.get("clv_home", 0) or 0) > 0)
        print(f"    Positive CLV rate: {positive_clv}/{len(clv_data)} ({positive_clv/len(clv_data):.1%})")
    
    # Odds snapshot coverage
    snap_count = conn.execute("""
        SELECT COUNT(DISTINCT game_id) as games, COUNT(*) as snaps,
               COUNT(DISTINCT game_date) as dates
        FROM odds_snapshots
    """).fetchone()
    if snap_count:
        snap_count = dict(snap_count)
        print(f"\n  ODDS TRACKING:")
        print(f"    Unique games: {snap_count['games']}")
        print(f"    Total snapshots: {snap_count['snaps']}")
        print(f"    Dates covered: {snap_count['dates']}")
    
    conn.close()


# ─── CLI ──────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="MLB Model Auto Pipeline")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("morning", help="Capture opening odds")
    sub.add_parser("pregame", help="Capture closing odds + compute CLV")

    close_p = sub.add_parser("close", help="Record end-of-day results")
    close_p.add_argument("--date", type=str, help="Game date (default: yesterday)")
    close_p.add_argument("--backfill", type=int, help="Backfill last N days")

    report_p = sub.add_parser("report", help="Print full report")
    report_p.add_argument("--days", type=int, default=30)

    backfill_p = sub.add_parser("backfill-odds", help="Link existing predictions to odds")
    backfill_p.add_argument("--date", type=str, help="Specific date")

    args = parser.parse_args()

    if args.command == "morning":
        morning_run()
    elif args.command == "pregame":
        pregame_run()
    elif args.command == "close":
        if args.backfill:
            for i in range(1, args.backfill + 1):
                d = (date.today() - timedelta(days=i)).isoformat()
                await close_run(d)
        else:
            await close_run(args.date)
    elif args.command == "report":
        print_full_report(args.days)
    elif args.command == "backfill-odds":
        result = backfill_predictions_odds(args.date)
        print(f"Backfill: linked {result['linked']}/{result['total_unlinked']} predictions to odds")
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
