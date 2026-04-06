#!/usr/bin/env python3
"""
daily_runner.py - End-of-day result recorder for MLB model backtesting.

Usage:
    python daily_runner.py                    # records yesterday's results  
    python daily_runner.py --date 2026-04-05  # records specific date
    python daily_runner.py --backfill 7       # backfill last 7 days
    python daily_runner.py --report           # print analytics report

Can also be imported and called programmatically:
    from backtesting.daily_runner import run_daily_close
    await run_daily_close("2026-04-05")
"""
import sys, argparse, asyncio, json
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtesting.backtester import record_actuals_for_date
from backtesting.analytics import summary_report
from backtesting.results_db import get_connection, get_summary_stats

async def run_daily_close(game_date: str, db_path: str = None, verbose: bool = True) -> dict:
    """
    Main function: record actuals for a game date.
    Safe to call multiple times (idempotent).
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  MLB Backtest — Daily Close for {game_date}")
        print(f"{'='*60}")
    
    result = await record_actuals_for_date(game_date, db_path)
    
    if verbose:
        print(f"  Games found:     {result.get('games_found', 0)}")
        print(f"  Results stored:  {result.get('results_recorded', 0)}")
    
    return result

async def backfill(n_days: int = 7, db_path: str = None):
    """Backfill the last n_days of results."""
    today = datetime.now(timezone.utc).date()
    for i in range(1, n_days + 1):
        date_str = (today - timedelta(days=i)).isoformat()
        print(f"\nBackfilling {date_str}...")
        result = await run_daily_close(date_str, db_path, verbose=False)
        print(f"  {date_str}: {result.get('results_recorded', 0)} games recorded")

def print_report(days: int = 30, db_path: str = None):
    """Print analytics report for last N days."""
    end = datetime.now(timezone.utc).date().isoformat()
    start = (datetime.now(timezone.utc).date() - timedelta(days=days)).isoformat()
    
    print(f"\n{'='*60}")
    print(f"  MLB Model Performance Report")
    print(f"  Period: {start} → {end} ({days} days)")
    print(f"{'='*60}")
    
    conn = get_connection(db_path)
    stats = get_summary_stats(conn)
    conn.close()
    
    if not stats.get("total_games_predicted"):
        print("  No predictions recorded yet.")
        return
    
    print(f"\n  Tracked games:  {stats.get('total_games_predicted', 0)} predicted")
    print(f"  Resolved games: {stats.get('total_games_resolved', 0)}")
    print(f"  Date range:     {stats.get('earliest_date','?')} → {stats.get('latest_date','?')}")
    
    report = summary_report(start, end, db_path)
    if "error" in report:
        print(f"\n  {report['error']}")
        return
    
    ml = report.get("ml_accuracy", {})
    if ml:
        print(f"\n  ML ACCURACY (all games)")
        print(f"    Correct:    {ml['correct']} / {ml['total_games']} ({ml['accuracy']:.1%})")
        print(f"    Break-even: {ml['break_even_accuracy']:.1%}")
    
    tot = report.get("total_accuracy", {})
    if tot:
        print(f"\n  TOTAL O/U ACCURACY")
        print(f"    Correct:    {tot['over_under_correct']} / {tot['total_games']} ({tot['over_under_accuracy']:.1%})")
        print(f"    Model avg:  {tot['model_mean_total']:.1f} runs/game")
        print(f"    Actual avg: {tot['actual_mean_total']:.1f} runs/game")
        print(f"    Bias:       {tot['avg_bias_runs']:+.2f} (+ = model over-projects)")
    
    ea = report.get("edge_accuracy_5pct", {})
    if ea and ea.get("games_with_edge", 0) > 0:
        print(f"\n  EDGE PLAYS (≥5% model edge)")
        print(f"    Games:    {ea['games_with_edge']}")
        print(f"    Correct:  {ea['correct']} ({ea['accuracy']:.1%})")
        print(f"    Avg edge: {ea['avg_edge_pct']:.1f}%")
    
    roi = report.get("roi", {})
    if roi and roi.get("total_bets", 0) > 0:
        print(f"\n  HYPOTHETICAL ROI (half-Kelly)")
        print(f"    Bets:     {roi['total_bets']}")
        print(f"    Wagered:  {roi['total_wagered_units']:.2f} units")
        print(f"    Profit:   {roi['total_profit_units']:+.2f} units")
        print(f"    ROI:      {roi['roi_pct']:+.1f}%")
    
    print(f"\n  CALIBRATION")
    for b in report.get("calibration", []):
        bar = "█" * int(abs(b["calibration_error"]) * 50)
        direction = "+" if b["calibration_error"] > 0 else "-"
        print(f"    {b['bucket']}: model {b['avg_model_prob']:.0%} → actual {b['actual_win_rate']:.0%}  err {b['calibration_error']:+.2f}  {direction}{bar}")
    
    print()

async def main():
    parser = argparse.ArgumentParser(description="MLB Model Daily Result Recorder")
    parser.add_argument("--date", type=str, help="Specific date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--backfill", type=int, metavar="N", help="Backfill last N days")
    parser.add_argument("--report", action="store_true", help="Print analytics report")
    parser.add_argument("--days", type=int, default=30, help="Days to include in report (default 30)")
    args = parser.parse_args()
    
    if args.report:
        print_report(args.days)
    elif args.backfill:
        await backfill(args.backfill)
    else:
        game_date = args.date or (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
        await run_daily_close(game_date)
        print_report(30)

if __name__ == "__main__":
    asyncio.run(main())
