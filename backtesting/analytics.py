import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import numpy as np
from backtesting.results_db import get_connection, get_predictions_with_actuals, get_bet_log

def load_joined_df(start_date: str, end_date: str, db_path: str = None) -> pd.DataFrame:
    """Load predictions+actuals as a DataFrame."""
    conn = get_connection(db_path)
    rows = get_predictions_with_actuals(conn, start_date, end_date)
    conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Derive helper columns
    df["model_fav"] = df.apply(lambda r: "away" if r["model_away_win"] > r["model_home_win"] else "home", axis=1)
    df["model_correct"] = df["model_fav"] == df["winner"]
    df["total_over"] = df["total_runs"] > df["mkt_total_line"]
    df["model_said_over"] = df["mkt_total_dir"] == "over"
    df["total_correct"] = (df["total_over"] == df["model_said_over"]) & df["mkt_total_line"].notna()
    df["model_edge_side"] = df.apply(
        lambda r: "away" if (r.get("edge_away") or 0) > (r.get("edge_home") or 0) else "home", axis=1
    )
    df["edge_side_correct"] = df["model_edge_side"] == df["winner"]
    return df

def ml_accuracy(df: pd.DataFrame) -> dict:
    """How often does the model's favorite win?"""
    if df.empty: return {}
    total = len(df)
    correct = df["model_correct"].sum()
    return {
        "total_games": total,
        "correct": int(correct),
        "accuracy": round(correct / total, 4) if total > 0 else 0,
        "break_even_accuracy": 0.5238,  # needed at -110 juice
    }

def calibration_report(df: pd.DataFrame, n_buckets: int = 10) -> list[dict]:
    """
    Bucket model win probabilities into deciles and compare to actual win rates.
    Perfect calibration: 60% model prob bucket should win 60% of the time.
    Returns list of bucket dicts.
    """
    if df.empty: return []
    buckets = []
    # Use the probability of the MODEL FAVORITE
    df["fav_prob"] = df.apply(
        lambda r: r["model_away_win"] if r["model_fav"] == "away" else r["model_home_win"], axis=1
    )
    edges = np.linspace(0.5, 1.0, n_buckets + 1)
    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        mask = (df["fav_prob"] >= lo) & (df["fav_prob"] < hi)
        subset = df[mask]
        if len(subset) == 0:
            continue
        actual_win_rate = subset["model_correct"].mean()
        avg_model_prob = subset["fav_prob"].mean()
        buckets.append({
            "bucket": f"{lo:.0%}–{hi:.0%}",
            "count": len(subset),
            "avg_model_prob": round(avg_model_prob, 3),
            "actual_win_rate": round(actual_win_rate, 3),
            "calibration_error": round(actual_win_rate - avg_model_prob, 3),
        })
    return buckets

def total_accuracy(df: pd.DataFrame) -> dict:
    """Over/under accuracy vs the model's projection."""
    if df.empty: return {}
    subset = df[df["mkt_total_line"].notna() & df["total_runs"].notna()]
    if subset.empty: return {}
    correct = subset["total_correct"].sum()
    total = len(subset)
    avg_model_error = (subset["model_total"] - subset["total_runs"]).abs().mean()
    avg_model_bias = (subset["model_total"] - subset["total_runs"]).mean()
    return {
        "total_games": total,
        "over_under_correct": int(correct),
        "over_under_accuracy": round(correct / total, 4) if total > 0 else 0,
        "avg_absolute_error_runs": round(avg_model_error, 3),
        "avg_bias_runs": round(avg_model_bias, 3),  # positive = model over-projects
        "model_mean_total": round(subset["model_total"].mean(), 2),
        "actual_mean_total": round(subset["total_runs"].mean(), 2),
    }

def edge_accuracy(df: pd.DataFrame, min_edge_pct: float = 3.0) -> dict:
    """Accuracy on games where model edge > min_edge_pct."""
    if df.empty: return {}
    min_edge = min_edge_pct / 100
    # Max edge per game
    df = df.copy()
    df["max_edge"] = df.apply(
        lambda r: max(abs(r.get("edge_away") or 0), abs(r.get("edge_home") or 0)), axis=1
    )
    high_edge = df[df["max_edge"] >= min_edge]
    if high_edge.empty:
        return {"games_with_edge": 0}
    correct = high_edge["edge_side_correct"].sum()
    total = len(high_edge)
    return {
        "min_edge_pct": min_edge_pct,
        "games_with_edge": total,
        "correct": int(correct),
        "accuracy": round(correct / total, 4) if total > 0 else 0,
        "avg_edge_pct": round(high_edge["max_edge"].mean() * 100, 2),
    }

def roi_report(df: pd.DataFrame, db_path: str = None,
               start_date: str = None, end_date: str = None) -> dict:
    """ROI from the bet_log table (half-Kelly hypothetical bets)."""
    if start_date and end_date:
        conn = get_connection(db_path)
        bets = get_bet_log(conn, start_date, end_date)
        conn.close()
        if not bets:
            return {"total_bets": 0}
        bet_df = pd.DataFrame(bets)
    else:
        return {"total_bets": 0}
    
    resolved = bet_df[bet_df["result"].notna() & (bet_df["result"] != "push")]
    if resolved.empty:
        return {"total_bets": 0}
    
    total_wagered = resolved["hypothetical_units"].sum()
    total_profit = resolved["profit_units"].sum()
    roi = total_profit / total_wagered if total_wagered > 0 else 0
    
    # By bet type
    by_type = resolved.groupby("bet_type").agg(
        count=("result", "count"),
        wins=("result", lambda x: (x == "win").sum()),
        profit=("profit_units", "sum"),
    ).round(4).to_dict("index")
    
    return {
        "total_bets": len(resolved),
        "total_wagered_units": round(total_wagered, 4),
        "total_profit_units": round(total_profit, 4),
        "roi_pct": round(roi * 100, 2),
        "by_type": by_type,
    }

def summary_report(start_date: str, end_date: str, db_path: str = None) -> dict:
    """Full summary report for a date range."""
    df = load_joined_df(start_date, end_date, db_path)
    if df.empty:
        return {"error": f"No resolved games between {start_date} and {end_date}"}
    
    return {
        "date_range": f"{start_date} to {end_date}",
        "ml_accuracy": ml_accuracy(df),
        "total_accuracy": total_accuracy(df),
        "edge_accuracy_3pct": edge_accuracy(df, 3.0),
        "edge_accuracy_5pct": edge_accuracy(df, 5.0),
        "calibration": calibration_report(df),
        "roi": roi_report(df, db_path, start_date, end_date),
    }
