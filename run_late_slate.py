"""
Run MLB model projections for the late slate (9:40 PM EDT+) games on March 31, 2026.
Outputs pitcher projections and batter stat projections for each game.
"""
import asyncio
import json
import sys
sys.path.insert(0, '/home/user/workspace/mlb-model')

from mlb_data import (
    get_schedule, get_batter_stats, get_pitcher_stats,
    get_team_roster, get_league_averages, TEAM_ABBREVS, TEAM_NAMES
)
from log5_engine import (
    team_win_probability_with_home, matchup_probability_vector,
    compute_expected_stats, pitcher_quality_adjustment
)
from monte_carlo import run_monte_carlo
from projections import (
    load_steamer_projections, get_blended_batter_probs, get_blended_pitcher_probs,
    get_bullpen_probs, estimate_starter_innings, TEAM_BULLPEN_2025
)
from model_adjustments import (
    get_park_factors, apply_park_factors,
    get_player_handedness, get_platoon_splits, apply_platoon_adjustment,
    get_batter_game_log, compute_recent_form, apply_form_adjustment,
    get_pitcher_game_log, compute_pitcher_workload, apply_workload_adjustment,
    get_weather_from_espn, compute_weather_adjustment, apply_weather_adjustment,
    PARK_FACTORS, CONTROLLED_CLIMATE
)
from game_context import (
    get_full_game_context, apply_umpire_adjustment, apply_bullpen_availability
)
from market_data import prob_to_american

LATE_GAME_IDS = [825107, 823322, 823158, 823969]

async def build_lineup_with_order(team_id, real_lineup, season=2025):
    """Build lineup using actual batting order from MLB live feed."""
    league_avg = get_league_averages()
    if real_lineup and len(real_lineup) >= 7:
        batters = []
        for slot in real_lineup[:9]:
            pid = slot["id"]
            stats = await get_batter_stats(pid, season)
            if not stats:
                stats = {
                    "player_id": pid, "name": slot["name"],
                    "position": slot["position"],
                    **league_avg,
                    "avg": ".243", "ops": ".700", "hr": 10, "sb": 5,
                    "p_sb_attempt": 0.05,
                }
            else:
                stats["position"] = slot["position"]
            
            try:
                blended = await get_blended_batter_probs(pid, slot["name"])
                for k in ["p_1b", "p_2b", "p_3b", "p_hr", "p_bb", "p_hbp", "p_out"]:
                    if k in blended:
                        stats[k] = blended[k]
                stats["_blend_info"] = blended.get("_blend_info", {})
            except Exception:
                pass
            
            try:
                stats["_platoon_splits"] = await get_platoon_splits(pid, season)
            except Exception:
                stats["_platoon_splits"] = {}
            try:
                log = await get_batter_game_log(pid, season)
                stats["_form"] = compute_recent_form(log)
            except Exception:
                stats["_form"] = {"form_multiplier": 1.0, "trend": "neutral"}
            
            stats["_order"] = slot["order"]
            batters.append(stats)
        
        while len(batters) < 9:
            batters.append({
                "player_id": 99900 + len(batters),
                "name": f"Replacement #{len(batters)+1}",
                "position": "UT", **league_avg,
                "avg": ".243", "ops": ".700", "hr": 10, "sb": 5,
                "p_sb_attempt": 0.05,
                "_platoon_splits": {}, "_form": {"form_multiplier": 1.0, "trend": "neutral"},
                "_order": len(batters) + 1,
            })
        return batters[:9]
    return []


async def run_game(game, league_avg):
    """Run full simulation for one game with all adjustments."""
    home_team_id = game["home_team_id"]
    away_team_id = game["away_team_id"]
    park = get_park_factors(home_team_id)
    
    # Weather
    weather_map = await get_weather_from_espn()
    short_name = f"{game['away_team']} @ {game['home_team']}"
    weather_data = weather_map.get(short_name, {})
    temp = weather_data.get("temperature", 72) if weather_data else 72
    weather_adj = compute_weather_adjustment(temp=temp, home_team_id=home_team_id)
    
    # Load Steamer
    await load_steamer_projections()
    
    # Game context
    game_ctx = await get_full_game_context(game["game_id"], away_team_id, home_team_id)
    ump = game_ctx.get("umpire", {})
    bp_avail = game_ctx.get("bullpen_availability", {})
    real_lineups = game_ctx.get("lineups", {})
    
    # Build lineups
    away_lineup = await build_lineup_with_order(away_team_id, real_lineups.get("away", []), season=2025)
    home_lineup = await build_lineup_with_order(home_team_id, real_lineups.get("home", []), season=2025)
    
    if not away_lineup or not home_lineup:
        print(f"  WARNING: Could not build lineups for {game['away_team']} @ {game['home_team']}")
        return None
    
    # Pitchers
    away_pitcher = None
    home_pitcher = None
    away_workload = {"fatigue_factor": 1.0, "workload_status": "unknown", "avg_ip_last3": 5.5}
    home_workload = {"fatigue_factor": 1.0, "workload_status": "unknown", "avg_ip_last3": 5.5}
    away_hand = {"pitch_hand": "R"}
    home_hand = {"pitch_hand": "R"}
    
    if game.get("away_pitcher_id"):
        ap_blended = await get_blended_pitcher_probs(game["away_pitcher_id"], game.get("away_pitcher_name", "TBD"))
        ap_raw = await get_pitcher_stats(game["away_pitcher_id"], 2025)
        if ap_raw:
            away_pitcher = {**ap_raw, **{k: v for k, v in ap_blended.items() if k.startswith('p_')}}
        else:
            away_pitcher = ap_blended
        away_hand = await get_player_handedness(game["away_pitcher_id"])
        a_log = await get_pitcher_game_log(game["away_pitcher_id"], 2025)
        away_workload = compute_pitcher_workload(a_log)
    
    if game.get("home_pitcher_id"):
        hp_blended = await get_blended_pitcher_probs(game["home_pitcher_id"], game.get("home_pitcher_name", "TBD"))
        hp_raw = await get_pitcher_stats(game["home_pitcher_id"], 2025)
        if hp_raw:
            home_pitcher = {**hp_raw, **{k: v for k, v in hp_blended.items() if k.startswith('p_')}}
        else:
            home_pitcher = hp_blended
        home_hand = await get_player_handedness(game["home_pitcher_id"])
        h_log = await get_pitcher_game_log(game["home_pitcher_id"], 2025)
        home_workload = compute_pitcher_workload(h_log)
    
    if not away_pitcher:
        away_pitcher = {"name": game.get("away_pitcher_name", "TBD"), **league_avg}
    if not home_pitcher:
        home_pitcher = {"name": game.get("home_pitcher_name", "TBD"), **league_avg}
    
    # Workload adjustment
    adj_away_pitcher = apply_workload_adjustment(away_pitcher, away_workload)
    adj_home_pitcher = apply_workload_adjustment(home_pitcher, home_workload)
    
    # Bullpen
    away_bp = get_bullpen_probs(away_team_id)
    home_bp = get_bullpen_probs(home_team_id)
    away_bp = apply_bullpen_availability(away_bp, bp_avail.get("away", {}))
    home_bp = apply_bullpen_availability(home_bp, bp_avail.get("home", {}))
    
    away_sp_ip = estimate_starter_innings(away_pitcher, away_workload)
    home_sp_ip = estimate_starter_innings(home_pitcher, home_workload)
    
    # Build matchup vectors with all adjustments
    away_matchups = []
    for b in away_lineup:
        m = matchup_probability_vector(b, adj_home_pitcher, league_avg)
        m = apply_platoon_adjustment(m, b.get("_platoon_splits", {}), home_hand.get("pitch_hand", "R"))
        m = apply_form_adjustment(m, b.get("_form", {"form_multiplier": 1.0}))
        m = apply_park_factors(m, park)
        m = apply_weather_adjustment(m, weather_adj)
        m = apply_umpire_adjustment(m, ump)
        away_matchups.append(m)
    
    home_matchups = []
    for b in home_lineup:
        m = matchup_probability_vector(b, adj_away_pitcher, league_avg)
        m = apply_platoon_adjustment(m, b.get("_platoon_splits", {}), away_hand.get("pitch_hand", "R"))
        m = apply_form_adjustment(m, b.get("_form", {"form_multiplier": 1.0}))
        m = apply_park_factors(m, park)
        m = apply_weather_adjustment(m, weather_adj)
        m = apply_umpire_adjustment(m, ump)
        home_matchups.append(m)
    
    # Bullpen matchups
    away_bp_matchups = []
    for b in away_lineup:
        bpm = matchup_probability_vector(b, home_bp, league_avg)
        bpm = apply_park_factors(bpm, park)
        bpm = apply_weather_adjustment(bpm, weather_adj)
        away_bp_matchups.append(bpm)
    
    home_bp_matchups = []
    for b in home_lineup:
        bpm = matchup_probability_vector(b, away_bp, league_avg)
        bpm = apply_park_factors(bpm, park)
        bpm = apply_weather_adjustment(bpm, weather_adj)
        home_bp_matchups.append(bpm)
    
    # Monte Carlo
    results = run_monte_carlo(
        away_lineup, home_lineup,
        away_matchups, home_matchups,
        away_pitcher, home_pitcher,
        n_sims=5000,
        away_bullpen_matchups=away_bp_matchups,
        home_bullpen_matchups=home_bp_matchups,
        away_starter_ip=away_sp_ip,
        home_starter_ip=home_sp_ip,
    )
    
    return {
        "game": game,
        "results": results,
        "away_pitcher": {
            "name": away_pitcher.get("name", "TBD"),
            "era": away_pitcher.get("era", "0.00"),
            "whip": away_pitcher.get("whip", "0.00"),
            "k_per_9": away_pitcher.get("k_per_9", "0.00"),
            "hand": away_hand.get("pitch_hand", "R"),
            "workload": away_workload,
            "est_ip": round(away_sp_ip, 1),
            "p_1b": away_pitcher.get("p_1b", 0),
            "p_2b": away_pitcher.get("p_2b", 0),
            "p_3b": away_pitcher.get("p_3b", 0),
            "p_hr": away_pitcher.get("p_hr", 0),
            "p_bb": away_pitcher.get("p_bb", 0),
            "p_hbp": away_pitcher.get("p_hbp", 0),
            "p_out": away_pitcher.get("p_out", 0),
        },
        "home_pitcher": {
            "name": home_pitcher.get("name", "TBD"),
            "era": home_pitcher.get("era", "0.00"),
            "whip": home_pitcher.get("whip", "0.00"),
            "k_per_9": home_pitcher.get("k_per_9", "0.00"),
            "hand": home_hand.get("pitch_hand", "R"),
            "workload": home_workload,
            "est_ip": round(home_sp_ip, 1),
            "p_1b": home_pitcher.get("p_1b", 0),
            "p_2b": home_pitcher.get("p_2b", 0),
            "p_3b": home_pitcher.get("p_3b", 0),
            "p_hr": home_pitcher.get("p_hr", 0),
            "p_bb": home_pitcher.get("p_bb", 0),
            "p_hbp": home_pitcher.get("p_hbp", 0),
            "p_out": home_pitcher.get("p_out", 0),
        },
        "away_lineup_detail": away_lineup,
        "home_lineup_detail": home_lineup,
        "park": park,
        "weather": weather_adj,
        "ump": ump,
        "lineup_source": "actual" if real_lineups.get("lineup_available") else "projected",
    }


async def main():
    games = await get_schedule("2026-03-31")
    league_avg = get_league_averages()
    
    late_games = [g for g in games if g["game_id"] in LATE_GAME_IDS]
    print(f"Found {len(late_games)} late-slate games\n")
    
    all_results = []
    for game in late_games:
        label = f"{game['away_team']} @ {game['home_team']}"
        print(f"Running: {label}...")
        result = await run_game(game, league_avg)
        if result:
            all_results.append(result)
            print(f"  Done: {result['results']['away_win_pct']:.1%} / {result['results']['home_win_pct']:.1%}")
        else:
            print(f"  SKIPPED")
    
    # Save full results as JSON for analysis
    output = []
    for r in all_results:
        g = r["game"]
        res = r["results"]
        
        game_output = {
            "matchup": f"{g['away_team']} @ {g['home_team']}",
            "away_team": g["away_team"],
            "home_team": g["home_team"],
            "away_win_pct": round(res["away_win_pct"], 4),
            "home_win_pct": round(res["home_win_pct"], 4),
            "away_odds": prob_to_american(res["away_win_pct"]),
            "home_odds": prob_to_american(res["home_win_pct"]),
            "total_runs_mean": round(res["total_runs_mean"], 2),
            "away_runs_mean": round(res["away_runs_mean"], 2),
            "home_runs_mean": round(res["home_runs_mean"], 2),
            "f5_away_win": round(res.get("f5_away_win", 0), 4),
            "f5_home_win": round(res.get("f5_home_win", 0), 4),
            "f5_draw": round(res.get("f5_draw", 0), 4),
            "away_pitcher": r["away_pitcher"],
            "home_pitcher": r["home_pitcher"],
            "park": r["park"],
            "lineup_source": r["lineup_source"],
        }
        
        # Build projection lookup by player_id
        away_proj_map = {}
        for p in res.get("away_projections", []):
            away_proj_map[p.get("player_id")] = p
        home_proj_map = {}
        for p in res.get("home_projections", []):
            home_proj_map[p.get("player_id")] = p
        
        # Batter projections
        away_batters = []
        for i, b in enumerate(r["away_lineup_detail"]):
            pid = b.get("player_id", 0)
            proj = away_proj_map.get(pid)
            # Fallback: match by name
            if not proj:
                for p in res.get("away_projections", []):
                    if p.get("name") == b.get("name"):
                        proj = p
                        break
            
            away_batters.append({
                "order": b.get("_order", i+1),
                "name": b.get("name", "?"),
                "position": b.get("position", "?"),
                "avg": b.get("avg", ".000"),
                "ops": b.get("ops", ".000"),
                "hr_season": b.get("hr", 0),
                "form": b.get("_form", {}).get("trend", "neutral"),
                "form_mult": round(b.get("_form", {}).get("form_multiplier", 1.0), 3),
                "sim_hits": round(proj.get("avg_hits", 0), 2) if proj else 0,
                "sim_hr": round(proj.get("avg_hr", 0), 3) if proj else 0,
                "sim_rbi": round(proj.get("avg_rbi", 0), 2) if proj else 0,
                "sim_runs": round(proj.get("avg_runs", 0), 2) if proj else 0,
                "sim_bb": round(proj.get("avg_bb", 0), 2) if proj else 0,
                "sim_sb": round(proj.get("avg_sb", 0), 3) if proj else 0,
                "dk_median": round(proj.get("dk_median", 0), 2) if proj else 0,
                "dk_p90": round(proj.get("dk_p90", 0), 2) if proj else 0,
                "dk_p99": round(proj.get("dk_p99", 0), 2) if proj else 0,
                "hit_rate": round(proj.get("hit_rate", 0), 3) if proj else 0,
                "hr_rate": round(proj.get("hr_rate", 0), 3) if proj else 0,
                "multi_hit_rate": round(proj.get("multi_hit_rate", 0), 3) if proj else 0,
            })
        
        home_batters = []
        for i, b in enumerate(r["home_lineup_detail"]):
            pid = b.get("player_id", 0)
            proj = home_proj_map.get(pid)
            if not proj:
                for p in res.get("home_projections", []):
                    if p.get("name") == b.get("name"):
                        proj = p
                        break
            
            home_batters.append({
                "order": b.get("_order", i+1),
                "name": b.get("name", "?"),
                "position": b.get("position", "?"),
                "avg": b.get("avg", ".000"),
                "ops": b.get("ops", ".000"),
                "hr_season": b.get("hr", 0),
                "form": b.get("_form", {}).get("trend", "neutral"),
                "form_mult": round(b.get("_form", {}).get("form_multiplier", 1.0), 3),
                "sim_hits": round(proj.get("avg_hits", 0), 2) if proj else 0,
                "sim_hr": round(proj.get("avg_hr", 0), 3) if proj else 0,
                "sim_rbi": round(proj.get("avg_rbi", 0), 2) if proj else 0,
                "sim_runs": round(proj.get("avg_runs", 0), 2) if proj else 0,
                "sim_bb": round(proj.get("avg_bb", 0), 2) if proj else 0,
                "sim_sb": round(proj.get("avg_sb", 0), 3) if proj else 0,
                "dk_median": round(proj.get("dk_median", 0), 2) if proj else 0,
                "dk_p90": round(proj.get("dk_p90", 0), 2) if proj else 0,
                "dk_p99": round(proj.get("dk_p99", 0), 2) if proj else 0,
                "hit_rate": round(proj.get("hit_rate", 0), 3) if proj else 0,
                "hr_rate": round(proj.get("hr_rate", 0), 3) if proj else 0,
                "multi_hit_rate": round(proj.get("multi_hit_rate", 0), 3) if proj else 0,
            })
        
        game_output["away_batters"] = sorted(away_batters, key=lambda x: x["order"])
        game_output["home_batters"] = sorted(home_batters, key=lambda x: x["order"])
        output.append(game_output)
    
    # Save to file
    with open("/home/user/workspace/mlb-model/late_slate_projections.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved {len(output)} games to late_slate_projections.json")
    
    # Print summary
    for g in output:
        print(f"\n{'='*80}")
        print(f"  {g['matchup']}   |   Total: {g['total_runs_mean']}")
        print(f"  {g['away_pitcher']['name']} ({g['away_pitcher']['hand']}HP, {g['away_pitcher']['era']} ERA, ~{g['away_pitcher']['est_ip']} IP)")
        print(f"  vs {g['home_pitcher']['name']} ({g['home_pitcher']['hand']}HP, {g['home_pitcher']['era']} ERA, ~{g['home_pitcher']['est_ip']} IP)")
        print(f"  Win%: {g['away_team']} {g['away_win_pct']:.1%} ({g['away_odds']})  /  {g['home_team']} {g['home_win_pct']:.1%} ({g['home_odds']})")
        print(f"  Runs: {g['away_team']} {g['away_runs_mean']:.1f} | {g['home_team']} {g['home_runs_mean']:.1f} | Total {g['total_runs_mean']:.1f}")
        print(f"  Lineups: {g['lineup_source']}")
        
        for side, batters, team in [('AWAY', g['away_batters'], g['away_team']), ('HOME', g['home_batters'], g['home_team'])]:
            print(f"\n  {team} ({side})")
            print(f"  {'#':<3} {'Name':<22} {'Pos':<4} {'AVG':<6} {'OPS':<6} {'Form':<7} {'Hits':<5} {'HR':<6} {'RBI':<5} {'R':<5} {'BB':<5} {'SB':<5} {'DK50':<6} {'DK90':<6} {'DK99':<6}")
            print(f"  {'-'*110}")
            for b in batters:
                print(f"  {b['order']:<3} {b['name']:<22} {b['position']:<4} {b['avg']:<6} {b['ops']:<6} {b['form']:<7} {b['sim_hits']:<5} {b['sim_hr']:<6.3f} {b['sim_rbi']:<5} {b['sim_runs']:<5} {b['sim_bb']:<5} {b['sim_sb']:<5.3f} {b['dk_median']:<6} {b['dk_p90']:<6} {b['dk_p99']:<6}")

if __name__ == "__main__":
    asyncio.run(main())
