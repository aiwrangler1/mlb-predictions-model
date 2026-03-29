"""
MLB Predictions Model v2 - FastAPI Server
Integrates market odds (Odds API, Kalshi, Polymarket) and model adjustments
(park factors, platoon splits, recent form, pitcher workload, weather).
"""
import asyncio
import math
import json as json_module
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from datetime import date
from typing import Optional
from mlb_data import (
    get_schedule, get_batter_stats, get_pitcher_stats,
    get_team_roster, get_league_averages, clear_cache,
    TEAM_ABBREVS, TEAM_NAMES
)
from log5_engine import (
    team_win_probability_with_home, matchup_probability_vector,
    compute_expected_stats, pitcher_quality_adjustment
)
from monte_carlo import run_monte_carlo
from market_data import (
    get_all_market_odds, get_odds_api_mlb, get_kalshi_mlb_games,
    get_polymarket_mlb, match_odds_to_game, set_odds_api_key,
    compute_edge, american_to_prob, prob_to_american,
    clear_market_cache
)
from game_context import (
    get_full_game_context, get_game_lineups, apply_umpire_adjustment,
    apply_bullpen_availability, get_pa_allocation_by_order,
    clear_ctx_cache
)
from projections import (
    load_steamer_projections, get_blended_batter_probs, get_blended_pitcher_probs,
    get_bullpen_probs, estimate_starter_innings, clear_proj_cache,
    TEAM_BULLPEN_2025
)
from model_adjustments import (
    get_park_factors, apply_park_factors,
    get_player_handedness, get_platoon_splits, apply_platoon_adjustment,
    get_batter_game_log, compute_recent_form, apply_form_adjustment,
    get_pitcher_game_log, compute_pitcher_workload, apply_workload_adjustment,
    get_weather_from_espn, compute_weather_adjustment, apply_weather_adjustment,
    get_all_adjustments, PARK_FACTORS, CONTROLLED_CLIMATE,
    clear_adj_cache
)
def sanitize(obj):
    """Replace NaN/Inf with 0 recursively for JSON safety."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj
app = FastAPI(title="MLB Predictions Model v2")
app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/")
async def root():
    return FileResponse("static/index.html")
# ─── Settings ──────────────────────────────────────────────────
@app.post("/api/settings/odds-api-key")
async def set_api_key(request: Request):
    body = await request.json()
    key = body.get("key", "")
    if key:
        set_odds_api_key(key)
        return {"status": "ok", "message": "Odds API key set"}
    return {"status": "error", "message": "No key provided"}
# ─── Schedule ──────────────────────────────────────────────────
@app.get("/api/schedule")
async def api_schedule(game_date: Optional[str] = None):
    """Get today's schedule with Log5 win probs + weather + park factors."""
    games = await get_schedule(game_date)
    weather_map = await get_weather_from_espn()
    
    for game in games:
        # Log5 win probability
        away_prob, home_prob = team_win_probability_with_home(
            game["away_wpct"], game["home_wpct"]
        )
        game["away_win_prob"] = round(away_prob, 3)
        game["home_win_prob"] = round(home_prob, 3)
        game["away_fair_odds"] = prob_to_american(away_prob)
        game["home_fair_odds"] = prob_to_american(home_prob)
        
        # Park factors
        park = get_park_factors(game["home_team_id"])
        game["park"] = park
        
        # Weather
        short_name = f"{game['away_team']} @ {game['home_team']}"
        full_name = f"{game['away_team_name']} at {game['home_team_name']}"
        weather_data = weather_map.get(short_name) or weather_map.get(full_name) or {}
        
        if weather_data:
            temp = weather_data.get("temperature", 72)
            w_adj = compute_weather_adjustment(
                temp=temp, home_team_id=game["home_team_id"]
            )
            game["weather"] = {
                "temp": temp,
                "condition": weather_data.get("condition_id", ""),
                "display": weather_data.get("display", ""),
                **w_adj
            }
        else:
            game["weather"] = {
                "temp": None, "description": "No data",
                "hr_factor": 1.0, "hit_factor": 1.0, "is_dome": game["home_team_id"] in CONTROLLED_CLIMATE
            }
        
        # Is dome/retractable?
        game["is_dome"] = game["home_team_id"] in CONTROLLED_CLIMATE
    
    return {"games": games, "date": game_date or date.today().isoformat()}
# ─── Market Odds ───────────────────────────────────────────────
@app.get("/api/market-odds")
async def api_market_odds():
    """Get odds from all market sources: sportsbooks, Kalshi, Polymarket."""
    data = await get_all_market_odds()
    return JSONResponse(content=sanitize(data))
@app.get("/api/sportsbook-odds")
async def api_sportsbook_odds():
    """Get detailed sportsbook odds from The Odds API."""
    odds = await get_odds_api_mlb()
    return JSONResponse(content=sanitize({"games": odds, "count": len(odds)}))
# ─── Simulation with Adjustments ──────────────────────────────
@app.get("/api/simulate/{game_id}")
async def api_simulate(game_id: int, n_sims: int = 5000, game_date: Optional[str] = None):
    """Run Monte Carlo simulation with all model adjustments applied."""
    games = await get_schedule(game_date)
    game = None
    for g in games:
        if g["game_id"] == game_id:
            game = g
            break
    
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    league_avg = get_league_averages()
    home_team_id = game["home_team_id"]
    park = get_park_factors(home_team_id)
    
    # Get weather
    weather_map = await get_weather_from_espn()
    short_name = f"{game['away_team']} @ {game['home_team']}"
    weather_data = weather_map.get(short_name, {})
    temp = weather_data.get("temperature", 72) if weather_data else 72
    weather_adj = compute_weather_adjustment(temp=temp, home_team_id=home_team_id)
    
    # Pre-load Steamer projections
    await load_steamer_projections()
    
    # Pull GAME CONTEXT: actual lineups, umpire, bullpen availability
    game_ctx = await get_full_game_context(
        game["game_id"], game["away_team_id"], game["home_team_id"]
    )
    ump = game_ctx.get("umpire", {})
    bp_avail = game_ctx.get("bullpen_availability", {})
    real_lineups = game_ctx.get("lineups", {})
    
    # Build lineups — use REAL batting order if available
    away_lineup = await build_lineup_with_order(
        game["away_team_id"], real_lineups.get("away", []), season=2025
    )
    home_lineup = await build_lineup_with_order(
        game["home_team_id"], real_lineups.get("home", []), season=2025
    )
    
    # Get pitcher stats with blended projections + workload
    away_pitcher = None
    home_pitcher = None
    away_workload = {"fatigue_factor": 1.0, "workload_status": "unknown", "avg_ip_last3": 5.5}
    home_workload = {"fatigue_factor": 1.0, "workload_status": "unknown", "avg_ip_last3": 5.5}
    
    if game.get("away_pitcher_id"):
        # Use blended pitcher projection
        away_pitcher_blended = await get_blended_pitcher_probs(
            game["away_pitcher_id"], game.get("away_pitcher_name", "TBD"))
        away_pitcher_raw = await get_pitcher_stats(game["away_pitcher_id"], 2025)
        # Merge: keep metadata from raw, probabilities from blended
        if away_pitcher_raw:
            away_pitcher = {**away_pitcher_raw, **{k: v for k, v in away_pitcher_blended.items() if k.startswith('p_')}}
        else:
            away_pitcher = away_pitcher_blended
        away_hand = await get_player_handedness(game["away_pitcher_id"])
        a_log = await get_pitcher_game_log(game["away_pitcher_id"], 2025)
        away_workload = compute_pitcher_workload(a_log)
    
    if game.get("home_pitcher_id"):
        home_pitcher_blended = await get_blended_pitcher_probs(
            game["home_pitcher_id"], game.get("home_pitcher_name", "TBD"))
        home_pitcher_raw = await get_pitcher_stats(game["home_pitcher_id"], 2025)
        if home_pitcher_raw:
            home_pitcher = {**home_pitcher_raw, **{k: v for k, v in home_pitcher_blended.items() if k.startswith('p_')}}
        else:
            home_pitcher = home_pitcher_blended
        home_hand = await get_player_handedness(game["home_pitcher_id"])
        h_log = await get_pitcher_game_log(game["home_pitcher_id"], 2025)
        home_workload = compute_pitcher_workload(h_log)
    
    if not away_pitcher:
        away_pitcher = {"name": game.get("away_pitcher_name", "TBD"), **league_avg}
        away_hand = {"pitch_hand": "R"}
    else:
        if 'away_hand' not in dir():
            away_hand = {"pitch_hand": "R"}
    
    if not home_pitcher:
        home_pitcher = {"name": game.get("home_pitcher_name", "TBD"), **league_avg}
        home_hand = {"pitch_hand": "R"}
    else:
        if 'home_hand' not in dir():
            home_hand = {"pitch_hand": "R"}
    
    # Apply workload adjustment to pitcher probabilities
    adjusted_home_pitcher = apply_workload_adjustment(home_pitcher, home_workload)
    adjusted_away_pitcher = apply_workload_adjustment(away_pitcher, away_workload)
    
    # Compute bullpen probability vectors, adjusted for availability
    away_bullpen_probs = get_bullpen_probs(game["away_team_id"])
    home_bullpen_probs = get_bullpen_probs(game["home_team_id"])
    away_bullpen_probs = apply_bullpen_availability(away_bullpen_probs, bp_avail.get("away", {}))
    home_bullpen_probs = apply_bullpen_availability(home_bullpen_probs, bp_avail.get("home", {}))
    
    # Estimate starter exit points
    away_sp_ip = estimate_starter_innings(away_pitcher, away_workload)
    home_sp_ip = estimate_starter_innings(home_pitcher, home_workload)
    
    # Use Retrosheet-calibrated 7-event league averages (from PBP mining)
    # 9-event model available but currently in testing — using proven 7-event path
    
    # Compute matchup probabilities WITH all adjustments
    away_matchups = []
    away_adjustments = []
    for batter in away_lineup:
        # Base Log5 matchup (7-event, proven stable)
        matchup = matchup_probability_vector(batter, adjusted_home_pitcher, league_avg)
        
        # Apply platoon splits
        platoon = batter.get("_platoon_splits", {})
        p_hand = home_hand.get("pitch_hand", "R")
        matchup = apply_platoon_adjustment(matchup, platoon, p_hand)
        
        # Apply recent form
        form = batter.get("_form", {"form_multiplier": 1.0})
        matchup = apply_form_adjustment(matchup, form)
        
        # Apply park factors
        matchup = apply_park_factors(matchup, park)
        
        # Apply weather
        matchup = apply_weather_adjustment(matchup, weather_adj)
        
        # Apply umpire tendency
        matchup = apply_umpire_adjustment(matchup, ump)
        
        away_matchups.append(matchup)
        away_adjustments.append({
            "name": batter.get("name", "?"),
            "platoon": "vs " + p_hand + "HP",
            "form": form.get("trend", "neutral"),
            "form_mult": form.get("form_multiplier", 1.0),
            "order": batter.get("_order", 0),
        })
    
    home_matchups = []
    home_adjustments = []
    for batter in home_lineup:
        matchup = matchup_probability_vector(batter, adjusted_away_pitcher, league_avg)
        platoon = batter.get("_platoon_splits", {})
        p_hand = away_hand.get("pitch_hand", "R")
        matchup = apply_platoon_adjustment(matchup, platoon, p_hand)
        form = batter.get("_form", {"form_multiplier": 1.0})
        matchup = apply_form_adjustment(matchup, form)
        matchup = apply_park_factors(matchup, park)
        matchup = apply_weather_adjustment(matchup, weather_adj)
        matchup = apply_umpire_adjustment(matchup, ump)
        home_matchups.append(matchup)
        home_adjustments.append({
            "name": batter.get("name", "?"),
            "platoon": "vs " + p_hand + "HP",
            "form": form.get("trend", "neutral"),
            "form_mult": form.get("form_multiplier", 1.0),
            "order": batter.get("_order", 0),
        })
    
    # Compute BULLPEN matchups (what batters do vs the other team's bullpen)
    # Away batters vs home bullpen
    away_bp_matchups = []
    for batter in away_lineup:
        bp_matchup = matchup_probability_vector(batter, home_bullpen_probs, league_avg)
        bp_matchup = apply_park_factors(bp_matchup, park)
        bp_matchup = apply_weather_adjustment(bp_matchup, weather_adj)
        away_bp_matchups.append(bp_matchup)
    
    # Home batters vs away bullpen
    home_bp_matchups = []
    for batter in home_lineup:
        bp_matchup = matchup_probability_vector(batter, away_bullpen_probs, league_avg)
        bp_matchup = apply_park_factors(bp_matchup, park)
        bp_matchup = apply_weather_adjustment(bp_matchup, weather_adj)
        home_bp_matchups.append(bp_matchup)
    
    
    # Run Monte Carlo with starter/bullpen split
    n_sims = min(n_sims, 10000)
    results = run_monte_carlo(
        away_lineup, home_lineup,
        away_matchups, home_matchups,
        away_pitcher, home_pitcher,
        n_sims=n_sims,
        away_bullpen_matchups=away_bp_matchups,
        home_bullpen_matchups=home_bp_matchups,
        away_starter_ip=away_sp_ip,
        home_starter_ip=home_sp_ip,
    )
    
    # Match sportsbook odds
    sportsbook_odds = await get_odds_api_mlb()
    matched_odds = match_odds_to_game(game, sportsbook_odds)
    
    # Match Kalshi
    kalshi_games = await get_kalshi_mlb_games()
    
    # Compute edges
    edges = {}
    if matched_odds and matched_odds.get("consensus"):
        consensus = matched_odds["consensus"]
        if consensus.get("away_ml"):
            edges["away"] = compute_edge(results["away_win_pct"], consensus["away_ml"])
            edges["home"] = compute_edge(results["home_win_pct"], consensus["home_ml"])
        if consensus.get("total"):
            edges["total_line"] = consensus["total"]
            edges["model_total"] = results["total_runs_mean"]
            edges["total_diff"] = round(results["total_runs_mean"] - consensus["total"], 2)
    
    # Add all context to results
    results["game"] = game
    results["adjustments"] = {
        "park": park,
        "weather": weather_adj,
        "bullpen": {
            "away": {"era": TEAM_BULLPEN_2025.get(game['away_team_id'], {}).get('era', 4.2), "team": game['away_team']},
            "home": {"era": TEAM_BULLPEN_2025.get(game['home_team_id'], {}).get('era', 4.2), "team": game['home_team']},
            "away_sp_ip": round(away_sp_ip, 1),
            "home_sp_ip": round(home_sp_ip, 1),
        },
        "data_blend": "Steamer + 2025 + Spring 2026 + 2026 Regular",
        "umpire": ump,
        "lineup_source": "actual" if real_lineups.get("lineup_available") else "projected",
        "bullpen_availability": bp_avail,
        "away_pitcher_workload": away_workload,
        "home_pitcher_workload": home_workload,
        "away_pitcher_hand": away_hand.get("pitch_hand", "R") if isinstance(away_hand, dict) else "R",
        "home_pitcher_hand": home_hand.get("pitch_hand", "R") if isinstance(home_hand, dict) else "R",
        "away_batters": away_adjustments,
        "home_batters": home_adjustments,
    }
    results["market_odds"] = matched_odds
    results["edges"] = edges
    
    results["away_lineup"] = [
        {"name": b.get("name", "?"), "position": b.get("position", "?"),
         "avg": b.get("avg", ".000"), "ops": b.get("ops", ".000"),
         "hr": b.get("hr", 0), "player_id": b.get("player_id", 0)}
        for b in away_lineup
    ]
    results["home_lineup"] = [
        {"name": b.get("name", "?"), "position": b.get("position", "?"),
         "avg": b.get("avg", ".000"), "ops": b.get("ops", ".000"),
         "hr": b.get("hr", 0), "player_id": b.get("player_id", 0)}
        for b in home_lineup
    ]
    results["away_pitcher"] = {
        "name": away_pitcher.get("name", "TBD"),
        "era": away_pitcher.get("era", "0.00"),
        "whip": away_pitcher.get("whip", "0.00"),
        "k_per_9": away_pitcher.get("k_per_9", "0.00"),
        "hand": away_hand.get("pitch_hand", "R") if isinstance(away_hand, dict) else "R",
        "workload": away_workload,
    }
    results["home_pitcher"] = {
        "name": home_pitcher.get("name", "TBD"),
        "era": home_pitcher.get("era", "0.00"),
        "whip": home_pitcher.get("whip", "0.00"),
        "k_per_9": home_pitcher.get("k_per_9", "0.00"),
        "hand": home_hand.get("pitch_hand", "R") if isinstance(home_hand, dict) else "R",
        "workload": home_workload,
    }
    
    return JSONResponse(content=sanitize(results))
# ─── DFS Projections ──────────────────────────────────────────
@app.get("/api/projections")
async def api_projections(game_date: Optional[str] = None):
    """DFS projections with all adjustments for all games."""
    games = await get_schedule(game_date)
    all_projections = []
    league_avg = get_league_averages()
    
    for game in games[:8]:
        try:
            home_team_id = game["home_team_id"]
            park = get_park_factors(home_team_id)
            
            away_lineup = await build_lineup_enhanced(game["away_team_id"], season=2025)
            home_lineup = await build_lineup_enhanced(game["home_team_id"], season=2025)
            
            away_pitcher = None
            home_pitcher = None
            away_hand_code = "R"
            home_hand_code = "R"
            
            if game.get("away_pitcher_id"):
                away_pitcher = await get_pitcher_stats(game["away_pitcher_id"], 2025)
                ah = await get_player_handedness(game["away_pitcher_id"])
                away_hand_code = ah.get("pitch_hand", "R")
            if game.get("home_pitcher_id"):
                home_pitcher = await get_pitcher_stats(game["home_pitcher_id"], 2025)
                hh = await get_player_handedness(game["home_pitcher_id"])
                home_hand_code = hh.get("pitch_hand", "R")
            
            if not away_pitcher:
                away_pitcher = {"name": game.get("away_pitcher_name", "TBD"), **league_avg}
            if not home_pitcher:
                home_pitcher = {"name": game.get("home_pitcher_name", "TBD"), **league_avg}
            
            # Apply adjustments to matchups
            away_matchups = []
            for b in away_lineup:
                m = matchup_probability_vector(b, home_pitcher, league_avg)
                m = apply_platoon_adjustment(m, b.get("_platoon_splits", {}), home_hand_code)
                m = apply_form_adjustment(m, b.get("_form", {"form_multiplier": 1.0}))
                m = apply_park_factors(m, park)
                away_matchups.append(m)
            
            home_matchups = []
            for b in home_lineup:
                m = matchup_probability_vector(b, away_pitcher, league_avg)
                m = apply_platoon_adjustment(m, b.get("_platoon_splits", {}), away_hand_code)
                m = apply_form_adjustment(m, b.get("_form", {"form_multiplier": 1.0}))
                m = apply_park_factors(m, park)
                home_matchups.append(m)
            
            results = run_monte_carlo(
                away_lineup, home_lineup,
                away_matchups, home_matchups,
                away_pitcher, home_pitcher,
                n_sims=2000
            )
            
            for proj in results["away_projections"]:
                proj["team"] = game["away_team"]
                proj["opp"] = game["home_team"]
                proj["opp_pitcher"] = home_pitcher.get("name", "TBD")
                proj["opp_pitcher_hand"] = home_hand_code
                proj["game_id"] = game["game_id"]
                proj["park_hr_factor"] = park.get("hr", 1.0)
                all_projections.append(proj)
            
            for proj in results["home_projections"]:
                proj["team"] = game["home_team"]
                proj["opp"] = game["away_team"]
                proj["opp_pitcher"] = away_pitcher.get("name", "TBD")
                proj["opp_pitcher_hand"] = away_hand_code
                proj["game_id"] = game["game_id"]
                proj["park_hr_factor"] = park.get("hr", 1.0)
                all_projections.append(proj)
                
        except Exception as e:
            print(f"Error processing game {game['game_id']}: {e}")
            continue
    
    all_projections.sort(key=lambda x: x.get("dk_median", 0), reverse=True)
    
    return JSONResponse(content=sanitize({
        "projections": all_projections,
        "date": game_date or date.today().isoformat(),
        "games_processed": min(len(games), 8)
    }))
# ─── Edges ─────────────────────────────────────────────────────
@app.get("/api/edges")
async def api_edges(game_date: Optional[str] = None):
    """Betting edges: model vs all market sources."""
    games = await get_schedule(game_date)
    sportsbook_odds = await get_odds_api_mlb()
    kalshi_games = await get_kalshi_mlb_games()
    
    edges = []
    for game in games:
        away_prob, home_prob = team_win_probability_with_home(
            game["away_wpct"], game["home_wpct"]
        )
        
        # Match sportsbook odds
        matched = match_odds_to_game(game, sportsbook_odds)
        consensus = matched.get("consensus", {}) if matched else {}
        
        edge_data = {
            "game_id": game["game_id"],
            "matchup": f"{game['away_team']} @ {game['home_team']}",
            "away_team": game["away_team"],
            "home_team": game["home_team"],
            "away_pitcher": game["away_pitcher_name"],
            "home_pitcher": game["home_pitcher_name"],
            "model_away_prob": round(away_prob, 3),
            "model_home_prob": round(home_prob, 3),
            "away_fair_odds": prob_to_american(away_prob),
            "home_fair_odds": prob_to_american(home_prob),
            "park": get_park_factors(game["home_team_id"]).get("name", ""),
            # Sportsbook consensus
            "consensus_away_ml": consensus.get("away_ml"),
            "consensus_home_ml": consensus.get("home_ml"),
            "consensus_total": consensus.get("total"),
            "num_books": consensus.get("num_books", 0),
            # Best available lines
            "best_away_ml": matched.get("best_away_ml") if matched else None,
            "best_home_ml": matched.get("best_home_ml") if matched else None,
            "best_away_book": matched.get("best_away_book", "") if matched else "",
            "best_home_book": matched.get("best_home_book", "") if matched else "",
            # Edge calculations
            "away_edge": None,
            "home_edge": None,
        }
        
        if consensus.get("away_ml"):
            edge_data["away_edge"] = compute_edge(away_prob, consensus["away_ml"])
            edge_data["home_edge"] = compute_edge(home_prob, consensus["home_ml"])
        
        edges.append(edge_data)
    
    return JSONResponse(content=sanitize({
        "edges": edges,
        "date": game_date or date.today().isoformat(),
        "sportsbook_count": len(sportsbook_odds),
        "kalshi_count": len(kalshi_games),
    }))
# ─── Matchup Detail ───────────────────────────────────────────
@app.get("/api/matchup/{batter_id}/{pitcher_id}")
async def api_matchup(batter_id: int, pitcher_id: int, 
                       home_team_id: int = 0, season: int = 2025):
    """Detailed matchup with all adjustments visible."""
    league_avg = get_league_averages()
    
    batter = await get_batter_stats(batter_id, season)
    pitcher = await get_pitcher_stats(pitcher_id, season)
    
    if not batter:
        raise HTTPException(status_code=404, detail="Batter stats not found")
    if not pitcher:
        raise HTTPException(status_code=404, detail="Pitcher stats not found")
    
    adjustments = await get_all_adjustments(batter_id, pitcher_id, home_team_id, season)
    
    # Base matchup
    base_matchup = matchup_probability_vector(batter, pitcher, league_avg)
    
    # Apply adjustments step by step
    after_platoon = apply_platoon_adjustment(
        base_matchup, adjustments["platoon_splits"], adjustments["pitcher_hand"]
    )
    after_form = apply_form_adjustment(after_platoon, adjustments["batter_form"])
    
    park = adjustments["park_factors"]
    after_park = apply_park_factors(after_form, park)
    
    return JSONResponse(content=sanitize({
        "batter": {"name": batter.get("name", ""), "avg": batter.get("avg", ""), "ops": batter.get("ops", "")},
        "pitcher": {"name": pitcher.get("name", ""), "era": pitcher.get("era", "")},
        "base_matchup": base_matchup,
        "after_platoon": after_platoon,
        "after_form": after_form,
        "after_park": after_park,
        "adjustments": adjustments,
    }))
# ─── Cache Management ─────────────────────────────────────────
@app.post("/api/cache/clear")
async def api_clear_cache():
    clear_cache()
    clear_market_cache()
    clear_adj_cache()
    clear_proj_cache()
    clear_ctx_cache()
    return {"status": "all caches cleared"}
# ─── Helpers ───────────────────────────────────────────────────
async def build_lineup_with_order(team_id: int, real_lineup: list, season: int = 2025) -> list:
    """Build lineup using ACTUAL batting order from MLB live feed when available."""
    if real_lineup and len(real_lineup) >= 7:
        # We have the real lineup — use it in order
        batters = []
        for slot in real_lineup[:9]:
            pid = slot["id"]
            stats = await get_batter_stats(pid, season)
            if not stats:
                # Try 2026 spring or use league average
                stats = {
                    "player_id": pid, "name": slot["name"],
                    "position": slot["position"],
                    **get_league_averages(),
                    "avg": ".243", "ops": ".700", "hr": 10, "sb": 5,
                    "p_sb_attempt": 0.05,
                }
            else:
                stats["position"] = slot["position"]
            
            # Blend with Steamer projections
            try:
                blended = await get_blended_batter_probs(pid, slot["name"])
                for k in ["p_1b", "p_2b", "p_3b", "p_hr", "p_bb", "p_hbp", "p_out"]:
                    if k in blended:
                        stats[k] = blended[k]
                stats["_blend_info"] = blended.get("_blend_info", {})
            except Exception:
                pass
            
            # Platoon splits and form
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
        
        # Pad to 9 if needed
        league_avg = get_league_averages()
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
    
    # Fallback: no real lineup available, use enhanced builder
    return await build_lineup_enhanced(team_id, season)
async def build_lineup_enhanced(team_id: int, season: int = 2025) -> list:
    """Build lineup with BLENDED projections, platoon splits, and recent form."""
    roster = await get_team_roster(team_id, 2026)
    position_players = [p for p in roster if p["position_type"] != "Pitcher"]
    
    batters = []
    for p in position_players[:15]:
        # Get base stats from 2025 (for metadata)
        stats = await get_batter_stats(p["id"], season)
        if stats:
            stats["position"] = p["position"]
            
            # Get BLENDED probability vector (Steamer + 2025 + Spring + 2026)
            try:
                blended = await get_blended_batter_probs(p["id"], p["name"])
                # Override probability fields with blended values
                for k in ["p_1b", "p_2b", "p_3b", "p_hr", "p_bb", "p_hbp", "p_out"]:
                    if k in blended:
                        stats[k] = blended[k]
                stats["_blend_info"] = blended.get("_blend_info", {})
            except Exception as e:
                stats["_blend_info"] = {"error": str(e)}
            
            # Fetch platoon splits
            try:
                splits = await get_platoon_splits(p["id"], season)
                stats["_platoon_splits"] = splits
            except Exception:
                stats["_platoon_splits"] = {}
            
            # Fetch recent form
            try:
                log = await get_batter_game_log(p["id"], season)
                stats["_form"] = compute_recent_form(log)
            except Exception:
                stats["_form"] = {"form_multiplier": 1.0, "trend": "neutral"}
            
            batters.append(stats)
    
    def safe_ops(x):
        try:
            v = x.get("ops", ".000")
            return float(v) if isinstance(v, str) else float(v or 0)
        except (ValueError, TypeError):
            return 0
    
    batters.sort(key=safe_ops, reverse=True)
    
    league_avg = get_league_averages()
    while len(batters) < 9:
        batters.append({
            "player_id": 99900 + len(batters),
            "name": f"Replacement #{len(batters)+1}",
            "position": "UT", **league_avg,
            "avg": ".243", "ops": ".700", "hr": 10, "sb": 5,
            "p_sb_attempt": 0.05,
            "_platoon_splits": {}, "_form": {"form_multiplier": 1.0, "trend": "neutral"},
        })
    
    return batters[:9]
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
