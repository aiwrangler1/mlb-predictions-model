"""
Play-by-Play Engine — Mines actual MLB PBP data for precise league averages
and per-player 9-event probability vectors.

9-event model: 1B, 2B, 3B, HR, BB, HBP, K, BIP_OUT, PRODUCTIVE_OUT
This is more accurate than the 7-event model because:
  - Strikeouts can't advance runners or produce sac flies
  - Ball-in-play outs CAN advance runners (sac fly, groundout advance)
  - GIDPs clear the bases (negative run value vs a K)
  - High-K pitchers suppress runs MORE than equivalent out rates from contact

Approach inspired by: mining actual PBP + pandas normalization for Log5 partitions.
"""
import httpx
import asyncio
import numpy as np
from typing import Optional
from datetime import date, timedelta

MLB_API = "https://statsapi.mlb.com/api/v1"

_pbp_cache = {}

# Event type mapping from MLB API to our 9-event model
EVENT_MAP = {
    "single": "1B",
    "double": "2B",
    "triple": "3B",
    "home_run": "HR",
    "walk": "BB",
    "intent_walk": "BB",
    "hit_by_pitch": "HBP",
    "strikeout": "K",
    "strikeout_double_play": "K",
    "sac_fly": "PROD_OUT",
    "sac_fly_double_play": "PROD_OUT",
    "sac_bunt": "PROD_OUT",
    "sac_bunt_double_play": "PROD_OUT",
    "field_out": "BIP_OUT",
    "force_out": "BIP_OUT",
    "grounded_into_double_play": "GIDP",
    "double_play": "GIDP",
    "triple_play": "GIDP",
    "fielders_choice": "BIP_OUT",
    "fielders_choice_out": "BIP_OUT",
    "field_error": "BIP_OUT",  # Reached on error — counted as BIP for the pitcher
    "catcher_interf": "BIP_OUT",
    "caught_stealing_2b": None,  # Not a PA
    "caught_stealing_3b": None,
    "caught_stealing_home": None,
    "stolen_base_2b": None,
    "stolen_base_3b": None,
    "stolen_base_home": None,
    "pickoff_1b": None,
    "pickoff_2b": None,
    "pickoff_3b": None,
    "wild_pitch": None,
    "passed_ball": None,
    "balk": None,
    "other_advance": None,
    "runner_double_play": None,
}

# The 9 events in our model
EVENTS_9 = ["1B", "2B", "3B", "HR", "BB", "HBP", "K", "BIP_OUT", "PROD_OUT"]
# GIDP is rolled into BIP_OUT for probability but tracked separately for runner logic

# 2025 league averages mined from Retrosheet PBP (186,640 plate appearances)
# Source: retrosheet.org/downloads/plays/2025plays.zip
# The information used here was obtained free of charge from and is
# copyrighted by Retrosheet.
LEAGUE_AVG_9_EVENT = {
    "p_1b":       0.142563,   # 14.26% singles (26,608 PA)
    "p_2b":       0.042258,   # 4.23% doubles (7,887 PA)
    "p_3b":       0.003408,   # 0.34% triples (636 PA)
    "p_hr":       0.030894,   # 3.09% home runs (5,766 PA)
    "p_bb":       0.084178,   # 8.42% walks incl IBB (15,711 PA)
    "p_hbp":      0.010577,   # 1.06% HBP (1,974 PA)
    "p_k":        0.222525,   # 22.25% strikeouts (41,532 PA)
    "p_bip_out":  0.453354,   # 45.33% ball-in-play outs (84,614 PA)
    "p_prod_out": 0.010244,   # 1.02% productive outs — sac fly + bunt (1,912 PA)
    "p_gidp_given_bip": 0.037559,  # 3.76% of BIP outs are GIDPs (3,178)
}


async def _fetch(url: str, timeout: float = 15.0):
    if url in _pbp_cache:
        return _pbp_cache[url]
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            _pbp_cache[url] = data
            return data
    except Exception:
        return None


async def mine_league_averages(season: int = 2025, sample_games: int = 50) -> dict:
    """
    Mine actual PBP data from a sample of games to compute league averages.
    Returns 9-event probability vector.
    
    For production, this should be pre-computed from ALL games. 
    For now, we sample enough games to get stable estimates.
    """
    # Get game IDs spread across the season
    dates = []
    for month in range(4, 10):  # April through September
        for day in [1, 15]:
            dates.append(f"{season}-{month:02d}-{day:02d}")
    
    game_ids = []
    for d in dates:
        try:
            sched = await _fetch(f"{MLB_API}/schedule?sportId=1&date={d}&gameType=R")
            if sched:
                for dt in sched.get("dates", []):
                    for g in dt.get("games", [])[:3]:
                        if g.get("status", {}).get("detailedState") == "Final":
                            game_ids.append(g["gamePk"])
        except Exception:
            continue
        if len(game_ids) >= sample_games:
            break
    
    # Mine PBP from these games
    counts = {e: 0 for e in EVENTS_9}
    counts["GIDP"] = 0
    total_pa = 0
    
    for gid in game_ids[:sample_games]:
        try:
            data = await _fetch(f"{MLB_API}/game/{gid}/playByPlay")
            if not data:
                continue
            for play in data.get("allPlays", []):
                if play.get("result", {}).get("type") != "atBat":
                    continue
                evt = play["result"].get("eventType", "")
                mapped = EVENT_MAP.get(evt)
                if mapped is None:
                    continue  # Not a PA (stolen base, etc.)
                if mapped == "GIDP":
                    counts["BIP_OUT"] += 1
                    counts["GIDP"] += 1
                else:
                    counts[mapped] += 1
                total_pa += 1
        except Exception:
            continue
    
    if total_pa < 100:
        return LEAGUE_AVG_9_EVENT  # Fallback
    
    result = {}
    for event in EVENTS_9:
        key = f"p_{event.lower()}"
        result[key] = counts[event] / total_pa
    
    result["p_gidp_given_bip"] = counts["GIDP"] / max(counts["BIP_OUT"], 1)
    result["_total_pa"] = total_pa
    result["_games_mined"] = len(game_ids[:sample_games])
    
    return result


def convert_7_to_9_event(probs_7: dict, k_rate: float = None) -> dict:
    """
    Convert a 7-event probability vector to 9-event by splitting OUT into K + BIP_OUT + PROD_OUT.
    
    If we know the player's K rate, use it. Otherwise estimate from the OUT rate.
    League average: ~34% of outs are Ks, ~64% are BIP outs, ~2% productive outs.
    """
    p_out = probs_7.get("p_out", 0.675)
    
    if k_rate is not None and k_rate > 0:
        p_k = k_rate
    else:
        # Estimate: 34% of outs are strikeouts in modern MLB
        p_k = p_out * 0.338
    
    p_bip_out = p_out - p_k - (p_out * 0.018)  # Remainder after K and productive outs
    p_prod_out = p_out * 0.018  # ~1.8% of outs are productive
    
    return {
        "p_1b": probs_7.get("p_1b", 0),
        "p_2b": probs_7.get("p_2b", 0),
        "p_3b": probs_7.get("p_3b", 0),
        "p_hr": probs_7.get("p_hr", 0),
        "p_bb": probs_7.get("p_bb", 0),
        "p_hbp": probs_7.get("p_hbp", 0),
        "p_k": max(0.01, p_k),
        "p_bip_out": max(0.01, p_bip_out),
        "p_prod_out": max(0.001, p_prod_out),
    }


def get_player_k_rate(player_stats: dict) -> float:
    """Extract K rate from player stats if available."""
    pa = player_stats.get("pa", 0)
    so = player_stats.get("so", 0)
    if pa > 0 and so > 0:
        return so / pa
    return None


def matchup_probability_9(batter_9: dict, pitcher_9: dict, league_9: dict) -> dict:
    """
    Log5 multi-event extension for 9-event model.
    
    P(Ei) = (xi * yi / zi) / SUM_j(xj * yj / zj)
    
    Same math as the 7-event version, just with 9 outcomes.
    """
    outcomes = ["p_1b", "p_2b", "p_3b", "p_hr", "p_bb", "p_hbp", "p_k", "p_bip_out", "p_prod_out"]
    
    raw = {}
    for o in outcomes:
        x = batter_9.get(o, 0)
        y = pitcher_9.get(o, 0)
        z = league_9.get(o, 0.001)
        raw[o] = (x * y) / z if z > 0 else 0
    
    total = sum(raw.values())
    if total == 0:
        return league_9
    
    return {o: raw[o] / total for o in outcomes}


def simulate_pa_9_event(matchup: dict, rng) -> str:
    """Simulate a plate appearance with 9 possible outcomes."""
    outcomes = ["1b", "2b", "3b", "hr", "bb", "hbp", "k", "bip_out", "prod_out"]
    probs = [
        matchup.get("p_1b", 0), matchup.get("p_2b", 0), matchup.get("p_3b", 0),
        matchup.get("p_hr", 0), matchup.get("p_bb", 0), matchup.get("p_hbp", 0),
        matchup.get("p_k", 0), matchup.get("p_bip_out", 0), matchup.get("p_prod_out", 0),
    ]
    total = sum(probs)
    if total <= 0:
        probs = [0.148, 0.043, 0.005, 0.030, 0.081, 0.012, 0.228, 0.441, 0.012]
        total = sum(probs)
    probs = [p / total for p in probs]
    
    r = rng.random()
    cumulative = 0
    for outcome, p in zip(outcomes, probs):
        cumulative += p
        if r < cumulative:
            return outcome
    return "bip_out"


def advance_runners_9_event(bases: list, outcome: str, rng, gidp_rate: float = 0.035) -> tuple:
    """
    Advance baserunners with 9-event logic.
    
    Key differences from 7-event:
    - K: NO runner advancement (can't sac fly on a K)
    - BIP_OUT: runners CAN advance (fly out = runner tags, ground out = runner advances)
      - Also chance of GIDP which erases a runner
    - PROD_OUT: runner on 3rd scores (sac fly), runner advances (sac bunt)
    """
    runs = 0
    new_bases = [0, 0, 0]
    
    if outcome == "hr":
        runs = 1 + sum(bases)
        new_bases = [0, 0, 0]
    
    elif outcome == "3b":
        runs = sum(bases)
        new_bases = [0, 0, 1]
    
    elif outcome == "2b":
        runs += bases[2]
        runs += bases[1]
        if bases[0]:
            if rng.random() < 0.40:
                runs += 1
            else:
                new_bases[2] = 1
        new_bases[1] = 1
    
    elif outcome == "1b":
        runs += bases[2]
        if bases[1]:
            if rng.random() < 0.60:
                runs += 1
            else:
                new_bases[2] = 1
        if bases[0]:
            if new_bases[2] == 0 and rng.random() < 0.25:
                new_bases[2] = 1
            else:
                new_bases[1] = 1 if new_bases[1] == 0 else 1
        new_bases[0] = 1
    
    elif outcome in ("bb", "hbp"):
        if bases[0] and bases[1] and bases[2]:
            runs += 1
            new_bases = [1, 1, 1]
        elif bases[0] and bases[1]:
            new_bases = [1, 1, 1]
        elif bases[0]:
            new_bases = [1, 1, bases[2]]
        else:
            new_bases = [1, bases[1], bases[2]]
    
    elif outcome == "k":
        # Strikeout: NO runner advancement at all
        # This is the key difference from the 7-event model
        new_bases = bases.copy()
    
    elif outcome == "bip_out":
        # Ball in play out — runners CAN advance
        new_bases = bases.copy()
        
        # GIDP check: if runner on 1st and less than 2 outs
        if bases[0] and rng.random() < gidp_rate * 3:  # Higher GIDP rate with runner on 1st
            # Double play: erase lead runner
            new_bases[0] = 0
            if bases[1]:
                new_bases[1] = 0
            # Extra out is handled by the inning simulator
        else:
            # Regular BIP out — runner advancement possible
            # Runner on 3rd scores on fly out ~50% of the time (sac fly equivalent)
            if bases[2] and rng.random() < 0.30:
                runs += 1
                new_bases[2] = 0
            # Runner on 2nd advances to 3rd ~25% of the time on ground out
            if bases[1] and rng.random() < 0.25:
                new_bases[2] = 1
                new_bases[1] = 0
    
    elif outcome == "prod_out":
        # Productive out (sac fly / sac bunt)
        new_bases = bases.copy()
        if bases[2]:
            runs += 1
            new_bases[2] = 0
        # Sac bunt: advance runner
        if bases[1] and new_bases[2] == 0:
            new_bases[2] = 1
            new_bases[1] = 0
        if bases[0] and new_bases[1] == 0:
            new_bases[1] = 1
            new_bases[0] = 0
    
    return new_bases, runs


def get_league_avg_9() -> dict:
    """Return the 9-event league average vector."""
    return LEAGUE_AVG_9_EVENT.copy()


def clear_pbp_cache():
    global _pbp_cache
    _pbp_cache = {}
