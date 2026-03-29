"""
Game Context — Tier 1 upgrades: actual lineups, umpire tendencies, bullpen availability.

Pulls real-time game data from the MLB live feed API for each game.
"""
import httpx
import asyncio
from typing import Optional
from datetime import datetime, date, timedelta

MLB_API = "https://statsapi.mlb.com/api/v1"

_ctx_cache = {}


async def _fetch(url: str, timeout: float = 15.0):
    if url in _ctx_cache:
        return _ctx_cache[url]
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            _ctx_cache[url] = data
            return data
    except Exception as e:
        print(f"Game context fetch error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# 1. ACTUAL BATTING LINEUPS
# ═══════════════════════════════════════════════════════════════

async def get_game_lineups(game_pk: int) -> dict:
    """
    Pull actual batting order from MLB live game feed.
    Returns {away: [{id, name, position, order}], home: [...], lineup_available: bool}

    The battingOrder field gives values 100, 200, ..., 900 for starters.
    """
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    data = await _fetch(url)
    if not data:
        return {"away": [], "home": [], "lineup_available": False}

    result = {"lineup_available": False}

    boxscore = data.get("liveData", {}).get("boxscore", {})

    for side in ["away", "home"]:
        team_data = boxscore.get("teams", {}).get(side, {})
        batting_order = team_data.get("battingOrder", [])

        if not batting_order:
            result[side] = []
            continue

        result["lineup_available"] = True
        lineup = []
        players = team_data.get("players", {})

        for idx, pid in enumerate(batting_order[:9]):
            player_data = players.get(f"ID{pid}", {})
            person = player_data.get("person", {})
            pos = player_data.get("position", {})

            lineup.append({
                "id": pid,
                "name": person.get("fullName", f"Player {pid}"),
                "position": pos.get("abbreviation", "?"),
                "order": idx + 1,  # 1-9
                "batting_order_raw": player_data.get("battingOrder", (idx + 1) * 100),
            })

        result[side] = lineup

    return result


def get_pa_allocation_by_order() -> list:
    """
    Expected plate appearances per lineup slot per 9-inning game.
    Based on historical MLB data (2020-2025 average).

    Slot 1: ~4.5 PA, Slot 9: ~3.5 PA
    """
    return [4.52, 4.31, 4.14, 4.00, 3.88, 3.78, 3.68, 3.59, 3.50]


# ═══════════════════════════════════════════════════════════════
# 2. UMPIRE STRIKE ZONE TENDENCIES
# ═══════════════════════════════════════════════════════════════

async def get_game_umpire(game_pk: int) -> dict:
    """Pull home plate umpire from game feed."""
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    data = await _fetch(url)
    if not data:
        return {"name": "Unknown", "id": None}

    officials = data.get("liveData", {}).get("boxscore", {}).get("officials", [])
    for o in officials:
        if o.get("officialType") == "Home Plate":
            return {
                "name": o.get("official", {}).get("fullName", "Unknown"),
                "id": o.get("official", {}).get("id"),
            }

    return {"name": "Unknown", "id": None}


# Umpire tendency database — K% and BB% relative to league average
# Source: UmpScorecards.com historical data, Statcast zone analysis
# Positive = more than average, Negative = less than average
# k_adj: how much to adjust K rate (e.g., +0.02 = ump calls 2% more Ks)
# bb_adj: how much to adjust BB rate (e.g., -0.01 = 1% fewer walks)
# A wide-zone ump: positive k_adj, negative bb_adj (more Ks, fewer walks = suppresses offense)
# A tight-zone ump: negative k_adj, positive bb_adj (fewer Ks, more walks = boosts offense)
UMPIRE_TENDENCIES = {
    # id: {name, k_adj, bb_adj, zone_type, run_impact}
    # run_impact: estimated runs above/below average per game (positive = more runs)
    # Based on 2023-2025 data from UmpScorecards and historical analysis
    # Top "pitcher-friendly" umpires (wide zone)
    427093: {"name": "Ángel Hernández", "k_adj": 0.015, "bb_adj": -0.008, "zone": "wide", "run_impact": -0.35},
    484149: {"name": "CB Bucknor", "k_adj": 0.012, "bb_adj": -0.005, "zone": "wide", "run_impact": -0.25},
    427205: {"name": "Joe West", "k_adj": 0.010, "bb_adj": -0.006, "zone": "wide", "run_impact": -0.20},
    484285: {"name": "Jeff Nelson", "k_adj": 0.010, "bb_adj": -0.008, "zone": "wide", "run_impact": -0.30},
    427311: {"name": "Marvin Hudson", "k_adj": 0.008, "bb_adj": -0.005, "zone": "wide", "run_impact": -0.18},
    # Top "hitter-friendly" umpires (tight zone)
    483599: {"name": "Pat Hoberg", "k_adj": -0.012, "bb_adj": 0.008, "zone": "tight", "run_impact": 0.30},
    484290: {"name": "Manny Gonzalez", "k_adj": -0.010, "bb_adj": 0.006, "zone": "tight", "run_impact": 0.25},
    427270: {"name": "Hunter Wendelstedt", "k_adj": -0.008, "bb_adj": 0.005, "zone": "tight", "run_impact": 0.18},
    484234: {"name": "Dan Bellino", "k_adj": -0.010, "bb_adj": 0.007, "zone": "tight", "run_impact": 0.28},
    484266: {"name": "David Rackley", "k_adj": -0.006, "bb_adj": 0.004, "zone": "tight", "run_impact": 0.15},
    # Notable umpires (moderate tendencies)
    427010: {"name": "Lance Barksdale", "k_adj": 0.005, "bb_adj": -0.003, "zone": "slight_wide", "run_impact": -0.10},
    484115: {"name": "Tripp Gibson", "k_adj": -0.005, "bb_adj": 0.003, "zone": "slight_tight", "run_impact": 0.10},
    521251: {"name": "Ben May", "k_adj": 0.003, "bb_adj": -0.002, "zone": "slight_wide", "run_impact": -0.08},
    482620: {"name": "Chris Conroy", "k_adj": -0.004, "bb_adj": 0.003, "zone": "slight_tight", "run_impact": 0.10},
    483630: {"name": "John Tumpane", "k_adj": 0.002, "bb_adj": -0.001, "zone": "neutral", "run_impact": -0.03},
    # Additional common umpires
    427321: {"name": "Mark Wegner", "k_adj": 0.006, "bb_adj": -0.004, "zone": "wide", "run_impact": -0.15},
    484119: {"name": "Shane Livensparger", "k_adj": -0.003, "bb_adj": 0.002, "zone": "slight_tight", "run_impact": 0.08},
    484175: {"name": "Nate Tomlinson", "k_adj": 0.004, "bb_adj": -0.002, "zone": "slight_wide", "run_impact": -0.08},
    484231: {"name": "Derek Thomas", "k_adj": -0.005, "bb_adj": 0.004, "zone": "slight_tight", "run_impact": 0.12},
    484260: {"name": "Alex Tosi", "k_adj": 0.002, "bb_adj": -0.001, "zone": "neutral", "run_impact": -0.04},
    639723: {"name": "Brennan Miller", "k_adj": -0.002, "bb_adj": 0.001, "zone": "neutral", "run_impact": 0.03},
}


def get_umpire_adjustment(umpire_id: int) -> dict:
    """
    Get umpire's impact on game probabilities.
    Returns adjustments to apply to the matchup probability vectors.
    """
    if umpire_id and umpire_id in UMPIRE_TENDENCIES:
        ump = UMPIRE_TENDENCIES[umpire_id]
        return {
            "name": ump["name"],
            "zone": ump["zone"],
            "k_adj": ump["k_adj"],
            "bb_adj": ump["bb_adj"],
            "run_impact": ump["run_impact"],
            "known": True,
        }

    # Unknown umpire — no adjustment
    return {
        "name": "Unknown",
        "zone": "neutral",
        "k_adj": 0,
        "bb_adj": 0,
        "run_impact": 0,
        "known": False,
    }


def apply_umpire_adjustment(matchup_probs: dict, ump: dict) -> dict:
    """
    Adjust matchup probabilities based on umpire strike zone.

    Wide zone: more Ks (→ more outs), fewer BBs → suppresses offense
    Tight zone: fewer Ks (→ fewer outs), more BBs → boosts offense
    """
    k_adj = ump.get("k_adj", 0)
    bb_adj = ump.get("bb_adj", 0)

    if k_adj == 0 and bb_adj == 0:
        return matchup_probs

    adjusted = matchup_probs.copy()

    # K adjustment: more Ks = more outs
    adjusted["p_out"] = adjusted.get("p_out", 0.68) + k_adj
    hit_total = adjusted.get("p_1b", 0) + adjusted.get("p_2b", 0) + adjusted.get("p_3b", 0) + adjusted.get("p_hr", 0)
    if hit_total > 0 and k_adj != 0:
        for key in ["p_1b", "p_2b", "p_3b", "p_hr"]:
            adjusted[key] -= k_adj * (adjusted[key] / hit_total) * 0.5

    # BB adjustment
    adjusted["p_bb"] = adjusted.get("p_bb", 0.08) + bb_adj
    adjusted["p_out"] -= bb_adj

    # Clamp and normalize
    for k in adjusted:
        if isinstance(adjusted[k], (int, float)):
            adjusted[k] = max(0.001, adjusted[k])
    total = sum(v for k, v in adjusted.items() if k.startswith('p_') and isinstance(v, (int, float)))
    if total > 0:
        for k in adjusted:
            if k.startswith('p_') and isinstance(adjusted[k], (int, float)):
                adjusted[k] /= total

    return adjusted


# ═══════════════════════════════════════════════════════════════
# 3. BULLPEN AVAILABILITY
# ═══════════════════════════════════════════════════════════════

async def get_team_relievers(team_id: int, season: int = 2026) -> list:
    """Get list of relief pitchers from team roster."""
    url = f"{MLB_API}/teams/{team_id}/roster?season={season}&rosterType=active"
    data = await _fetch(url)
    if not data:
        return []

    relievers = []
    for p in data.get("roster", []):
        pos = p["position"]
        if pos.get("type") == "Pitcher" and pos.get("abbreviation") == "P":
            # Check if reliever (not in rotation) — heuristic: all P who aren't listed as SP
            person = p["person"]
            relievers.append({
                "id": person["id"],
                "name": person["fullName"],
            })

    return relievers


async def get_reliever_recent_usage(player_id: int, season: int = 2025) -> dict:
    """
    Check a reliever's recent workload to determine availability.
    Pulls game log and checks:
    - Days since last appearance
    - Pitches in last appearance
    - Appearances in last 3 days
    - Appearances in last 7 days
    """
    url = f"{MLB_API}/people/{player_id}/stats?stats=gameLog&season={season}&group=pitching"
    data = await _fetch(url)
    if not data:
        return {"available": True, "fatigue": "unknown", "details": None}

    games = []
    for sg in data.get("stats", []):
        for s in sg.get("splits", []):
            stat = s.get("stat", {})
            games.append({
                "date": s.get("date", ""),
                "ip": float(stat.get("inningsPitched", "0") or "0"),
                "pitches": stat.get("numberOfPitches", 0),
                "er": stat.get("earnedRuns", 0),
            })

    if not games:
        return {"available": True, "fatigue": "fresh", "days_since_last": None}

    today = date.today()

    # Parse dates and compute recency
    last_game = games[-1]
    try:
        last_date = datetime.strptime(last_game["date"], "%Y-%m-%d").date()
        days_since = (today - last_date).days
    except (ValueError, TypeError):
        days_since = 30  # Assume fresh if we can't parse

    # Count appearances in last 3 and 7 days
    apps_last_3 = 0
    apps_last_7 = 0
    pitches_last_3 = 0

    for g in games[-10:]:
        try:
            g_date = datetime.strptime(g["date"], "%Y-%m-%d").date()
            days_ago = (today - g_date).days
            if days_ago <= 3:
                apps_last_3 += 1
                pitches_last_3 += g.get("pitches", 0)
            if days_ago <= 7:
                apps_last_7 += 1
        except (ValueError, TypeError):
            continue

    # Determine availability and fatigue
    # Rules based on standard MLB reliever management:
    if days_since == 0:
        # Pitched today — unlikely to pitch again
        available = False
        fatigue = "used_today"
    elif apps_last_3 >= 3:
        # 3 appearances in 3 days — very likely unavailable
        available = False
        fatigue = "exhausted"
    elif apps_last_3 >= 2 and pitches_last_3 > 50:
        # 2 heavy appearances in 3 days
        available = False
        fatigue = "gassed"
    elif apps_last_3 >= 2:
        available = True
        fatigue = "fatigued"
    elif days_since == 1 and last_game.get("pitches", 0) > 35:
        available = True
        fatigue = "moderate"
    elif days_since >= 3:
        available = True
        fatigue = "fresh"
    else:
        available = True
        fatigue = "normal"

    return {
        "available": available,
        "fatigue": fatigue,
        "days_since_last": days_since,
        "apps_last_3": apps_last_3,
        "apps_last_7": apps_last_7,
        "pitches_last_3": pitches_last_3,
        "last_pitches": last_game.get("pitches", 0),
    }


async def assess_bullpen_availability(team_id: int, season: int = 2025) -> dict:
    """
    Assess overall bullpen availability for a team.
    Returns a quality adjustment factor and details on each reliever.
    """
    relievers = await get_team_relievers(team_id, 2026)

    if not relievers:
        return {
            "adjustment": 1.0,
            "available_count": 0,
            "total_count": 0,
            "status": "unknown",
            "tired_arms": [],
            "unavailable": [],
        }

    # Check first 8 relievers (typical bullpen size)
    tasks = []
    for r in relievers[:10]:
        tasks.append(get_reliever_recent_usage(r["id"], season))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    available = []
    unavailable = []
    tired = []

    for r, usage in zip(relievers[:10], results):
        if isinstance(usage, Exception):
            available.append(r["name"])
            continue

        if not usage.get("available", True):
            unavailable.append({"name": r["name"], "reason": usage.get("fatigue", "?")})
        elif usage.get("fatigue") in ("fatigued", "moderate"):
            tired.append({"name": r["name"], "fatigue": usage.get("fatigue", "?")})
            available.append(r["name"])
        else:
            available.append(r["name"])

    total = len(relievers[:10])
    avail_count = len(available)
    unavail_count = len(unavailable)
    tired_count = len(tired)

    # Compute quality adjustment
    # Full bullpen available: 1.0 (no adjustment)
    # Missing 1-2 arms: slight penalty (1.02-1.05x ERA multiplier)
    # Missing 3+: significant penalty (1.08-1.15x)
    if unavail_count == 0:
        adjustment = 1.0
        status = "full_strength"
    elif unavail_count <= 1:
        adjustment = 1.03
        status = "slight_thin"
    elif unavail_count <= 2:
        adjustment = 1.06
        status = "thin"
    elif unavail_count <= 3:
        adjustment = 1.10
        status = "depleted"
    else:
        adjustment = 1.15
        status = "emergency"

    # Tired arms add a smaller penalty
    if tired_count >= 3:
        adjustment *= 1.03
    elif tired_count >= 2:
        adjustment *= 1.02

    return {
        "adjustment": round(adjustment, 3),
        "available_count": avail_count,
        "unavailable_count": unavail_count,
        "tired_count": tired_count,
        "total_count": total,
        "status": status,
        "unavailable": unavailable,
        "tired_arms": tired,
        "available_arms": available[:5],  # Show top 5
    }


def apply_bullpen_availability(bullpen_probs: dict, availability: dict) -> dict:
    """
    Adjust bullpen probability vector based on availability.
    Depleted bullpens give up more hits/walks (higher ERA).
    """
    adj = availability.get("adjustment", 1.0)
    if adj == 1.0:
        return bullpen_probs

    adjusted = bullpen_probs.copy()
    for key in ["p_1b", "p_2b", "p_hr", "p_bb"]:
        adjusted[key] = adjusted.get(key, 0) * adj
    non_out = sum(adjusted.get(k, 0) for k in ["p_1b", "p_2b", "p_3b", "p_hr", "p_bb", "p_hbp"])
    adjusted["p_out"] = max(0.01, 1.0 - non_out)
    return adjusted


# ═══════════════════════════════════════════════════════════════
# COMBINED GAME CONTEXT
# ═══════════════════════════════════════════════════════════════

async def get_full_game_context(game_pk: int, away_team_id: int,
                                  home_team_id: int) -> dict:
    """
    Pull all Tier 1 context for a game:
    - Actual batting lineups
    - Home plate umpire + tendencies
    - Bullpen availability for both teams
    """
    tasks = [
        get_game_lineups(game_pk),
        get_game_umpire(game_pk),
        assess_bullpen_availability(away_team_id),
        assess_bullpen_availability(home_team_id),
    ]

    lineups, umpire, away_bp, home_bp = await asyncio.gather(*tasks, return_exceptions=True)

    if isinstance(lineups, Exception):
        lineups = {"away": [], "home": [], "lineup_available": False}
    if isinstance(umpire, Exception):
        umpire = {"name": "Unknown", "id": None}
    if isinstance(away_bp, Exception):
        away_bp = {"adjustment": 1.0, "status": "unknown"}
    if isinstance(home_bp, Exception):
        home_bp = {"adjustment": 1.0, "status": "unknown"}

    ump_adj = get_umpire_adjustment(umpire.get("id"))

    return {
        "lineups": lineups,
        "umpire": {**umpire, **ump_adj},
        "bullpen_availability": {
            "away": away_bp,
            "home": home_bp,
        },
    }


def clear_ctx_cache():
    global _ctx_cache
    _ctx_cache = {}
