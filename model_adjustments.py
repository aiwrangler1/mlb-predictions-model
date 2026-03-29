"""
Model Adjustments - Park Factors, Platoon Splits, Recent Form,
Pitcher Workload, and Weather adjustments for the MLB predictions model.
"""
import httpx
import asyncio
from typing import Optional
from datetime import datetime, date, timedelta

MLB_API = "https://statsapi.mlb.com/api/v1"
ESPN_API = "http://site.api.espn.com/apis/site/v2/sports/baseball/mlb"

_adj_cache = {}


async def _fetch(url: str, timeout: float = 12.0):
    if url in _adj_cache:
        return _adj_cache[url]
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            _adj_cache[url] = data
            return data
    except Exception as e:
        return None


# ═══════════════════════════════════════════════════════════════
# 1. PARK FACTORS
# ═══════════════════════════════════════════════════════════════

# Statcast 3-year rolling park factors (100 = average)
# Source: Baseball Savant / FantasyPros 2023-2025
# Each factor adjusts the probability of that outcome
PARK_FACTORS = {
    # team_id: {hr, 1b, 2b, 3b, runs}  (100 = neutral)
    109: {"hr": 1.00, "1b": 1.00, "2b": 1.03, "3b": 0.80, "runs": 0.97, "name": "Chase Field"},              # ARI
    110: {"hr": 0.88, "1b": 1.08, "2b": 0.97, "3b": 2.04, "runs": 0.93, "name": "Camden Yards"},             # BAL
    111: {"hr": 0.85, "1b": 1.07, "2b": 1.21, "3b": 1.16, "runs": 1.05, "name": "Fenway Park"},              # BOS
    112: {"hr": 0.92, "1b": 0.96, "2b": 0.84, "3b": 1.19, "runs": 0.92, "name": "Wrigley Field"},            # CHC
    113: {"hr": 1.18, "1b": 0.98, "2b": 0.95, "3b": 0.85, "runs": 1.05, "name": "Great American Ball Park"}, # CIN
    114: {"hr": 0.87, "1b": 0.98, "2b": 0.96, "3b": 0.75, "runs": 0.90, "name": "Progressive Field"},        # CLE
    115: {"hr": 1.15, "1b": 1.06, "2b": 1.12, "3b": 1.40, "runs": 1.12, "name": "Coors Field"},              # COL
    116: {"hr": 1.05, "1b": 1.00, "2b": 1.02, "3b": 0.90, "runs": 1.00, "name": "Comerica Park"},            # DET
    117: {"hr": 0.99, "1b": 0.92, "2b": 1.00, "3b": 0.88, "runs": 0.94, "name": "Daikin Park"},              # HOU
    118: {"hr": 0.88, "1b": 1.04, "2b": 1.15, "3b": 1.88, "runs": 1.05, "name": "Kauffman Stadium"},         # KC
    119: {"hr": 1.21, "1b": 0.93, "2b": 0.91, "3b": 0.54, "runs": 0.98, "name": "Dodger Stadium"},           # LAD
    108: {"hr": 1.05, "1b": 0.99, "2b": 0.90, "3b": 0.98, "runs": 1.01, "name": "Angel Stadium"},            # LAA
    158: {"hr": 1.06, "1b": 0.97, "2b": 0.98, "3b": 0.85, "runs": 0.99, "name": "American Family Field"},    # MIL
    142: {"hr": 1.02, "1b": 0.97, "2b": 0.94, "3b": 0.90, "runs": 0.98, "name": "Target Field"},             # MIN
    146: {"hr": 0.98, "1b": 1.04, "2b": 1.08, "3b": 1.17, "runs": 1.06, "name": "loanDepot Park"},           # MIA
    121: {"hr": 1.14, "1b": 0.92, "2b": 0.84, "3b": 0.71, "runs": 0.93, "name": "Citi Field"},               # NYM
    147: {"hr": 1.12, "1b": 0.92, "2b": 0.89, "3b": 0.62, "runs": 0.96, "name": "Yankee Stadium"},           # NYY
    133: {"hr": 1.06, "1b": 1.08, "2b": 1.23, "3b": 0.84, "runs": 1.09, "name": "Sutter Health Park"},       # ATH
    143: {"hr": 1.16, "1b": 0.97, "2b": 0.91, "3b": 0.97, "runs": 1.04, "name": "Citizens Bank Park"},       # PHI
    134: {"hr": 0.84, "1b": 1.02, "2b": 0.97, "3b": 0.95, "runs": 0.92, "name": "PNC Park"},                 # PIT
    135: {"hr": 0.95, "1b": 0.97, "2b": 0.98, "3b": 0.78, "runs": 0.95, "name": "Petco Park"},               # SD
    137: {"hr": 0.82, "1b": 0.99, "2b": 1.01, "3b": 1.20, "runs": 0.92, "name": "Oracle Park"},              # SF
    136: {"hr": 0.90, "1b": 0.83, "2b": 0.80, "3b": 0.78, "runs": 0.84, "name": "T-Mobile Park"},            # SEA
    138: {"hr": 0.87, "1b": 1.05, "2b": 0.98, "3b": 0.74, "runs": 0.98, "name": "Busch Stadium"},            # STL
    139: {"hr": 1.09, "1b": 1.07, "2b": 0.85, "3b": 0.56, "runs": 1.00, "name": "Tropicana Field"},          # TB
    140: {"hr": 1.07, "1b": 0.95, "2b": 0.96, "3b": 0.70, "runs": 0.98, "name": "Globe Life Field"},         # TEX
    141: {"hr": 1.10, "1b": 0.96, "2b": 0.92, "3b": 0.65, "runs": 0.98, "name": "Rogers Centre"},            # TOR
    120: {"hr": 1.05, "1b": 1.00, "2b": 1.00, "3b": 1.00, "runs": 1.00, "name": "Nationals Park"},           # WSH
    145: {"hr": 0.95, "1b": 0.95, "2b": 0.95, "3b": 1.00, "runs": 0.97, "name": "Rate Field"},               # CHW
    144: {"hr": 1.03, "1b": 0.97, "2b": 0.94, "3b": 0.80, "runs": 0.97, "name": "Truist Park"},              # ATL
}


def _rebalance(probs: dict):
    """
    Rebalance probability vector so it sums to 1.0.
    Works for both 7-event (p_out) and 9-event (p_k, p_bip_out, p_prod_out).
    Adjusts out categories proportionally to absorb any excess/deficit.
    """
    non_out_keys = ["p_1b", "p_2b", "p_3b", "p_hr", "p_bb", "p_hbp"]
    non_out_sum = sum(probs.get(k, 0) for k in non_out_keys)
    
    if "p_k" in probs:
        # 9-event model
        out_keys = ["p_k", "p_bip_out", "p_prod_out"]
        current_out_sum = sum(probs.get(k, 0) for k in out_keys)
        target_out_sum = max(0.01, 1.0 - non_out_sum)
        
        if current_out_sum > 0:
            ratio = target_out_sum / current_out_sum
            for k in out_keys:
                probs[k] = probs.get(k, 0) * ratio
        else:
            probs["p_bip_out"] = target_out_sum
    else:
        # 7-event model
        probs["p_out"] = max(0.01, 1.0 - non_out_sum)


def get_park_factors(home_team_id: int) -> dict:
    """Get park factors for the home team's stadium. Returns multipliers (1.0 = neutral)."""
    return PARK_FACTORS.get(home_team_id, {
        "hr": 1.00, "1b": 1.00, "2b": 1.00, "3b": 1.00, "runs": 1.00, "name": "Unknown"
    })


def apply_park_factors(matchup_probs: dict, park: dict) -> dict:
    adjusted = matchup_probs.copy()
    adjusted["p_1b"] = adjusted.get("p_1b", 0) * park.get("1b", 1.0)
    adjusted["p_2b"] = adjusted.get("p_2b", 0) * park.get("2b", 1.0)
    adjusted["p_3b"] = adjusted.get("p_3b", 0) * park.get("3b", 1.0)
    adjusted["p_hr"] = adjusted.get("p_hr", 0) * park.get("hr", 1.0)
    total_non_out = sum(adjusted.get(k, 0) for k in ["p_1b", "p_2b", "p_3b", "p_hr", "p_bb", "p_hbp"])
    adjusted["p_out"] = max(0.01, 1.0 - total_non_out)
    return adjusted


# ═══════════════════════════════════════════════════════════════
# 2. PLATOON SPLITS (L/R)
# ═══════════════════════════════════════════════════════════════

async def get_player_handedness(player_id: int) -> dict:
    """Get batter/pitcher handedness from MLB API."""
    url = f"{MLB_API}/people/{player_id}"
    data = await _fetch(url)
    if not data:
        return {"bat_side": "R", "pitch_hand": "R"}
    
    person = data.get("people", [{}])[0]
    return {
        "bat_side": person.get("batSide", {}).get("code", "R"),
        "pitch_hand": person.get("pitchHand", {}).get("code", "R"),
    }


async def get_platoon_splits(player_id: int, season: int = 2025) -> dict:
    """
    Get batter's stats vs LHP and RHP from MLB API.
    Uses statSplits with sitCodes vr (vs right) and vl (vs left).
    """
    url = f"{MLB_API}/people/{player_id}/stats?stats=statSplits&season={season}&group=hitting&sitCodes=vr,vl"
    data = await _fetch(url)
    if not data:
        return {}
    
    splits = {}
    for sg in data.get("stats", []):
        for s in sg.get("splits", []):
            desc = s.get("split", {}).get("description", "").lower()
            stat = s.get("stat", {})
            pa = stat.get("plateAppearances", 0)
            if pa < 20:
                continue
            
            h = stat.get("hits", 0)
            doubles = stat.get("doubles", 0)
            triples = stat.get("triples", 0)
            hr = stat.get("homeRuns", 0)
            bb = stat.get("baseOnBalls", 0)
            hbp = stat.get("hitByPitch", 0)
            singles = h - doubles - triples - hr
            outs = pa - h - bb - hbp
            
            key = "vs_left" if "left" in desc else "vs_right"
            splits[key] = {
                "pa": pa,
                "avg": stat.get("avg", ".000"),
                "ops": stat.get("ops", ".000"),
                "hr": hr,
                "p_1b": singles / pa if pa > 0 else 0,
                "p_2b": doubles / pa if pa > 0 else 0,
                "p_3b": triples / pa if pa > 0 else 0,
                "p_hr": hr / pa if pa > 0 else 0,
                "p_bb": bb / pa if pa > 0 else 0,
                "p_hbp": hbp / pa if pa > 0 else 0,
                "p_out": outs / pa if pa > 0 else 1.0,
            }
    
    return splits


def apply_platoon_adjustment(batter_probs: dict, platoon_splits: dict,
                               pitcher_hand: str) -> dict:
    """
    Adjust batter probabilities based on pitcher handedness.
    If batter has split data vs the pitcher's hand, blend it in.
    
    Weighting: 70% overall + 30% split-specific
    (Keeps overall as anchor, uses splits for adjustment)
    """
    key = "vs_left" if pitcher_hand == "L" else "vs_right"
    
    if key not in platoon_splits:
        return batter_probs
    
    split = platoon_splits[key]
    if split.get("pa", 0) < 30:
        return batter_probs  # Not enough data
    
    adjusted = {}
    overall_weight = 0.70
    split_weight = 0.30
    
    for field in ["p_1b", "p_2b", "p_3b", "p_hr", "p_bb", "p_hbp", "p_out"]:
        overall_val = batter_probs.get(field, 0)
        split_val = split.get(field, 0)
        adjusted[field] = overall_val * overall_weight + split_val * split_weight
    
    # Normalize
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}
    
    return adjusted


# ═══════════════════════════════════════════════════════════════
# 3. RECENT FORM (Rolling 30-day window with decay)
# ═══════════════════════════════════════════════════════════════

async def get_batter_game_log(player_id: int, season: int = 2025) -> list:
    """Get batter's game-by-game stats for the season."""
    url = f"{MLB_API}/people/{player_id}/stats?stats=gameLog&season={season}&group=hitting"
    data = await _fetch(url)
    if not data:
        return []
    
    games = []
    for sg in data.get("stats", []):
        for s in sg.get("splits", []):
            stat = s.get("stat", {})
            games.append({
                "date": s.get("date", ""),
                "opponent": s.get("opponent", {}).get("name", ""),
                "ab": stat.get("atBats", 0),
                "pa": stat.get("plateAppearances", 0),
                "hits": stat.get("hits", 0),
                "hr": stat.get("homeRuns", 0),
                "rbi": stat.get("rbi", 0),
                "bb": stat.get("baseOnBalls", 0),
                "so": stat.get("strikeOuts", 0),
                "avg": stat.get("avg", ".000"),
                "ops": stat.get("ops", ".000"),
            })
    
    return games


def compute_recent_form(game_log: list, window_days: int = 30) -> dict:
    """
    Compute a form multiplier based on recent performance vs season average.
    Uses exponential decay weighting (more recent games weighted higher).
    
    Returns a multiplier: >1.0 = hot, <1.0 = cold
    """
    if not game_log or len(game_log) < 10:
        return {"form_multiplier": 1.0, "recent_ops": None, "games_used": 0, "trend": "neutral"}
    
    # Take last N games (approximately 30 days ~ 25-30 games)
    recent = game_log[-min(30, len(game_log)):]
    all_games = game_log
    
    # Season totals
    season_pa = sum(g["pa"] for g in all_games)
    season_hits = sum(g["hits"] for g in all_games)
    season_hr = sum(g["hr"] for g in all_games)
    season_bb = sum(g["bb"] for g in all_games)
    
    if season_pa == 0:
        return {"form_multiplier": 1.0, "recent_ops": None, "games_used": 0, "trend": "neutral"}
    
    season_obp = (season_hits + season_bb) / season_pa
    
    # Recent with exponential decay
    total_weight = 0
    weighted_hits = 0
    weighted_pa = 0
    weighted_hr = 0
    
    for i, g in enumerate(recent):
        # More recent = higher weight
        weight = 0.95 ** (len(recent) - 1 - i)  # Most recent has weight ~1.0
        total_weight += weight
        weighted_pa += g["pa"] * weight
        weighted_hits += g["hits"] * weight
        weighted_hr += g["hr"] * weight
    
    if weighted_pa == 0 or total_weight == 0:
        return {"form_multiplier": 1.0, "recent_ops": None, "games_used": len(recent), "trend": "neutral"}
    
    recent_obp = (weighted_hits + sum(g["bb"] * (0.95 ** (len(recent) - 1 - i)) for i, g in enumerate(recent))) / weighted_pa
    
    # Form multiplier: ratio of recent to season (clamped)
    if season_obp > 0:
        form_raw = recent_obp / season_obp
    else:
        form_raw = 1.0
    
    # Clamp to reasonable range (0.8 to 1.2)
    form = max(0.80, min(1.20, form_raw))
    
    # Trend classification
    if form > 1.05:
        trend = "hot"
    elif form < 0.95:
        trend = "cold"
    else:
        trend = "neutral"
    
    return {
        "form_multiplier": round(form, 3),
        "recent_ops": None,
        "games_used": len(recent),
        "trend": trend,
    }


def apply_form_adjustment(matchup_probs: dict, form: dict) -> dict:
    mult = form.get("form_multiplier", 1.0)
    if mult == 1.0:
        return matchup_probs
    adjusted = matchup_probs.copy()
    for key in ["p_1b", "p_2b", "p_3b", "p_hr"]:
        adjusted[key] = adjusted.get(key, 0) * mult
    total_non_out = sum(adjusted.get(k, 0) for k in ["p_1b", "p_2b", "p_3b", "p_hr", "p_bb", "p_hbp"])
    adjusted["p_out"] = max(0.01, 1.0 - total_non_out)
    return adjusted


# ═══════════════════════════════════════════════════════════════
# 4. PITCHER WORKLOAD
# ═══════════════════════════════════════════════════════════════

async def get_pitcher_game_log(player_id: int, season: int = 2025) -> list:
    """Get pitcher's game-by-game stats."""
    url = f"{MLB_API}/people/{player_id}/stats?stats=gameLog&season={season}&group=pitching"
    data = await _fetch(url)
    if not data:
        return []
    
    games = []
    for sg in data.get("stats", []):
        for s in sg.get("splits", []):
            stat = s.get("stat", {})
            games.append({
                "date": s.get("date", ""),
                "opponent": s.get("opponent", {}).get("name", ""),
                "ip": float(stat.get("inningsPitched", "0") or "0"),
                "hits": stat.get("hits", 0),
                "er": stat.get("earnedRuns", 0),
                "k": stat.get("strikeOuts", 0),
                "bb": stat.get("baseOnBalls", 0),
                "pitches": stat.get("numberOfPitches", 0),
                "era": stat.get("era", "0.00"),
            })
    
    return games


def compute_pitcher_workload(game_log: list, game_date_str: Optional[str] = None) -> dict:
    """
    Analyze pitcher workload:
    - Days rest since last start
    - Recent pitch count trend
    - Fatigue indicator (high pitch counts in recent starts)
    """
    if not game_log:
        return {
            "days_rest": None, "avg_pitches_last3": None,
            "fatigue_factor": 1.0, "workload_status": "unknown"
        }
    
    today = datetime.now().date()
    if game_date_str:
        try:
            today = datetime.strptime(game_date_str, "%Y-%m-%d").date()
        except ValueError:
            pass
    
    # Days rest
    last_game = game_log[-1]
    try:
        last_date = datetime.strptime(last_game["date"], "%Y-%m-%d").date()
        days_rest = (today - last_date).days
    except (ValueError, TypeError):
        days_rest = 5  # Default
    
    # Average pitches in last 3 starts
    recent_starts = game_log[-3:] if len(game_log) >= 3 else game_log
    avg_pitches = sum(g["pitches"] for g in recent_starts) / len(recent_starts) if recent_starts else 90
    avg_ip = sum(g["ip"] for g in recent_starts) / len(recent_starts) if recent_starts else 5.5
    
    # Recent ERA (last 5 starts)
    recent_5 = game_log[-5:] if len(game_log) >= 5 else game_log
    total_er = sum(g["er"] for g in recent_5)
    total_ip = sum(g["ip"] for g in recent_5)
    recent_era = (total_er * 9 / total_ip) if total_ip > 0 else 4.50
    
    # Fatigue factor
    # Short rest (< 4 days): penalty
    # Normal rest (4-5): neutral
    # Extra rest (6+): slight bonus (fresher arm)
    # High recent pitch counts: penalty
    if days_rest is not None:
        if days_rest <= 3:
            rest_factor = 0.90  # Short rest = 10% penalty
        elif days_rest == 4:
            rest_factor = 0.97  # Slightly short
        elif days_rest == 5:
            rest_factor = 1.00  # Normal
        elif days_rest <= 7:
            rest_factor = 1.02  # Extra rest bonus
        else:
            rest_factor = 0.98  # Too long = possible rust
    else:
        rest_factor = 1.0
    
    # Pitch count fatigue
    if avg_pitches > 105:
        pitch_factor = 0.95
    elif avg_pitches > 95:
        pitch_factor = 0.98
    else:
        pitch_factor = 1.0
    
    fatigue = rest_factor * pitch_factor
    
    # Status
    if fatigue >= 1.01:
        status = "fresh"
    elif fatigue >= 0.97:
        status = "normal"
    elif fatigue >= 0.93:
        status = "fatigued"
    else:
        status = "exhausted"
    
    return {
        "days_rest": days_rest,
        "avg_pitches_last3": round(avg_pitches, 0),
        "avg_ip_last3": round(avg_ip, 1),
        "recent_era_last5": round(recent_era, 2),
        "fatigue_factor": round(fatigue, 3),
        "workload_status": status,
    }


def apply_workload_adjustment(pitcher_probs: dict, workload: dict) -> dict:
    fatigue = workload.get("fatigue_factor", 1.0)
    if fatigue == 1.0:
        return pitcher_probs
    adjusted = pitcher_probs.copy()
    batter_boost = 1.0 / fatigue
    for key in ["p_1b", "p_2b", "p_hr", "p_bb"]:
        adjusted[key] = adjusted.get(key, 0) * batter_boost
    total_non_out = sum(adjusted.get(k, 0) for k in ["p_1b", "p_2b", "p_3b", "p_hr", "p_bb", "p_hbp"])
    adjusted["p_out"] = max(0.01, 1.0 - total_non_out)
    return adjusted


# ═══════════════════════════════════════════════════════════════
# 5. WEATHER ADJUSTMENTS
# ═══════════════════════════════════════════════════════════════

async def get_weather_from_espn() -> dict:
    """
    Get weather data for today's games from ESPN scoreboard.
    Returns dict keyed by a game identifier.
    """
    url = f"{ESPN_API}/scoreboard"
    data = await _fetch(url)
    if not data:
        return {}
    
    weather_map = {}
    for event in data.get("events", []):
        name = event.get("shortName", event.get("name", ""))
        weather = event.get("weather", {})
        
        if weather:
            temp = weather.get("temperature", 72)
            condition = weather.get("conditionId", "")
            
            weather_map[name] = {
                "temperature": temp,
                "high_temp": weather.get("highTemperature", temp),
                "condition_id": condition,
                "display": weather.get("displayValue", ""),
            }
    
    return weather_map


# Stadium coordinates for weather lookups
STADIUM_COORDS = {
    109: (33.445, -112.067, "Chase Field"),        # ARI - retractable roof
    110: (39.284, -76.622, "Camden Yards"),
    111: (42.346, -71.098, "Fenway Park"),
    112: (41.948, -87.656, "Wrigley Field"),
    113: (39.097, -84.508, "Great American"),
    114: (41.496, -81.685, "Progressive Field"),
    115: (39.756, -104.994, "Coors Field"),
    116: (42.339, -83.049, "Comerica Park"),
    117: (29.757, -95.355, "Daikin Park"),         # retractable roof
    118: (39.051, -94.480, "Kauffman Stadium"),
    119: (34.074, -118.240, "Dodger Stadium"),
    108: (33.800, -117.883, "Angel Stadium"),
    158: (43.028, -87.971, "American Family"),     # retractable roof
    142: (44.982, -93.278, "Target Field"),
    146: (25.778, -80.220, "loanDepot Park"),      # retractable roof
    121: (40.757, -73.846, "Citi Field"),
    147: (40.829, -73.926, "Yankee Stadium"),
    133: (38.580, -121.499, "Sutter Health Park"),  # ATH Sacramento
    143: (39.906, -75.166, "Citizens Bank Park"),
    134: (40.447, -80.006, "PNC Park"),
    135: (32.707, -117.157, "Petco Park"),
    137: (37.778, -122.389, "Oracle Park"),
    136: (47.591, -122.333, "T-Mobile Park"),       # retractable roof
    138: (38.623, -90.193, "Busch Stadium"),
    139: (27.768, -82.653, "Tropicana Field"),      # dome
    140: (32.751, -97.083, "Globe Life Field"),      # retractable roof
    141: (43.641, -79.389, "Rogers Centre"),         # retractable roof
    120: (38.873, -77.007, "Nationals Park"),
    145: (41.830, -87.634, "Rate Field"),
    144: (33.891, -84.468, "Truist Park"),
}

# Domed/retractable roof stadiums (weather has minimal effect)
CONTROLLED_CLIMATE = {117, 136, 139, 140, 141, 146, 158}
# 117=HOU, 136=SEA, 139=TB, 140=TEX, 141=TOR, 146=MIA, 158=MIL


def compute_weather_adjustment(temp: int, wind_mph: float = 5.0,
                                 wind_dir: str = "calm",
                                 home_team_id: int = 0) -> dict:
    """
    Compute weather-based adjustments to game outcomes.
    
    Research-backed effects:
    - Temperature: every 10°F above 70°F adds ~2% HR probability
    - Cold (<55°F): reduces offense ~5-8%
    - Wind blowing out: +8-12% HR boost
    - Wind blowing in: -8-12% HR reduction
    - Altitude (Coors): already captured in park factors
    """
    # Skip for domed/controlled stadiums
    if home_team_id in CONTROLLED_CLIMATE:
        return {
            "hr_factor": 1.0, "hit_factor": 1.0,
            "description": "Retractable roof / dome — weather neutral",
            "temp": temp, "is_dome": True
        }
    
    # Temperature effect on HR
    # Baseline: 72°F
    temp_diff = temp - 72
    hr_temp_factor = 1.0 + (temp_diff * 0.002)  # ~0.2% per degree
    
    # Cold penalty on overall offense
    if temp < 55:
        hit_cold_factor = 0.94 + (temp - 32) * 0.003  # Scales from ~0.94 at 32°F to ~1.0 at 55°F
    elif temp < 65:
        hit_cold_factor = 0.97 + (temp - 55) * 0.003
    else:
        hit_cold_factor = 1.0
    
    # Hot bonus
    if temp > 85:
        hit_cold_factor = 1.0 + (temp - 85) * 0.001  # Slight boost in heat
    
    # Wind effect (simplified without detailed direction)
    # wind_dir could be: "out", "in", "cross", "calm"
    if wind_dir == "out" and wind_mph > 8:
        wind_hr_factor = 1.0 + min(wind_mph, 25) * 0.005  # up to ~12% at 25mph
    elif wind_dir == "in" and wind_mph > 8:
        wind_hr_factor = 1.0 - min(wind_mph, 25) * 0.004
    else:
        wind_hr_factor = 1.0
    
    hr_factor = max(0.75, min(1.25, hr_temp_factor * wind_hr_factor))
    hit_factor = max(0.85, min(1.15, hit_cold_factor))
    
    # Description
    parts = []
    if temp < 55:
        parts.append(f"Cold ({temp}°F) — offense suppressed")
    elif temp > 85:
        parts.append(f"Hot ({temp}°F) — slight offense boost")
    else:
        parts.append(f"{temp}°F")
    
    if wind_dir == "out" and wind_mph > 8:
        parts.append(f"Wind out {wind_mph:.0f}mph — HR boost")
    elif wind_dir == "in" and wind_mph > 8:
        parts.append(f"Wind in {wind_mph:.0f}mph — HR suppressed")
    
    return {
        "hr_factor": round(hr_factor, 3),
        "hit_factor": round(hit_factor, 3),
        "description": " | ".join(parts) if parts else "Normal conditions",
        "temp": temp,
        "is_dome": False,
    }


def apply_weather_adjustment(matchup_probs: dict, weather: dict) -> dict:
    hr_f = weather.get("hr_factor", 1.0)
    hit_f = weather.get("hit_factor", 1.0)
    if hr_f == 1.0 and hit_f == 1.0:
        return matchup_probs
    adjusted = matchup_probs.copy()
    adjusted["p_hr"] = adjusted.get("p_hr", 0) * hr_f
    adjusted["p_1b"] = adjusted.get("p_1b", 0) * hit_f
    adjusted["p_2b"] = adjusted.get("p_2b", 0) * hit_f
    adjusted["p_3b"] = adjusted.get("p_3b", 0) * hit_f
    total_non_out = sum(adjusted.get(k, 0) for k in ["p_1b", "p_2b", "p_3b", "p_hr", "p_bb", "p_hbp"])
    adjusted["p_out"] = max(0.01, 1.0 - total_non_out)
    return adjusted


# ═══════════════════════════════════════════════════════════════
# COMBINED ADJUSTMENT PIPELINE
# ═══════════════════════════════════════════════════════════════

async def get_all_adjustments(batter_id: int, pitcher_id: int,
                                home_team_id: int, season: int = 2025,
                                game_date: Optional[str] = None) -> dict:
    """
    Fetch all adjustment data for a batter-pitcher matchup.
    Returns all adjustment components for the dashboard.
    """
    tasks = [
        get_player_handedness(pitcher_id),
        get_platoon_splits(batter_id, season),
        get_batter_game_log(batter_id, season),
        get_pitcher_game_log(pitcher_id, season),
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    pitcher_hand = results[0] if isinstance(results[0], dict) else {"pitch_hand": "R"}
    platoon = results[1] if isinstance(results[1], dict) else {}
    batter_log = results[2] if isinstance(results[2], list) else []
    pitcher_log = results[3] if isinstance(results[3], list) else []
    
    park = get_park_factors(home_team_id)
    form = compute_recent_form(batter_log)
    workload = compute_pitcher_workload(pitcher_log, game_date)
    
    return {
        "park_factors": park,
        "pitcher_hand": pitcher_hand.get("pitch_hand", "R"),
        "platoon_splits": platoon,
        "batter_form": form,
        "pitcher_workload": workload,
    }


def clear_adj_cache():
    global _adj_cache
    _adj_cache = {}
