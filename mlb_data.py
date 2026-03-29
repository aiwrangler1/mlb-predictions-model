"""
MLB Data Layer - Fetches data from MLB Stats API and ESPN API.
Handles team rosters, player stats, schedules, and league averages.
"""
import httpx
import asyncio
from datetime import datetime, date
from typing import Optional
import json

MLB_API = "https://statsapi.mlb.com/api/v1"
ESPN_API = "http://site.api.espn.com/apis/site/v2/sports/baseball/mlb"

# Cache for API responses
_cache = {}

TEAM_ABBREVS = {
    109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC", 113: "CIN",
    114: "CLE", 115: "COL", 116: "DET", 117: "HOU", 118: "KC",
    119: "LAD", 108: "LAA", 158: "MIL", 142: "MIN", 146: "MIA",
    121: "NYM", 147: "NYY", 133: "ATH", 143: "PHI", 134: "PIT",
    135: "SD", 137: "SF", 136: "SEA", 138: "STL", 139: "TB",
    140: "TEX", 141: "TOR", 120: "WSH", 145: "CHW", 144: "ATL"
}

TEAM_NAMES = {
    109: "Arizona Diamondbacks", 110: "Baltimore Orioles", 111: "Boston Red Sox",
    112: "Chicago Cubs", 113: "Cincinnati Reds", 114: "Cleveland Guardians",
    115: "Colorado Rockies", 116: "Detroit Tigers", 117: "Houston Astros",
    118: "Kansas City Royals", 119: "Los Angeles Dodgers", 108: "Los Angeles Angels",
    158: "Milwaukee Brewers", 142: "Minnesota Twins", 146: "Miami Marlins",
    121: "New York Mets", 147: "New York Yankees", 133: "Athletics",
    143: "Philadelphia Phillies", 134: "Pittsburgh Pirates", 135: "San Diego Padres",
    137: "San Francisco Giants", 136: "Seattle Mariners", 138: "St. Louis Cardinals",
    139: "Tampa Bay Rays", 140: "Texas Rangers", 141: "Toronto Blue Jays",
    120: "Washington Nationals", 145: "Chicago White Sox", 144: "Atlanta Braves"
}

# 2025 Final standings win percentages (for early season blending)
TEAM_2025_WPCT = {
    119: .623, 147: .611, 144: .605, 143: .599, 117: .593,  # Top teams
    114: .586, 136: .580, 110: .574, 158: .568, 112: .562,
    142: .556, 111: .549, 135: .543, 121: .537, 140: .531,
    109: .525, 116: .519, 139: .512, 138: .506, 118: .500,
    137: .494, 134: .488, 141: .481, 108: .475, 113: .469,
    115: .463, 120: .456, 146: .450, 133: .444, 145: .432
}


async def fetch_json(url: str, timeout: float = 15.0) -> dict:
    """Fetch JSON from URL with caching."""
    if url in _cache:
        return _cache[url]
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
        _cache[url] = data
        return data


async def get_schedule(game_date: Optional[str] = None) -> list:
    """Get today's MLB schedule with probable pitchers."""
    if game_date is None:
        game_date = date.today().isoformat()
    url = f"{MLB_API}/schedule?sportId=1&date={game_date}&hydrate=probablePitcher,team"
    data = await fetch_json(url)
    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            away = g["teams"]["away"]
            home = g["teams"]["home"]
            away_team_id = away["team"]["id"]
            home_team_id = home["team"]["id"]
            
            away_pitcher = away.get("probablePitcher", {})
            home_pitcher = home.get("probablePitcher", {})
            
            games.append({
                "game_id": g["gamePk"],
                "game_date": g.get("gameDate", ""),
                "status": g.get("status", {}).get("detailedState", ""),
                "away_team_id": away_team_id,
                "away_team": TEAM_ABBREVS.get(away_team_id, "???"),
                "away_team_name": TEAM_NAMES.get(away_team_id, "Unknown"),
                "home_team_id": home_team_id,
                "home_team": TEAM_ABBREVS.get(home_team_id, "???"),
                "home_team_name": TEAM_NAMES.get(home_team_id, "Unknown"),
                "away_pitcher_id": away_pitcher.get("id"),
                "away_pitcher_name": away_pitcher.get("fullName", "TBD"),
                "home_pitcher_id": home_pitcher.get("id"),
                "home_pitcher_name": home_pitcher.get("fullName", "TBD"),
                "away_wpct": TEAM_2025_WPCT.get(away_team_id, .500),
                "home_wpct": TEAM_2025_WPCT.get(home_team_id, .500),
            })
    return games


async def get_team_roster(team_id: int, season: int = 2026) -> list:
    """Get team roster with player IDs."""
    url = f"{MLB_API}/teams/{team_id}/roster?season={season}&rosterType=active"
    data = await fetch_json(url)
    players = []
    for p in data.get("roster", []):
        person = p["person"]
        pos = p["position"]
        players.append({
            "id": person["id"],
            "name": person["fullName"],
            "position": pos["abbreviation"],
            "position_type": pos.get("type", ""),
        })
    return players


async def get_batter_stats(player_id: int, season: int = 2025) -> Optional[dict]:
    """Get batter's season stats. Returns probability vector for Log5."""
    url = f"{MLB_API}/people/{player_id}/stats?stats=season&season={season}&group=hitting"
    try:
        data = await fetch_json(url)
    except Exception:
        return None
    
    for sg in data.get("stats", []):
        for split in sg.get("splits", []):
            s = split["stat"]
            pa = s.get("plateAppearances", 0)
            if pa < 20:  # Need minimum PA for reliability
                continue
            
            h = s.get("hits", 0)
            doubles = s.get("doubles", 0)
            triples = s.get("triples", 0)
            hr = s.get("homeRuns", 0)
            bb = s.get("baseOnBalls", 0)
            hbp = s.get("hitByPitch", 0)
            so = s.get("strikeOuts", 0)
            sb = s.get("stolenBases", 0)
            singles = h - doubles - triples - hr
            outs = pa - h - bb - hbp
            
            return {
                "player_id": player_id,
                "name": split.get("player", {}).get("fullName", ""),
                "team": split.get("team", {}).get("name", ""),
                "season": season,
                "pa": pa,
                "ab": s.get("atBats", 0),
                "hits": h,
                "singles": singles,
                "doubles": doubles,
                "triples": triples,
                "hr": hr,
                "bb": bb,
                "hbp": hbp,
                "so": so,
                "sb": sb,
                "rbi": s.get("rbi", 0),
                "runs": s.get("runs", 0),
                "avg": s.get("avg", ".000"),
                "obp": s.get("obp", ".000"),
                "slg": s.get("slg", ".000"),
                "ops": s.get("ops", ".000"),
                # Probability vector (7 outcomes)
                "p_1b": singles / pa if pa > 0 else 0,
                "p_2b": doubles / pa if pa > 0 else 0,
                "p_3b": triples / pa if pa > 0 else 0,
                "p_hr": hr / pa if pa > 0 else 0,
                "p_bb": bb / pa if pa > 0 else 0,
                "p_hbp": hbp / pa if pa > 0 else 0,
                "p_out": outs / pa if pa > 0 else 1.0,
                "p_sb_attempt": sb / max(h + bb + hbp, 1),  # SB rate when on base
            }
    return None


async def get_pitcher_stats(player_id: int, season: int = 2025) -> Optional[dict]:
    """Get pitcher's season stats. Returns probability vector for Log5."""
    url = f"{MLB_API}/people/{player_id}/stats?stats=season&season={season}&group=pitching"
    try:
        data = await fetch_json(url)
    except Exception:
        return None
    
    for sg in data.get("stats", []):
        for split in sg.get("splits", []):
            s = split["stat"]
            bf = s.get("battersFaced", 0)
            if bf < 50:
                continue
            
            h = s.get("hits", 0)
            doubles = s.get("doubles", 0)
            triples = s.get("triples", 0)
            hr = s.get("homeRuns", 0)
            bb = s.get("baseOnBalls", 0)
            hbp = s.get("hitByPitch", 0)
            so = s.get("strikeOuts", 0)
            singles = h - doubles - triples - hr
            outs = bf - h - bb - hbp
            
            ip_str = s.get("inningsPitched", "0")
            ip = float(ip_str) if ip_str else 0
            
            return {
                "player_id": player_id,
                "name": split.get("player", {}).get("fullName", ""),
                "team": split.get("team", {}).get("name", ""),
                "season": season,
                "bf": bf,
                "ip": ip,
                "hits": h,
                "singles": singles,
                "doubles": doubles,
                "triples": triples,
                "hr": hr,
                "bb": bb,
                "hbp": hbp,
                "so": so,
                "era": s.get("era", "0.00"),
                "whip": s.get("whip", "0.00"),
                "wins": s.get("wins", 0),
                "losses": s.get("losses", 0),
                "k_per_9": s.get("strikeoutsPer9Inn", "0.00"),
                "bb_per_9": s.get("walksPer9Inn", "0.00"),
                # Probability vector (what batters do against this pitcher)
                "p_1b": singles / bf if bf > 0 else 0,
                "p_2b": doubles / bf if bf > 0 else 0,
                "p_3b": triples / bf if bf > 0 else 0,
                "p_hr": hr / bf if bf > 0 else 0,
                "p_bb": bb / bf if bf > 0 else 0,
                "p_hbp": hbp / bf if bf > 0 else 0,
                "p_out": outs / bf if bf > 0 else 1.0,
            }
    return None


def get_league_averages(season: int = 2025) -> dict:
    """
    League average probability vector (7-event).
    Mined from 186,640 PA of Retrosheet 2025 PBP data.
    """
    return {
        "p_1b": 0.142563,
        "p_2b": 0.042258,
        "p_3b": 0.003408,
        "p_hr": 0.030894,
        "p_bb": 0.084178,
        "p_hbp": 0.010577,
        "p_out": 0.686122,  # K + BIP_OUT + PROD_OUT combined
    }


async def get_lineup(team_id: int, season: int = 2026) -> list:
    """
    Get projected batting order for a team.
    Uses roster and sorts by position to approximate a lineup.
    """
    roster = await get_team_roster(team_id, season)
    
    # Filter to position players
    position_players = [p for p in roster if p["position_type"] != "Pitcher" 
                       or p["position"] == "DH"]
    
    # Also include pitchers for the roster lookup
    pitchers = [p for p in roster if p["position_type"] == "Pitcher"]
    
    # Try to get stats for each position player
    batters = []
    tasks = []
    for p in position_players[:12]:  # Limit to avoid too many API calls
        tasks.append(get_batter_stats(p["id"], season))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for p, result in zip(position_players[:12], results):
        if isinstance(result, dict) and result is not None:
            result["position"] = p["position"]
            batters.append(result)
        else:
            # Try previous season
            prev = await get_batter_stats(p["id"], season - 1)
            if prev:
                prev["position"] = p["position"]
                prev["name"] = p["name"]
                batters.append(prev)
    
    # Sort by OPS descending to approximate batting order importance
    batters.sort(key=lambda x: float(x.get("ops", "0").replace(".", "0.", 1) if x.get("ops") else "0"), reverse=True)
    
    return batters[:9]  # Return top 9 as lineup


def clear_cache():
    """Clear the API response cache."""
    global _cache
    _cache = {}
