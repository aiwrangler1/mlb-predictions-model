"""
Projections Engine - Steamer/FanGraphs projections, spring training data,
2026 data blending, and bullpen modeling.

Handles the Bayesian blending of multiple data sources:
1. FanGraphs Steamer 2026 projections (preseason baseline)
2. 2025 regular season actuals (prior season)
3. 2026 spring training stats (light weight)
4. 2026 regular season stats (increasing weight as sample grows)
"""
import httpx
import asyncio
import math
from typing import Optional

MLB_API = "https://statsapi.mlb.com/api/v1"
FANGRAPHS_API = "https://www.fangraphs.com/api/projections"

_proj_cache = {}


async def _fetch(url: str, timeout: float = 15.0):
    if url in _proj_cache:
        return _proj_cache[url]
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            _proj_cache[url] = data
            return data
    except Exception as e:
        print(f"Projections fetch error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# 1. FANGRAPHS STEAMER PROJECTIONS
# ═══════════════════════════════════════════════════════════════

_steamer_batters = None
_steamer_pitchers = None


async def load_steamer_projections():
    """Load 2026 Steamer projections from FanGraphs API."""
    global _steamer_batters, _steamer_pitchers

    if _steamer_batters is not None:
        return

    # Batting projections
    bat_data = await _fetch(f"{FANGRAPHS_API}?type=steamer&stats=bat&pos=all&team=0&players=0&lg=all")
    if bat_data and isinstance(bat_data, list):
        _steamer_batters = {}
        for p in bat_data:
            pid = p.get("playerid")
            if not pid:
                continue
            pa = p.get("PA", 0)
            if pa < 50:
                continue
            h = p.get("H", 0)
            singles = p.get("1B", 0)
            doubles = p.get("2B", 0)
            triples = p.get("3B", 0)
            hr = p.get("HR", 0)
            bb = p.get("BB", 0)
            hbp = p.get("HBP", 0)
            outs = pa - h - bb - hbp
            sb = p.get("SB", 0)

            _steamer_batters[pid] = {
                "fg_id": pid,
                "name": p.get("PlayerName", ""),
                "team": p.get("Team", ""),
                "pa": pa,
                "avg": p.get("AVG", 0),
                "obp": p.get("OBP", 0),
                "slg": p.get("SLG", 0),
                "ops": p.get("OPS", 0),
                "hr": hr,
                "sb": sb,
                "p_1b": singles / pa if pa > 0 else 0,
                "p_2b": doubles / pa if pa > 0 else 0,
                "p_3b": triples / pa if pa > 0 else 0,
                "p_hr": hr / pa if pa > 0 else 0,
                "p_bb": bb / pa if pa > 0 else 0,
                "p_hbp": hbp / pa if pa > 0 else 0,
                "p_out": outs / pa if pa > 0 else 0.675,
            }
        print(f"Loaded {len(_steamer_batters)} Steamer batting projections")

    # Pitching projections
    pit_data = await _fetch(f"{FANGRAPHS_API}?type=steamer&stats=pit&pos=all&team=0&players=0&lg=all")
    if pit_data and isinstance(pit_data, list):
        _steamer_pitchers = {}
        for p in pit_data:
            pid = p.get("playerid")
            if not pid:
                continue
            ip = p.get("IP", 0)
            if ip < 10:
                continue
            # For pitchers, compute what batters do against them
            bf = p.get("TBF", 0) or int(ip * 4.3)  # Approximate BF
            h = p.get("H", 0)
            hr = p.get("HR", 0)
            bb = p.get("BB", 0)
            hbp = p.get("HBP", 0)
            so = p.get("SO", 0)
            # Estimate singles from H - HR - 2B - 3B
            # FanGraphs doesn't always have 2B/3B for pitchers in projections
            doubles_est = h * 0.19  # ~19% of hits are doubles
            triples_est = h * 0.015
            singles_est = h - doubles_est - triples_est - hr
            outs = bf - h - bb - hbp if bf > 0 else ip * 3

            _steamer_pitchers[pid] = {
                "fg_id": pid,
                "name": p.get("PlayerName", ""),
                "team": p.get("Team", ""),
                "ip": ip,
                "era": p.get("ERA", 4.50),
                "whip": p.get("WHIP", 1.30),
                "bf": bf,
                "k": so,
                "p_1b": singles_est / bf if bf > 0 else 0.152,
                "p_2b": doubles_est / bf if bf > 0 else 0.044,
                "p_3b": triples_est / bf if bf > 0 else 0.004,
                "p_hr": hr / bf if bf > 0 else 0.031,
                "p_bb": bb / bf if bf > 0 else 0.083,
                "p_hbp": hbp / bf if bf > 0 else 0.011,
                "p_out": outs / bf if bf > 0 else 0.675,
            }
        print(f"Loaded {len(_steamer_pitchers)} Steamer pitching projections")


def get_steamer_batter(player_name: str, team: str = "") -> Optional[dict]:
    """Look up Steamer projection by player name."""
    if not _steamer_batters:
        return None
    name_lower = player_name.lower().strip()
    for pid, proj in _steamer_batters.items():
        pname = proj["name"].lower().strip()
        if pname == name_lower:
            return proj
        # Partial match: last name
        if name_lower.split()[-1] == pname.split()[-1] and name_lower[0] == pname[0]:
            if team and proj["team"] and team.upper() not in proj["team"].upper():
                continue
            return proj
    return None


def get_steamer_pitcher(player_name: str, team: str = "") -> Optional[dict]:
    """Look up Steamer pitching projection by name."""
    if not _steamer_pitchers:
        return None
    name_lower = player_name.lower().strip()
    for pid, proj in _steamer_pitchers.items():
        pname = proj["name"].lower().strip()
        if pname == name_lower:
            return proj
        if name_lower.split()[-1] == pname.split()[-1] and name_lower[0] == pname[0]:
            if team and proj["team"] and team.upper() not in proj["team"].upper():
                continue
            return proj
    return None


# ═══════════════════════════════════════════════════════════════
# 2. SPRING TRAINING DATA (from MLB API)
# ═══════════════════════════════════════════════════════════════

async def get_spring_batter_stats(player_id: int, season: int = 2026) -> Optional[dict]:
    """Get batter's spring training stats."""
    url = f"{MLB_API}/people/{player_id}/stats?stats=season&season={season}&group=hitting&gameType=S"
    data = await _fetch(url)
    if not data:
        return None

    for sg in data.get("stats", []):
        for split in sg.get("splits", []):
            s = split["stat"]
            pa = s.get("plateAppearances", 0)
            if pa < 10:
                return None

            h = s.get("hits", 0)
            doubles = s.get("doubles", 0)
            triples = s.get("triples", 0)
            hr = s.get("homeRuns", 0)
            bb = s.get("baseOnBalls", 0)
            hbp = s.get("hitByPitch", 0)
            singles = h - doubles - triples - hr
            outs = pa - h - bb - hbp

            return {
                "pa": pa, "source": "spring_2026",
                "avg": s.get("avg", ".000"), "ops": s.get("ops", ".000"),
                "p_1b": singles / pa if pa > 0 else 0,
                "p_2b": doubles / pa if pa > 0 else 0,
                "p_3b": triples / pa if pa > 0 else 0,
                "p_hr": hr / pa if pa > 0 else 0,
                "p_bb": bb / pa if pa > 0 else 0,
                "p_hbp": hbp / pa if pa > 0 else 0,
                "p_out": outs / pa if pa > 0 else 1.0,
            }
    return None


async def get_spring_pitcher_stats(player_id: int, season: int = 2026) -> Optional[dict]:
    """Get pitcher's spring training stats."""
    url = f"{MLB_API}/people/{player_id}/stats?stats=season&season={season}&group=pitching&gameType=S"
    data = await _fetch(url)
    if not data:
        return None

    for sg in data.get("stats", []):
        for split in sg.get("splits", []):
            s = split["stat"]
            bf = s.get("battersFaced", 0)
            if bf < 20:
                return None

            h = s.get("hits", 0)
            doubles = s.get("doubles", 0)
            triples = s.get("triples", 0)
            hr = s.get("homeRuns", 0)
            bb = s.get("baseOnBalls", 0)
            hbp = s.get("hitByPitch", 0)
            singles = h - doubles - triples - hr
            outs = bf - h - bb - hbp

            return {
                "bf": bf, "source": "spring_2026",
                "era": s.get("era", "0.00"), "whip": s.get("whip", "0.00"),
                "p_1b": singles / bf if bf > 0 else 0,
                "p_2b": doubles / bf if bf > 0 else 0,
                "p_3b": triples / bf if bf > 0 else 0,
                "p_hr": hr / bf if bf > 0 else 0,
                "p_bb": bb / bf if bf > 0 else 0,
                "p_hbp": hbp / bf if bf > 0 else 0,
                "p_out": outs / bf if bf > 0 else 1.0,
            }
    return None


async def get_2026_regular_stats(player_id: int, group: str = "hitting") -> Optional[dict]:
    """Get 2026 regular season stats (accumulates as season progresses)."""
    url = f"{MLB_API}/people/{player_id}/stats?stats=season&season=2026&group={group}&gameType=R"
    data = await _fetch(url)
    if not data:
        return None

    for sg in data.get("stats", []):
        for split in sg.get("splits", []):
            s = split["stat"]
            if group == "hitting":
                pa = s.get("plateAppearances", 0)
                if pa < 5:
                    return None
                h = s.get("hits", 0)
                doubles = s.get("doubles", 0)
                triples = s.get("triples", 0)
                hr = s.get("homeRuns", 0)
                bb = s.get("baseOnBalls", 0)
                hbp = s.get("hitByPitch", 0)
                singles = h - doubles - triples - hr
                outs = pa - h - bb - hbp
                return {
                    "pa": pa, "source": "regular_2026",
                    "p_1b": singles / pa, "p_2b": doubles / pa,
                    "p_3b": triples / pa, "p_hr": hr / pa,
                    "p_bb": bb / pa, "p_hbp": hbp / pa,
                    "p_out": outs / pa,
                }
            else:
                bf = s.get("battersFaced", 0)
                if bf < 10:
                    return None
                h = s.get("hits", 0)
                doubles = s.get("doubles", 0)
                triples = s.get("triples", 0)
                hr = s.get("homeRuns", 0)
                bb = s.get("baseOnBalls", 0)
                hbp = s.get("hitByPitch", 0)
                singles = h - doubles - triples - hr
                outs = bf - h - bb - hbp
                return {
                    "bf": bf, "source": "regular_2026",
                    "p_1b": singles / bf, "p_2b": doubles / bf,
                    "p_3b": triples / bf, "p_hr": hr / bf,
                    "p_bb": bb / bf, "p_hbp": hbp / bf,
                    "p_out": outs / bf,
                }
    return None


# ═══════════════════════════════════════════════════════════════
# 3. BAYESIAN BLENDING ENGINE
# ═══════════════════════════════════════════════════════════════

def blend_probability_vectors(sources: list) -> dict:
    """
    Bayesian-weighted blend of multiple probability vector sources.

    Each source is a dict with:
        - probability keys (p_1b, p_2b, etc.)
        - 'weight': float (relative importance)
        - 'pa' or 'bf': sample size (used for credibility weighting)

    Weighting scheme:
        - Steamer projections: high baseline weight, regression target
        - 2025 actuals: strong prior, decays as 2026 data accumulates
        - Spring training: light weight (small sample, different context)
        - 2026 regular season: increases with sample size (credibility)

    Uses a simplified Marcel-style credibility formula:
        credibility = pa / (pa + reliability_threshold)
    """
    if not sources:
        return _league_avg_probs()

    outcomes = ["p_1b", "p_2b", "p_3b", "p_hr", "p_bb", "p_hbp", "p_out"]
    total_weight = sum(s.get("weight", 1.0) for s in sources)

    if total_weight == 0:
        return _league_avg_probs()

    blended = {}
    for o in outcomes:
        val = sum(s.get(o, 0) * s.get("weight", 1.0) for s in sources) / total_weight
        blended[o] = max(0, val)

    # Normalize to sum to 1
    total = sum(blended.values())
    if total > 0:
        blended = {k: v / total for k, v in blended.items()}

    return blended


def compute_blend_weights(steamer: Optional[dict],
                           season_2025: Optional[dict],
                           spring_2026: Optional[dict],
                           regular_2026: Optional[dict]) -> list:
    """
    Compute dynamic blend weights based on available data and sample sizes.

    Early season (Opening Day): Steamer dominates
    Mid-season (100+ PA): 2026 actuals dominate
    """
    sources = []

    # 2026 regular season (highest priority as sample grows)
    if regular_2026:
        pa = regular_2026.get("pa", regular_2026.get("bf", 0))
        # Credibility: pa / (pa + 200) for batters
        # At 50 PA: 20% credibility. At 200 PA: 50%. At 400 PA: 67%.
        cred = pa / (pa + 200)
        weight = 5.0 * cred  # Max weight = 5.0 at full credibility
        if weight > 0.1:
            regular_2026["weight"] = weight
            sources.append(regular_2026)

    # Steamer projections (baseline — always contributes)
    if steamer:
        # Steamer weight decreases as 2026 data accumulates
        reg_pa = regular_2026.get("pa", regular_2026.get("bf", 0)) if regular_2026 else 0
        steamer_weight = max(0.5, 4.0 * (1 - reg_pa / (reg_pa + 300)))
        steamer["weight"] = steamer_weight
        sources.append(steamer)

    # 2025 actuals (decays as 2026 data grows)
    if season_2025:
        reg_pa = regular_2026.get("pa", regular_2026.get("bf", 0)) if regular_2026 else 0
        # Strong early, fades fast
        s25_weight = max(0.3, 3.0 * (1 - reg_pa / (reg_pa + 150)))
        season_2025["weight"] = s25_weight
        sources.append(season_2025)

    # Spring training (small sample, light weight always)
    if spring_2026:
        spring_pa = spring_2026.get("pa", spring_2026.get("bf", 0))
        spring_weight = min(0.8, spring_pa / 100)  # Max 0.8, typical 0.2-0.5
        spring_2026["weight"] = spring_weight
        sources.append(spring_2026)

    return sources


def _league_avg_probs():
    return {
        "p_1b": 0.152, "p_2b": 0.044, "p_3b": 0.004, "p_hr": 0.031,
        "p_bb": 0.083, "p_hbp": 0.011, "p_out": 0.675,
    }


async def get_blended_batter_probs(player_id: int, player_name: str,
                                     team: str = "", season: int = 2025) -> dict:
    """
    Get the best available blended probability vector for a batter.
    Combines Steamer + 2025 actuals + spring + 2026 regular season.
    """
    await load_steamer_projections()

    # Gather all sources
    from mlb_data import get_batter_stats
    season_2025 = await get_batter_stats(player_id, 2025)
    spring = await get_spring_batter_stats(player_id, 2026)
    regular = await get_2026_regular_stats(player_id, "hitting")
    steamer = get_steamer_batter(player_name, team)

    sources = compute_blend_weights(steamer, season_2025, spring, regular)

    if not sources:
        # Fallback to whatever we have from 2025
        if season_2025:
            return season_2025
        return _league_avg_probs()

    blended = blend_probability_vectors(sources)

    # Attach metadata about what was blended
    blended["_blend_info"] = {
        "sources": [s.get("source", "steamer" if "fg_id" in s else "unknown") for s in sources],
        "weights": [round(s.get("weight", 0), 2) for s in sources],
        "steamer_available": steamer is not None,
        "spring_pa": spring.get("pa", 0) if spring else 0,
        "regular_pa": regular.get("pa", 0) if regular else 0,
    }

    return blended


async def get_blended_pitcher_probs(player_id: int, player_name: str,
                                      team: str = "", season: int = 2025) -> dict:
    """Get blended probability vector for a pitcher."""
    await load_steamer_projections()

    from mlb_data import get_pitcher_stats
    season_2025 = await get_pitcher_stats(player_id, 2025)
    spring = await get_spring_pitcher_stats(player_id, 2026)
    regular = await get_2026_regular_stats(player_id, "pitching")
    steamer = get_steamer_pitcher(player_name, team)

    sources = compute_blend_weights(steamer, season_2025, spring, regular)

    if not sources:
        if season_2025:
            return season_2025
        return _league_avg_probs()

    blended = blend_probability_vectors(sources)
    blended["_blend_info"] = {
        "sources": [s.get("source", "steamer" if "fg_id" in s else "unknown") for s in sources],
        "weights": [round(s.get("weight", 0), 2) for s in sources],
    }

    # Carry forward metadata from best source
    if steamer:
        blended["era"] = steamer.get("era", "4.50")
        blended["whip"] = steamer.get("whip", "1.30")
        blended["name"] = steamer.get("name", player_name)
    elif season_2025:
        blended["era"] = season_2025.get("era", "4.50")
        blended["whip"] = season_2025.get("whip", "1.30")
        blended["name"] = season_2025.get("name", player_name)

    return blended


# ═══════════════════════════════════════════════════════════════
# 4. BULLPEN MODELING
# ═══════════════════════════════════════════════════════════════

# 2025 team bullpen ERA and rates (from Covers.com / Baseball Reference)
# Used to model innings after the starter is pulled
TEAM_BULLPEN_2025 = {
    # team_id: {era, whip, k_rate, bb_rate, hr_rate, ip_per_game (bullpen portion)}
    109: {"era": 4.82, "whip": 1.41, "name": "ARI"},
    110: {"era": 4.57, "whip": 1.42, "name": "BAL"},
    111: {"era": 3.41, "whip": 1.25, "name": "BOS"},
    112: {"era": 3.78, "whip": 1.23, "name": "CHC"},
    113: {"era": 3.89, "whip": 1.30, "name": "CIN"},
    114: {"era": 3.44, "whip": 1.23, "name": "CLE"},
    115: {"era": 5.18, "whip": 1.51, "name": "COL"},
    116: {"era": 4.05, "whip": 1.30, "name": "DET"},
    117: {"era": 3.70, "whip": 1.22, "name": "HOU"},
    118: {"era": 3.63, "whip": 1.27, "name": "KC"},
    119: {"era": 4.27, "whip": 1.33, "name": "LAD"},
    108: {"era": 4.86, "whip": 1.41, "name": "LAA"},
    158: {"era": 3.63, "whip": 1.24, "name": "MIL"},
    142: {"era": 4.60, "whip": 1.39, "name": "MIN"},
    146: {"era": 4.28, "whip": 1.33, "name": "MIA"},
    121: {"era": 3.93, "whip": 1.29, "name": "NYM"},
    147: {"era": 4.37, "whip": 1.32, "name": "NYY"},
    133: {"era": 4.53, "whip": 1.38, "name": "ATH"},
    143: {"era": 4.25, "whip": 1.32, "name": "PHI"},
    134: {"era": 3.83, "whip": 1.25, "name": "PIT"},
    135: {"era": 3.06, "whip": 1.15, "name": "SD"},
    137: {"era": 3.48, "whip": 1.24, "name": "SF"},
    136: {"era": 3.72, "whip": 1.28, "name": "SEA"},
    138: {"era": 3.74, "whip": 1.31, "name": "STL"},
    139: {"era": 3.81, "whip": 1.23, "name": "TB"},
    140: {"era": 3.62, "whip": 1.22, "name": "TEX"},
    141: {"era": 3.98, "whip": 1.28, "name": "TOR"},
    120: {"era": 5.59, "whip": 1.52, "name": "WSH"},
    145: {"era": 4.16, "whip": 1.38, "name": "CHW"},
    144: {"era": 4.19, "whip": 1.26, "name": "ATL"},
}


def get_bullpen_probs(team_id: int) -> dict:
    """
    Convert team bullpen ERA/WHIP into a 7-event probability vector.
    Uses sqrt-dampened ERA scaling so bad bullpens aren't wildly inflated.
    """
    bp = TEAM_BULLPEN_2025.get(team_id, {"era": 3.90, "whip": 1.28})
    era = bp["era"]
    whip = bp["whip"]

    # Dampened scaling using sqrt: prevents extreme values
    era_ratio = (era / 3.90) ** 0.5
    whip_ratio = (whip / 1.28) ** 0.5

    # Retrosheet 2025 league avg as base, slight bullpen adjustment
    base = {
        "p_1b": 0.143, "p_2b": 0.042, "p_3b": 0.003, "p_hr": 0.033,
        "p_bb": 0.088, "p_hbp": 0.011, "p_out": 0.680,
    }

    adjusted = base.copy()
    adjusted["p_1b"] *= era_ratio
    adjusted["p_2b"] *= era_ratio
    adjusted["p_hr"] *= era_ratio
    adjusted["p_bb"] *= whip_ratio

    non_out = sum(adjusted[k] for k in ["p_1b", "p_2b", "p_3b", "p_hr", "p_bb", "p_hbp"])
    adjusted["p_out"] = max(0.01, 1.0 - non_out)

    return adjusted


def estimate_starter_innings(pitcher_stats: dict, workload: dict) -> float:
    """
    Estimate how many innings the starter will pitch.
    Based on their typical workload and fatigue level.

    MLB average starter goes ~5.4 IP per start (2024-2025).
    """
    # Base from their average IP in recent starts
    avg_ip = workload.get("avg_ip_last3", 5.5)
    if avg_ip == 0 or avg_ip is None:
        avg_ip = 5.5

    # Adjust for fatigue
    fatigue = workload.get("fatigue_factor", 1.0)
    if fatigue < 0.95:
        avg_ip *= 0.90  # Tired pitchers get pulled earlier
    elif fatigue > 1.01:
        avg_ip *= 1.05  # Fresh pitchers go deeper

    # Clamp to realistic range
    return max(3.0, min(7.5, avg_ip))


def clear_proj_cache():
    global _proj_cache, _steamer_batters, _steamer_pitchers
    _proj_cache = {}
    _steamer_batters = None
    _steamer_pitchers = None
