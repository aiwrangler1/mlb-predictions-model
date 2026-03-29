"""
Log5 Engine - Team win probability and batter-vs-pitcher matchup calculations.

Based on:
- Bill James / Dallas Adams Log5 formula for team matchups
- SABR "Matchup Probabilities in Major League Baseball" multi-event extension
- Wayne Winston's Mathletics Monte Carlo approach
"""


def team_win_probability(team_a_wpct: float, team_b_wpct: float, 
                          home_advantage: float = 0.04) -> float:
    """
    Log5 formula for probability that Team A beats Team B.
    
    P(A beats B) = (pA * (1 - pB)) / (pA * (1 - pB) + pB * (1 - pA))
    
    Equivalent to: (pA - pA*pB) / (pA + pB - 2*pA*pB)
    
    Args:
        team_a_wpct: Team A's winning percentage
        team_b_wpct: Team B's winning percentage  
        home_advantage: Additional probability for home team (typically 0.04 in MLB)
    
    Returns:
        Probability that Team A wins
    """
    pA = team_a_wpct
    pB = team_b_wpct
    
    denom = pA + pB - 2 * pA * pB
    if denom == 0:
        return 0.5
    
    prob = (pA - pA * pB) / denom
    
    # Clamp to reasonable range
    return max(0.01, min(0.99, prob))


def team_win_probability_with_home(away_wpct: float, home_wpct: float,
                                     home_advantage: float = 0.04) -> tuple:
    """
    Log5 with home field advantage applied.
    
    Returns (away_win_prob, home_win_prob)
    """
    # Adjust home team's effective winning pct upward
    adj_home = min(0.99, home_wpct + home_advantage / 2)
    adj_away = max(0.01, away_wpct - home_advantage / 2)
    
    away_prob = team_win_probability(adj_away, adj_home)
    home_prob = 1.0 - away_prob
    
    return away_prob, home_prob


def matchup_probability_vector(batter_probs: dict, pitcher_probs: dict, 
                                 league_probs: dict) -> dict:
    """
    Log5 Multi-Event Extension for batter vs pitcher matchup.
    
    From SABR article Equation 13:
    P(Ei) = (xi * yi / zi) / SUM_j(xj * yj / zj)
    
    Where:
    - xi = batter's probability of outcome i
    - yi = pitcher's probability of allowing outcome i  
    - zi = league average probability of outcome i
    
    Events: 1B, 2B, 3B, HR, BB, HBP, OUT
    
    Args:
        batter_probs: dict with keys p_1b, p_2b, p_3b, p_hr, p_bb, p_hbp, p_out
        pitcher_probs: dict with same keys (from pitcher's perspective - what batters do against him)
        league_probs: dict with same keys (league averages)
    
    Returns:
        dict with adjusted probability for each outcome
    """
    outcomes = ["p_1b", "p_2b", "p_3b", "p_hr", "p_bb", "p_hbp", "p_out"]
    
    # Compute numerator for each outcome: (xi * yi) / zi
    raw = {}
    for o in outcomes:
        x = batter_probs.get(o, 0)
        y = pitcher_probs.get(o, 0)
        z = league_probs.get(o, 0.001)  # Avoid division by zero
        
        if z > 0:
            raw[o] = (x * y) / z
        else:
            raw[o] = 0
    
    # Normalize so probabilities sum to 1
    total = sum(raw.values())
    if total == 0:
        # Fallback to league average
        return {o: league_probs.get(o, 0) for o in outcomes}
    
    result = {}
    for o in outcomes:
        result[o] = raw[o] / total
    
    return result


def compute_expected_stats(matchup_probs: dict, num_pa: float = 4.0) -> dict:
    """
    Given matchup probabilities and expected plate appearances,
    compute expected stats for a batter in this game.
    
    Args:
        matchup_probs: Log5-adjusted probability vector
        num_pa: Expected plate appearances (default 4.0 for a typical game)
    
    Returns:
        dict with expected hits, HR, RBI, runs, etc.
    """
    p_hit = matchup_probs["p_1b"] + matchup_probs["p_2b"] + matchup_probs["p_3b"] + matchup_probs["p_hr"]
    p_on_base = p_hit + matchup_probs["p_bb"] + matchup_probs["p_hbp"]
    
    return {
        "expected_pa": num_pa,
        "expected_hits": num_pa * p_hit,
        "expected_1b": num_pa * matchup_probs["p_1b"],
        "expected_2b": num_pa * matchup_probs["p_2b"],
        "expected_3b": num_pa * matchup_probs["p_3b"],
        "expected_hr": num_pa * matchup_probs["p_hr"],
        "expected_bb": num_pa * matchup_probs["p_bb"],
        "expected_hbp": num_pa * matchup_probs["p_hbp"],
        "expected_obp": p_on_base,
        "expected_slg": (matchup_probs["p_1b"] + 2*matchup_probs["p_2b"] + 
                         3*matchup_probs["p_3b"] + 4*matchup_probs["p_hr"]) / max(1 - matchup_probs["p_bb"] - matchup_probs["p_hbp"], 0.001),
    }


def pitcher_quality_adjustment(pitcher_stats: dict) -> float:
    """
    Returns a quality multiplier for the pitcher.
    Above 1.0 = better than average (suppress offense)
    Below 1.0 = worse than average (boost offense)
    
    Based on ERA, WHIP, and K/BB ratio relative to league average.
    """
    if not pitcher_stats:
        return 1.0
    
    era = float(pitcher_stats.get("era", "4.50"))
    whip = float(pitcher_stats.get("whip", "1.30"))
    
    # League average ERA ~4.30, WHIP ~1.28
    era_factor = 4.30 / max(era, 1.0)  # Higher ERA = lower quality
    whip_factor = 1.28 / max(whip, 0.5)
    
    # Blend factors (ERA more important)
    quality = 0.6 * era_factor + 0.4 * whip_factor
    
    # Clamp to reasonable range
    return max(0.5, min(1.5, quality))
