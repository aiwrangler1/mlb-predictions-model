"""
Monte Carlo Game Simulator - Mathletics style plate-appearance-by-plate-appearance simulation.

Simulates full 9-inning baseball games with:
- Baserunner tracking
- Run scoring
- Per-player stat accumulation
- DFS point calculation
"""
import numpy as np
from typing import Optional
from log5_engine import matchup_probability_vector, team_win_probability_with_home
from mlb_data import get_league_averages



# DraftKings Classic scoring
DK_SCORING = {
    "1b": 3, "2b": 5, "3b": 8, "hr": 10,
    "rbi": 2, "run": 2, "bb": 2, "hbp": 2,
    "sb": 5,
    # Pitching
    "pitcher_win": 4, "er": -2, "k": 2,
    "ip_bonus": 2.5,  # >= 6 IP
    "cg_bonus": 2.5, "cgso_bonus": 2.5, "no_hitter": 5,
}


def simulate_plate_appearance(matchup_probs: dict, rng: np.random.Generator) -> str:
    """
    Simulate a single plate appearance outcome using Log5-adjusted probabilities.
    Returns one of: '1b', '2b', '3b', 'hr', 'bb', 'hbp', 'out'
    """
    outcomes = ['1b', '2b', '3b', 'hr', 'bb', 'hbp', 'out']
    probs = [
        matchup_probs.get("p_1b", 0),
        matchup_probs.get("p_2b", 0),
        matchup_probs.get("p_3b", 0),
        matchup_probs.get("p_hr", 0),
        matchup_probs.get("p_bb", 0),
        matchup_probs.get("p_hbp", 0),
        matchup_probs.get("p_out", 0.675),
    ]
    total = sum(probs)
    if total <= 0:
        probs = [0.143, 0.042, 0.003, 0.031, 0.084, 0.011, 0.686]
        total = sum(probs)
    probs = [p / total for p in probs]
    r = rng.random()
    cumulative = 0
    for outcome, p in zip(outcomes, probs):
        cumulative += p
        if r < cumulative:
            return outcome
    return 'out'


def advance_runners(bases: list, outcome: str, rng: np.random.Generator) -> tuple:
    """
    Advance baserunners based on plate appearance outcome.
    
    bases: [1st, 2nd, 3rd] - 0 or 1 indicating runner presence
    
    Returns: (new_bases, runs_scored)
    
    Runner advancement rules (simplified but realistic):
    - 1B: runners advance 1 base; runner on 2nd scores ~60% of time, 
           runner on 3rd always scores
    - 2B: runners advance 2 bases; runners on 2nd and 3rd score, 
          runner on 1st scores ~40% of time  
    - 3B: all runners score
    - HR: all runners + batter score
    - BB/HBP: force advancement only
    - OUT: runners hold (simplified - no sac flies/GIDPs for speed)
    """
    runs = 0
    new_bases = [0, 0, 0]
    
    if outcome == 'hr':
        # Everyone scores, including batter
        runs = 1 + sum(bases)
        new_bases = [0, 0, 0]
        
    elif outcome == '3b':
        # All runners score, batter to 3rd
        runs = sum(bases)
        new_bases = [0, 0, 1]
        
    elif outcome == '2b':
        # Runner on 3rd scores
        runs += bases[2]
        # Runner on 2nd scores
        runs += bases[1]
        # Runner on 1st: scores ~40%, else to 3rd
        if bases[0]:
            if rng.random() < 0.40:
                runs += 1
            else:
                new_bases[2] = 1
        # Batter to 2nd
        new_bases[1] = 1
        
    elif outcome == '1b':
        # Runner on 3rd scores
        runs += bases[2]
        # Runner on 2nd: scores ~60%, else to 3rd
        if bases[1]:
            if rng.random() < 0.60:
                runs += 1
            else:
                new_bases[2] = 1
        # Runner on 1st to 2nd (or 3rd sometimes)
        if bases[0]:
            if new_bases[2] == 0 and rng.random() < 0.25:
                new_bases[2] = 1
            else:
                new_bases[1] = 1 if new_bases[1] == 0 else new_bases[1]
                if new_bases[1] == 0:
                    new_bases[1] = 1
        # Batter to 1st
        new_bases[0] = 1
        
    elif outcome in ('bb', 'hbp'):
        # Force advancement only
        if bases[0] and bases[1] and bases[2]:
            runs += 1  # Bases loaded walk
            new_bases = [1, 1, 1]
        elif bases[0] and bases[1]:
            new_bases = [1, 1, 1]
        elif bases[0]:
            new_bases = [1, 1, bases[2]]
        else:
            new_bases = [1, bases[1], bases[2]]
            
    elif outcome == 'out':
        # Check for sacrifice fly with runner on 3rd and less than 2 outs
        # This is handled at the inning level
        new_bases = bases.copy()
    
    return new_bases, runs


def simulate_half_inning(lineup: list, lineup_idx: int, pitcher_matchups: list,
                          rng: np.random.Generator, player_stats: dict) -> tuple:
    """
    Simulate a half inning for one team.
    
    Args:
        lineup: list of 9 batter dicts with matchup probabilities
        lineup_idx: current position in batting order (0-8)
        pitcher_matchups: list of 9 matchup probability vectors
        rng: random number generator
        player_stats: dict to accumulate stats {player_id: {stat: count}}
    
    Returns:
        (runs_scored, new_lineup_idx)
    """
    outs = 0
    bases = [0, 0, 0]
    runs = 0
    
    while outs < 3:
        batter_idx = lineup_idx % 9
        batter = lineup[batter_idx]
        matchup = pitcher_matchups[batter_idx]
        
        outcome = simulate_plate_appearance(matchup, rng)
        pid = batter.get("player_id", batter_idx)
        
        if pid not in player_stats:
            player_stats[pid] = {
                "name": batter.get("name", f"Batter {batter_idx+1}"),
                "pa": 0, "ab": 0, "h": 0, "1b": 0, "2b": 0, "3b": 0,
                "hr": 0, "rbi": 0, "r": 0, "bb": 0, "hbp": 0, "so": 0,
                "sb": 0, "dk_pts": 0
            }
        
        ps = player_stats[pid]
        ps["pa"] += 1
        
        if outcome == 'out':
            outs += 1
            ps["ab"] += 1
            # Sac fly: runner on 3rd scores with < 2 outs ~30% of the time
            if bases[2] == 1 and outs < 3 and rng.random() < 0.30:
                bases[2] = 0
                runs += 1
                ps["rbi"] += 1
                ps["dk_pts"] += DK_SCORING["rbi"]
        else:
            old_bases = bases.copy()
            new_bases, pa_runs = advance_runners(bases, outcome, rng)
            bases = new_bases
            runs += pa_runs
            
            if outcome in ('1b', '2b', '3b', 'hr'):
                ps["ab"] += 1
                ps["h"] += 1
                ps[outcome] += 1
                ps["dk_pts"] += DK_SCORING[outcome]
                
                # RBI: runs scored on this play
                ps["rbi"] += pa_runs
                ps["dk_pts"] += pa_runs * DK_SCORING["rbi"]
                
                # The batter scores on HR
                if outcome == 'hr':
                    ps["r"] += 1
                    ps["dk_pts"] += DK_SCORING["run"]
                    
            elif outcome == 'bb':
                ps["bb"] += 1
                ps["dk_pts"] += DK_SCORING["bb"]
                ps["rbi"] += pa_runs
                ps["dk_pts"] += pa_runs * DK_SCORING["rbi"]
                
            elif outcome == 'hbp':
                ps["hbp"] += 1
                ps["dk_pts"] += DK_SCORING["hbp"]
                ps["rbi"] += pa_runs
                ps["dk_pts"] += pa_runs * DK_SCORING["rbi"]
            
            # Stolen base attempt (simplified)
            if outcome in ('1b', 'bb', 'hbp') and outs < 2:
                sb_rate = batter.get("p_sb_attempt", 0.05)
                if rng.random() < sb_rate * 0.7:  # 70% success rate
                    ps["sb"] += 1
                    ps["dk_pts"] += DK_SCORING["sb"]
        
        lineup_idx += 1
    
    # Score runs for runners who were on base
    # (they need to be attributed to the batters who reached)
    
    return runs, lineup_idx


def simulate_game(away_lineup: list, home_lineup: list,
                   away_matchups: list, home_matchups: list,
                   away_pitcher: dict, home_pitcher: dict,
                   rng: np.random.Generator,
                   away_bullpen_matchups: list = None,
                   home_bullpen_matchups: list = None,
                   away_starter_ip: float = 5.5,
                   home_starter_ip: float = 5.5) -> dict:
    """
    Simulate a complete baseball game with starter/bullpen transition.
    
    Starters pitch for ~starter_ip innings (with variance), then
    bullpen takes over with different matchup probabilities.
    
    Returns dict with:
    - away_score, home_score
    - player_stats for each player
    - pitcher_stats
    """
    away_stats = {}
    home_stats = {}
    away_score = 0
    home_score = 0
    away_idx = 0
    home_idx = 0
    
    innings = 9
    
    # Add variance to starter IP (±1 inning)
    away_sp_exit = max(3, min(8, away_starter_ip + rng.normal(0, 0.8)))
    home_sp_exit = max(3, min(8, home_starter_ip + rng.normal(0, 0.8)))
    
    # Use bullpen matchups if not provided, fall back to starter matchups
    if away_bullpen_matchups is None:
        away_bullpen_matchups = away_matchups
    if home_bullpen_matchups is None:
        home_bullpen_matchups = home_matchups
    
    for inning in range(1, innings + 1):
        # Top of inning: away team bats
        # Choose matchup set based on whether home starter is still in
        if inning <= home_sp_exit:
            matchups = away_matchups  # vs home starter
        else:
            matchups = away_bullpen_matchups  # vs home bullpen
        
        runs, away_idx = simulate_half_inning(
            away_lineup, away_idx, matchups, rng, away_stats
        )
        away_score += runs
        
        # Bottom of inning: home team bats
        if inning == 9 and home_score > away_score:
            break
        
        if inning <= away_sp_exit:
            matchups = home_matchups  # vs away starter
        else:
            matchups = home_bullpen_matchups  # vs away bullpen
        
        runs, home_idx = simulate_half_inning(
            home_lineup, home_idx, matchups, rng, home_stats
        )
        home_score += runs
        
        if inning >= 9 and home_score > away_score:
            break
    
    # Extra innings (bullpen always)
    extra = 0
    while away_score == home_score and extra < 5:
        extra += 1
        runs, away_idx = simulate_half_inning(
            away_lineup, away_idx, away_bullpen_matchups, rng, away_stats
        )
        away_score += runs
        
        runs, home_idx = simulate_half_inning(
            home_lineup, home_idx, home_bullpen_matchups, rng, home_stats
        )
        home_score += runs
    
    # Pitcher stats
    away_pitcher_stats = {
        "name": away_pitcher.get("name", "Away SP"),
        "ip": 0, "h": 0, "er": 0, "k": 0, "bb": 0, "win": 0,
        "dk_pts": 0
    }
    home_pitcher_stats = {
        "name": home_pitcher.get("name", "Home SP"),
        "ip": 0, "h": 0, "er": 0, "k": 0, "bb": 0, "win": 0,
        "dk_pts": 0
    }
    
    for pid, ps in home_stats.items():
        away_pitcher_stats["h"] += ps["h"]
        away_pitcher_stats["bb"] += ps["bb"]
        away_pitcher_stats["k"] += ps["pa"] - ps["h"] - ps["bb"] - ps["hbp"]
    away_pitcher_stats["er"] = home_score
    
    for pid, ps in away_stats.items():
        home_pitcher_stats["h"] += ps["h"]
        home_pitcher_stats["bb"] += ps["bb"]
        home_pitcher_stats["k"] += ps["pa"] - ps["h"] - ps["bb"] - ps["hbp"]
    home_pitcher_stats["er"] = away_score
    
    # Starter IP based on simulated exit point
    total_innings = innings + extra
    away_pitcher_stats["ip"] = min(away_sp_exit, total_innings)
    home_pitcher_stats["ip"] = min(home_sp_exit, total_innings)
    
    # K count: ~23% of outs are Ks for starters, weight by IP proportion
    for ps, sp_exit in [(away_pitcher_stats, away_sp_exit), (home_pitcher_stats, home_sp_exit)]:
        sp_ratio = min(1.0, sp_exit / total_innings) if total_innings > 0 else 0.6
        ps["k"] = int(ps["k"] * 0.23 * sp_ratio * 1.5)  # Starters K more than bullpen average
    
    # Pitcher DK scoring
    for ps, opp_score in [(away_pitcher_stats, home_score), (home_pitcher_stats, away_score)]:
        ps["dk_pts"] += ps["k"] * DK_SCORING["k"]
        # Only charge starter with their portion of earned runs
        sp_ip = ps["ip"]
        total_ip = total_innings if total_innings > 0 else 9
        er_share = opp_score * (sp_ip / total_ip)
        ps["er"] = int(er_share)
        ps["dk_pts"] += ps["er"] * DK_SCORING["er"]
        if sp_ip >= 6:
            ps["dk_pts"] += DK_SCORING["ip_bonus"]
    
    # Win attribution
    if away_score > home_score:
        away_pitcher_stats["win"] = 1
        away_pitcher_stats["dk_pts"] += DK_SCORING["pitcher_win"]
    elif home_score > away_score:
        home_pitcher_stats["win"] = 1
        home_pitcher_stats["dk_pts"] += DK_SCORING["pitcher_win"]
    
    # Assign runs scored to batters who reached base
    # (simplified - distribute proportionally based on times on base)
    for stats_dict, total_runs in [(away_stats, away_score), (home_stats, home_score)]:
        total_on_base = sum(ps["h"] + ps["bb"] + ps["hbp"] for ps in stats_dict.values())
        if total_on_base > 0:
            remaining_runs = total_runs - sum(ps["r"] for ps in stats_dict.values())
            if remaining_runs > 0:
                for pid, ps in stats_dict.items():
                    on_base = ps["h"] + ps["bb"] + ps["hbp"]
                    extra_runs = int(remaining_runs * on_base / total_on_base)
                    ps["r"] += extra_runs
                    ps["dk_pts"] += extra_runs * DK_SCORING["run"]
    
    return {
        "away_score": away_score,
        "home_score": home_score,
        "away_player_stats": away_stats,
        "home_player_stats": home_stats,
        "away_pitcher_stats": away_pitcher_stats,
        "home_pitcher_stats": home_pitcher_stats,
        "innings": total_innings,
    }


def run_monte_carlo(away_lineup: list, home_lineup: list,
                     away_matchups: list, home_matchups: list,
                     away_pitcher: dict, home_pitcher: dict,
                     n_sims: int = 5000, seed: int = 42,
                     away_bullpen_matchups: list = None,
                     home_bullpen_matchups: list = None,
                     away_starter_ip: float = 5.5,
                     home_starter_ip: float = 5.5) -> dict:
    """
    Run N Monte Carlo simulations of a game and aggregate results.
    
    Returns comprehensive simulation results including:
    - Win probabilities
    - Run distributions
    - Player projection distributions
    - DFS point projections with percentiles
    """
    rng = np.random.default_rng(seed)
    
    away_wins = 0.0
    home_wins = 0.0
    ties = 0
    total_runs = []
    away_runs_dist = []
    home_runs_dist = []
    
    # Per-player accumulation
    all_away_stats = {}
    all_home_stats = {}
    
    for sim in range(n_sims):
        result = simulate_game(
            away_lineup, home_lineup,
            away_matchups, home_matchups,
            away_pitcher, home_pitcher,
            rng,
            away_bullpen_matchups=away_bullpen_matchups,
            home_bullpen_matchups=home_bullpen_matchups,
            away_starter_ip=away_starter_ip,
            home_starter_ip=home_starter_ip,
        )
        
        if result["away_score"] > result["home_score"]:
            away_wins += 1.0
        elif result["home_score"] > result["away_score"]:
            home_wins += 1.0
        else:
            # The sim caps extra innings, so unresolved ties should not be
            # credited as home wins. Split the outcome evenly instead.
            ties += 1
            away_wins += 0.5
            home_wins += 0.5
        
        away_runs_dist.append(result["away_score"])
        home_runs_dist.append(result["home_score"])
        total_runs.append(result["away_score"] + result["home_score"])
        
        # Accumulate per-player stats across sims
        for pid, ps in result["away_player_stats"].items():
            if pid not in all_away_stats:
                all_away_stats[pid] = {"name": ps["name"], "sims": []}
            all_away_stats[pid]["sims"].append(ps.copy())
        
        for pid, ps in result["home_player_stats"].items():
            if pid not in all_home_stats:
                all_home_stats[pid] = {"name": ps["name"], "sims": []}
            all_home_stats[pid]["sims"].append(ps.copy())
    
    # Compute player percentiles
    def compute_player_projections(all_stats):
        projections = []
        for pid, data in all_stats.items():
            sims = data["sims"]
            dk_pts = [s["dk_pts"] for s in sims]
            hits = [s["h"] for s in sims]
            hrs = [s["hr"] for s in sims]
            rbis = [s["rbi"] for s in sims]
            runs = [s["r"] for s in sims]
            bbs = [s["bb"] for s in sims]
            sbs = [s["sb"] for s in sims]
            
            projections.append({
                "player_id": pid,
                "name": data["name"],
                "dk_median": float(np.median(dk_pts)),
                "dk_mean": float(np.mean(dk_pts)),
                "dk_p10": float(np.percentile(dk_pts, 10)),
                "dk_p25": float(np.percentile(dk_pts, 25)),
                "dk_p50": float(np.percentile(dk_pts, 50)),
                "dk_p75": float(np.percentile(dk_pts, 75)),
                "dk_p85": float(np.percentile(dk_pts, 85)),
                "dk_p95": float(np.percentile(dk_pts, 95)),
                "dk_p90": float(np.percentile(dk_pts, 90)),
                "dk_p99": float(np.percentile(dk_pts, 99)),
                "dk_std": float(np.std(dk_pts)),
                "avg_hits": float(np.mean(hits)),
                "avg_hr": float(np.mean(hrs)),
                "avg_rbi": float(np.mean(rbis)),
                "avg_runs": float(np.mean(runs)),
                "avg_bb": float(np.mean(bbs)),
                "avg_sb": float(np.mean(sbs)),
                "hit_rate": float(np.mean([h > 0 for h in hits]) if hits else 0),
                "hr_rate": float(np.mean([h > 0 for h in hrs]) if hrs else 0),
                "multi_hit_rate": float(np.mean([h >= 2 for h in hits]) if hits else 0),
            })
        
        projections.sort(key=lambda x: x["dk_median"], reverse=True)
        return projections
    
    away_projections = compute_player_projections(all_away_stats)
    home_projections = compute_player_projections(all_home_stats)
    
    # Run distribution histogram
    away_runs_arr = np.array(away_runs_dist)
    home_runs_arr = np.array(home_runs_dist)
    total_runs_arr = np.array(total_runs)
    
    return {
        "n_sims": n_sims,
        "away_win_pct": away_wins / n_sims,
        "home_win_pct": home_wins / n_sims,
        "tie_pct": ties / n_sims,
        "away_runs_mean": float(np.mean(away_runs_arr)),
        "home_runs_mean": float(np.mean(home_runs_arr)),
        "total_runs_mean": float(np.mean(total_runs_arr)),
        "away_runs_median": float(np.median(away_runs_arr)),
        "home_runs_median": float(np.median(home_runs_arr)),
        "total_runs_median": float(np.median(total_runs_arr)),
        "away_runs_std": float(np.std(away_runs_arr)),
        "home_runs_std": float(np.std(home_runs_arr)),
        # Distribution data for charts
        "away_runs_dist": {int(k): int(v) for k, v in zip(*np.unique(away_runs_arr, return_counts=True))},
        "home_runs_dist": {int(k): int(v) for k, v in zip(*np.unique(home_runs_arr, return_counts=True))},
        "total_runs_dist": {int(k): int(v) for k, v in zip(*np.unique(total_runs_arr, return_counts=True))},
        # Player projections
        "away_projections": away_projections,
        "home_projections": home_projections,
    }
