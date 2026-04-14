"""
Multi-lineup portfolio construction for DFS GPP contests.
Builds diversified portfolios maximizing expected portfolio win rate.
"""
import numpy as np
from typing import Optional
from player_model import PlayerModel, load_player_pool, compute_gpp_scores
from contest_model import ContestParams, compute_geo_mean, compute_lineup_ev
from lineup_optimizer import optimize_lineup, LineupResult


def build_portfolio(
    csv_path: str,
    contest: ContestParams,
    n_lineups: int = 10,
    salary_cap: int = 50000,
    max_exposure: float = 0.7,       # max fraction of lineups a player can appear in
    min_lineup_proj: float = 80.0,   # minimum acceptable lineup projection
    run_ev_sim: bool = False,         # compute EWR per lineup (slow)
    ev_n_sims: int = 1000,
    locked_players: Optional[list] = None,
    excluded_players: Optional[list] = None,
    seed: int = 42,
) -> list:
    """
    Build a portfolio of n_lineups optimized for the given contest.
    Uses iterative exclusion to force diversity.
    """
    players = load_player_pool(csv_path)
    compute_gpp_scores(players)

    if not players:
        print("WARNING: No players loaded from CSV!")
        return []

    player_name_to_count = {}  # track exposure per player
    lineups = []
    # Base exclusions: user-specified + diversity-driven
    base_excluded = set(excluded_players or [])
    diversity_excluded = set()  # players excluded to force diversity

    # Max times any single player can appear
    max_appearances = max(1, int(n_lineups * max_exposure))

    for attempt in range(n_lineups * 6):  # allow extra attempts for diversity
        if len(lineups) >= n_lineups:
            break

        # Combine: user exclusions + diversity exclusions + exposure-capped players
        exposure_excluded = {
            name for name, count in player_name_to_count.items()
            if count >= max_appearances
        }
        current_excluded = base_excluded | diversity_excluded | exposure_excluded

        lineup = optimize_lineup(
            players, contest, salary_cap,
            locked_players=locked_players,
            excluded_players=list(current_excluded),
        )

        if lineup is None:
            # If we can't find more lineups, stop
            if len(lineups) >= max(1, n_lineups // 2):
                break
            # Try relaxing diversity exclusions
            if diversity_excluded:
                # Remove the most recently added diversity exclusion
                diversity_excluded = set(list(diversity_excluded)[:-1])
            else:
                break
            continue

        if lineup.ss_proj < min_lineup_proj:
            # Try relaxing min proj threshold
            if min_lineup_proj > 50.0:
                min_lineup_proj = min_lineup_proj * 0.85
            # Still add this lineup rather than break

        # Check for near-duplicates (>80% player overlap)
        is_dup = False
        lineup_names = {p.name for p in lineup.players}
        for prev in lineups:
            prev_names = {p.name for p in prev.players}
            overlap = len(lineup_names & prev_names) / len(lineup_names)
            if overlap > 0.8:
                is_dup = True
                # Permanently exclude the highest-ceiling overlap player to force next solution to differ
                overlap_players = [p for p in lineup.players if p.name in prev_names]
                if overlap_players:
                    top_overlap = max(overlap_players, key=lambda p: p.dk_p99)
                    diversity_excluded.add(top_overlap.name)
                break

        if is_dup:
            continue

        lineups.append(lineup)
        for p in lineup.players:
            player_name_to_count[p.name] = player_name_to_count.get(p.name, 0) + 1

    # Run EV sim if requested
    if run_ev_sim and players and lineups:
        for lineup in lineups:
            ev_stats = compute_lineup_ev(
                lineup.players, contest, players, n_sims=ev_n_sims, seed=seed
            )
            lineup._ev_stats = ev_stats

    return lineups


def print_portfolio_report(lineups: list, contest: ContestParams):
    """Print a comprehensive portfolio summary."""
    print(f"\n{'='*90}")
    print(f"  PORTFOLIO REPORT: {contest.name}")
    print(f"  {contest.n_entries} entries | {contest.top_heavy_ratio:.0%} to 1st | λ={contest.lambda_ownership:.3f}")
    print(f"  Target GeoMean ≤ {contest.target_geo_mean:.4f}")
    print(f"{'='*90}")

    if not lineups:
        print("  No lineups generated!")
        return

    print(f"\n{'#':<3} {'Salary':>8} {'Proj':>6} {'Ceil':>6} {'GeoMean':>9} {'ExpDupes':>9} {'Players (ownership%)'}")
    print("-" * 90)

    all_players = {}
    for i, lu in enumerate(lineups):
        gm_ok = "✓" if lu.geo_mean <= contest.target_geo_mean * 1.05 else "⚠"
        players_str = lu.summary_line().split('|')
        players_display = players_str[1].strip() if len(players_str) > 1 else ''
        print(
            f"{i+1:<3} ${lu.salary:>7,} {lu.ss_proj:>6.1f} {lu.ceiling:>6.0f} "
            f"{lu.geo_mean:>9.4f}{gm_ok} {lu.expected_dupes:>9.2f}  "
            f"{players_display}"
        )

        for p in lu.players:
            if p.name not in all_players:
                all_players[p.name] = {'count': 0, 'player': p}
            all_players[p.name]['count'] += 1

    # Exposure report
    print(f"\n{'='*60}")
    print(f"  PLAYER EXPOSURE (out of {len(lineups)} lineups)")
    print(f"{'='*60}")
    print(f"  {'Player':<25} {'Team':<5} {'Own%':>6} {'Exp%':>6} {'GppScore':>9}")
    for name, data in sorted(all_players.items(), key=lambda x: -x[1]['count']):
        p = data['player']
        exp_pct = data['count'] / len(lineups) * 100
        print(f"  {p.name:<25} {p.team:<5} {p.adj_own:>5.1f}% {exp_pct:>5.0f}% {p.gpp_score:>9.2f}")
