"""
ILP-based DFS lineup optimizer.
Uses PuLP for integer programming with ownership penalty and GeoMean constraints.
"""
import numpy as np
import pulp
from dataclasses import dataclass
from typing import Optional
from player_model import PlayerModel, compute_gpp_scores
from contest_model import ContestParams, compute_geo_mean

DK_CLASSIC_SLOTS = {
    'P': 2, 'C': 1, '1B': 1, '2B': 1, '3B': 1, 'SS': 1, 'OF': 3
}
MAX_PER_TEAM = 5  # DK MLB Classic rule


@dataclass
class LineupResult:
    players: list
    assignments: dict   # name -> assigned position slot
    salary: int
    ss_proj: float
    geo_mean: float
    expected_dupes: float
    ceiling: float    # sum of dk_p99
    leverage_sum: float

    def format_dk_slots(self) -> list:
        """Return 10 player strings in DK upload format, ordered by roster slot."""
        slot_order = ['P', 'P', 'C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF']
        name_to_player = {p.name: p for p in self.players}

        # Group players by their assigned slot
        slot_players = {}
        for name, slot in self.assignments.items():
            if slot not in slot_players:
                slot_players[slot] = []
            slot_players[slot].append(name)

        result = []
        p_count, of_count = 0, 0
        for slot in slot_order:
            if slot == 'P':
                players_in_slot = slot_players.get('P', [])
                if p_count < len(players_in_slot):
                    name = players_in_slot[p_count]
                    result.append(name_to_player[name].format_dk())
                    p_count += 1
                else:
                    result.append('')
            elif slot == 'OF':
                players_in_slot = slot_players.get('OF', [])
                if of_count < len(players_in_slot):
                    name = players_in_slot[of_count]
                    result.append(name_to_player[name].format_dk())
                    of_count += 1
                else:
                    result.append('')
            else:
                players_in_slot = slot_players.get(slot, [])
                result.append(name_to_player[players_in_slot[0]].format_dk() if players_in_slot else '')

        return result

    def summary_line(self) -> str:
        names = ', '.join(
            f"{p.name}({p.adj_own:.0f}%)"
            for p in sorted(self.players, key=lambda x: -x.dk_p99)
        )
        return (
            f"Sal:${self.salary:,} Proj:{self.ss_proj:.1f} "
            f"Ceil:{self.ceiling:.0f} GeoMean:{self.geo_mean:.4f} "
            f"ExpDupes:{self.expected_dupes:.2f} | {names}"
        )


def optimize_lineup(
    players: list,
    contest: ContestParams,
    salary_cap: int = 50000,
    locked_players: Optional[list] = None,
    excluded_players: Optional[list] = None,
    min_teams: int = 2,
    enforce_geo_mean: bool = True,
) -> Optional[LineupResult]:
    """
    Build a single optimal DFS lineup using Integer Linear Programming.

    Objective: maximize (ceiling score) - λ × (ownership log sum)
    Subject to: salary, position, team, GeoMean constraints.
    """
    locked = set(locked_players or [])
    excluded = set(excluded_players or [])

    # Filter to usable players
    pool = [p for p in players if p.name not in excluded and p.salary > 0]

    slots_needed = DK_CLASSIC_SLOTS  # {'P': 2, 'C': 1, ...}
    n = len(pool)
    player_idx = {p.name: i for i, p in enumerate(pool)}

    # Create LP problem
    prob = pulp.LpProblem("DFS_Lineup", pulp.LpMaximize)

    # y_i: binary, player i is selected
    y = [pulp.LpVariable(f"y_{i}", cat='Binary') for i in range(n)]

    # z[i][slot]: binary, player i fills position slot
    z = {}
    for i, p in enumerate(pool):
        for slot in slots_needed:
            if p.is_eligible(slot):
                z[(i, slot)] = pulp.LpVariable(f"z_{i}_{slot}", cat='Binary')

    # === OBJECTIVE ===
    # Maximize: Σ(y_i × ceiling_i) + λ × Σ(y_i × log_own_i)
    # Note: log_own is negative (since own < 1), so λ > 0 penalizes high ownership
    lambda_own = contest.lambda_ownership

    obj = pulp.lpSum([
        y[i] * (pool[i].dk_p99 + lambda_own * pool[i].log_own)
        for i in range(n)
    ])
    prob += obj

    # === CONSTRAINTS ===

    # 1. Salary cap
    prob += pulp.lpSum([y[i] * pool[i].salary for i in range(n)]) <= salary_cap

    # 2. Each player assigned to exactly one slot (if selected), none if not selected
    for i in range(n):
        eligible_slots = [s for s in slots_needed if (i, s) in z]
        if eligible_slots:
            prob += pulp.lpSum([z[(i, s)] for s in eligible_slots]) == y[i]
        else:
            prob += y[i] == 0

    # 3. Exactly the right number of each slot filled
    for slot, count in slots_needed.items():
        eligible_players = [i for i in range(n) if (i, slot) in z]
        prob += pulp.lpSum([z[(i, slot)] for i in eligible_players]) == count

    # 4. Total players = 10
    prob += pulp.lpSum(y) == sum(slots_needed.values())

    # 5. Max 5 hitters from one team (pitchers excluded from this constraint)
    teams = set(p.team for p in pool)
    for team in teams:
        team_hitters = [
            i for i, p in enumerate(pool)
            if p.team == team and 'P' not in p.positions
        ]
        if team_hitters:
            prob += pulp.lpSum([y[i] for i in team_hitters]) <= MAX_PER_TEAM

    # 6. Locked players
    for name in locked:
        if name in player_idx:
            prob += y[player_idx[name]] == 1

    # 7. GeoMean constraint (optional hard constraint)
    # Σ(y_i × log_own_i) / n ≤ log(geo_mean_target)
    if enforce_geo_mean and contest.target_geo_mean > 0:
        geo_mean_limit = np.log(contest.target_geo_mean)
        n_lineup = sum(slots_needed.values())
        prob += (
            pulp.lpSum([y[i] * pool[i].log_own for i in range(n)]) / n_lineup
            <= geo_mean_limit
        )

    # === SOLVE ===
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=30)
    status = prob.solve(solver)

    if pulp.LpStatus[prob.status] != 'Optimal':
        # Try without GeoMean constraint if it caused infeasibility
        if enforce_geo_mean:
            return optimize_lineup(
                players, contest, salary_cap, locked_players,
                excluded_players, min_teams, enforce_geo_mean=False
            )
        return None

    # Extract solution
    selected = []
    assignments = {}

    # Build slot_to_players for tracking duplicate slot assignments
    slot_fill_count = {slot: 0 for slot in slots_needed}

    for i, p in enumerate(pool):
        if pulp.value(y[i]) is not None and pulp.value(y[i]) > 0.5:
            selected.append(p)
            # Find assigned slot
            for slot in slots_needed:
                if (i, slot) in z and pulp.value(z[(i, slot)]) is not None and pulp.value(z[(i, slot)]) > 0.5:
                    assignments[p.name] = slot
                    slot_fill_count[slot] = slot_fill_count.get(slot, 0) + 1
                    break

    total_needed = sum(slots_needed.values())
    if len(selected) != total_needed:
        # Try again without geo mean constraint
        if enforce_geo_mean:
            return optimize_lineup(
                players, contest, salary_cap, locked_players,
                excluded_players, min_teams, enforce_geo_mean=False
            )
        return None

    total_salary = sum(p.salary for p in selected)
    total_proj = sum(p.ss_proj for p in selected)
    total_ceil = sum(p.dk_p99 for p in selected)
    geo = compute_geo_mean(selected)
    exp_dupes = contest.expected_dupes(geo)

    return LineupResult(
        players=selected,
        assignments=assignments,
        salary=total_salary,
        ss_proj=round(total_proj, 2),
        geo_mean=round(geo, 4),
        expected_dupes=round(exp_dupes, 2),
        ceiling=round(total_ceil, 1),
        leverage_sum=round(sum(p.leverage_score for p in selected), 2),
    )
