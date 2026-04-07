"""
Showdown DFS lineup optimizer for DraftKings MLB showdown slates.

This reuses the same distribution engine as the classic optimizer, but it
builds lineups in CPT/FLEX format and keeps the same human player out of both
captain and flex slots.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

from dfs_optimizer import ContestSpec, Player, geo_mean, safe_float, sample_player_draws, weighted_choice
from outcome_distribution import empirical_projection_profile, first_present_float
from projection_overlay import load_projection_rows


SALARY_CAP = 50000
SHOWDOWN_CAPTAIN_MULTIPLIER = 1.5
SHOWDOWN_FLEX_SLOTS = 5


def first_positive(row: dict, keys: Sequence[str], default: float = 0.0) -> float:
    for key in keys:
        value = safe_float(row.get(key), 0.0)
        if value > 0:
            return value
    return default


def normalize_key(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


def build_projection_lookup(path: str) -> Dict[str, dict]:
    rows = load_projection_rows(path)

    lookup: Dict[str, dict] = {}
    for row in rows:
        team = normalize_key(row.get("Team") or row.get("team") or row.get("TeamAbbrev") or "")
        for key_field in ("player_id", "DFS ID", "DFS ID ", "ID", "Name"):
            raw = row.get(key_field)
            if not raw:
                continue
            key = normalize_key(raw)
            if key and key not in lookup:
                lookup[key] = row
            if team:
                team_key = normalize_key(f"{team}:{raw}")
                if team_key and team_key not in lookup:
                    lookup[team_key] = row
    return lookup


def projection_row_for_player(group: Sequence[dict], lookup: Optional[Dict[str, dict]]) -> Optional[dict]:
    if not lookup:
        return None

    candidates = []
    for row in group:
        team = normalize_key(row.get("Team") or row.get("team") or row.get("TeamAbbrev") or "")
        for key_field in ("player_id", "DFS ID", "DFS ID ", "ID", "Name"):
            raw = row.get(key_field)
            if raw:
                candidates.append(normalize_key(raw))
                if team:
                    candidates.append(normalize_key(f"{team}:{raw}"))
    for key in candidates:
        if key in lookup:
            return lookup[key]
    return None


@dataclass(frozen=True)
class ShowdownPlayer:
    name: str
    team: str
    opp: str
    base: Player
    flex_id: str
    captain_id: str
    flex_salary: int
    captain_salary: int
    flex_projection: float
    captain_projection: float
    captain_p95: float

    @property
    def own(self) -> float:
        return self.base.own

    @property
    def base_p95(self) -> float:
        return self.base.p95


@dataclass(frozen=True)
class ShowdownLineup:
    captain: ShowdownPlayer
    flex: Tuple[ShowdownPlayer, ...]

    @property
    def salary(self) -> int:
        return self.captain.captain_salary + sum(p.flex_salary for p in self.flex)

    @property
    def players(self) -> Tuple[ShowdownPlayer, ...]:
        return (self.captain,) + self.flex

    def player_ids(self) -> Tuple[str, ...]:
        return (self.captain.captain_id,) + tuple(p.flex_id for p in self.flex)

    def human_names(self) -> Tuple[str, ...]:
        return (self.captain.name,) + tuple(p.name for p in self.flex)

    def teams(self) -> Counter:
        return Counter(p.team for p in self.players)

    def geo_own(self) -> float:
        values = [max(self.captain.own * SHOWDOWN_CAPTAIN_MULTIPLIER / 100.0, 1e-9)]
        values.extend(max(p.own / 100.0, 1e-9) for p in self.flex)
        return geo_mean(values)


@dataclass
class ShowdownResult:
    lineup: ShowdownLineup
    mean_score: float
    ceiling_score: float
    ownership_geo_mean: float
    duplication_index: float
    raw_win_rate: float
    de_duped_win_rate: float
    expected_payout: float
    expected_value: float
    cash_rate: float
    percentile_90: float
    percentile_95: float
    expected_rank: float


def load_showdown_players(path: str, projections_path: Optional[str] = None) -> List[ShowdownPlayer]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    projection_lookup = build_projection_lookup(projections_path) if projections_path else None
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        if row.get("Name"):
            grouped[row["Name"]].append(row)

    players: List[ShowdownPlayer] = []
    for name, group in grouped.items():
        ordered = sorted(group, key=lambda r: safe_float(r.get("Salary"), 0.0))
        flex_row = ordered[0]
        captain_row = ordered[-1]
        source_row = projection_row_for_player(group, projection_lookup) or flex_row
        profile = empirical_projection_profile(source_row, fallback_mean=0.0)
        flex_mean = profile["mean"]
        if flex_mean <= 0:
            flex_mean = first_positive(
                source_row,
                ["dk_points", "My Proj", "SS Proj", "Live Proj", "AvgPointsPerGame", "Actual"],
                default=safe_float(source_row.get("AvgPointsPerGame"), safe_float(source_row.get("Actual"), 0.0)),
            )
        captain_mean = flex_mean * SHOWDOWN_CAPTAIN_MULTIPLIER
        captain_p95 = max(profile["p95"] * SHOWDOWN_CAPTAIN_MULTIPLIER, captain_mean)
        team = (
            source_row.get("Team")
            or source_row.get("team")
            or source_row.get("TeamAbbrev")
            or flex_row.get("Team")
            or flex_row.get("team")
            or flex_row.get("TeamAbbrev")
            or ""
        )
        opp = source_row.get("Opp") or source_row.get("opp") or flex_row.get("Opp") or flex_row.get("opp") or ""

        base = Player(
            player_id=str(flex_row.get("DFS ID") or flex_row.get("ID") or flex_row.get("Name")),
            name=name,
            positions=tuple(),
            team=team,
            opp=opp,
            salary=int(safe_float(flex_row.get("Salary"), 0.0)),
            mean=flex_mean,
            p10=profile["p10"],
            p25=profile["p25"],
            p50=profile["p50"],
            p75=profile["p75"],
            p85=profile["p85"],
            p90=profile["p90"],
            p95=profile["p95"],
            p99=profile["p99"],
            own=first_present_float(source_row, ["Adj Own"], default=first_present_float(flex_row, ["Adj Own"], default=0.0)),
            saber_proj=first_positive(
                source_row,
                ["SS Proj", "My Proj", "dk_mean", "dk_median", "dk_points", "AvgPointsPerGame"],
                default=flex_mean,
            ),
            value=first_positive(source_row, ["Value"], default=0.0),
            status=(flex_row.get("Status") or "").strip(),
        )

        players.append(
            ShowdownPlayer(
                name=name,
                team=team,
                opp=opp,
                base=base,
                flex_id=str(flex_row.get("DFS ID") or flex_row.get("Name")),
                captain_id=str(captain_row.get("DFS ID") or captain_row.get("Name")),
                flex_salary=int(safe_float(flex_row.get("Salary"), 0.0)),
                captain_salary=int(safe_float(captain_row.get("Salary"), 0.0)),
                flex_projection=flex_mean,
                captain_projection=captain_mean,
                captain_p95=captain_p95,
            )
        )

    return players


def showdown_slot_score(player: ShowdownPlayer, slot: str, contest: ContestSpec) -> float:
    top_heavy = contest.top_heaviness
    mean_w = 0.76 - 0.14 * top_heavy
    ceiling_w = 0.24 + 0.26 * top_heavy
    own_w = 0.05 + 0.08 * top_heavy

    if slot == "CPT":
        mean_proj = player.captain_projection
        ceiling = max(player.captain_p95 - player.captain_projection, 0.0)
        ownership = math.log1p(max(player.own * 1.4, 0.0))
        return mean_w * mean_proj + ceiling_w * ceiling - own_w * ownership

    mean_proj = player.flex_projection
    ceiling = max(player.base_p95 - player.flex_projection, 0.0)
    ownership = math.log1p(max(player.own, 0.0))
    return mean_w * mean_proj + ceiling_w * ceiling - own_w * ownership


def showdown_stack_bonus(lineup, contest: ContestSpec) -> float:
    if isinstance(lineup, dict):
        captain = lineup.get("captain")
        flex = lineup.get("flex", [])
        players = [p for p in [captain, *flex] if p is not None]
        counts = Counter(p.team for p in players)
        captain_team = captain.team if captain is not None else None
    else:
        counts = lineup.teams()
        captain_team = lineup.captain.team
    if not counts:
        return 0.0

    top_team, top_count = max(counts.items(), key=lambda kv: kv[1])
    other_count = sum(counts.values()) - top_count
    top_heavy = contest.top_heaviness

    bonus = 0.0
    bonus += max(0, top_count - 2) * (0.22 + 0.14 * top_heavy)
    bonus += max(0, other_count - 1) * 0.08

    if captain_team == top_team:
        bonus += 0.10 + 0.06 * max(0, top_count - 2)
    return bonus


def state_score(state, contest: ContestSpec) -> float:
    salary_penalty = max(0, state["salary"] - 45000) * 0.00004
    return state["score"] + showdown_stack_bonus(state["lineup"], contest) - salary_penalty


def optimistic_min_remaining_salary(
    players: Sequence[ShowdownPlayer],
    used_names: set,
    captain_chosen: Optional[ShowdownPlayer],
    remaining_flex_slots: int,
) -> int:
    available = [p for p in players if p.name not in used_names]
    if captain_chosen is None:
        best = None
        for captain in available:
            flex_pool = [p for p in available if p.name != captain.name]
            if len(flex_pool) < remaining_flex_slots:
                continue
            flex_min = sum(sorted(p.flex_salary for p in flex_pool)[:remaining_flex_slots])
            total = captain.captain_salary + flex_min
            if best is None or total < best:
                best = total
        return best if best is not None else 10**9

    if len(available) < remaining_flex_slots:
        return 10**9
    return sum(sorted(p.flex_salary for p in available)[:remaining_flex_slots])


def candidate_pool(players: Sequence[ShowdownPlayer], slot: str, contest: ContestSpec, pool_size: int) -> List[ShowdownPlayer]:
    if slot == "CPT":
        ordered = sorted(players, key=lambda p: (showdown_slot_score(p, "CPT", contest), p.captain_projection), reverse=True)
    else:
        ordered = sorted(players, key=lambda p: (showdown_slot_score(p, "FLEX", contest), p.flex_projection), reverse=True)

    merged: List[ShowdownPlayer] = []
    seen = set()
    for player in ordered[:pool_size]:
        if player.name in seen:
            continue
        merged.append(player)
        seen.add(player.name)
    return merged


def beam_search_lineups(
    players: Sequence[ShowdownPlayer],
    contest: ContestSpec,
    target_count: int,
    candidate_size: int = 18,
    beam_width: int = 500,
) -> List[ShowdownLineup]:
    cpt_pool = candidate_pool(players, "CPT", contest, max(8, candidate_size // 2))
    flex_pool = candidate_pool(players, "FLEX", contest, candidate_size)
    if not cpt_pool or not flex_pool:
        return []

    states = [
        {
            "lineup": {"captain": None, "flex": []},
            "used_names": set(),
            "salary": 0,
            "score": 0.0,
        }
    ]

    for stage in range(1 + SHOWDOWN_FLEX_SLOTS):
        next_states = []
        for state in states:
            if stage == 0:
                slot = "CPT"
                pool = cpt_pool
                for player in pool:
                    salary = state["salary"] + player.captain_salary
                    if salary + optimistic_min_remaining_salary(players, {player.name}, player, SHOWDOWN_FLEX_SLOTS) > SALARY_CAP:
                        continue
                    next_states.append(
                        {
                            "lineup": {"captain": player, "flex": []},
                            "used_names": {player.name},
                            "salary": salary,
                            "score": state["score"] + showdown_slot_score(player, slot, contest),
                        }
                    )
                continue

            slot = "FLEX"
            captain = state["lineup"]["captain"]
            flex_chosen = state["lineup"]["flex"]
            remaining = SHOWDOWN_FLEX_SLOTS - len(flex_chosen)
            if remaining <= 0:
                next_states.append(state)
                continue

            for player in flex_pool:
                if player.name in state["used_names"]:
                    continue
                salary = state["salary"] + player.flex_salary
                new_used = set(state["used_names"]) | {player.name}
                min_remaining = optimistic_min_remaining_salary(players, new_used, captain, remaining - 1)
                if salary + min_remaining > SALARY_CAP:
                    continue
                new_flex = list(flex_chosen) + [player]
                next_states.append(
                    {
                        "lineup": {"captain": captain, "flex": new_flex},
                        "used_names": new_used,
                        "salary": salary,
                        "score": state["score"] + showdown_slot_score(player, slot, contest),
                    }
                )

        if not next_states:
            return []
        next_states.sort(key=lambda s: state_score(s, contest), reverse=True)
        states = next_states[:beam_width]

    lineups: List[ShowdownLineup] = []
    seen = set()
    for state in states:
        captain = state["lineup"]["captain"]
        flex = tuple(state["lineup"]["flex"])
        if captain is None or len(flex) != SHOWDOWN_FLEX_SLOTS:
            continue
        lineup = ShowdownLineup(captain=captain, flex=flex)
        key = lineup.player_ids()
        if key in seen:
            continue
        seen.add(key)
        lineups.append(lineup)
        if len(lineups) >= max(target_count * 8, 40):
            break

    lineups.sort(
        key=lambda lu: (
            showdown_slot_score(lu.captain, "CPT", contest)
            + sum(showdown_slot_score(p, "FLEX", contest) for p in lu.flex)
            + showdown_stack_bonus(lu, contest),
            -lu.salary,
        ),
        reverse=True,
    )
    return lineups


def lineup_similarity(a: ShowdownLineup, b: ShowdownLineup) -> float:
    a_ids = set(a.player_ids())
    b_ids = set(b.player_ids())
    return len(a_ids & b_ids) / max(len(a_ids), 1)


def duplication_index(lineup: ShowdownLineup) -> float:
    values = [max(lineup.captain.own * 1.4 / 100.0, 1e-6)]
    values.extend(max(p.own / 100.0, 1e-6) for p in lineup.flex)
    return geo_mean(values)


def showdown_lineup_score_vector(lineup: ShowdownLineup, draws: Dict[str, List[float]], n_sims: int) -> List[float]:
    scores = [0.0] * n_sims
    captain_draws = draws[lineup.captain.base.player_id]
    for i in range(n_sims):
        scores[i] += captain_draws[i] * SHOWDOWN_CAPTAIN_MULTIPLIER
    for player in lineup.flex:
        player_draws = draws[player.base.player_id]
        for i in range(n_sims):
            scores[i] += player_draws[i]
    return scores


def build_field_population(
    players: Sequence[ShowdownPlayer],
    contest: ContestSpec,
    n_lineups: int,
    seed: int,
    candidate_size: int = 15,
) -> List[ShowdownLineup]:
    rng = random.Random(seed)
    cpt_pool = candidate_pool(players, "CPT", contest, max(8, candidate_size // 2))
    flex_pool = candidate_pool(players, "FLEX", contest, candidate_size)

    def field_weight(player: ShowdownPlayer, slot: str) -> float:
        own = max(player.own / 100.0, 1e-6)
        if slot == "CPT":
            return math.exp(0.18 * player.captain_projection + 4.8 * own + 0.02 * player.captain_p95)
        return math.exp(0.20 * player.flex_projection + 4.2 * own + 0.02 * player.base_p95)

    lineups: List[ShowdownLineup] = []
    seen = set()
    attempts = 0
    max_attempts = n_lineups * 100

    while len(lineups) < n_lineups and attempts < max_attempts:
        attempts += 1
        used = set()
        captain = weighted_choice(
            cpt_pool,
            [field_weight(p, "CPT") for p in cpt_pool],
            rng,
        )
        used.add(captain.name)
        flex: List[ShowdownPlayer] = []
        salary = captain.captain_salary
        team_counts = Counter([captain.team])
        target_stack_size = 4 if contest.top_heaviness < 0.85 else 5

        for _ in range(SHOWDOWN_FLEX_SLOTS):
            pool = [p for p in flex_pool if p.name not in used]
            if not pool:
                break
            weights = []
            for p in pool:
                w = field_weight(p, "FLEX")
                if p.team == captain.team:
                    w *= 1.45 if team_counts[p.team] < target_stack_size else 0.9
                else:
                    w *= 1.12 if team_counts[captain.team] >= 3 else 1.0
                weights.append(w)
            player = weighted_choice(pool, weights, rng)
            used.add(player.name)
            flex.append(player)
            salary += player.flex_salary
            team_counts[player.team] += 1

        if len(flex) != SHOWDOWN_FLEX_SLOTS or salary > SALARY_CAP:
            continue

        lineup = ShowdownLineup(captain=captain, flex=tuple(flex))
        key = lineup.player_ids()
        if key not in seen:
            seen.add(key)
            lineups.append(lineup)

    return lineups


def estimate_lineup_metrics(
    lineup: ShowdownLineup,
    contest: ContestSpec,
    lineup_scores: List[float],
    field_score_matrix: List[List[float]],
    entry_fee: float,
) -> ShowdownResult:
    field_size = contest.field_size
    n_sims = len(lineup_scores)
    assert n_sims > 0

    duplicate_index = duplication_index(lineup)
    duplicate_multiplier = 1.0 / (1.0 + field_size * duplicate_index * 0.0009)

    payouts = []
    ranks = []
    wins = 0
    cash = 0
    paid_cutoff = contest.payout_bands[-1].max_rank if contest.payout_bands else max(1, int(field_size * contest.cash_rate))

    for sim_idx in range(n_sims):
        cand_score = lineup_scores[sim_idx]
        field_scores = [scores[sim_idx] for scores in field_score_matrix if sim_idx < len(scores)]
        beaten = sum(1 for s in field_scores if s < cand_score)
        tied = sum(1 for s in field_scores if s == cand_score)
        population_n = max(len(field_scores), 1)
        percentile = (beaten + 0.5 * tied) / population_n
        rank_est = 1.0 + (1.0 - percentile) * (field_size - 1)
        rank_bucket = max(1, int(math.floor(rank_est + 0.5)))
        ranks.append(rank_est)

        if rank_est <= 1.0:
            wins += 1
        if rank_est <= paid_cutoff:
            cash += 1

        gross = contest.payout_for_rank(rank_bucket)
        payouts.append(gross * duplicate_multiplier)

    ordered = sorted(lineup_scores)
    mean_score = mean(lineup_scores)
    q90 = ordered[int(0.90 * (n_sims - 1))]
    q95 = ordered[int(0.95 * (n_sims - 1))]
    ceiling_score = mean(ordered[int(0.95 * (n_sims - 1)) :]) if n_sims > 1 else ordered[0]

    expected_payout = mean(payouts) if payouts else 0.0
    expected_value = expected_payout - entry_fee
    raw_win_rate = wins / n_sims
    de_duped_win_rate = raw_win_rate * duplicate_multiplier

    return ShowdownResult(
        lineup=lineup,
        mean_score=mean_score,
        ceiling_score=ceiling_score,
        ownership_geo_mean=lineup.geo_own(),
        duplication_index=duplicate_index,
        raw_win_rate=raw_win_rate,
        de_duped_win_rate=de_duped_win_rate,
        expected_payout=expected_payout,
        expected_value=expected_value,
        cash_rate=cash / n_sims,
        percentile_90=q90,
        percentile_95=q95,
        expected_rank=mean(ranks) if ranks else float(field_size),
    )


def optimize_showdown(
    players: Sequence[ShowdownPlayer],
    contest: ContestSpec,
    lineup_count: int,
    sims: int = 300,
    seed: int = 7,
    field_population_size: int = 400,
    candidate_pool_size: int = 18,
    beam_width: int = 500,
) -> List[ShowdownResult]:
    candidate_lineups = beam_search_lineups(
        players=players,
        contest=contest,
        target_count=max(lineup_count * 10, 40),
        candidate_size=candidate_pool_size,
        beam_width=beam_width,
    )
    if not candidate_lineups:
        return []

    field_population = build_field_population(
        players=players,
        contest=contest,
        n_lineups=field_population_size,
        seed=seed + 1000,
        candidate_size=max(12, candidate_pool_size - 3),
    )
    if not field_population:
        return []

    base_players = [p.base for p in players]
    draws = sample_player_draws(base_players, sims, seed=seed)
    field_score_matrix = [showdown_lineup_score_vector(lineup, draws, sims) for lineup in field_population]

    evaluated: List[ShowdownResult] = []
    for lineup in candidate_lineups:
        lineup_scores = showdown_lineup_score_vector(lineup, draws, sims)
        evaluated.append(
            estimate_lineup_metrics(
                lineup=lineup,
                contest=contest,
                lineup_scores=lineup_scores,
                field_score_matrix=field_score_matrix,
                entry_fee=contest.entry_fee,
            )
        )

    evaluated.sort(
        key=lambda r: (
            r.expected_value,
            r.de_duped_win_rate,
            r.percentile_95,
            r.mean_score,
        ),
        reverse=True,
    )

    selected: List[ShowdownResult] = []
    used_keys = set()
    threshold = 0.66
    while len(selected) < lineup_count and threshold <= 0.88:
        for result in evaluated:
            key = result.lineup.player_ids()
            if key in used_keys:
                continue
            if any(lineup_similarity(result.lineup, other.lineup) >= threshold for other in selected):
                continue
            used_keys.add(key)
            selected.append(result)
            if len(selected) >= lineup_count:
                break
        if len(selected) < lineup_count:
            threshold += 0.05

    if len(selected) < lineup_count:
        for result in evaluated:
            key = result.lineup.player_ids()
            if key in used_keys:
                continue
            selected.append(result)
            used_keys.add(key)
            if len(selected) >= lineup_count:
                break

    return selected[:lineup_count]


def print_results(contest: ContestSpec, results: Sequence[ShowdownResult]) -> None:
    print(f"\nContest: {contest.name} | field={contest.field_size} | entry_fee=${contest.entry_fee:.2f}")
    print(f"Top-heaviness: {contest.top_heaviness:.2f}")
    for idx, r in enumerate(results, 1):
        counts = r.lineup.teams()
        top_team = max(counts.items(), key=lambda kv: kv[1])[0] if counts else ""
        print(
            f"{idx}. EV=${r.expected_value:.2f}  payout=${r.expected_payout:.2f}  "
            f"win={r.raw_win_rate:.3%}  dup-win={r.de_duped_win_rate:.3%}  "
            f"mean={r.mean_score:.2f}  p95={r.percentile_95:.2f}  "
            f"salary=${r.lineup.salary}  geo-own={r.ownership_geo_mean:.4f}  "
            f"dup={r.duplication_index:.6f}  stack={top_team}:{counts.get(top_team, 0)}"
        )


def write_output_csv(path: str, results: Sequence[ShowdownResult]) -> None:
    header = [
        "Rank",
        "EV",
        "Expected Payout",
        "Win Rate",
        "De-duped Win Rate",
        "Mean",
        "P95",
        "Salary",
        "CPT ID",
        "CPT Name",
        "FLEX1 ID",
        "FLEX1 Name",
        "FLEX2 ID",
        "FLEX2 Name",
        "FLEX3 ID",
        "FLEX3 Name",
        "FLEX4 ID",
        "FLEX4 Name",
        "FLEX5 ID",
        "FLEX5 Name",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, r in enumerate(results, 1):
            row = [
                i,
                round(r.expected_value, 4),
                round(r.expected_payout, 4),
                round(r.raw_win_rate, 6),
                round(r.de_duped_win_rate, 6),
                round(r.mean_score, 4),
                round(r.percentile_95, 4),
                r.lineup.salary,
                r.lineup.captain.captain_id,
                r.lineup.captain.name,
            ]
            for flex in r.lineup.flex:
                row.extend([flex.flex_id, flex.name])
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Showdown DFS lineup optimizer")
    parser.add_argument("--slate", required=True, help="Showdown slate CSV")
    parser.add_argument("--projections", help="Optional empirical projection JSON/CSV from the game model")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--lineups", type=int, default=4, help="Number of lineups to generate")
    parser.add_argument("--field-size", type=int, default=1783, help="Contest field size")
    parser.add_argument("--cash-rate", type=float, default=0.2361, help="Percent of field that cashes")
    parser.add_argument("--first-place-share", type=float, default=0.10, help="Prize pool share for 1st place")
    parser.add_argument("--entry-fee", type=float, default=1.0, help="Entry fee used for EV normalization")
    parser.add_argument("--sims", type=int, default=240, help="Monte Carlo simulations per lineup")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--candidate-pool", type=int, default=18, help="Candidates per slot")
    parser.add_argument("--beam-width", type=int, default=500, help="Beam search width")
    parser.add_argument("--field-population", type=int, default=350, help="Approximate field lineups")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    players = load_showdown_players(args.slate, projections_path=args.projections)
    contest = ContestSpec.fallback_top_heavy(
        contest_id="showdown",
        name="Tonight Showdown",
        field_size=args.field_size,
        entry_fee=args.entry_fee,
        cash_rate=args.cash_rate,
        first_place_share=args.first_place_share,
    )
    results = optimize_showdown(
        players=players,
        contest=contest,
        lineup_count=args.lineups,
        sims=args.sims,
        seed=args.seed,
        field_population_size=args.field_population,
        candidate_pool_size=args.candidate_pool,
        beam_width=args.beam_width,
    )
    print_results(contest, results)
    write_output_csv(args.out, results)
    print(f"\nWrote optimized showdown entries to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
