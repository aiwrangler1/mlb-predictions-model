"""
Contest-aware DFS lineup optimizer.

This module is intentionally separate from the FastAPI app so it can be used as:

1. A CLI tool that reads a DK slate file plus DraftKings entry upload template.
2. A reusable optimizer that scores lineups with contest size, payout bands,
   player probability distributions, and ownership-based duplication penalties.

The core idea is:
- Use player projection distributions, not just point estimates.
- Build a field population from ownership-weighted lineups.
- Evaluate candidate lineups against a payout ladder for the contest.
- Penalize duplicated / overly chalky constructions.

The contest payout schedule must be provided for exact EV. If it is omitted,
the optimizer falls back to a heuristic top-heavy payout model so it still runs.
That fallback is useful for ranking, but it is not a substitute for a real
contest payout table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from outcome_distribution import empirical_projection_profile, first_present_float, inverse_cdf_sample
from projection_overlay import load_projection_rows, merge_projection_rows


SALARY_CAP = 50000
ROSTER_SLOTS = ["P1", "P2", "C", "1B", "2B", "3B", "SS", "OF1", "OF2", "OF3"]
SLOT_TO_ROSTER = {
    "P1": "P",
    "P2": "P",
    "C": "C",
    "1B": "1B",
    "2B": "2B",
    "3B": "3B",
    "SS": "SS",
    "OF1": "OF",
    "OF2": "OF",
    "OF3": "OF",
}
HITTER_SLOTS = ["C", "1B", "2B", "3B", "SS", "OF", "UTIL"]


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def geo_mean(values: Sequence[float]) -> float:
    values = [max(v, 1e-9) for v in values]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else 0.0


def percentile_to_std(
    p10: float,
    p25: float,
    p50: float,
    p75: float,
    p85: float,
    p90: float,
    p95: float,
    p99: float,
) -> float:
    """
    Estimate a score distribution standard deviation from DK percentile columns.
    """
    estimates = []
    if p90 and p10:
        estimates.append((p90 - p10) / 2.563)
    if p75 and p25:
        estimates.append((p75 - p25) / 1.349)
    if p95 and p50:
        estimates.append((p95 - p50) / 1.645)
    if p85 and p50:
        estimates.append((p85 - p50) / 1.036)
    if p99 and p50:
        estimates.append((p99 - p50) / 2.326)
    estimates = [e for e in estimates if e > 0]
    if not estimates:
        return 3.5
    return max(1.5, sum(estimates) / len(estimates))


def weighted_choice(items: Sequence, weights: Sequence[float], rng: random.Random):
    total = sum(weights)
    if total <= 0:
        return rng.choice(list(items))
    roll = rng.random() * total
    cumulative = 0.0
    for item, weight in zip(items, weights):
        cumulative += weight
        if roll <= cumulative:
            return item
    return items[-1]


@dataclass(frozen=True)
class Player:
    player_id: str
    name: str
    positions: Tuple[str, ...]
    team: str
    opp: str
    salary: int
    mean: float
    p10: float
    p25: float
    p50: float
    p75: float
    p85: float
    p90: float
    p95: float
    p99: float
    own: float
    saber_proj: float
    value: float
    status: str = ""

    @property
    def is_pitcher(self) -> bool:
        return self.positions == ("P",)

    @property
    def std(self) -> float:
        return percentile_to_std(self.p10, self.p25, self.p50, self.p75, self.p85, self.p90, self.p95, self.p99)

    def field_weight(self) -> float:
        """
        Weight used to approximate what the field is likely to roster.
        This intentionally favors projection and ownership.
        """
        own = max(self.own / 100.0, 1e-6)
        return math.exp(0.16 * self.mean + 2.0 * own + 0.025 * self.p95)

    def candidate_score(self, contest: "ContestSpec") -> float:
        """
        Contest-aware lineup construction score.
        Top-heavy contests should lean more ceiling-heavy and less ownership-sensitive.
        """
        top_heavy = contest.top_heaviness
        mean_w = 0.78 - 0.18 * top_heavy
        ceiling_w = 0.22 + 0.28 * top_heavy
        own_w = 0.06 + 0.10 * top_heavy
        ceiling = self.p95 - self.mean
        ownership = math.log1p(max(self.own, 0.0))
        score = mean_w * self.mean + ceiling_w * ceiling - own_w * ownership

        # Treat Sabersim as a sanity check, not a replacement for the model.
        # If the projections agree, the player gets a small boost; if they
        # diverge sharply, the player is still eligible but loses a bit of rank.
        if self.saber_proj > 0:
            gap = abs(self.mean - self.saber_proj)
            consensus_scale = max(6.0, 0.35 * max(self.mean, self.saber_proj))
            consensus = clamp(1.0 - gap / consensus_scale, 0.0, 1.0)
            score += 0.06 * self.mean * consensus
        return score


@dataclass(frozen=True)
class PayoutBand:
    min_rank: int
    max_rank: int
    payout: float

    def contains(self, rank: float) -> bool:
        return self.min_rank <= rank <= self.max_rank


@dataclass
class ContestSpec:
    contest_id: str
    name: str
    field_size: int
    entry_fee: float
    cash_rate: float = 0.25
    first_place_share: float = 0.10
    payout_bands: List[PayoutBand] = field(default_factory=list)

    @property
    def total_prize_pool(self) -> float:
        return sum(b.payout * (b.max_rank - b.min_rank + 1) for b in self.payout_bands)

    @property
    def paid_places(self) -> int:
        return sum(b.max_rank - b.min_rank + 1 for b in self.payout_bands)

    @property
    def top_heaviness(self) -> float:
        """
        Rough measure of how top-heavy the payout structure is.
        Higher values -> more incentive to chase ceiling and uniqueness.
        """
        if not self.payout_bands:
            base = self.first_place_share / max(self.cash_rate, 0.05)
            return clamp(base * 2.5, 0.2, 3.0)
        first = self.payout_bands[0].payout
        avg_paid = self.total_prize_pool / max(self.paid_places, 1)
        if avg_paid <= 0:
            return 0.6
        return clamp(first / avg_paid, 0.2, 3.0)

    def payout_for_rank(self, rank: float) -> float:
        for band in self.payout_bands:
            if band.contains(rank):
                return band.payout
        return 0.0

    @classmethod
    def fallback_top_heavy(
        cls,
        contest_id: str,
        name: str,
        field_size: int,
        entry_fee: float,
        cash_rate: float = 0.25,
        first_place_share: float = 0.10,
    ) -> "ContestSpec":
        """
        Heuristic fallback when the real payout ladder is not available.
        This keeps the optimizer usable, but it is not exact EV.
        """
        prize_pool = max(field_size * entry_fee * 0.80, entry_fee * 20)
        paid_places = max(1, int(round(field_size * cash_rate)))
        first = prize_pool * first_place_share
        remainder = prize_pool - first

        band_templates = [
            (2, 3, 0.18),
            (4, 10, 0.16),
            (11, 25, 0.14),
            (26, 50, 0.12),
            (51, 100, 0.10),
            (101, 250, 0.08),
            (251, 500, 0.06),
            (501, 1000, 0.04),
            (1001, paid_places, 0.02),
        ]

        usable = [(lo, hi, weight) for lo, hi, weight in band_templates if lo <= paid_places]
        total_weight = sum(weight for _, _, weight in usable) or 1.0
        bands = [PayoutBand(1, 1, first)]
        for lo, hi, weight in usable:
            hi = min(hi, paid_places)
            if hi < lo:
                continue
            count = hi - lo + 1
            per_place = remainder * (weight / total_weight) / count
            bands.append(PayoutBand(lo, hi, per_place))

        return cls(
            contest_id=contest_id,
            name=name,
            field_size=field_size,
            entry_fee=entry_fee,
            cash_rate=cash_rate,
            first_place_share=first_place_share,
            payout_bands=bands,
        )


@dataclass
class Lineup:
    slots: Dict[str, Player]

    @property
    def salary(self) -> int:
        return sum(p.salary for p in self.slots.values())

    @property
    def players(self) -> Tuple[Player, ...]:
        return tuple(self.slots[slot] for slot in ROSTER_SLOTS)

    def player_ids(self) -> Tuple[str, ...]:
        pitchers = sorted([self.slots["P1"].player_id, self.slots["P2"].player_id])
        hitters = [
            self.slots["C"].player_id,
            self.slots["1B"].player_id,
            self.slots["2B"].player_id,
            self.slots["3B"].player_id,
            self.slots["SS"].player_id,
        ]
        outfielders = sorted(
            [self.slots["OF1"].player_id, self.slots["OF2"].player_id, self.slots["OF3"].player_id]
        )
        hitters.extend(outfielders)
        return tuple(pitchers + hitters)

    def teams(self) -> Counter:
        return Counter(p.team for p in self.players if p.positions != ("P",))

    def geo_own(self) -> float:
        hitter_owns = [(p.own / 100.0) for p in self.players if p.positions != ("P",)]
        return geo_mean(hitter_owns) if hitter_owns else 0.0

    def stack_profile(self) -> Dict[str, int]:
        return dict(self.teams())


@dataclass
class LineupResult:
    lineup: Lineup
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


def parse_positions(raw: str) -> Tuple[str, ...]:
    if not raw:
        return ()
    raw = raw.strip().upper()
    if raw == "P":
        return ("P",)
    parts = tuple(part.strip() for part in raw.split("/") if part.strip())
    return parts if parts else (raw,)


def load_players(
    path: str,
    status_filter: Optional[Sequence[str]] = None,
    sim_projection_path: Optional[str] = None,
) -> List[Player]:
    players: List[Player] = []
    status_filter_set = {s.lower() for s in status_filter} if status_filter else None

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    if sim_projection_path:
        rows = merge_projection_rows(rows, load_projection_rows(sim_projection_path))

    for row in rows:
        profile = empirical_projection_profile(row, fallback_mean=0.0)
        mean_proj = profile["mean"]
        salary = int(safe_float(row.get("Salary"), 0.0))
        if mean_proj <= 0 or salary <= 0:
            continue
        status = (row.get("Status") or "").strip()
        if status_filter_set and status.lower() not in status_filter_set:
            continue

        player = Player(
            player_id=str(row.get("DFS ID") or row.get("DFS ID ") or row.get("ID") or row.get("player_id") or row.get("Name")),
            name=row.get("Name", ""),
            positions=parse_positions(row.get("Pos", "")),
            team=row.get("Team", ""),
            opp=row.get("Opp", ""),
            salary=salary,
            mean=mean_proj,
            p10=profile["p10"],
            p25=profile["p25"],
            p50=profile["p50"],
            p75=profile["p75"],
            p85=profile["p85"],
            p90=profile["p90"],
            p95=profile["p95"],
            p99=profile["p99"],
            own=first_present_float(row, ["Adj Own"], default=0.0),
            saber_proj=first_present_float(
                row,
                ["SS Proj", "My Proj", "dk_mean", "dk_median", "dk_points"],
                default=mean_proj,
                positive_only=True,
            ),
            value=first_present_float(row, ["Value"], default=0.0),
            status=status,
        )
        players.append(player)

    return players


def read_contest_entries(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    actual = [r for r in rows if r.get("Entry ID") and r["Entry ID"].strip().isdigit()]
    return actual


def group_entries_by_contest(entries: Sequence[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in entries:
        grouped[row["Contest ID"]].append(row)
    return grouped


def load_contest_specs(
    contest_json_path: Optional[str],
    entries: Sequence[Dict[str, str]],
    default_field_size: int,
) -> Dict[str, ContestSpec]:
    """
    Load contest metadata from JSON.

    Format:
    {
      "contest_id": {
        "name": "Contest Name",
        "field_size": 5000,
        "entry_fee": 20.0,
        "payout_bands": [
          {"min_rank": 1, "max_rank": 1, "payout": 100000.0},
          {"min_rank": 2, "max_rank": 2, "payout": 50000.0}
        ]
      }
    }
    """
    grouped = group_entries_by_contest(entries)
    specs: Dict[str, ContestSpec] = {}
    config = {}
    if contest_json_path:
        with open(contest_json_path) as f:
            config = json.load(f)

    for contest_id, rows in grouped.items():
        first = rows[0]
        cfg = config.get(contest_id, {})
        name = cfg.get("name") or first.get("Contest Name") or f"Contest {contest_id}"
        field_size = int(cfg.get("field_size") or default_field_size)
        entry_fee = safe_float(cfg.get("entry_fee"), safe_float(first.get("Entry Fee", "$1").replace("$", ""), 1.0))
        cash_rate = safe_float(cfg.get("cash_rate"), 0.25)
        first_place_share = safe_float(cfg.get("first_place_share"), 0.10)
        payout_bands = []
        for band in cfg.get("payout_bands", []) or []:
            payout_bands.append(
                PayoutBand(
                    min_rank=int(band["min_rank"]),
                    max_rank=int(band["max_rank"]),
                    payout=float(band["payout"]),
                )
            )
        if not payout_bands:
            specs[contest_id] = ContestSpec.fallback_top_heavy(
                contest_id,
                name,
                field_size,
                entry_fee,
                cash_rate=cash_rate,
                first_place_share=first_place_share,
            )
        else:
            specs[contest_id] = ContestSpec(
                contest_id=contest_id,
                name=name,
                field_size=field_size,
                entry_fee=entry_fee,
                cash_rate=cash_rate,
                first_place_share=first_place_share,
                payout_bands=payout_bands,
            )

    return specs


def eligible_for_slot(player: Player, slot: str) -> bool:
    slot = SLOT_TO_ROSTER.get(slot, slot)
    if slot == "UTIL":
        return not player.is_pitcher
    if player.is_pitcher:
        return slot == "P"
    return slot in player.positions


def candidate_pool(players: Sequence[Player], slot: str, contest: ContestSpec, pool_size: int) -> List[Player]:
    eligible = [p for p in players if eligible_for_slot(p, slot)]
    if not eligible:
        return []

    if slot == "P":
        # For pitchers, keep the blend of projection and leverage plus a few
        # cheap pivots. Pitching is where lineup construction often starts.
        score_sorted = sorted(eligible, key=lambda p: (p.candidate_score(contest), p.mean - 0.02 * p.own), reverse=True)
        value_sorted = sorted(
            eligible,
            key=lambda p: (
                p.mean / max(p.salary, 1) * 1000.0 + p.p95 / max(p.salary, 1) * 500.0 - 0.01 * p.own,
            ),
            reverse=True,
        )
    else:
        score_sorted = sorted(eligible, key=lambda p: (p.candidate_score(contest), p.mean - 0.015 * p.own), reverse=True)
        value_sorted = sorted(
            eligible,
            key=lambda p: (
                p.mean / max(p.salary, 1) * 1000.0 + p.p95 / max(p.salary, 1) * 350.0 - 0.02 * p.own,
            ),
            reverse=True,
        )

    merged: List[Player] = []
    seen = set()
    for player in score_sorted[:pool_size]:
        if player.player_id not in seen:
            merged.append(player)
            seen.add(player.player_id)
    for player in value_sorted[: max(8, pool_size // 2)]:
        if player.player_id not in seen:
            merged.append(player)
            seen.add(player.player_id)
    return merged


def min_salary_by_remaining_slots(candidates: Dict[str, List[Player]]) -> Dict[str, int]:
    remaining = {}
    order = ROSTER_SLOTS
    cheapest = [min((p.salary for p in candidates[slot]), default=999999) for slot in order]
    suffix = [0] * (len(order) + 1)
    for i in range(len(order) - 1, -1, -1):
        suffix[i] = suffix[i + 1] + cheapest[i]
    for i, slot in enumerate(order):
        remaining[slot] = suffix[i]
    return remaining


def stack_bonus(team_counts: Counter, contest: ContestSpec) -> float:
    if not team_counts:
        return 0.0

    top_team_count = max(team_counts.values())
    stacked_teams = sum(1 for c in team_counts.values() if c >= 3)
    secondary_teams = sum(1 for c in team_counts.values() if c >= 2)

    top_heavy = contest.top_heaviness
    bonus = 0.0
    bonus += max(0, top_team_count - 3) * (0.42 + 0.35 * top_heavy)
    bonus += stacked_teams * (0.18 + 0.10 * top_heavy)
    bonus += secondary_teams * 0.04
    return bonus


def partial_score(state, contest: ContestSpec) -> float:
    # Salary efficiency matters, but the lineups need to compete for top prizes.
    salary_penalty = max(0, state["salary"] - 45000) * 0.00003
    return state["score"] + stack_bonus(state["team_counts"], contest) - salary_penalty


def beam_search_lineups(
    players: Sequence[Player],
    contest: ContestSpec,
    target_count: int,
    candidate_size: int = 18,
    beam_width: int = 600,
    stack_floor: int = 3,
) -> List[Lineup]:
    """
    Generate valid lineups via beam search.

    The search objective is contest-aware, but the final ranking will be
    determined by the full EV simulation.
    """
    candidates: Dict[str, List[Player]] = {
        slot: candidate_pool(players, slot, contest, candidate_size if slot != "P" else max(12, candidate_size // 2))
        for slot in ROSTER_SLOTS
    }

    slot_order = sorted(ROSTER_SLOTS, key=lambda s: len(candidates[s]))
    slot_index = {slot: i for i, slot in enumerate(slot_order)}
    min_remaining_salary = min_salary_by_remaining_slots(candidates)

    states = [
        {
            "lineup": {},
            "used": set(),
            "salary": 0,
            "score": 0.0,
            "team_counts": Counter(),
        }
    ]

    for depth, slot in enumerate(slot_order):
        next_states = []
        remaining_floor = min_remaining_salary[slot]
        for state in states:
            for player in candidates[slot]:
                if player.player_id in state["used"]:
                    continue
                salary = state["salary"] + player.salary
                if salary + remaining_floor > SALARY_CAP:
                    continue
                new_team_counts = state["team_counts"].copy()
                if not player.is_pitcher:
                    new_team_counts[player.team] += 1
                new_state = {
                    "lineup": dict(state["lineup"], **{slot: player}),
                    "used": set(state["used"]) | {player.player_id},
                    "salary": salary,
                    "score": state["score"] + player.candidate_score(contest),
                    "team_counts": new_team_counts,
                }
                next_states.append(new_state)

        if not next_states:
            return []

        next_states.sort(key=lambda s: partial_score(s, contest), reverse=True)
        states = next_states[:beam_width]

    lineups: List[Lineup] = []
    seen = set()
    for state in states:
        lineup = state["lineup"]
        if len(lineup) != len(ROSTER_SLOTS):
            continue
        if state["salary"] > SALARY_CAP:
            continue
        team_counts = state["team_counts"]
        lineup_obj = Lineup({slot: lineup[slot] for slot in ROSTER_SLOTS})
        key = lineup_obj.player_ids()
        if key in seen:
            continue
        seen.add(key)
        lineups.append(lineup_obj)
        if len(lineups) >= max(target_count * 25, 120):
            break

    # Sort by a quick proxy so we can evaluate better candidates first.
    lineups.sort(
        key=lambda lu: (
            sum(p.candidate_score(contest) for p in lu.players) + stack_bonus(lu.teams(), contest),
            -lu.salary,
        ),
        reverse=True,
    )
    return lineups


def sample_player_draws(players: Sequence[Player], n_sims: int, seed: int) -> Dict[str, List[float]]:
    rng = random.Random(seed)
    draws: Dict[str, List[float]] = {}
    for player in players:
        quantiles = {
            0.10: player.p10,
            0.25: player.p25,
            0.50: player.p50,
            0.75: player.p75,
            0.85: player.p85,
            0.90: player.p90,
            0.95: player.p95,
            0.99: player.p99,
        }
        vals = [inverse_cdf_sample(quantiles, rng, floor=0.0) for _ in range(n_sims)]
        if not any(v > 0 for v in vals):
            sd = max(player.std, 1.0)
            vals = [max(0.0, rng.gauss(player.mean, sd)) for _ in range(n_sims)]
        draws[player.player_id] = vals
    return draws


def lineup_score_vector(lineup: Lineup, draws: Dict[str, List[float]], n_sims: int) -> List[float]:
    scores = [0.0] * n_sims
    for player in lineup.players:
        player_draws = draws[player.player_id]
        for i in range(n_sims):
            scores[i] += player_draws[i]
    return scores


def build_field_population(
    players: Sequence[Player],
    contest: ContestSpec,
    n_lineups: int,
    seed: int,
    candidate_size: int = 15,
) -> List[Lineup]:
    """
    Build a population of likely field lineups.

    This is not an exact reconstruction of the field. It is a probabilistic
    approximation that weights projection and ownership heavily, which is good
    enough for contest EV ranking.
    """
    rng = random.Random(seed)
    candidates: Dict[str, List[Player]] = {
        slot: candidate_pool(players, slot, contest, candidate_size if slot != "P" else max(10, candidate_size))
        for slot in ROSTER_SLOTS
    }

    # Put more weight on ownership than the optimizer does.
    def field_weight(player: Player) -> float:
        own = max(player.own / 100.0, 1e-6)
        return math.exp(0.22 * player.mean + 4.5 * own + 0.02 * player.p95)

    lineups: List[Lineup] = []
    seen = set()
    attempts = 0
    max_attempts = n_lineups * 80

    while len(lineups) < n_lineups and attempts < max_attempts:
        attempts += 1
        used = set()
        lineup = {}
        salary = 0
        team_counts = Counter()

        # Encourage popular stack shapes.
        target_stack_size = 4 if contest.top_heaviness < 0.85 else 5
        stack_team = None
        hitter_teams = Counter(p.team for p in players if not p.is_pitcher)
        if hitter_teams:
            teams = list(hitter_teams.keys())
            weights = [sum(p.own for p in players if p.team == t and not p.is_pitcher) + 1 for t in teams]
            stack_team = weighted_choice(teams, weights, rng)

        for slot in ROSTER_SLOTS:
            pool = [p for p in candidates[slot] if p.player_id not in used]
            if not pool:
                break

            weights = []
            for p in pool:
                w = field_weight(p)
                if not p.is_pitcher and stack_team and p.team == stack_team:
                    w *= 1.45 if team_counts[p.team] < target_stack_size else 0.9
                weights.append(w)
            player = weighted_choice(pool, weights, rng)

            used.add(player.player_id)
            lineup[slot] = player
            salary += player.salary
            if not player.is_pitcher:
                team_counts[player.team] += 1

        if len(lineup) != len(ROSTER_SLOTS):
            continue
        if salary > SALARY_CAP:
            continue

        lineup_obj = Lineup({slot: lineup[slot] for slot in ROSTER_SLOTS})
        key = lineup_obj.player_ids()
        if key not in seen:
            seen.add(key)
            lineups.append(lineup_obj)

    return lineups


def duplication_index(lineup: Lineup) -> float:
    """
    A compact proxy for lineup duplication.
    Lower ownership lineups have lower duplication risk.
    """
    hitter_owns = [max(p.own / 100.0, 1e-6) for p in lineup.players if not p.is_pitcher]
    if not hitter_owns:
        return 0.0
    own_geo = geo_mean(hitter_owns)
    team_counts = lineup.teams()
    stack_factor = 1.0 + max(0, max(team_counts.values() or [0]) - 3) * 0.22
    return own_geo * stack_factor


def estimate_lineup_metrics(
    lineup: Lineup,
    contest: ContestSpec,
    lineup_scores: List[float],
    field_score_matrix: List[List[float]],
    entry_fee: float,
) -> LineupResult:
    """
    Convert a lineup score distribution into contest EV.
    """
    field_size = contest.field_size
    n_sims = len(lineup_scores)
    assert n_sims > 0

    duplicate_index = duplication_index(lineup)
    duplicate_multiplier = 1.0 / (1.0 + field_size * duplicate_index * 0.0006)

    payouts = []
    ranks = []
    wins = 0
    cash = 0

    paid_cutoff = contest.payout_bands[-1].max_rank if contest.payout_bands else max(1, int(field_size * 0.2))

    for sim_idx in range(n_sims):
        cand_score = lineup_scores[sim_idx]
        field_scores = field_score_matrix[sim_idx]
        beaten = sum(1 for s in field_scores if s < cand_score)
        tied = sum(1 for s in field_scores if s == cand_score)

        # Extrapolate from sampled field population to the full contest.
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
        net = gross * duplicate_multiplier
        payouts.append(net)

    mean_score = mean(lineup_scores)
    ceiling_score = sum(sorted(lineup_scores)[int(0.95 * (n_sims - 1)) :]) / max(1, n_sims - int(0.95 * (n_sims - 1)))
    q90 = sorted(lineup_scores)[int(0.90 * (n_sims - 1))]
    q95 = sorted(lineup_scores)[int(0.95 * (n_sims - 1))]

    expected_payout = mean(payouts) if payouts else 0.0
    expected_value = expected_payout - entry_fee
    raw_win_rate = wins / n_sims
    de_duped_win_rate = raw_win_rate * duplicate_multiplier

    return LineupResult(
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


def optimize_contest(
    players: Sequence[Player],
    contest: ContestSpec,
    lineup_count: int,
    sims: int = 300,
    seed: int = 7,
    field_population_size: int = 400,
    candidate_pool_size: int = 18,
    beam_width: int = 600,
) -> List[LineupResult]:
    """
    Generate and rank lineups for a contest.
    """
    candidate_lineups = beam_search_lineups(
        players=players,
        contest=contest,
        target_count=max(lineup_count * 10, 40),
        candidate_size=candidate_pool_size,
        beam_width=beam_width,
    )
    if not candidate_lineups:
        return []

    # Field population is shared across all candidate evaluations.
    field_population = build_field_population(
        players=players,
        contest=contest,
        n_lineups=field_population_size,
        seed=seed + 1000,
        candidate_size=max(12, candidate_pool_size - 3),
    )
    if not field_population:
        return []

    draws = sample_player_draws(players, sims, seed=seed)
    field_score_matrix: List[List[float]] = []
    for field_lineup in field_population:
        field_score_matrix.append(lineup_score_vector(field_lineup, draws, sims))

    evaluated: List[LineupResult] = []
    for lineup in candidate_lineups:
        lineup_scores = lineup_score_vector(lineup, draws, sims)
        result = estimate_lineup_metrics(
            lineup=lineup,
            contest=contest,
            lineup_scores=lineup_scores,
            field_score_matrix=field_score_matrix,
            entry_fee=contest.entry_fee,
        )
        evaluated.append(result)

    # Contest-aware ordering.
    evaluated.sort(
        key=lambda r: (
            r.expected_value,
            r.de_duped_win_rate,
            r.percentile_95,
            r.mean_score,
        ),
        reverse=True,
    )

    selected: List[LineupResult] = []
    used_keys = set()
    # Small-entry contests should be diversified harder; large sets can keep
    # a little more overlap while still varying the main construction.
    base_threshold = 0.62 if lineup_count <= 3 else 0.80
    threshold = base_threshold
    while len(selected) < lineup_count and threshold <= 0.85:
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

    # Final backfill if we still could not satisfy the requested count.
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


def lineup_similarity(a: Lineup, b: Lineup) -> float:
    a_ids = set(a.player_ids())
    b_ids = set(b.player_ids())
    return len(a_ids & b_ids) / max(len(a_ids), 1)


def contest_similarity(a: ContestSpec, b: ContestSpec) -> float:
    """
    Contest-level similarity score in [0, 1].

    Similar contests can legitimately share the same optimal lineup. The score
    intentionally focuses on the pieces that drive payout math:
    field size, payout concentration, cash rate, and entry fee.
    """
    field_component = min(abs(math.log(max(a.field_size, 1) / max(b.field_size, 1))), 3.0) / 3.0
    top_component = min(abs(a.top_heaviness - b.top_heaviness) / 1.5, 1.0)
    cash_component = min(abs(a.cash_rate - b.cash_rate) / 0.25, 1.0)
    first_component = min(abs(a.first_place_share - b.first_place_share) / 0.10, 1.0)
    fee_component = min(abs(math.log1p(a.entry_fee) - math.log1p(b.entry_fee)) / 1.5, 1.0)
    distance = 0.35 * field_component + 0.30 * top_component + 0.15 * cash_component + 0.10 * first_component + 0.10 * fee_component
    return clamp(1.0 - distance, 0.0, 1.0)


def lineup_repeat_penalty(
    lineup_key: Tuple[str, ...],
    current_contest_id: str,
    contests: Dict[str, ContestSpec],
    lineup_usage: Dict[Tuple[str, ...], List[str]],
) -> float:
    """
    Penalize cross-contest lineup reuse only when the contests are dissimilar.
    """
    prior_contests = lineup_usage.get(lineup_key, [])
    if not prior_contests:
        return 0.0

    current = contests[current_contest_id]
    best_similarity = 0.0
    for prev_contest_id in prior_contests:
        prev = contests[prev_contest_id]
        best_similarity = max(best_similarity, contest_similarity(current, prev))

    # Similar contests can share lineups; dissimilar contests should pay a price.
    repeat_scale = 2.5 + 0.25 * current.top_heaviness
    return (1.0 - best_similarity) * repeat_scale


def lineup_selection_score(
    result: LineupResult,
    current_contest_id: str,
    contests: Dict[str, ContestSpec],
    lineup_usage: Dict[Tuple[str, ...], List[str]],
) -> float:
    """
    Contest-aware score for portfolio assignment.
    """
    key = result.lineup.player_ids()
    penalty = lineup_repeat_penalty(key, current_contest_id, contests, lineup_usage)
    return result.expected_value - penalty


def lineup_saber_gap(lineup: Lineup) -> float:
    gaps = [abs(p.mean - p.saber_proj) for p in lineup.players if p.saber_proj > 0]
    return mean(gaps) if gaps else 0.0


def assign_lineups_to_entries(
    contests: Dict[str, ContestSpec],
    grouped_entries: Dict[str, List[Dict[str, str]]],
    results_by_contest: Dict[str, List[LineupResult]],
) -> List[Dict[str, str]]:
    """
    Convert optimized lineups into a DraftKings upload-friendly CSV.
    """
    output_rows: List[Dict[str, str]] = []
    for contest_id, entries in grouped_entries.items():
        lineups = results_by_contest.get(contest_id, [])
        if not lineups:
            continue
        for idx, entry in enumerate(entries):
            lineup = lineups[idx % len(lineups)].lineup
            row = dict(entry)
            row.update(
                {
                    "P": lineup.slots["P"].player_id,
                    "P ": lineup.slots["P"].player_id,
                    "C": lineup.slots["C"].player_id if lineup.slots["C"].positions == ("C",) else lineup.slots["C"].player_id,
                    "1B": lineup.slots["1B"].player_id,
                    "2B": lineup.slots["2B"].player_id,
                    "3B": lineup.slots["3B"].player_id,
                    "SS": lineup.slots["SS"].player_id,
                    "OF": lineup.slots["OF"].player_id,
                }
            )
            output_rows.append(row)
    return output_rows


def write_dk_upload_from_template(
    template_path: str,
    grouped_entries: Dict[str, List[Dict[str, str]]],
    results_by_contest: Dict[str, List[LineupResult]],
    contests: Dict[str, ContestSpec],
    out_path: str,
) -> None:
    """
    Preserve the DraftKings upload template exactly, including duplicate headers.
    """
    with open(template_path, newline="") as f:
        raw_rows = list(csv.reader(f))

    if not raw_rows:
        raise ValueError("Empty DraftKings template")

    header = raw_rows[0]
    data_rows = raw_rows[1:]

    header_map: Dict[str, List[int]] = defaultdict(list)
    for idx, name in enumerate(header):
        header_map[name].append(idx)

    # Map Entry ID -> raw template row so we can preserve Contest Name, Instructions, etc.
    entry_id_idx = header_map["Entry ID"][0]
    raw_by_entry_id: Dict[str, List[str]] = {}
    for row in data_rows:
        if not row or len(row) <= entry_id_idx:
            continue
        entry_id = row[entry_id_idx].strip()
        if entry_id.isdigit():
            raw_by_entry_id[entry_id] = row

    p_indices = header_map.get("P", [])
    of_indices = header_map.get("OF", [])
    c_idx = header_map.get("C", [None])[0]
    one_idx = header_map.get("1B", [None])[0]
    two_idx = header_map.get("2B", [None])[0]
    three_idx = header_map.get("3B", [None])[0]
    ss_idx = header_map.get("SS", [None])[0]

    if len(p_indices) < 2 or len(of_indices) < 3:
        raise ValueError("DraftKings template does not look like an MLB upload template")

    output_rows: List[List[str]] = []
    lineup_usage: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    selected_by_contest: Dict[str, List[LineupResult]] = defaultdict(list)

    contest_items = sorted(
        grouped_entries.items(),
        key=lambda item: (
            contests[item[0]].field_size * contests[item[0]].entry_fee * max(contests[item[0]].top_heaviness, 0.5),
            contests[item[0]].field_size,
        ),
        reverse=True,
    )

    for contest_id, entry_rows in contest_items:
        lineups = results_by_contest.get(contest_id, [])
        if not lineups:
            continue

        contest = contests[contest_id]
        used_in_contest = set()
        for idx, entry in enumerate(entry_rows):
            base = list(raw_by_entry_id[entry["Entry ID"]])
            candidate_results = [result for result in lineups if result.lineup.player_ids() not in used_in_contest]
            if not candidate_results:
                candidate_results = list(lineups)

            chosen = None
            best_score = float("-inf")
            for result in candidate_results:
                key = result.lineup.player_ids()
                score = lineup_selection_score(result, contest_id, contests, lineup_usage)
                if key in used_in_contest:
                    score -= 1000.0
                if score > best_score:
                    best_score = score
                    chosen = result

            if chosen is None:
                chosen = lineups[idx % len(lineups)]

            lineup = chosen.lineup
            key = lineup.player_ids()
            used_in_contest.add(key)
            lineup_usage[key].append(contest_id)
            selected_by_contest[contest_id].append(chosen)
            base[p_indices[0]] = lineup.slots["P1"].player_id
            base[p_indices[1]] = lineup.slots["P2"].player_id
            if c_idx is not None:
                base[c_idx] = lineup.slots["C"].player_id
            if one_idx is not None:
                base[one_idx] = lineup.slots["1B"].player_id
            if two_idx is not None:
                base[two_idx] = lineup.slots["2B"].player_id
            if three_idx is not None:
                base[three_idx] = lineup.slots["3B"].player_id
            if ss_idx is not None:
                base[ss_idx] = lineup.slots["SS"].player_id
            base[of_indices[0]] = lineup.slots["OF1"].player_id
            base[of_indices[1]] = lineup.slots["OF2"].player_id
            base[of_indices[2]] = lineup.slots["OF3"].player_id
            output_rows.append(base)

    print_portfolio_checks(contests, grouped_entries, results_by_contest, selected_by_contest, lineup_usage)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(output_rows)


def print_portfolio_checks(
    contests: Dict[str, ContestSpec],
    grouped_entries: Dict[str, List[Dict[str, str]]],
    results_by_contest: Dict[str, List[LineupResult]],
    selected_by_contest: Dict[str, List[LineupResult]],
    lineup_usage: Dict[Tuple[str, ...], List[str]],
) -> None:
    print("\nPortfolio repeat check:")
    repeated = False
    for lineup_key, contest_ids in lineup_usage.items():
        unique_contests = sorted(set(contest_ids))
        if len(unique_contests) <= 1:
            continue
        repeated = True
        pair_sims = []
        for i in range(len(unique_contests)):
            for j in range(i + 1, len(unique_contests)):
                a = contests[unique_contests[i]]
                b = contests[unique_contests[j]]
                pair_sims.append(contest_similarity(a, b))
        min_sim = min(pair_sims) if pair_sims else 1.0
        max_sim = max(pair_sims) if pair_sims else 1.0
        status = "OK" if min_sim >= 0.65 else "REVIEW"
        print(
            f"{status}: lineup repeated across {len(unique_contests)} contests "
            f"(contest sim {min_sim:.2f}-{max_sim:.2f})"
        )
    if not repeated:
        print("No cross-contest lineup repeats.")

    print("\nContest regret check:")
    for contest_id, entry_rows in grouped_entries.items():
        chosen = selected_by_contest.get(contest_id, [])
        pool = results_by_contest.get(contest_id, [])
        if not chosen or not pool:
            continue
        best_ev = max(r.expected_value for r in pool)
        avg_regret = mean(best_ev - r.expected_value for r in chosen)
        max_regret = max(best_ev - r.expected_value for r in chosen)
        print(
            f"{contests[contest_id].name}: best EV=${best_ev:.2f}, "
            f"avg regret=${avg_regret:.2f}, max regret=${max_regret:.2f}, entries={len(entry_rows)}"
        )


def write_upload_csv(path: str, rows: Sequence[Dict[str, str]], template_header: Sequence[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(template_header))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in template_header})


def lineup_to_row(result: LineupResult) -> Dict[str, str]:
    lineup = result.lineup
    return {
        "P1": f"{lineup.slots['P1'].player_id} {lineup.slots['P1'].name}",
        "P2": f"{lineup.slots['P2'].player_id} {lineup.slots['P2'].name}",
        "C": f"{lineup.slots['C'].player_id} {lineup.slots['C'].name}",
        "1B": f"{lineup.slots['1B'].player_id} {lineup.slots['1B'].name}",
        "2B": f"{lineup.slots['2B'].player_id} {lineup.slots['2B'].name}",
        "3B": f"{lineup.slots['3B'].player_id} {lineup.slots['3B'].name}",
        "SS": f"{lineup.slots['SS'].player_id} {lineup.slots['SS'].name}",
        "OF1": f"{lineup.slots['OF1'].player_id} {lineup.slots['OF1'].name}",
        "OF2": f"{lineup.slots['OF2'].player_id} {lineup.slots['OF2'].name}",
        "OF3": f"{lineup.slots['OF3'].player_id} {lineup.slots['OF3'].name}",
    }


def print_results(contest: ContestSpec, results: Sequence[LineupResult]) -> None:
    print(f"\nContest: {contest.name} | field={contest.field_size} | entry_fee=${contest.entry_fee:.2f}")
    print(f"Top-heaviness: {contest.top_heaviness:.2f}")
    for idx, r in enumerate(results, 1):
        teams = r.lineup.stack_profile()
        top_team = max(teams.items(), key=lambda kv: kv[1])[0] if teams else ""
        print(
            f"{idx}. EV=${r.expected_value:.2f}  payout=${r.expected_payout:.2f}  "
            f"win={r.raw_win_rate:.3%}  dup-win={r.de_duped_win_rate:.3%}  "
            f"mean={r.mean_score:.2f}  p95={r.percentile_95:.2f}  "
            f"salary=${r.lineup.salary}  geo-own={r.ownership_geo_mean:.4f}  "
            f"dup={r.duplication_index:.6f}  saber-gap={lineup_saber_gap(r.lineup):.2f}  "
            f"stack={top_team}:{teams.get(top_team, 0)}"
        )


def print_sanity_check(players: Sequence[Player], limit: int = 10) -> None:
    diffs = sorted(
        [p for p in players if p.saber_proj > 0],
        key=lambda p: abs(p.mean - p.saber_proj),
        reverse=True,
    )
    if not diffs:
        return
    print("\nSabersim sanity check (largest model deltas):")
    for player in diffs[:limit]:
        gap = player.mean - player.saber_proj
        print(
            f"{player.name:24s} model={player.mean:5.2f}  saber={player.saber_proj:5.2f}  "
            f"delta={gap:+5.2f}  own={player.own:5.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Contest-aware DFS lineup optimizer")
    parser.add_argument("--slate", required=True, help="DK slate CSV containing projections and ownership")
    parser.add_argument("--sim-projections", help="Optional empirical sim projections JSON/CSV to overlay onto the slate")
    parser.add_argument("--entries", required=True, help="DK entry upload template CSV")
    parser.add_argument("--contest-json", help="Optional JSON file with contest field size and payout bands")
    parser.add_argument("--out", required=True, help="Output CSV path for DraftKings upload")
    parser.add_argument("--field-size", type=int, default=1000, help="Fallback contest field size if not provided")
    parser.add_argument("--sims", type=int, default=300, help="Monte Carlo simulations per lineup")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--candidate-pool", type=int, default=18, help="Candidates per slot")
    parser.add_argument("--beam-width", type=int, default=600, help="Beam search width")
    parser.add_argument("--field-population", type=int, default=400, help="Approximate field lineups to simulate")
    parser.add_argument("--status", action="append", help="Optional player status filter, e.g. Confirmed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    players = load_players(args.slate, status_filter=args.status, sim_projection_path=args.sim_projections)
    print_sanity_check(players)
    entries = read_contest_entries(args.entries)
    grouped = group_entries_by_contest(entries)
    contest_specs = load_contest_specs(args.contest_json, entries, default_field_size=args.field_size)

    results_by_contest: Dict[str, List[LineupResult]] = {}
    for contest_id, entry_rows in grouped.items():
        contest = contest_specs[contest_id]
        lineup_count = len(entry_rows)
        pool_count = max(lineup_count * 8, lineup_count + 6, 10)
        results = optimize_contest(
            players=players,
            contest=contest,
            lineup_count=pool_count,
            sims=args.sims,
            seed=args.seed,
            field_population_size=args.field_population,
            candidate_pool_size=args.candidate_pool,
            beam_width=args.beam_width,
        )
        results_by_contest[contest_id] = results
        print_results(contest, results[:lineup_count])

    write_dk_upload_from_template(
        template_path=args.entries,
        grouped_entries=grouped,
        results_by_contest=results_by_contest,
        contests=contest_specs,
        out_path=args.out,
    )

    print(f"\nWrote optimized entries to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
