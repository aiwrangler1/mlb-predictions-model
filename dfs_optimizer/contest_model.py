"""
Contest parameters, payout parsing, and EWR/EV computation.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ContestParams:
    name: str
    n_entries: int
    entry_fee: float
    prize_pool: float
    payout_structure: dict  # {rank: prize_amount} e.g. {1: 1000, 2: 500, ...}
    max_entries_per_user: int = 1
    n_players_per_lineup: int = 10  # 10 for Classic, 6 for Showdown

    # Derived
    top_heavy_ratio: float = field(init=False)   # prize_1st / prize_pool
    target_geo_mean: float = field(init=False)   # computed from contest size
    lambda_ownership: float = field(init=False)  # ILP penalty weight

    def __post_init__(self):
        max_prize = max(self.payout_structure.values()) if self.payout_structure else self.prize_pool
        self.top_heavy_ratio = max_prize / self.prize_pool

        # Target GeoMean: want ≤ max_desired_dupes duplicates
        # More top-heavy → fewer acceptable dupes
        if self.top_heavy_ratio > 0.5:  # winner-take-all style
            max_dupes = max(1, self.n_entries // 500)
        elif self.top_heavy_ratio > 0.25:
            max_dupes = max(2, self.n_entries // 300)
        else:
            max_dupes = max(3, self.n_entries // 200)

        self.target_geo_mean = (max_dupes / max(self.n_entries, 1)) ** (1.0 / self.n_players_per_lineup)

        # Lambda: ownership penalty weight for ILP
        # Scales with log(n_entries) and top_heavy_ratio
        base_lambda = np.log10(max(self.n_entries, 10)) / 4  # 0.25 for 100 entries, ~1.0 for 10K
        self.lambda_ownership = base_lambda * self.top_heavy_ratio

    def expected_dupes(self, geo_mean: float) -> float:
        """Expected duplicate lineups in the field for a given GeoMean."""
        product_own = geo_mean ** self.n_players_per_lineup
        return product_own * self.n_entries

    def prize_for_rank(self, rank: int) -> float:
        """Get prize for finishing at a given rank (0 if not paid)."""
        if rank in self.payout_structure:
            return self.payout_structure[rank]
        # Find the highest rank key that's >= rank
        paid_ranks = sorted(self.payout_structure.keys())
        for k in paid_ranks:
            if rank <= k:
                return self.payout_structure[k]
        return 0.0

    @classmethod
    def from_simple(cls, name: str, n_entries: int, entry_fee: float,
                    prize_pool: float, top_pct_to_first: float = 0.10,
                    max_entries: int = 1) -> 'ContestParams':
        """Quick constructor for common GPP structures."""
        first_place = prize_pool * top_pct_to_first
        # Approximate payout distribution (top 20% pay roughly)
        n_paid = max(1, int(n_entries * 0.18))
        payout = {}
        remaining = prize_pool

        # Top prizes: 1st gets top_pct, then diminishing
        payout[1] = first_place
        remaining -= first_place

        if n_paid > 1:
            top10_pool = prize_pool * 0.30 - first_place  # next 10% get 30% total
            top10 = max(1, n_paid // 10)
            per_top10 = max(0, top10_pool / max(top10, 1))
            for r in range(2, top10 + 2):
                payout[r] = max(entry_fee, per_top10)
                remaining -= payout[r]

            # Rest of paid spots get minimum cash (1.5x entry fee usually)
            for r in range(top10 + 2, n_paid + 1):
                payout[r] = entry_fee * 1.5

        return cls(name=name, n_entries=n_entries, entry_fee=entry_fee,
                   prize_pool=prize_pool, payout_structure=payout,
                   max_entries_per_user=max_entries)


def compute_geo_mean(lineup_players: list) -> float:
    """Compute geometric mean of player ownerships for a lineup."""
    if not lineup_players:
        return 0.0
    log_owns = [p.log_own for p in lineup_players]
    return float(np.exp(np.mean(log_owns)))


def compute_lineup_ev(
    lineup_players: list,
    contest: ContestParams,
    field_players: list,
    n_sims: int = 2000,
    seed: int = 42,
) -> dict:
    """
    Compute expected value, EWR, and score stats for a lineup via Monte Carlo.

    Field simulation:
    1. Sample N-1 field lineups by sampling players proportional to ownership
    2. Score your lineup and each field lineup
    3. Count how often your lineup finishes 1st
    4. Compute expected prize payout
    """
    rng = np.random.default_rng(seed)
    n_field = contest.n_entries - 1
    n_slots = contest.n_players_per_lineup

    # Build ownership-weighted field sampling distribution
    owns = np.array([p.own_frac for p in field_players])
    owns = owns / owns.sum()  # normalize

    # Pre-sample all player scores for speed: shape (n_sims, n_players)
    all_scores = np.column_stack([
        p.sample_score(rng, n_sims) for p in field_players
    ])  # (n_sims, n_field_players)

    # Your lineup scores — find indices in field_players
    my_indices = []
    field_name_idx = {p.name: i for i, p in enumerate(field_players)}
    for p in lineup_players:
        if p.name in field_name_idx:
            my_indices.append(field_name_idx[p.name])

    if not my_indices:
        # Fallback: sample directly
        my_scores = np.array([
            p.sample_score(rng, n_sims) for p in lineup_players
        ]).sum(axis=0)
    else:
        my_scores = all_scores[:, my_indices].sum(axis=1)

    # Sample field lineups and their scores
    # For memory efficiency, batch field sampling
    # Each field lineup: sample n_slots players (with replacement, weighted by own)
    # Cap field size for memory
    n_field_sample = min(n_field, 500)

    field_lineup_indices = rng.choice(
        len(field_players),
        size=(n_sims, n_field_sample, n_slots),
        p=owns
    )  # (n_sims, n_field_sample, n_slots)

    # Score each field lineup using pre-sampled scores
    field_scores = all_scores[
        np.arange(n_sims)[:, None, None],
        field_lineup_indices
    ].sum(axis=-1)  # (n_sims, n_field_sample)

    # Find rank for each sim
    my_scores_2d = my_scores[:, None]  # (n_sims, 1)
    beats_me = (field_scores > my_scores_2d).sum(axis=1)  # (n_sims,)
    # Scale beats_me to reflect actual field size
    if n_field_sample < n_field:
        beats_me = (beats_me * n_field / n_field_sample).astype(int)
    my_ranks = beats_me + 1  # 1-indexed rank

    # Compute prize for each rank
    prizes = np.vectorize(contest.prize_for_rank)(my_ranks)

    # Key metrics
    win_rate = float(np.mean(my_ranks == 1))
    ev = float(np.mean(prizes) - contest.entry_fee)
    avg_score = float(np.mean(my_scores))
    score_p50 = float(np.percentile(my_scores, 50))
    score_p90 = float(np.percentile(my_scores, 90))
    score_p99 = float(np.percentile(my_scores, 99))

    geo_mean = compute_geo_mean(lineup_players)
    expected_dupes = contest.expected_dupes(geo_mean)

    return {
        'ev': round(ev, 4),
        'win_rate': round(win_rate, 6),
        'win_rate_pct': round(win_rate * 100, 4),
        'avg_score': round(avg_score, 2),
        'score_p50': round(score_p50, 2),
        'score_p90': round(score_p90, 2),
        'score_p99': round(score_p99, 2),
        'geo_mean': round(geo_mean, 4),
        'expected_dupes': round(expected_dupes, 2),
        'top_heavy_ratio': contest.top_heavy_ratio,
        'avg_rank': round(float(np.mean(my_ranks)), 1),
    }
