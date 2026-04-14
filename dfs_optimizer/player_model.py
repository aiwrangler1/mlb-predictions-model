"""
Player score distribution modeling for DFS optimization.
Fits log-normal distributions from SaberSim percentile data.
"""
import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PlayerModel:
    name: str
    dfs_id: int
    team: str
    opp: str
    pos: str          # e.g. "3B/SS" — slash-separated eligible positions
    salary: int
    ss_proj: float    # SaberSim median projection
    adj_own: float    # projected ownership (0-100 scale, NOT fraction)
    dk_std: float
    dk_p25: float
    dk_p50: float
    dk_p75: float
    dk_p95: float
    dk_p99: float
    saber_total: float   # game total
    saber_team: float    # team total
    order: Optional[int] = None  # batting order, None for pitchers

    # Derived fields (computed at init)
    own_frac: float = field(init=False)       # adj_own / 100
    log_own: float = field(init=False)        # log(own_frac), for GeoMean calc
    ceiling_ratio: float = field(init=False)  # dk_p99 / max(dk_p50, 1)
    leverage_score: float = field(init=False) # dk_p99 / (adj_own + 0.1)
    gpp_score: float = field(init=False)      # set externally after pool is known

    def __post_init__(self):
        self.own_frac = max(self.adj_own / 100, 0.001)
        self.log_own = np.log(self.own_frac)
        self.ceiling_ratio = self.dk_p99 / max(self.dk_p50, 1.0)
        self.leverage_score = self.dk_p99 / max(self.adj_own, 0.1)
        self.gpp_score = 0.0

    @property
    def positions(self) -> list:
        return [p.strip() for p in self.pos.split('/')]

    def is_eligible(self, slot: str) -> bool:
        return slot in self.positions

    def sample_score(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        """Sample DK fantasy scores from empirical percentile distribution."""
        # Fit log-normal from p50 and dk_std using moment matching
        # But protect against zero-projection pitchers
        if self.ss_proj <= 0:
            return rng.exponential(self.dk_std + 0.1, size=n)

        # Use normal distribution approximation (dk scores can be negative for pitchers)
        mu = self.ss_proj
        sigma = max(self.dk_std, 0.1)
        samples = rng.normal(mu, sigma, size=n)
        return samples

    def format_dk(self) -> str:
        """Format as DraftKings player string: 'Name (ID)'"""
        return f"{self.name} ({self.dfs_id})"


def load_player_pool(csv_path: str, status_filter: str = 'Confirmed') -> list:
    """Load and filter player pool from SaberSim CSV."""
    import pandas as pd
    df = pd.read_csv(csv_path)

    if status_filter:
        df = df[df['Status'] == status_filter]

    # Filter to players with meaningful projections
    df = df[df['SS Proj'] > 0].copy()

    players = []
    for _, row in df.iterrows():
        try:
            p = PlayerModel(
                name=str(row['Name']),
                dfs_id=int(row['DFS ID']),
                team=str(row['Team']),
                opp=str(row['Opp']),
                pos=str(row['Pos']),
                salary=int(row['Salary']),
                ss_proj=float(row['SS Proj']),
                adj_own=float(row.get('Adj Own', 0.1)),
                dk_std=float(row.get('dk_std', 5.0)),
                dk_p25=float(row.get('dk_25_percentile', 0)),
                dk_p50=float(row.get('dk_50_percentile', 0)),
                dk_p75=float(row.get('dk_75_percentile', 0)),
                dk_p95=float(row.get('dk_95_percentile', 0)),
                dk_p99=float(row.get('dk_99_percentile', 0)),
                saber_total=float(row.get('Saber Total', 8.5)),
                saber_team=float(row.get('Saber Team', 4.25)),
                order=int(row['Order']) if 'Order' in row and not _is_na(row.get('Order')) else None,
            )
            players.append(p)
        except Exception as e:
            continue

    return players


def _is_na(val):
    """Check if a value is NaN or None."""
    import math
    if val is None:
        return True
    try:
        return math.isnan(float(val))
    except (TypeError, ValueError):
        return False


def compute_gpp_scores(players: list) -> None:
    """Compute GPP scores (target ownership vs projected) for all players."""
    total_p99 = sum(max(p.dk_p99, 0.01) for p in players)
    for p in players:
        target_own = (p.dk_p99 / total_p99) * 100  # target % ownership
        p.gpp_score = target_own / max(p.adj_own, 0.1)
        # >1.0 means underowned relative to ceiling, <1.0 means overowned
