# DFS True Optimizer — Mathematical Specification

## Core Concepts

### 1. Geometric Mean Ownership (GeoMean)
```
GeoMean = (∏ own_i)^(1/n) = exp( mean(log(own_i)) )

Target GeoMean from contest size:
  max_dupes = desired maximum duplicate lineups in field
  geo_mean_target = (max_dupes / n_entries)^(1/n_players)

  e.g., 4300-entry contest, 10 players, want ≤5 dupes:
    geo_mean_target = (5/4300)^(1/10) = 0.4968
  e.g., 1189-entry contest, want ≤2 dupes:
    geo_mean_target = (2/1189)^(1/10) = 0.5303
```

### 2. Product Ownership
```
product_own = ∏ own_i  (raw probability of exact lineup duplication)
expected_dupes = product_own × n_entries
```

### 3. Lineup Score Distribution
Model each player's DK score from SaberSim percentiles:
- Fit a log-normal: μ and σ from dk_50_percentile and dk_std
- Or use empirical CDF from [dk_25, dk_50, dk_75, dk_95, dk_99]
- Lineup score = sum of player scores (with game correlation)

### 4. Expected Win Rate (EWR)
```
EWR = P(lineup_score > max(field_score_1, ..., field_score_{N-1}))

Via Monte Carlo (n_sims iterations):
  1. For each sim: sample your lineup's total score from player distributions
  2. Sample N-1 field lineups by: for each slot, weight by player ownership
  3. Score each field lineup from player distributions
  4. EWR += I(your_score > all field scores) / n_sims
```

### 5. Expected Value (EV)
```
EV = Σ_k (prize_k × P(finish in prize bucket k)) - entry_fee

For top-heavy payout (10% to first):
  P(win) ≈ EWR (above)
  EV ≈ prize_1st × EWR - entry_fee
  (simplified, ignoring smaller prizes for dominantly top-heavy structures)
```

### 6. Leverage Score (per player)
```
leverage = (dk_99_percentile / dk_50_percentile) × (1 / adj_own)
         = ceiling_ratio / ownership

Higher = more upside per unit of field exposure
```

### 7. GPP Score (FTN-style)
```
target_own_i = dk_p99_i / Σ(dk_p99_j for all j)  # proportional to ceiling
gpp_score_i = target_own_i / adj_own_i             # >1 = underowned, <1 = overowned
```

### 8. ILP Optimizer Objective
```
maximize: Σ(x_i × score_ceiling_i) + λ × Σ(x_i × log(adj_own_i + ε))
         [ceiling term]               [negative = ownership penalty]

where x_i ∈ {0,1}, λ = f(contest_size, payout_top_heavy_ratio)
λ increases with field size and decreases with flat payout structures

Constraints:
  salary:    Σ(x_i × salary_i) ≤ 50000
  positions: Σ(x_i [for pos]) = required_count per position
  team:      Σ(x_i [for team]) ≤ max_per_team (default 5 for MLB)
  geomean:   Σ(x_i × log(own_i)) / n ≤ log(geo_mean_target)  [optional hard constraint]
  lock/fade: x_i = 1 for locked players, x_i = 0 for faded
```

## Files to Build

### /home/user/workspace/mlb-model/dfs_optimizer/
```
__init__.py
player_model.py     — score distribution fitting, correlation model
contest_model.py    — payout parsing, EWR/EV computation, field simulation  
lineup_optimizer.py — ILP with PuLP, GeoMean constraint, leverage objective
portfolio_builder.py — multi-lineup portfolio, diversification, contest routing
utils.py            — DK formatting, salary validation, roster checks
```

## DraftKings Roster Rules (Classic)
- Slots: P P C 1B 2B 3B SS OF OF OF (10 players)
- Salary cap: $50,000
- Max 5 hitters from one team
- Multi-eligible: a player with "3B/SS" can fill 3B or SS slot
- Player ID format: "Player Name (DFS_ID)"

## SaberSim CSV Format
Columns used:
- DFS ID, Name, Pos, Order, Team, Opp, Status, Salary
- SS Proj, Adj Own, dk_std
- dk_25_percentile, dk_50_percentile, dk_75_percentile, dk_95_percentile, dk_99_percentile
- HR, SB, Saber Team, Saber Total
