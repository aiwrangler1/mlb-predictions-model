# MLB Predictions Model — Sprint Backlog

## Tier 2 — Medium Impact (Next Sprint)

### Velocity / Stuff Quality Trends
- Pull Statcast pitch-level data from Baseball Savant
- Track fastball velocity over last 3 starts vs season average
- A 1-2 mph velocity drop is a leading indicator of blowup starts
- Apply as a pitcher quality multiplier (similar to fatigue factor)
- Source: `https://baseballsavant.mlb.com/statcast_search` CSV exports

### Travel / Rest Schedule
- Compute days since last game and travel distance between cities
- Cross-country flights (e.g., SEA→NYY) for day games after night games = penalty
- West Coast teams playing East Coast day games historically underperform
- Use stadium coordinates (already in model_adjustments.py STADIUM_COORDS)
- Haversine distance between consecutive game venues
- Adjustment: 2-3% offensive penalty for long-haul travel + short rest

### Catcher Framing
- Elite framers (Austin Hedges tier) add ~1-2% called strike rate on borderline pitches
- Worth ~15 runs/year vs poor framers
- Lineup data already gives us the catcher
- Source: Baseball Savant catcher framing leaderboard
- Implementation: adjust pitcher K% and BB% based on catcher framing runs

### Batter vs Specific Pitcher History
- MLB API: `https://statsapi.mlb.com/api/v1/people/{batter_id}/stats?stats=vsPlayer&opposingPlayerId={pitcher_id}&group=hitting`
- Only use when sample >= 15 PA (below that, noise dominates)
- Blend with Log5 matchup: 80% Log5 + 20% head-to-head (if sufficient sample)

## Tier 3 — Smaller Edge, Accumulates Over Volume

### Day/Night Splits
- MLB API statSplits with sitCodes for day/night
- Some hitters have 50+ OPS point gaps between day and night
- Apply as batter adjustment similar to platoon splits

### Home/Away Splits
- Beyond team-level home field advantage
- Individual players with extreme home/away splits (e.g., Coors-boosted Rockies hitters)
- Already partially captured by park factors, but individual splits add signal

### Pitch Mix / Arsenal Matching
- A batter who crushes sliders vs a slider-heavy pitcher = different matchup
- Requires Statcast pitch-type data and batter performance by pitch type
- Highest complexity to implement, but real edge for player props
- Source: Baseball Savant pitch-level data

### Divisional Familiarity
- Teams in same division play 19 times/year
- Familiarity can suppress offense (hitters see same pitchers repeatedly)
- Or boost it (hitters learn pitcher tendencies)
- Historically a wash on aggregate, but could matter for specific matchups

### Managerial Tendencies
- Quick-hook managers (Kevin Cash style) = earlier bullpen transition
- Some managers platoon heavily = different lineup vs LHP/RHP
- Could adjust starter IP estimate based on manager profile

### Stolen Base Modeling
- Current model uses a flat SB attempt rate
- Better: model SB probability based on runner speed, pitcher hold time, catcher pop time
- Matters more for DFS than for game totals

### Injury Reports / IL Tracking
- Monitor daily injury reports for key players
- A team missing their best hitter or closer fundamentally changes the line
- Could pull from MLB API transactions endpoint or Rotowire
