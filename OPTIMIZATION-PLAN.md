# MLB Model v3.1 — Optimization Plan
**Based on backtest analysis: April 1–6, 2026 (62 resolved games)**

---

## What the Data Says

### The Total Problem — Both Model AND Vegas Are Systematically Biased

| Source | Avg Total | Gap from Actual |
|--------|-----------|-----------------|
| Our model (Apr 6) | 9.65 | **+0.78 over** |
| Vegas lines (Apr 6) | 8.23 | **-0.64 under** |
| Actual 2026 avg | 8.87 | — |
| 2025 full season | 8.52 | — |

**Key finding: Neither is right. The actual sits between them.**
- Our model overshoots by ~0.78 runs/game
- Vegas undershoots by ~0.64 runs/game
- Vegas is closer to correct, but BOTH are wrong

This partially explains the early-season "overs" trend:
- [TeamRankings shows](https://www.teamrankings.com/mlb/trends/ou_trends/) CWS overs at 87.5%, TB 85.7%, HOU 77.8%, WSH 75%
- Vegas set totals too low → over is hitting at high rate
- Our model set totals too high → we would have flagged few "over" edges

### The Variance Problem — Total Uncertainty is Huge

The 2026 early-season distribution is not a clean bell curve:

| Range | Count | % |
|-------|-------|---|
| 0–4 runs | 13 | 21% |
| 5–6 runs | 8 | 13% |
| 7–8 runs | 9 | 15% |
| 9–10 runs | 11 | 18% |
| 11–14 runs | 13 | 21% |
| 15+ runs | 8 | 13% |

Standard deviation: 5.0 runs. That's enormous — nearly ±5 on an 8.87 mean.

The distribution is **bimodal with fat tails**: lots of 1-4 run games AND lots of 11+ run games. The model projecting smooth 9.5 averages will always miss the extremes. This is partially a feature of early-season baseball, not a pure model flaw.

### The ML Accuracy — To Be Determined

We have 62 actuals but need to backfill April 1–5 predictions to measure ML accuracy. Backfill requires retroactively running the model on those dates (done below in the plan).

---

## Root Causes of Total Over-Projection

### 1. Steamer Projections Are Pre-Season
The Steamer projections we blend from FanGraphs are based on projected full-season performance. Early-season hitters start cold (March–April OPS is historically ~5–8% lower than full-season averages). Our model blends Steamer too heavily in April.

**Fix**: Add an early-season calibration multiplier. April OPS is historically ~5% lower than the Steamer projection. Reduce batter contact probabilities by ~3–5% in April.

### 2. League Average Calibration Uses 2025 Data
`get_league_averages()` returns 2025 averages (p_1b, p_2b, p_hr, etc.). Early 2026 is slightly different:
- Actual 2026 avg: 8.87 (vs 2025: 8.52) — **2026 is actually scoring MORE**
- But our model still over-projects because Steamer is too optimistic on individuals

**Fix**: After ~2 weeks of 2026 data, blend 2026 YTD rates into league averages.

### 3. Bullpen Transition May Over-Count Runs
The starter→bullpen transition happens at `avg_sp_ip`. If the bullpen era is too generous or bullpen usage is mis-timed, late-inning scoring inflates.

**Fix**: Audit `TEAM_BULLPEN_2025` ERAs against actual 2026 early performance.

### 4. Park Factors Are Stale
`PARK_FACTORS` uses multi-year averages. Coors Field in April is genuinely different from Coors in July (altitude effect smaller when cold). Several other parks have had changes between seasons.

**Fix**: Weight park factors more toward recent/same-season data when available.

---

## What to Do About Betting NOW

### Totals Strategy

Our model says 9.65 average total, Vegas says 8.23, actual is 8.87. The real implied edge:

- **OVER on low-set lines (<7.5):** Vegas is undershooting by ~0.6. When a line is set at 7.0–7.5 (like DET@MIN today), reality is likely 8.5–9. **Lean over on the lowest-set lines.**
- **Avoid over on high-set lines (>9.5):** Our model and Vegas both price these high. The distribution's fat low tail (21% of games under 4 runs) means these are traps.
- **Don't use model total as an absolute — use the gap to Vegas:** A +2.0 model-vs-Vegas gap means both are probably pointing at a higher total than the line. A +0.1 gap means the line is probably right.

### Moneyline Strategy

The model's ML edges (STL, KC, ATL today) are more reliable than totals because:
- The actual ML direction doesn't change with total projection errors
- A game can have 5 runs OR 15 runs, but STL still wins or loses
- The calibration error on totals doesn't compound into ML accuracy

Proceed with ML bets at stated confidence levels. The total over-projection doesn't invalidate the win probability model.

---

## Optimization Roadmap

### Priority 1 — Seasonal Calibration Multiplier (Quick, High Impact)
Add to `model_adjustments.py`:

```python
def get_early_season_multiplier(game_date: str) -> float:
    """
    Reduce batter contact probabilities in April (cold weather, rust).
    Historical: April OPS ~5% lower than full-season.
    """
    month = int(game_date[5:7])
    if month == 3:   return 0.94  # spring training warmup effect
    if month == 4:   return 0.97  # slight April depression
    if month == 5:   return 0.99  # near full strength
    return 1.00                    # June+ at full season rates
```

Expected impact: Reduces model total projection by ~0.3–0.5 runs. Gets us closer to actual.

### Priority 2 — Live 2026 League Average Blending (Medium, High Impact)
Modify `mlb_data.py` to blend current-season running averages into league baselines:

```python
def get_calibrated_league_averages(season_data_path=None):
    """After ~3 weeks, blend 2026 YTD rates (70%) + 2025 full season (30%)."""
```

Will self-correct as the season progresses. The model becomes more accurate by May.

### Priority 3 — Backfill April 1–5 Predictions (Analytical)
Re-run the model on April 1–5 games retroactively to get ML accuracy numbers:

```python
# Run for each past date to fill predictions table
await record_actuals_for_date("2026-04-01")  # already done
# Then re-simulate each game to add predictions
```

Once predictions exist for those 62 games, we can compute:
- ML accuracy rate
- Calibration (does 60% → 60% win rate?)
- Edge accuracy (when we had 5%+ edge, did we win 55%+ of those?)

### Priority 4 — Total Model Recalibration (Medium)
Add a post-hoc calibration layer that adjusts final total projection based on historical model bias:

```python
def calibrate_total(raw_model_total: float, game_date: str) -> float:
    """
    Apply learned calibration to model total projection.
    Based on 62-game analysis: model is +0.78 over actual in early 2026.
    """
    month = int(game_date[5:7])
    # April adjustment: model historically over-projects by ~0.8 in April
    if month == 4: 
        return raw_model_total * 0.92   # ~8% downward calibration
    return raw_model_total
```

### Priority 5 — Variance Modeling for Totals (Harder, Later)
The bimodal distribution (21% games under 4, 13% games over 14) is real and can be modeled:
- Add pitcher dominant game probability (when ace has high K% and opponent OPS < .680)
- Add "blowout game" probability (when one team is 65%+ favorite)
- Price "total extremes" differently than the mean projection

---

## Validation Protocol

### Week 1 (Now — April 13):
1. Implement Priority 1 (seasonal multiplier) on a `feature/calibration` branch
2. Backfill April 1–5 predictions — get our ML accuracy baseline
3. Run the backtest report daily after games finish

### Week 2 (April 13–20):
1. With 80+ games in the DB, ML accuracy will have statistical meaning
2. If ML accuracy > 53%, the model has genuine edge on picks
3. If total calibration error drops below 0.5 runs, calibration is working
4. Merge calibration branch to main

### Month 1 Checkpoint (May 1):
1. Run full analytics report — calibration curves, ROI by bet type
2. Decide if Kelly-sized ML bets are warranted based on accuracy data
3. Determine if total model needs deeper structural fix or just calibration

---

## Bottom Line on Betting Today

| Bet Type | Confidence | Why |
|---|---|---|
| ML favorites with 5%+ edge | **Medium-High** | Model direction is likely right; total noise doesn't affect ML |
| Over on lines set ≤7.5 | **Medium** | Vegas is undershooting vs actual (−0.64 gap), model and Vegas both say over |
| Over on lines set >9.0 | **Low** | Model over-projects; fat lower tail means many games come in under |
| Under on lines set 8.0–8.5 | **Low-Medium** | Actual 2026 avg is 8.87 — right at these lines, too close to bet |
| Exact totals | **Don't bother** | ±5.0 std dev makes totals a coinflip |
