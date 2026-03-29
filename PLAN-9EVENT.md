# Plan: 9-Event Model Upgrade

## Architecture

### Core Concept: `EventVector` dict with exactly 9 keys
Every probability vector in the entire system will have these 9 keys:
```
p_1b, p_2b, p_3b, p_hr, p_bb, p_hbp, p_k, p_bip_out, p_prod_out
```
**No `p_out` anywhere.** The old 7-event `p_out` is eliminated system-wide.

### Branch: `feature/9-event-model`

## Implementation Steps (in order)

### Step 1: `event_model.py` — Central abstraction (NEW FILE)
- Define `KEYS_9 = ["p_1b", "p_2b", "p_3b", "p_hr", "p_bb", "p_hbp", "p_k", "p_bip_out", "p_prod_out"]`
- `make_9event(probs_7: dict, k_rate: float = None) -> dict` — converts any 7-event dict
- `normalize_9event(vec: dict) -> dict` — ensures sum = 1.0, all 9 keys present
- `rebalance_outs(vec: dict) -> dict` — after adjusting hit/walk probs, redistributes out categories proportionally
- `validate(vec: dict) -> bool` — asserts all 9 keys present, sum ≈ 1.0, no `p_out`
- `LEAGUE_AVG_9` — the Retrosheet-mined averages (already computed: 186k PA)

### Step 2: Update `mlb_data.py` — Data ingestion
- `get_batter_stats()` → return dict with 9-event keys (convert at source)
- `get_pitcher_stats()` → return dict with 9-event keys (convert at source)
- `get_league_averages()` → return 9-event dict
- Each function calls `event_model.make_9event()` before returning

### Step 3: Update `projections.py` — Steamer + blending
- `get_bullpen_probs()` → return 9-event dict
- `blend_probability_vectors()` → blend all 9 keys
- Steamer parsing → produce 9-event vectors

### Step 4: Update `model_adjustments.py` — ALL adjustment functions
- `apply_park_factors()` → adjust hit keys, call `rebalance_outs()`
- `apply_platoon_adjustment()` → blend 9 keys, normalize
- `apply_form_adjustment()` → adjust hit keys, call `rebalance_outs()`
- `apply_weather_adjustment()` → adjust hit keys, call `rebalance_outs()`
- `apply_workload_adjustment()` → adjust hit keys, call `rebalance_outs()`
- **Every function uses `rebalance_outs()` instead of manually setting `p_out`**

### Step 5: Update `game_context.py` — Umpire + BP availability
- `apply_umpire_adjustment()` → adjust `p_k` directly (not `p_out`), `rebalance_outs()`
- `apply_bullpen_availability()` → adjust hit keys, `rebalance_outs()`

### Step 6: Update `log5_engine.py` — Matchup computation
- `matchup_probability_vector()` → renamed to `matchup_probability()`, works with 9 keys
- Uses `event_model.KEYS_9` for the Log5 formula iteration

### Step 7: Update `monte_carlo.py` — Simulation engine
- `simulate_plate_appearance()` → draws from 9 outcomes
- `simulate_half_inning()` → handle K (no advancement), BIP_OUT (advancement possible), PROD_OUT (sac fly/bunt)
- `advance_runners()` → replaced with `advance_runners_9event()` that handles all 9 outcomes
- Remove old 7-event `advance_runners()` entirely

### Step 8: Update `server.py` — Pipeline wiring
- Remove all `matchup_probability_vector` calls → use new `matchup_probability`
- Remove all `convert_7_to_9_event` calls (data comes in as 9-event natively)
- Adjustment functions just work (they all use `rebalance_outs()`)

### Step 9: Tests
- Unit test: `event_model.validate()` on every vector produced by every function
- Integration test: simulate 1000 half-innings with league avg → should produce ~0.5 runs/half-inning
- Integration test: full game simulation → total runs should be 8-10 for league avg teams
- Comparison test: run LAA @ HOU and compare to v3.1 output and DK line

## Parallelization Strategy

Three independent subagents:
1. **Agent A**: Steps 1-2 (event_model.py + mlb_data.py)
2. **Agent B**: Steps 3-5 (projections.py + model_adjustments.py + game_context.py)  
3. **Agent C**: Steps 6-7 (log5_engine.py + monte_carlo.py)

Agent B and C depend on Agent A's `event_model.py`, so:
- Agent A runs first and saves `event_model.py` to workspace
- Then Agents B and C run in parallel, importing from the saved file

Step 8 (server.py) runs after all three complete.
Step 9 (tests) runs last.

## Safety Rules
1. NO function may reference `p_out` — grep for it before committing
2. EVERY function that returns a probability vector must call `validate()` 
3. The branch does NOT merge to main until all tests pass
4. Compare output against v3.1 and DK lines before merge
