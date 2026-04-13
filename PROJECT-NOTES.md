# MLB Model — Project Notes & Lessons Learned

> Reference file to prevent repeated mistakes. READ THIS before any dashboard build, model run, or data pipeline work.

---

## Critical Rules

### 1. NEVER Hardcode Player Data
- **Pitchers**: Always pull from MLB Stats API (`statsapi.mlb.com`). Never guess or assume starters.
- **Hitters**: Must come from model sim output or verified stats. No fabricated projections.
- **Lesson**: Dashboard v1 had 25/31 wrong pitcher names because they were hardcoded from assumptions. Cost significant tokens to diagnose and fix.

### 2. Verify Model Inputs Before Deploying
- Before presenting projections as actionable, cross-check that the **pitchers the model used** match **actual confirmed starters**.
- Check MLB probable pitchers endpoint: `https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=YYYY-MM-DD&hydrate=probablePitcher`
- **Lesson**: ARI@PHI model used Andrew Painter but actual starter was Zach Pop. The projections were meaningless for that game.

### 3. Flag Uncertainty, Don't Hide It
- If a starter is TBD, flag it visibly in the dashboard (⚠️ icon + warning banner).
- If model inputs don't match reality, mark the game as unreliable rather than showing it clean.

### 4. The Odds API — Be Efficient
- Free tier: 500 requests/month. Currently ~421 remaining.
- Use `bookmakers=draftkings` filter to minimize response size.
- `/v4/sports` endpoint is free (doesn't count toward quota).
- Cache responses to `/tmp/dk_odds_cache.json` to avoid redundant calls.
- Historical odds require paid tier — not available.

### 5. Model Known Biases
- **Total over-projection**: +0.97 runs/game. Apply -0.78 calibration constant.
- **Backtest results** (36 games): 55.6% ML accuracy, 0.329 correlation on totals.

### 6. DraftKings Rules (Don't Violate)
- **Classic**: P,P,C,1B,2B,3B,SS,OF,OF,OF. $50K cap. Max 5 hitters from one team. Min 2 teams.
- **Showdown**: 1 CPT (1.5x) + 5 FLEX. $50K cap. Min 1 player per team.
- Always validate roster legality BEFORE optimizing for projection upside.

### 7. Dashboard Data Pipeline (Correct Order)
1. Fetch confirmed starters from MLB Stats API
2. Run model sims with verified starters
3. Fetch DK odds from The Odds API (single call, cache result)
4. Build K props from real pitcher stats (K/start, IP, ERA, K/9)
5. Build hitter props from model sim output
6. Assemble dashboard HTML with all verified data
7. Deploy and screenshot-verify before sharing

### 8. AGENTS.md
- Always read `/home/user/workspace/mlb-model/AGENTS.md` before making repo changes.
- If another agent is working on the repo, do NOT make changes.

---

## File Locations

| File | Purpose |
|------|---------|
| `/tmp/dk_odds_cache.json` | Cached DK odds (refresh each slate) |
| `/tmp/pitcher_props_correct.json` | Verified pitcher stats from MLB API |
| `/tmp/apr12_sims.json` | Model sim output for current slate |
| `/home/user/workspace/mlb-betting-dashboard/index.html` | Live dashboard |
| `/home/user/workspace/mlb-model/data/mlb_backtest.db` | Backtest database |

---

*Last updated: April 12, 2026*
