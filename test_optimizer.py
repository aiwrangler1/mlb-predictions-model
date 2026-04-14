import sys
sys.path.insert(0, '/home/user/workspace/mlb-model/dfs_optimizer')
from player_model import load_player_pool, compute_gpp_scores
from contest_model import ContestParams, compute_geo_mean
from lineup_optimizer import optimize_lineup
from portfolio_builder import build_portfolio, print_portfolio_report
from utils import load_dk_entries, fill_entries_csv, validate_lineup

CSV_PATH = '/home/user/workspace/MLB_2026-04-06-707pm_DK_Main.csv'
ENTRIES_PATH = '/home/user/workspace/DKEntries-12-2.csv'

# Contest definitions (from the entries file)
contests = {
    'solo_shot': ContestParams.from_simple(
        name='MLB $7.5K Solo Shot', n_entries=4000, entry_fee=1.0,
        prize_pool=7500.0, top_pct_to_first=0.10, max_entries=1
    ),
    'five_tool': ContestParams.from_simple(
        name='MLB $4K Five-Tool Player', n_entries=1200, entry_fee=5.0,
        prize_pool=4000.0, top_pct_to_first=0.10, max_entries=5
    ),
    'hot_corner': ContestParams.from_simple(
        name='MLB $5K Hot Corner', n_entries=1200, entry_fee=3.0,
        prize_pool=5000.0, top_pct_to_first=0.10, max_entries=5
    ),
}

print("Contest parameters:")
for name, c in contests.items():
    print(f"  {c.name}: top_heavy={c.top_heavy_ratio:.3f}, target_geo_mean={c.target_geo_mean:.4f}, lambda={c.lambda_ownership:.4f}")

# Build Solo Shot portfolio (3 entries)
print("\nBuilding Solo Shot portfolio (3 lineups, 4000 entries, 10% to 1st)...")
solo_lineups = build_portfolio(
    CSV_PATH, contests['solo_shot'],
    n_lineups=3, max_exposure=0.67,
)
print_portfolio_report(solo_lineups, contests['solo_shot'])

# Validate
print("\nValidation:")
for i, lu in enumerate(solo_lineups):
    errs = validate_lineup(lu)
    status = "OK" if not errs else "ERRORS: " + "; ".join(errs)
    print(f"  L{i+1}: {status}")

# Build Five-Tool (1 entry)
print("\nBuilding Five-Tool portfolio (1 lineup, 1200 entries)...")
ft_lineups = build_portfolio(CSV_PATH, contests['five_tool'], n_lineups=1)
print_portfolio_report(ft_lineups, contests['five_tool'])

# Validate
print("\nValidation:")
for i, lu in enumerate(ft_lineups):
    errs = validate_lineup(lu)
    status = "OK" if not errs else "ERRORS: " + "; ".join(errs)
    print(f"  L{i+1}: {status}")

# Build Hot Corner (1 entry)
print("\nBuilding Hot Corner portfolio (1 lineup, 1200 entries, contrarian)...")
hc_lineups = build_portfolio(CSV_PATH, contests['hot_corner'], n_lineups=1)
print_portfolio_report(hc_lineups, contests['hot_corner'])

# Validate
print("\nValidation:")
for i, lu in enumerate(hc_lineups):
    errs = validate_lineup(lu)
    status = "OK" if not errs else "ERRORS: " + "; ".join(errs)
    print(f"  L{i+1}: {status}")

# Fill and write the entries CSV
all_lineups = solo_lineups + ft_lineups + hc_lineups  # 5 total
print(f"\nTotal lineups built: {len(all_lineups)}")

if all_lineups:
    fill_entries_csv(ENTRIES_PATH, '/home/user/workspace/DKEntries-OPTIMIZER.csv', all_lineups)
    print("\nDone! Lineups written to DKEntries-OPTIMIZER.csv")
else:
    print("ERROR: No lineups were generated!")
