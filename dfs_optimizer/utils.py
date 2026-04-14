"""
Utility functions: DK CSV formatting, validation, contest parsing.
"""
import csv
from collections import Counter
from lineup_optimizer import LineupResult, DK_CLASSIC_SLOTS


def load_dk_entries(entries_csv: str) -> list:
    """Load the DraftKings entries file."""
    rows = []
    with open(entries_csv, newline='', encoding='utf-8-sig') as f:
        all_rows = list(csv.reader(f))

    header = all_rows[0]
    for row in all_rows[1:]:
        if row and row[0].strip().isdigit():
            rows.append({
                'entry_id': row[0],
                'contest_name': row[1],
                'contest_id': row[2],
                'entry_fee': row[3],
                'slots': row[4:14],
                'raw': row,
            })
    return rows


def fill_entries_csv(
    entries_csv: str,
    output_csv: str,
    lineups: list,  # list of LineupResult
):
    """
    Fill the DraftKings entries CSV with the given lineups.

    entries_csv: original DK upload file
    output_csv: where to write the filled version
    lineups: list of LineupResult, one per entry (in order)
    """
    all_rows = []
    with open(entries_csv, newline='', encoding='utf-8-sig') as f:
        all_rows = list(csv.reader(f))

    # Find actual entry rows (those where first column is a digit = entry ID)
    entry_indices = []
    for i, row in enumerate(all_rows):
        if row and row[0].strip().isdigit():
            entry_indices.append(i)

    # Fill each entry row with corresponding lineup
    for idx, (row_i, lineup) in enumerate(zip(entry_indices, lineups)):
        slots = lineup.format_dk_slots()
        # Ensure we have exactly 10 slots
        while len(slots) < 10:
            slots.append('')
        # Expand row if needed
        while len(all_rows[row_i]) < 14:
            all_rows[row_i].append('')
        all_rows[row_i][4:14] = slots[:10]

    # Write output
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(all_rows)

    print(f"Written {len(lineups)} lineups to {output_csv}")


def validate_lineup(lineup: LineupResult, salary_cap: int = 50000) -> list:
    """Return list of validation errors (empty = valid)."""
    errors = []

    if lineup.salary > salary_cap:
        errors.append(f"Over salary cap: ${lineup.salary:,} > ${salary_cap:,}")

    teams = set(p.team for p in lineup.players)
    if len(teams) < 2:
        errors.append(f"Only {len(teams)} team(s) — need at least 2")

    # Check max 5 non-pitchers per team
    team_hitter_counts = Counter(
        p.team for p in lineup.players if 'P' not in p.positions
    )
    for team, count in team_hitter_counts.items():
        if count > 5:
            errors.append(f"Too many hitters from {team}: {count} > 5")

    # Check total player count
    if len(lineup.players) != sum(DK_CLASSIC_SLOTS.values()):
        errors.append(
            f"Wrong player count: {len(lineup.players)} vs {sum(DK_CLASSIC_SLOTS.values())}"
        )

    # Check assignments cover all required slots
    slot_counts = Counter(lineup.assignments.values())
    for slot, required in DK_CLASSIC_SLOTS.items():
        actual = slot_counts.get(slot, 0)
        if actual != required:
            errors.append(f"Slot {slot}: need {required}, have {actual}")

    return errors
