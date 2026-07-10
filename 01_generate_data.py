"""
01_generate_data.py
--------------------
Generates a synthetic dataset that simulates a Jira changelog export
for a software team: 90 features + 120 bugs across 12 sprints.

Why synthetic data?
This project reproduces, with sample data, the same kind of analysis
performed on a real team's Jira history. Using synthetic data keeps
the client's real information private while preserving the exact
statistical patterns needed to demonstrate the full analysis pipeline.

Columns generated:
- ticket_id: unique identifier (FEAT-### / BUG-###)
- type: "Feature" or "Bug"
- sprint: sprint number (1-12)
- story_points: Fibonacci-like scale (1, 2, 3, 5, 8, 13)
- dev_days: days spent in development
- testing_days: days spent in QA/testing (target variable, includes rework)
- rejections: number of times QA sent the ticket back to development
- staging_days: days stuck in "Pending Staging" (deployment) status
- assignee: synthetic developer name
- created_date / resolved_date: synthetic timestamps

Injected patterns (on purpose, to be discovered during EDA):
1. Tickets with 5 story points take proportionally LONGER in testing
   than 8-point tickets, because 5-point tickets skip an intermediate
   code review step in this simulated team's workflow.
2. Each QA rejection adds ~1.6 days of rework overhead to testing_days,
   compounding roughly 0.8x per rejection (this is the relationship the
   analysis later "discovers" and expresses as a formula).
3. ~12% of tickets get randomly stuck in "Pending Staging" for extra
   days, simulating an intermittent deployment bottleneck.
4. A handful of dirty-data issues are injected on purpose so the
   cleaning step (02_clean_data.py) has real work to do:
   - a few negative/impossible durations
   - a few missing values in staging_days and assignee
   - a few duplicate ticket rows
   - inconsistent text casing in "type"
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

N_FEATURES = 90
N_BUGS = 120
N_TOTAL = N_FEATURES + N_BUGS

STORY_POINTS = [1, 2, 3, 5, 8, 13]
SP_WEIGHTS = [0.10, 0.18, 0.22, 0.22, 0.18, 0.10]

DEVS = ["A. Torres", "M. Gimenez", "R. Fernandez", "L. Suarez",
        "J. Molina", "C. Herrera", "P. Alvarez", "N. Rios"]

rows = []
ticket_counter = {"Feature": 1, "Bug": 1}

for i in range(N_TOTAL):
    is_feature = i < N_FEATURES
    ttype = "Feature" if is_feature else "Bug"
    prefix = "FEAT" if is_feature else "BUG"
    ticket_id = f"{prefix}-{ticket_counter[ttype]:03d}"
    ticket_counter[ttype] += 1

    sprint = int(RNG.integers(1, 13))
    story_points = int(RNG.choice(STORY_POINTS, p=SP_WEIGHTS))

    # Base development time scales with story points
    dev_days = max(0.5, RNG.normal(loc=story_points * 0.7, scale=1.0))

    # Number of QA rejections (Poisson-distributed, most tickets pass clean)
    rejections = int(RNG.poisson(lam=0.6))
    rejections = min(rejections, 5)

    # Base testing time scales with story points EXCEPT for the
    # 5-point "sweet spot of risk" anomaly: these skip a code review
    # step in this team's process, so they run proportionally longer.
    base_testing = story_points * 0.55
    if story_points == 5:
        base_testing *= 2.1  # anomaly: under-reviewed tier

    # Compound delay formula (ground truth being simulated):
    # Total Testing Time ≈ Base Testing Time * (1 + 0.8 * rejections)
    # plus ~1.6 days of rework overhead per rejection, with noise.
    rejection_overhead = rejections * 1.6
    testing_days = base_testing * (1 + 0.8 * rejections) + rejection_overhead
    testing_days += RNG.normal(0, 0.6)
    testing_days = max(0.3, testing_days)

    # Staging bottleneck: ~12% chance of getting stuck in deployment
    stuck_in_staging = RNG.random() < 0.12
    staging_days = float(RNG.uniform(2, 6)) if stuck_in_staging else float(RNG.uniform(0.1, 1.0))

    assignee = RNG.choice(DEVS)

    created_offset = sprint * 14 + int(RNG.integers(0, 5))
    created_date = pd.Timestamp("2025-01-06") + pd.Timedelta(days=created_offset)
    resolved_date = created_date + pd.Timedelta(days=float(dev_days + testing_days + staging_days))

    rows.append({
        "ticket_id": ticket_id,
        "type": ttype,
        "sprint": sprint,
        "story_points": story_points,
        "dev_days": round(dev_days, 2),
        "testing_days": round(testing_days, 2),
        "rejections": rejections,
        "staging_days": round(staging_days, 2),
        "assignee": assignee,
        "created_date": created_date.date().isoformat(),
        "resolved_date": resolved_date.date().isoformat(),
    })

df = pd.DataFrame(rows)

# --- Inject realistic "dirty data" issues on purpose ---

# 1. A few negative/impossible durations (data entry errors)
dirty_idx = RNG.choice(df.index, size=4, replace=False)
df.loc[dirty_idx[:2], "testing_days"] = -1.0
df.loc[dirty_idx[2:], "dev_days"] = -0.5

# 2. Missing values in staging_days and assignee
missing_idx = RNG.choice(df.index, size=10, replace=False)
df.loc[missing_idx[:6], "staging_days"] = np.nan
df.loc[missing_idx[6:], "assignee"] = None

# 3. Duplicate rows (same ticket logged twice due to a sync bug)
dupes = df.sample(5, random_state=1)
df = pd.concat([df, dupes], ignore_index=True)

# 4. Inconsistent text casing in "type"
case_idx = RNG.choice(df.index, size=8, replace=False)
df.loc[case_idx, "type"] = df.loc[case_idx, "type"].str.lower()

df = df.sample(frac=1, random_state=7).reset_index(drop=True)

df.to_csv("data/jira_changelog_raw.csv", index=False)
print(f"Generated {len(df)} rows (with intentional data-quality issues) -> data/jira_changelog_raw.csv")
print(df.head(10))
