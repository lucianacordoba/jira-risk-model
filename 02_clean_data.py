"""
02_clean_data.py
-----------------
Cleans the raw Jira changelog export (data/jira_changelog_raw.csv).

Cleaning steps performed (in order), each printed with a before/after
row count so the impact of every step is auditable:

1. Standardize the "type" column casing (Feature/Bug), since Jira
   exports sometimes mix case depending on the automation rule that
   created the ticket.
2. Drop exact duplicate rows (tickets double-logged by a webhook sync
   issue between Jira and the reporting pipeline).
3. Fix impossible negative durations in dev_days / testing_days.
   These are data-entry/timestamp errors (e.g. a ticket reopened and
   the timer logic went negative). We treat them as missing rather
   than guessing a value, then impute using the median for that
   ticket's story-point tier (more robust than a global median).
4. Impute missing staging_days using the median staging time for that
   ticket's type (Feature vs Bug), since Bugs and Features go through
   slightly different release trains in this simulated workflow.
5. Fill missing assignee with "Unassigned" rather than dropping the
   row, since the ticket's timing data is still valid and useful for
   the delay analysis even without a known owner.
6. Cast dates to datetime and sanity-check that resolved_date is
   always after created_date.
7. Add a derived column `total_cycle_days` = dev_days + testing_days +
   staging_days, used throughout the rest of the analysis.

Output: data/jira_changelog_clean.csv
"""

import numpy as np
import pandas as pd

df = pd.read_csv("data/jira_changelog_raw.csv")
n_start = len(df)
print(f"Raw rows: {n_start}")

# 1. Standardize casing
df["type"] = df["type"].str.strip().str.title()
assert set(df["type"].unique()) <= {"Feature", "Bug"}

# 2. Drop exact duplicates
before = len(df)
df = df.drop_duplicates(subset=["ticket_id", "sprint", "story_points"], keep="first")
print(f"Removed {before - len(df)} duplicate rows")

# 3. Fix impossible negative durations -> treat as missing, impute by
#    median within the same story_points tier
for col in ["dev_days", "testing_days"]:
    n_bad = (df[col] < 0).sum()
    df.loc[df[col] < 0, col] = np.nan
    medians_by_sp = df.groupby("story_points")[col].transform("median")
    df[col] = df[col].fillna(medians_by_sp)
    print(f"Fixed {n_bad} negative/invalid values in '{col}' "
          f"(imputed with median for that story-point tier)")

# 4. Impute missing staging_days by median per ticket type
n_missing_staging = df["staging_days"].isna().sum()
medians_by_type = df.groupby("type")["staging_days"].transform("median")
df["staging_days"] = df["staging_days"].fillna(medians_by_type)
print(f"Imputed {n_missing_staging} missing 'staging_days' values "
      f"(median per ticket type)")

# 5. Fill missing assignee
n_missing_assignee = df["assignee"].isna().sum()
df["assignee"] = df["assignee"].fillna("Unassigned")
print(f"Filled {n_missing_assignee} missing 'assignee' values with 'Unassigned'")

# 6. Dates + sanity check
df["created_date"] = pd.to_datetime(df["created_date"])
df["resolved_date"] = pd.to_datetime(df["resolved_date"])
bad_dates = (df["resolved_date"] < df["created_date"]).sum()
if bad_dates:
    df = df[df["resolved_date"] >= df["created_date"]]
print(f"Removed {bad_dates} rows with resolved_date before created_date")

# 7. Derived column
df["total_cycle_days"] = (df["dev_days"] + df["testing_days"] + df["staging_days"]).round(2)

df = df.reset_index(drop=True)
df.to_csv("data/jira_changelog_clean.csv", index=False)

print(f"\nClean rows: {len(df)} (started with {n_start})")
print(df.select_dtypes(include=[np.number]).describe().round(2))
