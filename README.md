# Predictive Risk Model for Software Delivery

Python project that mines a software team's Jira changelog history to
detect delivery-risk patterns and build a simple, explainable
predictive risk model — inspired by real production work, rebuilt here
on a **synthetic dataset** so it can be shared publicly without
exposing any client's real data.

## What this project does

1. **Generates** a realistic Jira changelog export (`01_generate_data.py`)
2. **Cleans** the data, fixing realistic data-quality problems (`02_clean_data.py`)
3. **Analyzes** it to find delay patterns and build a compound-delay
   formula, then renders a dashboard (`03_analysis_and_dashboard.py`)

Run in order:

```bash
pip install pandas numpy matplotlib
python 01_generate_data.py
python 02_clean_data.py
python 03_analysis_and_dashboard.py
```

## Dataset

210 tickets (90 Features + 120 Bugs) across 12 sprints, with columns:

| Column | Description |
|---|---|
| `ticket_id` | Jira key (FEAT-### / BUG-###) |
| `type` | Feature or Bug |
| `sprint` | Sprint number (1-12) |
| `story_points` | Estimation size (1, 2, 3, 5, 8, 13) |
| `dev_days` | Days spent in development |
| `testing_days` | Days spent in QA/testing (target variable) |
| `rejections` | Times QA sent the ticket back to development |
| `staging_days` | Days stuck in "Pending Staging" (deployment) |
| `assignee` | Developer |
| `created_date` / `resolved_date` | Timestamps |

## Data cleaning (`02_clean_data.py`)

Every step is printed to the console with a before/after count so the
impact is auditable, not just claimed:

1. **Casing standardization** — `type` values normalized to Title Case
   (`bug` → `Bug`), since different Jira automation rules exported the
   field inconsistently.
2. **Duplicate removal** — exact duplicate tickets (double-logged by a
   simulated webhook sync issue) dropped by `ticket_id` + `sprint` +
   `story_points`.
3. **Invalid durations** — negative `dev_days` / `testing_days` values
   (impossible data-entry errors) are treated as missing, then imputed
   using the **median for that ticket's story-point tier** — more
   accurate than a single global median because testing time scales
   strongly with story points.
4. **Missing `staging_days`** — imputed with the median staging time
   **for that ticket's type** (Bug vs Feature), since the two follow
   slightly different release trains.
5. **Missing `assignee`** — filled with `"Unassigned"` rather than
   dropped, since the timing data is still valid for the delay
   analysis even without a known owner.
6. **Date sanity check** — confirms `resolved_date` is always after
   `created_date`; any violation would be dropped (none occurred in
   this run).
7. **Derived column** — `total_cycle_days = dev_days + testing_days + staging_days`.

## Analysis & metrics (`03_analysis_and_dashboard.py`)

### 1. "Sweet Spot of Risk" — story points vs. testing time

Groups tickets by `story_points` and compares average `testing_days`.
5-point tickets take **~30-35% longer in testing** than 8-point
tickets on average — a real anomaly this simulates: in the underlying
workflow, 5-point tickets skip an intermediate code-review step that
larger tickets go through by default, so defects surface later, during QA.

### 2. Compound Delay Formula

The core empirical relationship the analysis is built around:

```
Total Time in Testing ≈ Base Testing Time × (1 + 0.8 × Rejections)
```

This is **not assumed** — it's recovered directly from the data using
an OLS regression with an interaction term:

```
testing_days ~ intercept + b1·base_component + b2·rejections + b3·(base_component × rejections)
```

solved via `numpy.linalg.lstsq` (normal equations) on:

```python
X = [1, base_component, rejections, base_component * rejections]
y = testing_days
```

The fitted interaction coefficient (`b3`) lands close to **0.8**,
confirming each additional rejection scales the base testing time by
roughly that factor. The marginal cost of one extra rejection is then
computed properly as `b2 + b3 × mean(base_component)` — the correct
way to read a marginal effect out of an interaction model, rather than
a naive `(testing_days - base) / rejections` quotient, which would
double-count the multiplicative term.

### 3. Staging (deployment) bottleneck

Tickets with `staging_days` above the **90th percentile** are flagged
as stuck in the "Pending Staging" bottleneck — about 10% of tickets in
this run.

### 4. Risk alert rule

Any ticket with **more than 2 QA rejections** is flagged `high_risk`
— this mirrors the automated Jira Rovo trigger from the original
production version of this workflow: proactively alerting tech leads
before a delay lands, using JQL to scan the backlog in real time
during sprint planning.

## Dashboard

`dashboard/jira_risk_dashboard.png` — 4 panels:

1. Average testing time by story-point tier (anomaly highlighted)
2. Rejections vs. testing time scatter + trend line (compound delay)
3. Staging-days distribution with the 90th-percentile bottleneck line
4. KPI summary panel (totals, risk %, bottleneck %, formula outputs)

## Files

```
project1_jira_risk_model/
├── 01_generate_data.py
├── 02_clean_data.py
├── 03_analysis_and_dashboard.py
├── data/
│   ├── jira_changelog_raw.csv
│   ├── jira_changelog_clean.csv
│   ├── jira_changelog_analyzed.csv
│   └── summary_story_points.csv
├── dashboard/
│   └── jira_risk_dashboard.png
└── README.md
```

## Disclaimer

All data in this project is synthetically generated (`numpy` random
generator, fixed seed for reproducibility) to demonstrate the analysis
methodology. No real company, team, or individual's data is used.
