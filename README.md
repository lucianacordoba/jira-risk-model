# Predictive Risk Model for Software Delivery

I built this to answer a question that kept coming up on a team I worked with: which Jira tickets are actually going to blow past their estimate, and can you tell before it happens instead of after.

It's a synthetic version of a real analysis — same methodology, same kind of findings, just rebuilt on generated data so I can share it without touching a client's actual Jira history.

## What it does

1. `01_generate_data.py` builds a synthetic Jira changelog: 210 tickets, 90 features and 120 bugs, spread across 12 sprints.
2. `02_clean_data.py` fixes the data-quality problems any real export has.
3. `03_analysis_and_dashboard.py` finds the delay pattern, fits a formula for it, and renders a dashboard.

```bash
pip install pandas numpy matplotlib
python 01_generate_data.py
python 02_clean_data.py
python 03_analysis_and_dashboard.py
```

## The dataset

| Column | What it is |
|---|---|
| `ticket_id` | Jira key (FEAT-### / BUG-###) |
| `type` | Feature or Bug |
| `sprint` | Sprint number, 1 through 12 |
| `story_points` | 1, 2, 3, 5, 8, 13 |
| `dev_days` | Days in development |
| `testing_days` | Days in QA — the number I'm trying to explain |
| `rejections` | Times QA sent it back |
| `staging_days` | Days stuck waiting for deployment |
| `assignee` | Developer |
| `created_date` / `resolved_date` | Timestamps |

## Cleaning it up

Nothing here is analysis-ready straight out of the export:

- `type` had inconsistent casing (`bug` vs `Bug`), normalized to Title Case — otherwise a groupby splits one category into two without telling you.
- A handful of duplicate rows, from a simulated sync glitch, get dropped.
- Some `dev_days` / `testing_days` values come in negative, which is obviously wrong. I don't just clamp those to zero — I treat them as missing and fill them with the median for that ticket's story-point tier, since testing time depends heavily on ticket size.
- Missing `staging_days` gets filled with the median for that ticket type (bugs and features move through slightly different release paths).
- Missing `assignee` becomes `"Unassigned"` instead of getting dropped. The timing data is still useful even without knowing who worked it.
- Quick sanity check that `resolved_date` never lands before `created_date`.

## What the analysis found

5-point tickets take about a third longer in testing than 8-point tickets. That's backwards — bigger tickets should take longer, not less. Digging into why, it turns out 5-point tickets skip an intermediate code review step that larger tickets go through automatically. A single blended "average testing time" would never surface this; it only shows up once you split by story points.

From there I fit a regression (testing_days on ticket size and rejections, with an interaction term, solved with `numpy.linalg.lstsq` — no sklearn, so the math stays visible) to turn "rejections cause delays" into something a team can actually plan around:

```
Total Time in Testing ≈ Base Testing Time × (1 + 0.8 × Rejections)
```

The interaction coefficient lands close to 0.8, confirming each rejection isn't adding a fixed chunk of time, it's multiplying the base time. Anything stuck above the 90th percentile in `staging_days` gets flagged as a deployment bottleneck (about 1 in 10 tickets), and anything with more than 2 rejections gets a `high_risk` flag — simple enough that a tech lead can act on it in a planning meeting without needing to trust a black box.

## Dashboard

`dashboard/jira_risk_dashboard.png`, four panels: testing time by story-point tier, rejections vs. testing time with the trend line, the staging bottleneck distribution, and a KPI summary.

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

## A note on the data

Everything here is generated with numpy, fixed seed, so it's reproducible. No real company's data is in this repo.
