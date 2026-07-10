# How I Built This: Predictive Risk Model for Software Delivery

A step-by-step walkthrough of the process — what I did, and why each
decision mattered. Built end-to-end in Python (pandas, numpy,
matplotlib).

## 1. Defining the question

Before writing any code, I framed the business question: *"Can we
predict which tickets are at risk of delaying a sprint, using only
data Jira already collects?"* This matters because most teams treat
Jira as a task tracker, not a dataset — the goal here was to prove it
can also answer *why* delivery slips, not just *what* is late.

## 2. Data collection

I worked from a changelog export (ticket type, story points, dev
time, QA rejections, staging time, sprint) covering 210 tickets across
12 sprints. In a real engagement this comes straight from Jira's API
or changelog export; here I generated a synthetic version with the
same structure so the process is fully reproducible without exposing
a client's data.

## 3. Data cleaning

Raw exports are never analysis-ready. I went through the data
systematically and fixed, in order:

- **Inconsistent categorical values** (`bug` vs `Bug`) — standardized
  so `groupby` operations don't silently split one category into two.
- **Duplicate rows** from a simulated sync issue — removed, since they
  would double-count tickets and bias every average.
- **Impossible values** (negative durations) — treated as missing
  rather than clamped to zero, then imputed using the median *for that
  ticket's story-point tier*, because testing time correlates strongly
  with story points; a single global median would have been less
  accurate.
- **Missing values** in `staging_days` and `assignee` — imputed or
  filled with `"Unassigned"` rather than dropping rows, because
  dropping would have thrown away otherwise-valid timing data.

Why this step matters: any metric built on top of dirty data is wrong
regardless of how good the analysis is. This is usually 60-70% of the
real work on a project like this, and it's the part that's easiest to
skip when you're in a hurry — which is exactly when it causes the
most damage down the line.

## 4. Exploratory data analysis (EDA)

I started by grouping tickets by `story_points` and comparing average
testing time per tier. This surfaced a pattern that wouldn't be
visible in a simple "average testing time" KPI: 5-point tickets were
taking **33% longer** in testing than 8-point tickets — the opposite
of what you'd expect, since bigger tickets should take longer.
Digging into *why* (not just reporting the number) led to the root
cause: 5-point tickets were skipping an intermediate code-review step
in the workflow.

**Why this matters:** a single blended "average testing time" metric
would have hidden this completely. Segmenting by story points is what
turned a vague "testing is slow sometimes" complaint into a specific,
actionable fix (add the missing review step for 5-point tickets).

## 5. Quantifying the relationship (regression)

Instead of eyeballing "more rejections = more delay," I fit a linear
regression with an interaction term:

```
testing_days ~ intercept + base_component + rejections + (base_component × rejections)
```

using `numpy.linalg.lstsq`. This let me state the relationship as a
precise, defensible formula:

```
Total Time in Testing ≈ Base Testing Time × (1 + 0.8 × Rejections)
```

**Why this matters:** "rejections cause delays" is an opinion. A
fitted coefficient is a number a stakeholder can plan around — for
example, estimating in advance how much a ticket with 2 known
rejections will realistically cost the sprint, instead of finding out
when it's already late.

## 6. Building the risk-alert rule

From the regression and the data distribution, I defined a simple,
explainable rule: tickets with more than 2 rejections are flagged
`high_risk`. I deliberately kept this rule simple (a threshold, not a
black-box model) because in a real Sprint Planning meeting, a tech
lead needs to trust and act on a flag immediately — a rule they can
explain in one sentence is more useful than a marginally more accurate
model nobody can interpret under time pressure.

## 7. Visualizing the results

I built a 4-panel dashboard (matplotlib) rather than a single chart,
because each panel answers a different stakeholder question:

- **Panel 1** (story points vs. testing time): answers *"where is the
  process broken?"* for engineering leads.
- **Panel 2** (rejections vs. testing time + trend line): answers
  *"how much does quality control cost us in time?"* for anyone
  estimating sprint capacity.
- **Panel 3** (staging bottleneck distribution): answers *"is this a
  code problem or a deployment problem?"* — a distinction that
  determines who should own the fix.
- **Panel 4** (KPI summary): the 10-second version for anyone who
  doesn't have time to read the other three panels.

## Key takeaway

The most valuable output of this project wasn't a chart — it was a
formula stakeholders could use to make decisions *before* a delay
happened, instead of explaining it after the fact.
