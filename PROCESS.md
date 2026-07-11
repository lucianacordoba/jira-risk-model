# How I Built This: Predictive Risk Model for Software Delivery

## Starting point

Most teams treat Jira as a place to track tasks, not as a dataset. I wanted to see if the changelog data it already collects — story points, rejections, time in each status — was enough to predict which tickets would blow up a sprint, without adding any new tracking.

## Getting the data

In the real version of this, the data came straight from a Jira export. Here I generated a synthetic dataset with the same shape and the same statistical relationships (210 tickets, 12 sprints) so I could share the whole pipeline without touching anyone's actual Jira history.

## Cleaning

Usually the least glamorous part of a project like this, and also the part that decides whether anything downstream is trustworthy. I standardized inconsistent category casing, dropped duplicate rows from a simulated sync bug, and — the part I think matters most — treated negative durations as missing instead of zeroing them out, then filled them with the median for that ticket's story-point tier rather than one global median, since testing time scales a lot with size.

## Finding the pattern

I grouped by story points and compared average testing time per tier. 5-point tickets came out taking about 33% longer than 8-point tickets, the opposite of what you'd expect. That sent me looking for a cause instead of just reporting the anomaly, and it turned out 5-point tickets were skipping a code review step that bigger tickets went through by default. A blended average across all tickets would have hidden this completely.

## Turning it into a formula

"Rejections slow things down" isn't something a team can plan around. I fit a regression with an interaction term — testing_days as a function of ticket size and rejections, solved by hand with `numpy.linalg.lstsq` rather than a library, so the mechanics stay visible — and got:

```
Total Time in Testing ≈ Base Testing Time × (1 + 0.8 × Rejections)
```

Now a tech lead can look at a ticket with two known rejections and estimate what it's actually going to cost the sprint, instead of finding out once it's already late.

## The risk rule

I kept the actual alert simple on purpose: more than 2 rejections gets flagged high-risk. A slightly more accurate model nobody can explain in a planning meeting isn't more useful than a threshold everyone trusts.

## The dashboard

Four panels instead of one chart, because engineering leads, capacity planners, and whoever owns the deployment pipeline are asking different questions, and one chart can't answer all three at once.

## What I'd call the actual result

The formula, not the chart. Being able to say "this ticket will cost you X extra days" before the sprint is already behind is worth more than any visualization of what already happened.
