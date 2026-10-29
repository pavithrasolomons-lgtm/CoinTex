# COMP6009 Final Assignment Part A: Results

**Colombo-group-08:** Pavithra Solomons (24079123), Ravindu Chandrasekara (24100201), Rashmika Dilshan (20543909)

Author: Pavithra Solomons  
Last modified by: Pavithra Solomons  
Last modified: 29 October 2026

These are the results behind Part 3 of our report. All three agents played the real CoinTex game: the original greedy agent, the CoinTex tutorial GA and our danger-aware A* agent. Each played 24 levels with 5 runs per level, so 360 games in total, with a 60 second limit per game. The code is in the [code](../code) folder and the report and slides are in the [main folder](..).

## Files

| File | What it is |
|:---|:---|
| `partA_results.csv` | One row per game (360 rows). This is the data used in the report |
| `partA_summary.csv` | One row per agent |
| `partA_per_level.csv` | One row per agent and level |
| `partA_results_raw_run1.csv` | Our first batch, before we removed the duplicated greedy rows for levels 1 to 3. Kept as a record, not used in the report |
| `chart_overall.png` | Success rate, completion time and RACR for each agent |
| `chart_success_by_level.png` | Success rate on each level |
| `chart_time_by_level.png` | Completion time on each level |

## Summary

| | Greedy | GA | A* (ours) |
|:---|---:|---:|---:|
| Games won (of 120) | 96 | 99 | 120 |
| Success rate | 80.0% | 82.5% | 100.0% |
| Levels completed (of 24) | 21 | 21 | 24 |
| Deaths | 24 | 21 | 0 |
| RACR (our new metric) | 1.48 | 0.81 | 3.10 |

RACR (risk-adjusted collection rate) = coins collected per second × fraction of life left. It is the new metric we proposed.

## Columns in partA_results.csv

| Column | Meaning |
|:---|:---|
| `agent` | greedy, ga or astar |
| `level`, `run` | level number (1 to 24) and run number (1 to 5) |
| `coins_total`, `coins_collected`, `collection_rate` | coins in the level, coins collected and the fraction collected |
| `success` | 1 if every coin was collected |
| `died`, `timed_out` | 1 if the character died, or ran out of time (60 s) |
| `time_s` | seconds from level start to the last coin, or to death |
| `life_remaining` | life left at the end, from 0 to 1 |
| `hits` | frames where the character touched a monster or fire |
| `moves`, `decisions` | jumps made and decisions taken |
| `mean_plan_ms` | A* planning time per decision in milliseconds (blank for greedy and GA) |
| `timestamp` | when the game finished |
