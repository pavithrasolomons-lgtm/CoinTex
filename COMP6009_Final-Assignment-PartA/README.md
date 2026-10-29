# COMP6009 Final Assignment Part A: CoinTex

**Colombo-group-08:** Pavithra Solomons (24079123), Ravindu Chandrasekara (24100201), Rashmika Dilshan (20543909)

This folder has our work for Part A of the COMP6009 final assignment. We built a new danger-aware A* agent for CoinTex and compared it with the original greedy agent and the CoinTex genetic algorithm (GA) on all 24 levels.

## Contents

| Item | What it is |
|:---|:---|
| `Report - COMP6009_Final-Assignment-PartA_Colombo-group-08.docx` | Our report |
| `Group Presentation - COMP6009_Final-Assignment-PartA_Colombo-group-08.pptx` | Our presentation slides |
| [`code`](code) | The code we wrote, with a README on how to set it up and run it |
| [`results`](results) | Results from all 360 games (CSV files) and charts |

## Results

360 games in the real game: 3 agents, 24 levels and 5 runs per level.

| | Greedy | GA | A* (ours) |
|:---|---:|---:|---:|
| Games won (of 120) | 96 | 99 | 120 |
| Success rate | 80.0% | 82.5% | 100.0% |
| Levels completed (of 24) | 21 | 21 | 24 |
| Deaths | 24 | 21 | 0 |
| RACR (our new metric) | 1.48 | 0.81 | 3.10 |

RACR (risk-adjusted collection rate) = coins collected per second × fraction of life left. It is the new metric we proposed.

## How to run

Copy the files in [`code`](code) into `PlayerGA/TutorialProject`, set up the PlayerGA virtual environment and run `main_partA.py`. The steps are in [code/README.md](code/README.md).
